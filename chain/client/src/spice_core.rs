use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use itertools::Itertools;
use near_async::messaging::{CanSend as _, Sender};
use near_chain::chain::{NewChunkData, StorageContext};
use near_chain::sharding::shuffle_receipt_proofs;
use near_chain::stateless_validation::chunk_validation::{
    MainStateTransitionCache, MainTransition, PreValidationOutput, get_resharding_transition,
    validate_chunk_state_witness,
};
use near_chain::types::{RuntimeAdapter, StorageDataSource};
use near_chain::{Block, Chain, ChainStore, Error};
use near_epoch_manager::EpochManagerAdapter;
use near_epoch_manager::shard_assignment::shard_id_to_uid;
use near_network::types::{NetworkRequests, PeerManagerMessageRequest};
use near_primitives::hash::{CryptoHash, hash};
use near_primitives::merkle::merklize;
use near_primitives::receipt::Receipt;
use near_primitives::sharding::{ChunkHash, ReceiptProof, ShardChunkHeader};
use near_primitives::spice::ExecutionResult;
use near_primitives::stateless_validation::chunk_endorsement::ChunkEndorsement;
use near_primitives::stateless_validation::state_witness::ChunkStateWitness;
use near_primitives::types::ShardId;
use near_primitives::types::chunk_extra::ChunkExtra;
use near_primitives::validator_signer::ValidatorSigner;
use near_store::adapter::StoreAdapter as _;
use near_store::{PartialStorage, Store, get_genesis_state_roots};

use crate::chunk_executor_actor::{ChunkExecutorAdapter, ExecutorAllExecutionResultsAvailable};
use crate::stateless_validation::validate::validate_chunk_endorsement;

#[derive(Clone)]
pub struct CoreStatementsProcessor(Arc<RwLock<CoreStatementsTracker>>);

// FIXME(spice): one idea is to have a separarte abstraction with read-only access for places that
// shouldn't be able to modify core state. (To potentially limit core state modification to a
// single agent which may make things a bit simpler to reason about, no write race conditions.)
impl CoreStatementsProcessor {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(CoreStatementsTracker::new())))
    }

    pub fn write(&self) -> RwLockWriteGuard<CoreStatementsTracker> {
        self.0.write().unwrap()
    }

    pub fn read(&self) -> RwLockReadGuard<CoreStatementsTracker> {
        self.0.read().unwrap()
    }
}

/// Tracks core statements that node knows about. This includes core statements that correspond to
/// core state as well as additional core statements that may not be included in the core state.
/// Core state is based purely on core statement we include in blocks. However, we need to make
/// decisions about endorsement before blocks are producers to allow execution to catch up to
/// consensus.
// FIXME(spice): Store in storage, instead of in-memory.
pub struct CoreStatementsTracker {
    block_execution_results: HashMap<CryptoHash, HashMap<ChunkHash, ExecutionResult>>,
    // FIXME(spice): Record endorsements & record execution result only after enough endorsements
    // are present.
    // block -> chunk hash -> signature of execution result.
    // execution_results_endorsements: HashMap<CryptoHash, HashMap<ChunkHash, Vec<Endorsement>>>,

    // FIXME(spice): For each block record pending (not-included in any previous block)
    // endorsements. Have API to retrieve next endorsements & record_block to update this
    // structure. Each endorsement included in the block would have to include information about
    // associated chunk.
}

impl CoreStatementsTracker {
    pub fn new() -> Self {
        Self { block_execution_results: HashMap::new() }
    }

    pub fn record_execution_result(&mut self, execution_result: ExecutionResult) {
        self.block_execution_results
            .entry(execution_result.block_hash)
            .or_default()
            .insert(execution_result.chunk_hash.clone(), execution_result);
    }

    fn get_execution_results(
        &self,
        block_hash: &CryptoHash,
    ) -> Result<&HashMap<ChunkHash, ExecutionResult>, Error> {
        self.block_execution_results
            .get(block_hash)
            .ok_or_else(|| Error::Other(format!("no execution results exist for {block_hash}")))
    }
}

// FIXME(spice): This may need refactoring and more thought about the structure.
impl CoreStatementsProcessor {
    pub fn process_chunk_endorsement(
        &self,
        epoch_manager: &dyn EpochManagerAdapter,
        endorsement: ChunkEndorsement,
        store: &Store,
        chunk_executor_adapter: &ChunkExecutorAdapter,
    ) -> Result<(), Error> {
        // FIXME(spice): same as in  chunk_endorsement.rs can have some logic/cache to avoid
        // processing the same endorsement twice based on:
        // let key = endorsement.chunk_production_key();
        // let account_id = endorsement.account_id();

        tracing::debug!(target: "spice_core", ?endorsement, "process_chunk_endorsement");

        if !validate_chunk_endorsement(epoch_manager, &endorsement, store)? {
            tracing::debug!(target: "spice_core", ?endorsement, "invalid endorsement");
            return Err(Error::Other(format!("received invalid endorsement")));
        }

        let ChunkEndorsement::V3(endorsement) = endorsement else {
            return Err(Error::Other(format!("received non-spice endorsement")));
        };
        // FIXME(spice): Record endorsements themselves (core statements).

        let execution_result = endorsement.into_execution_result();

        // FIXME(spice): Record execution result only after enough endorsements are available.
        self.record_execution_result(execution_result, chunk_executor_adapter);

        Ok(())
    }

    pub fn record_execution_result(
        &self,
        execution_result: ExecutionResult,
        chunk_executor_adapter: &ChunkExecutorAdapter,
    ) {
        let block_hash = execution_result.block_hash;

        let mut core_statements = self.write();
        core_statements.record_execution_result(execution_result);

        // FIXME(spice): Before sending this message, make a check here to make sure enough
        // execution results are present.
        chunk_executor_adapter.send(ExecutorAllExecutionResultsAvailable { block_hash });
    }

    // FIXME(spice): potentially rename to better reflect special genesis handling
    pub fn do_all_execution_results_exist(
        &self,
        block_hash: &CryptoHash,
        store: &ChainStore,
    ) -> Result<bool, Error> {
        let block = store.get_block(block_hash)?;
        // For genesis we don't use execution results so we can do execution with no execution
        // results.
        if block.header().is_genesis() {
            return Ok(true);
        }
        let core_statements = self.read();
        let results = core_statements.get_execution_results(block_hash)?;
        for chunk in block.chunks().iter_raw() {
            if results.get(&chunk.chunk_hash()).is_none() {
                return Ok(false);
            }
        }
        assert_eq!(results.len(), block.chunks().len());
        return Ok(true);
    }

    // FIXME(spice): Make async & rename to start.* similar to how witness validation works right now.
    pub fn validate_state_witness_and_send_endorsements(
        &self,
        witness: ChunkStateWitness,
        store: &ChainStore,
        // FIXME(spice): might make sense to have processor own these
        epoch_manager: &dyn EpochManagerAdapter,
        runtime_adapter: &dyn RuntimeAdapter,
        network_sender: &Sender<PeerManagerMessageRequest>,
        signer: &ValidatorSigner,
    ) -> Result<(), Error> {
        let block_hash = witness.main_state_transition.block_hash;

        let (chunk_header, chunk_extra, outgoing_receipts_root) =
            self.validate_state_witness(witness, store, epoch_manager, runtime_adapter)?;

        // FIXME(spice): if we need endorsement, record it here.
        send_chunk_endorsement(
            &chunk_header,
            ExecutionResult {
                block_hash,
                chunk_hash: chunk_header.chunk_hash(),
                shard_id: chunk_header.shard_id(),
                chunk_extra: chunk_extra.into(),
                outgoing_receipts_root,
            },
            epoch_manager,
            network_sender,
            signer,
        )?;

        Ok(())
    }

    pub fn validate_state_witness(
        &self,
        witness: ChunkStateWitness,
        store: &ChainStore,
        epoch_manager: &dyn EpochManagerAdapter,
        runtime_adapter: &dyn RuntimeAdapter,
    ) -> Result<(ShardChunkHeader, ChunkExtra, CryptoHash), Error> {
        let block = store.get_block(&witness.main_state_transition.block_hash)?;
        let prev_block_header = store.get_block_header(block.header().prev_hash())?;
        let core_statements = self.read();
        let prev_execution_results = if prev_block_header.is_genesis() {
            &HashMap::new()
        } else {
            core_statements.get_execution_results(block.header().prev_hash())?
        };

        // If we don't have all previous execution results we cannot do validation.
        if !self.do_all_execution_results_exist(block.header().prev_hash(), store)? {
            // FIXME(spice): Don't rely on DBNotFoundErr, but have neccessary checks on the caller side explicitly.
            // DBNotFoundErr to allow orphan handling up the stack
            return Err(Error::DBNotFoundErr(format!(
                "missing previous execution results required for chunk validation for block_hash={}",
                block.header().prev_hash()
            )));
        }

        let pre_validation_result = pre_validate_chunk_state_witness(
            &witness,
            &block,
            &prev_execution_results,
            epoch_manager,
            store,
        )?;
        let chunk_header = witness.chunk_header.clone();
        let (chunk_extra, outgoing_receipts_root) = validate_chunk_state_witness(
            witness,
            pre_validation_result,
            epoch_manager,
            runtime_adapter,
            &MainStateTransitionCache::default(),
        )?;
        Ok((chunk_header, chunk_extra, outgoing_receipts_root))
    }
}

fn send_chunk_endorsement(
    chunk_header: &ShardChunkHeader,
    execution_result: ExecutionResult,
    epoch_manager: &dyn EpochManagerAdapter,
    network_sender: &Sender<PeerManagerMessageRequest>,
    signer: &ValidatorSigner,
) -> Result<(), Error> {
    let epoch_id =
        epoch_manager.get_epoch_id_from_prev_block(chunk_header.prev_block_hash()).unwrap();

    // FIXME(spice): May need to send to validators from the next epoch as well
    // (since this execution may finish after epoch switch).

    // Everyone should be aware of all core statements.
    let validators = epoch_manager.get_epoch_all_validators(&epoch_id)?;
    let endorsement = ChunkEndorsement::new_with_execution_result(
        epoch_id,
        chunk_header,
        execution_result,
        signer,
    );
    for validator_stake in validators {
        let account = validator_stake.destructure().0;
        if &account == signer.validator_id() {
            continue;
        }
        network_sender.send(PeerManagerMessageRequest::NetworkRequests(
            NetworkRequests::ChunkEndorsement(account, endorsement.clone()),
        ));
    }
    Ok(())
}

// FIXME(spice): May be able to dedup some of this with current implementation.
fn pre_validate_chunk_state_witness(
    state_witness: &ChunkStateWitness,
    block: &Block,
    prev_execution_results: &HashMap<ChunkHash, ExecutionResult>,
    epoch_manager: &dyn EpochManagerAdapter,
    store: &ChainStore,
) -> Result<PreValidationOutput, Error> {
    let epoch_id = epoch_manager.get_epoch_id(block.header().hash())?;

    // FIXME(spice): Don't use asserts; Also in future don't need to pass in epoch_id
    assert_eq!(epoch_id, state_witness.epoch_id);
    let chunk_header = state_witness.chunk_header.clone();
    assert!(block.chunks().iter_raw().contains(&chunk_header));

    // Ensure that the chunk header version is supported in this protocol version
    let protocol_version = epoch_manager.get_epoch_info(&epoch_id)?.protocol_version();
    chunk_header.validate_version(protocol_version)?;

    let prev_block_header = store.get_block_header(block.header().prev_hash())?;

    let shard_id = chunk_header.shard_id();
    let shard_uid = { shard_id_to_uid(epoch_manager, shard_id, &epoch_id)? };

    let implicit_transition_params = if let Some(transition) =
        get_resharding_transition(epoch_manager, block.header(), shard_uid, 0)?
    {
        vec![transition]
    } else {
        Vec::new()
    };

    let receipts_to_apply = validate_source_receipts_proofs(
        &state_witness.source_receipt_proofs,
        prev_execution_results,
        shard_id,
        prev_block_header.hash(),
        &block,
    )?;
    let applied_receipts_hash = hash(&borsh::to_vec(receipts_to_apply.as_slice()).unwrap());
    if applied_receipts_hash != state_witness.applied_receipts_hash {
        return Err(Error::InvalidChunkStateWitness(format!(
            "Receipts hash {:?} does not match expected receipts hash {:?}",
            applied_receipts_hash, state_witness.applied_receipts_hash
        )));
    }

    let (tx_root_from_state_witness, _) = merklize(&state_witness.transactions);
    if chunk_header.tx_root() != tx_root_from_state_witness {
        return Err(Error::InvalidChunkStateWitness(format!(
            "Transaction root {:?} does not match expected transaction root {:?}",
            tx_root_from_state_witness,
            chunk_header.tx_root()
        )));
    }

    let transaction_validity_check_results = state_witness
        .transactions
        .iter()
        .map(|tx| {
            store
                .check_transaction_validity_period(&prev_block_header, tx.transaction.block_hash())
                .is_ok()
        })
        .collect::<Vec<_>>();

    let get_genesis_chunk_extra = || -> Result<ChunkExtra, Error> {
        let epoch_id = prev_block_header.epoch_id();
        let shard_layout = epoch_manager.get_shard_layout(&epoch_id)?;
        // FIXME(spice): get block right away above (instead of getting both header and block).
        let prev_block = store.get_block(prev_block_header.hash())?;
        let congestion_info =
            prev_block.block_congestion_info().get(&shard_id).map(|info| info.congestion_info);
        let genesis_protocol_version = epoch_manager.get_epoch_protocol_version(&epoch_id)?;

        // FIXME(spice): This should be rewritten; would need genesis information here and may be
        // good to dedup parts with implementation used in non-spice validation.
        // Consider if we'd want to have chunk_extra always created in storage on
        // initialization to make logic  here simpler.
        let chunk_extra = {
            let shard_index = shard_layout.get_shard_index(shard_id)?;
            let state_root = *get_genesis_state_roots(&store.store())?
                .ok_or_else(|| Error::Other("genesis state roots do not exist in the db".to_owned()))?
                .get(shard_index)
                .ok_or_else(|| {
                    Error::Other(format!("genesis state root does not exist for shard id {shard_id} shard index {shard_index}"))
                })?;
            // FIXME(spice): don't use a hardcoded constant here.
            let gas_limit = 1000000000000000;
            Chain::create_genesis_chunk_extra(
                &state_root,
                gas_limit,
                genesis_protocol_version,
                congestion_info,
            )
        };
        Ok(chunk_extra)
    };

    let main_transition_params = if block.header().is_genesis() {
        // FIXME(spice): there may be no need to handle genesis here at all. Not sure if we execute
        // genesis. This may be dead code.
        let chunk_extra = get_genesis_chunk_extra()?;
        MainTransition::Genesis { chunk_extra, block_hash: *block.header().hash(), shard_id }
    } else {
        // For correct application we need to convert chunk_header into spice_chunk_header.
        let spice_chunk_header = if prev_block_header.is_genesis() {
            // Cannot use chunk_extra from store since it may not be there if head progresed.
            let chunk_extra = get_genesis_chunk_extra()?;
            chunk_header.clone().into_spice_chunk_execution_header(&chunk_extra, Default::default())
        } else {
            let prev_shard_id = epoch_manager
                .get_prev_shard_id_from_prev_hash(prev_block_header.hash(), shard_id)?
                .1;
            let prev_execution_results = prev_execution_results
                .iter()
                .find(|exec_result| exec_result.1.shard_id == prev_shard_id)
                // FIXME(spice): restructure to avoid unwrap.
                .unwrap()
                .1;
            chunk_header.clone().into_spice_chunk_execution_header(
                &prev_execution_results.chunk_extra,
                prev_execution_results.outgoing_receipts_root,
            )
        };

        let storage_context = if prev_block_header.is_genesis() {
            // FIXME(spice): For some reason, PartialStorage from witness doesn't include genesis.
            // Genesis trie should be available so it may not be a big issue.
            // Would have to double check how the previous implementation worked.
            // It may have relied on chunk_extra being available for genesis so there was never
            // need to transition from genesis to the next block (up the stack).
            StorageContext {
                storage_data_source: StorageDataSource::Db,
                state_patch: Default::default(),
            }
        } else {
            StorageContext {
                storage_data_source: StorageDataSource::Recorded(PartialStorage {
                    nodes: state_witness.main_state_transition.base_state.clone(),
                }),
                state_patch: Default::default(),
            }
        };
        MainTransition::NewChunk(NewChunkData {
            chunk_header: spice_chunk_header,
            transactions: state_witness.transactions.clone(),
            transaction_validity_check_results,
            receipts: receipts_to_apply,
            block: Chain::get_apply_chunk_block_context(&block, &prev_block_header, true)?,
            storage_context,
        })
    };

    Ok(PreValidationOutput { main_transition_params, implicit_transition_params })
}

fn validate_source_receipts_proofs(
    source_receipt_proofs: &HashMap<ChunkHash, ReceiptProof>,
    prev_execution_results: &HashMap<ChunkHash, ExecutionResult>,
    shard_id: ShardId,
    prev_block_hash: &CryptoHash,
    block: &Block,
) -> Result<Vec<Receipt>, Error> {
    if block.header().is_genesis() {
        if !source_receipt_proofs.is_empty() {
            return Err(Error::InvalidChunkStateWitness(format!(
                "genesis source_receipt_proofs should be empty, actual len is {}",
                source_receipt_proofs.len()
            )));
        }
        return Ok(vec![]);
    }

    if prev_execution_results.len() != source_receipt_proofs.len() {
        return Err(Error::InvalidChunkStateWitness(format!(
            "source_receipt_proofs contains incorrect number of proofs. Expected {} proofs, found {}",
            prev_execution_results.len(),
            source_receipt_proofs.len()
        )));
    }

    let mut receipt_proofs = Vec::new();
    for (chunk_hash, prev_execution_result) in prev_execution_results {
        let Some(receipt_proof) = source_receipt_proofs.get(&chunk_hash) else {
            return Err(Error::InvalidChunkStateWitness(format!(
                "Missing source receipt proof for chunk {:?}",
                chunk_hash
            )));
        };

        validate_receipt_proof(receipt_proof, prev_execution_result, shard_id)?;

        receipt_proofs.push(receipt_proof.clone());
    }
    // FIXME(spice): receipt_proofs is random so we order them. A better approach would be above to
    // iterate of chunks from prev block.
    // Right now in executor and here both we rely on this sorting.
    receipt_proofs.sort_by_key(|proof| proof.1.from_shard_id);

    // FIXME(spice): Note - we don't filter here since we expect state witness to contain only proper
    // receipts. If it doesn't witness validation would fail. For early failure might make sense to
    // do filtering here still (same as in non-spice implementation).
    let receipts_shuffle_salt = prev_block_hash;
    shuffle_receipt_proofs(&mut receipt_proofs, receipts_shuffle_salt);
    tracing::debug!(target: "adhoc_receipts", ?prev_block_hash, ?receipts_shuffle_salt, ?receipt_proofs, "validation");
    Ok(receipt_proofs.into_iter().map(|proof| proof.0).flatten().collect())
}

fn validate_receipt_proof(
    receipt_proof: &ReceiptProof,
    prev_execution_result: &ExecutionResult,
    target_shard_id: ShardId,
) -> Result<(), Error> {
    if receipt_proof.1.from_shard_id != prev_execution_result.shard_id {
        return Err(Error::InvalidChunkStateWitness(format!(
            "Receipt proof for chunk {:?} is from shard {}, expected shard {}",
            prev_execution_result.chunk_hash,
            receipt_proof.1.from_shard_id,
            prev_execution_result.shard_id,
        )));
    }
    if receipt_proof.1.to_shard_id != target_shard_id {
        return Err(Error::InvalidChunkStateWitness(format!(
            "Receipt proof for chunk {:?} is for shard {}, expected shard {}",
            prev_execution_result.chunk_hash, receipt_proof.1.to_shard_id, target_shard_id
        )));
    }

    if !receipt_proof.verify_against_receipt_root(prev_execution_result.outgoing_receipts_root) {
        return Err(Error::InvalidChunkStateWitness(format!(
            "Receipt proof for chunk {:?} has invalid merkle path, doesn't match outgoing receipts root",
            prev_execution_result.chunk_hash
        )));
    }

    Ok(())
}
