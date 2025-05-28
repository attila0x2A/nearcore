use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ops::Deref as _;
use std::sync::Arc;

use itertools::Itertools as _;
use lru::LruCache;
use near_async::MultiSend;
use near_async::MultiSenderFrom;
use near_async::messaging::CanSend;
use near_async::messaging::Handler;
use near_async::messaging::IntoSender as _;
use near_async::messaging::Sender;
use near_chain::chain::ChunkStateWitnessMessage;
use near_chain::chain::{
    ApplyChunksMode, NewChunkData, NewChunkResult, OldChunkData, OldChunkResult, ShardContext,
    StorageContext, UpdateShardJob, do_apply_chunks, get_should_apply_chunk,
};
use near_chain::sharding::shuffle_receipt_proofs;
use near_chain::types::{ApplyChunkBlockContext, RuntimeAdapter, StorageDataSource};
use near_chain::update_shard::{ShardUpdateReason, ShardUpdateResult, process_shard_update};
use near_chain::{
    Block, Chain, ChainGenesis, ChainStore, ChainUpdate, DoomslugThresholdMode, Error,
    collect_receipts, get_chunk_clone_from_header,
};
use near_chain_configs::MutableValidatorSigner;
use near_chunks::logic::make_outgoing_receipts_proofs;
use near_epoch_manager::EpochManagerAdapter;
use near_epoch_manager::shard_assignment::shard_id_to_uid;
use near_epoch_manager::shard_tracker::ShardTracker;
use near_network::types::{NetworkRequests, PeerManagerAdapter, PeerManagerMessageRequest};
use near_primitives::block::Chunks;
use near_primitives::hash::CryptoHash;
use near_primitives::hash::hash;
use near_primitives::merkle::merklize;
use near_primitives::optimistic_block::{BlockToApply, CachedShardUpdateKey};
use near_primitives::receipt::Receipt;
use near_primitives::sandbox::state_patch::SandboxStatePatch;
use near_primitives::shard_layout::ShardLayout;
use near_primitives::sharding::ChunkHash;
use near_primitives::sharding::ReceiptProof;
use near_primitives::spice::ExecutionResult;
use near_primitives::stateless_validation::contract_distribution::ContractUpdates;
use near_primitives::stateless_validation::state_witness::ChunkStateTransition;
use near_primitives::stateless_validation::state_witness::ChunkStateWitness;
use near_primitives::stateless_validation::state_witness::ChunkStateWitnessAck;
use near_primitives::stateless_validation::state_witness::ChunkStateWitnessSize;
use near_primitives::stateless_validation::stored_chunk_state_transition_data::StoredChunkStateTransitionData;
use near_primitives::stateless_validation::stored_chunk_state_transition_data::StoredChunkStateTransitionDataV1;
use near_primitives::types::chunk_extra::ChunkExtra;
use near_primitives::types::{AccountId, EpochId, ShardId, ShardIndex};
use near_primitives::validator_signer::ValidatorSigner;
use near_store::Store;
use near_store::adapter::StoreAdapter as _;
use tracing::instrument;

use crate::DistributeStateWitnessRequest;
use crate::PartialWitnessSenderForClient;
use crate::spice_core::CoreStatementsProcessor;
use crate::stateless_validation::chunk_validator::orphan_witness_pool::OrphanStateWitnessPool;

#[derive(Clone, MultiSend, MultiSenderFrom)]
pub struct ChunkExecutorAdapter {
    pub block_sender: Sender<ExecutorBlock>,
    pub receipts_sender: Sender<ExecutorIncomingReceipt>,
    pub execution_result_available_sender: Sender<ExecutorAllExecutionResultsAvailable>,
    pub witness_sender: Sender<ChunkStateWitnessMessage>,
}

pub struct ChunkExecutorActor {
    chain_store: ChainStore,
    runtime_adapter: Arc<dyn RuntimeAdapter>,
    epoch_manager: Arc<dyn EpochManagerAdapter>,
    /// Contains validator info about this node. This field is mutable and optional. Use with caution!
    /// Lock the value of mutable validator signer for the duration of a request to ensure consistency.
    /// Please note that the locked value should not be stored anywhere or passed through the thread boundary.
    validator_signer: MutableValidatorSigner,
    shard_tracker: ShardTracker,
    network_adapter: PeerManagerAdapter,

    /// Receipts originating from block keyed by block hash.
    block_receipts_cache: LruCache<CryptoHash, Vec<ReceiptProof>>,

    core_processor: CoreStatementsProcessor,

    partial_witness_adapter: PartialWitnessSenderForClient,

    orphan_witness_pool: OrphanStateWitnessPool,
    myself_sender: ChunkExecutorAdapter,
}

impl ChunkExecutorActor {
    pub fn new(
        store: Store,
        genesis: &ChainGenesis,
        runtime_adapter: Arc<dyn RuntimeAdapter>,
        epoch_manager: Arc<dyn EpochManagerAdapter>,
        validator_signer: MutableValidatorSigner,
        shard_tracker: ShardTracker,
        network_adapter: PeerManagerAdapter,
        block_receipts_cache_capacity: NonZeroUsize,
        partial_witness_adapter: PartialWitnessSenderForClient,
        core_processor: CoreStatementsProcessor,
        myself_sender: ChunkExecutorAdapter,
    ) -> Self {
        let orphan_witness_pool_size = 1000;
        Self {
            // FIXME(spice): get size from outside.
            orphan_witness_pool: OrphanStateWitnessPool::new(orphan_witness_pool_size),
            chain_store: ChainStore::new(store, true, genesis.transaction_validity_period),
            runtime_adapter,
            epoch_manager,
            validator_signer,
            shard_tracker,
            network_adapter,
            partial_witness_adapter,
            core_processor,
            block_receipts_cache: LruCache::new(block_receipts_cache_capacity),
            myself_sender,
        }
    }

    fn calculate_receipts_root(
        &self,
        shard_layout: &ShardLayout,
        receipts: &[Receipt],
    ) -> Result<CryptoHash, Error> {
        let receipts_hashes = Chain::build_receipts_hashes(&receipts, &shard_layout)?;
        let (receipts_root, _) = merklize(&receipts_hashes);
        Ok(receipts_root)
    }
}

impl near_async::messaging::Actor for ChunkExecutorActor {}

/// Message with incoming receipts corresponding to the block.
/// Eventually this would be handled properly with data availability layer.
/// For now this is useful to do testing with test loop.
#[derive(actix::Message, Debug)]
#[rtype(result = "()")]
pub struct ExecutorIncomingReceipt {
    pub block_hash: CryptoHash,
    pub receipt_proofs: Vec<ReceiptProof>,
}

/// Message that should be sent once block is processed to indicate that it's available for
/// execution.
#[derive(actix::Message, Debug)]
#[rtype(result = "()")]
pub struct ExecutorBlock {
    pub block_hash: CryptoHash,
}

// FIXME(spice): not sure if this is the best approach.
#[derive(actix::Message, Debug)]
#[rtype(result = "()")]
pub struct ExecutorAllExecutionResultsAvailable {
    pub block_hash: CryptoHash,
}

impl Handler<ExecutorIncomingReceipt> for ChunkExecutorActor {
    fn handle(
        &mut self,
        ExecutorIncomingReceipt { block_hash, receipt_proofs }: ExecutorIncomingReceipt,
    ) {
        let block_receipts = self.block_receipts_cache.get_or_insert_mut(block_hash, || Vec::new());
        block_receipts.extend(receipt_proofs.into_iter());

        let me = self.validator_signer.get().map(|signer| signer.validator_id().clone());

        if let Err(err) = self.try_save_incoming_receipts(me.as_ref(), &block_hash) {
            tracing::error!(target: "chunk_executor", %block_hash, ?err, "failed to save incoming receipts");
        }

        let next_block_hash = self.chain_store.get_next_block_hash(&block_hash);
        let next_block_hash = match next_block_hash {
            Ok(hash) => hash,
            Err(err) => {
                if matches!(err, Error::DBNotFoundErr(_)) {
                    // Next block wasn't processed yet.
                    tracing::debug!(target: "chunk_executor", %block_hash, ?err, "no next block hash is available");
                    return;
                }
                tracing::error!(target: "chunk_executor", %block_hash, ?err, "failed to get next block hash");
                return;
            }
        };
        if let Err(err) = self.try_apply_chunks(&next_block_hash, me.as_ref()) {
            tracing::error!(target: "chunk_executor", ?err, ?block_hash, "failed to apply chunk for block hash");
        };
    }
}

impl Handler<ExecutorBlock> for ChunkExecutorActor {
    fn handle(&mut self, ExecutorBlock { block_hash }: ExecutorBlock) {
        let me = self.validator_signer.get().map(|signer| signer.validator_id().clone());
        // We may have received receipts before the corresponding block.
        let block = match self.chain_store.get_block(&block_hash) {
            Ok(block) => block,
            Err(err) => {
                tracing::error!(target: "chunk_executor", %block_hash, ?err, "failed to get block");
                return;
            }
        };

        // FIXME(spice): Not sure if this is the best place for this.
        if let Some(signer) = self.validator_signer.get() {
            if let Err(err) = self.process_ready_orphan_state_witnesses(&block, signer) {
                tracing::error!(target: "chunk_executor", %block_hash, ?err, "failed to process orphan state witnesses");
            }
        }

        let header = block.header();
        let prev_block_hash = header.prev_hash();
        if let Err(err) = self.try_save_incoming_receipts(me.as_ref(), &prev_block_hash) {
            tracing::error!(target: "chunk_executor", %prev_block_hash, ?err, "failed to save incoming receipts");
            return;
        }

        if let Err(err) = self.try_apply_chunks(&block_hash, me.as_ref()) {
            tracing::error!(target: "chunk_executor", ?err, ?block_hash, "failed to apply chunk for block hash");
        };
    }
}

impl Handler<ExecutorAllExecutionResultsAvailable> for ChunkExecutorActor {
    fn handle(
        &mut self,
        ExecutorAllExecutionResultsAvailable { block_hash }: ExecutorAllExecutionResultsAvailable,
    ) {
        // FIXME(spice): dedup with incoming receipts handling.
        let me = self.validator_signer.get().map(|signer| signer.validator_id().clone());

        let next_block_hash = self.chain_store.get_next_block_hash(&block_hash);
        let next_block_hash = match next_block_hash {
            Ok(hash) => hash,
            Err(err) => {
                if matches!(err, Error::DBNotFoundErr(_)) {
                    // Next block wasn't processed yet.
                    tracing::debug!(target: "chunk_executor", %block_hash, ?err, "no next block hash is available");
                    return;
                }
                tracing::error!(target: "chunk_executor", %block_hash, ?err, "failed to get next block hash");
                return;
            }
        };

        // FIXME(spice): Not sure if this is the best place for this.
        {
            let block = self.chain_store.get_block(&next_block_hash).unwrap();
            if let Some(signer) = self.validator_signer.get() {
                if let Err(err) = self.process_ready_orphan_state_witnesses(&block, signer) {
                    tracing::error!(target: "chunk_executor", %block_hash, ?err, "failed to process orphan state witnesses");
                }
            }
        }
        if let Err(err) = self.try_apply_chunks(&next_block_hash, me.as_ref()) {
            tracing::error!(target: "chunk_executor", ?err, ?block_hash, "failed to apply chunk for block hash");
        };
    }
}

impl ChunkExecutorActor {
    #[instrument(target = "chunk_executor", level = "debug", skip_all, fields(%block_hash, ?me))]
    fn try_apply_chunks(
        &mut self,
        block_hash: &CryptoHash,
        me: Option<&AccountId>,
    ) -> Result<(), Error> {
        let epoch_id = self.epoch_manager.get_epoch_id(block_hash)?;
        let block = self.chain_store.get_block(block_hash)?;
        let header = block.header();
        let prev_block_hash = header.prev_hash();
        for shard_id in self.epoch_manager.shard_ids(&epoch_id)? {
            let is_me = true;
            if self
                .shard_tracker
                .cares_about_shard_this_or_next_epoch(me, block_hash, shard_id, is_me)
                && !self.chain_store.incoming_receipts_exist(&prev_block_hash, shard_id)?
            {
                tracing::debug!(target: "chunk_executor", %block_hash, %prev_block_hash, "missing receipts to apply all tracked chunks for a block");
                return Ok(());
            }
        }

        if !self
            .core_processor
            .do_all_execution_results_exist(prev_block_hash, &self.chain_store)?
        {
            tracing::debug!(target: "chunk_executor", %block_hash, %prev_block_hash, "missing previous block's execution results");
            return Ok(());
        }

        self.apply_chunks(me, block, SandboxStatePatch::default())
    }

    fn get_incoming_receipts(
        &self,
        prev_block_hash: &CryptoHash,
        shard_id: ShardId,
    ) -> Result<Arc<Vec<ReceiptProof>>, Error> {
        self.chain_store.get_incoming_receipts(prev_block_hash, shard_id)
    }

    // Logic here is based on Chain::apply_chunk_preprocessing
    fn apply_chunks(
        &mut self,
        me: Option<&AccountId>,
        block: Block,
        mut state_patch: SandboxStatePatch,
    ) -> Result<(), Error> {
        let block_hash = block.hash();
        let header = block.header();
        let prev_hash = header.prev_hash();
        let prev_block = self.chain_store.get_block(prev_hash)?;

        let prev_chunk_headers =
            Chain::get_prev_chunk_headers(self.epoch_manager.as_ref(), &prev_block)?;

        let epoch_id = block.header().epoch_id();
        let shard_layout = self.epoch_manager.get_shard_layout(&epoch_id)?;

        let chunk_headers = &block.chunks();
        let mut jobs = Vec::new();
        for (shard_index, _prev_chunk_header) in prev_chunk_headers.iter().enumerate() {
            // XXX: This is a bit questionable -- sandbox state patching works
            // only for a single shard. This so far has been enough.
            let state_patch = state_patch.take();
            let shard_id = shard_layout.get_shard_id(shard_index)?;

            let chunk_header =
                chunk_headers.get(shard_index).ok_or(Error::InvalidShardId(shard_id))?;

            let is_new_chunk = chunk_header.is_new_chunk(block.header().height());

            let block_context = Chain::get_apply_chunk_block_context_from_block_header(
                block.header(),
                &chunk_headers,
                prev_block.header(),
                is_new_chunk,
            )?;

            // If we don't care about shard we wouldn't have relevant incoming receipts.
            let is_me = true;
            if !self
                .shard_tracker
                .cares_about_shard_this_or_next_epoch(me, block_hash, shard_id, is_me)
            {
                continue;
            }
            let receipt_proofs = self.get_incoming_receipts(prev_hash, shard_id)?;
            let incoming_receipts = Some(receipt_proofs.deref());

            let storage_context =
                StorageContext { storage_data_source: StorageDataSource::Db, state_patch };

            let cached_shard_update_key =
                Chain::get_cached_shard_update_key(&block_context, chunk_headers, shard_id)?;

            let job = self.get_update_shard_job(
                me,
                cached_shard_update_key,
                block_context,
                chunk_headers,
                shard_index,
                &prev_block,
                ApplyChunksMode::IsCaughtUp,
                incoming_receipts,
                storage_context,
            );
            match job {
                Ok(Some(job)) => jobs.push(job),
                Ok(None) => {}
                Err(e) => panic!("{e:?}"),
            }
        }

        let apply_result =
            do_apply_chunks(BlockToApply::Normal(*block.hash()), block.header().height(), jobs);
        let apply_result = apply_result.into_iter().map(|res| (res.0, res.2)).collect_vec();
        let results = apply_result.into_iter().map(|(shard_id, x)| {
            if let Err(err) = &x {
                tracing::warn!(target: "chunk_executor", ?shard_id, hash = %block.hash(), %err, "error in applying chunk for block");
            }
            x
        }).collect::<Result<Vec<_>, Error>>()?;

        let mut chain_update = self.chain_update();
        // FIXME(spice): Consider if it would be better to extract state transition information
        // directly from apply result.
        let should_save_state_transition_data = true;
        chain_update.apply_chunk_postprocessing(
            &block,
            results.clone(),
            should_save_state_transition_data,
        )?;
        chain_update.commit()?;

        // FIXME(spice): create some helpers to make it easier to follow?
        for result in &results {
            let (shard_uid, apply_result) = match result {
                ShardUpdateResult::NewChunk(NewChunkResult {
                    shard_uid,
                    gas_limit: _,
                    apply_result,
                }) => (shard_uid, apply_result),
                ShardUpdateResult::OldChunk(OldChunkResult { shard_uid, apply_result }) => {
                    (shard_uid, apply_result)
                }
            };
            let shard_id = shard_uid.shard_id();
            let shard_index = shard_layout.get_shard_index(shard_id)?;
            let chunk_header =
                chunk_headers.get(shard_index).ok_or(Error::InvalidShardId(shard_id))?;

            let prev_chunk_header =
                Chain::get_prev_chunk_header(self.epoch_manager.as_ref(), &prev_block, shard_id)
                    .unwrap();

            let prev_outgoing_receipts = self.chain_store.get_outgoing_receipts_for_shard(
                self.epoch_manager.as_ref(),
                *prev_block.hash(),
                shard_id,
                prev_chunk_header.height_included(),
            )?;
            let chunk_extra = self.chain_store.get_chunk_extra(&prev_block.hash(), shard_uid)?;
            let prev_outgoing_receipts_root =
                self.calculate_receipts_root(&shard_layout, &prev_outgoing_receipts)?;
            let spice_chunk_header = chunk_header
                .clone()
                .into_spice_chunk_execution_header(&chunk_extra, prev_outgoing_receipts_root);

            let receipt_proofs = make_outgoing_receipts_proofs(
                &spice_chunk_header,
                apply_result.outgoing_receipts.clone(),
                self.epoch_manager.as_ref(),
            )?;

            {
                // FIXME(spice): consider if recording of execution results shoud happen here.
                let chunk_extra = self.chain_store.get_chunk_extra(&block_hash, shard_uid)?;
                let outgoing_receipts_root =
                    self.calculate_receipts_root(&shard_layout, &apply_result.outgoing_receipts)?;
                self.core_processor.record_execution_result(
                    ExecutionResult {
                        block_hash: *block_hash,
                        chunk_hash: chunk_header.chunk_hash(),
                        shard_id,
                        chunk_extra,
                        outgoing_receipts_root,
                    },
                    &self.myself_sender,
                )
            }
            self.send_outgoing_receipts(*block_hash, receipt_proofs);

            // FIXME(spice): Consider refactoring to better handle data that is common
            // FIXME(spice): Make helpers to deal with witnesses

            let (main_transition, applied_receipts_hash, contract_updates) = if chunk_header
                .is_genesis()
            {
                (
                    ChunkStateTransition {
                        block_hash: *block_hash,
                        base_state: Default::default(),
                        post_state_root: apply_result.new_root,
                    },
                    hash(&borsh::to_vec::<[Receipt]>(&[]).unwrap()),
                    ContractUpdates::default(),
                )
            } else {
                let stored_chunk_state_transition_data = self
                        .chain_store
                        .store()
                        .get_ser(
                            near_store::DBCol::StateTransitionData,
                            &near_primitives::utils::get_block_shard_id(block_hash, shard_id),
                        )?
                        .ok_or_else(|| {
                            let message = format!(
                                "Missing transition state proof for block {block_hash} and shard {shard_id}"
                            );
                            Error::Other(message)
                        })?;
                let StoredChunkStateTransitionData::V1(StoredChunkStateTransitionDataV1 {
                    base_state,
                    receipts_hash,
                    contract_accesses,
                    contract_deploys,
                }) = stored_chunk_state_transition_data;
                let contract_updates = ContractUpdates {
                    contract_accesses: contract_accesses.into_iter().collect(),
                    contract_deploys: contract_deploys.into_iter().map(|c| c.into()).collect(),
                };
                (
                    ChunkStateTransition {
                        block_hash: *block_hash,
                        base_state,
                        // FIXME(spice): If can, use data from apply_result here.
                        post_state_root: *self
                            .chain_store
                            .get_chunk_extra(block_hash, &shard_uid)?
                            .state_root(),
                    },
                    receipts_hash,
                    contract_updates,
                )
            };

            // FIXME(spice): Make sure this logic is correct with resharding.
            let source_receipt_proofs: HashMap<ChunkHash, ReceiptProof> = {
                let receipt_proofs = self.get_incoming_receipts(prev_hash, shard_id)?;
                let prev_block_shard_layout =
                    self.epoch_manager.get_shard_layout(prev_block.header().epoch_id())?;
                receipt_proofs
                    .iter()
                    .map(|proof| -> Result<_, Error> {
                        let from_shard_id = proof.1.from_shard_id;
                        let from_shard_index =
                            prev_block_shard_layout.get_shard_index(from_shard_id)?;
                        let from_chunk_hash = prev_block
                            .chunks()
                            .get(from_shard_index)
                            .ok_or(Error::InvalidShardId(proof.1.from_shard_id))?
                            .chunk_hash();
                        Ok((from_chunk_hash, proof.clone()))
                    })
                    .try_collect()?
            };
            // FIXME(spice): although implicit_transitions are mostly used for missing chunks, they
            // are also used for resharding so we need to include resharding in
            // implicit_transitions when it happens. It would also mean updating
            // main_transition_shard_id accordingly.
            let implicit_transitions = Vec::new();
            let main_transition_shard_id = shard_id;

            let chunk = get_chunk_clone_from_header(&self.chain_store, chunk_header)?;
            let state_witness = ChunkStateWitness::new(
                // FIXME(spice): refactor to avoid unwrap.
                me.unwrap().clone(),
                *epoch_id,
                // FIXME(spice): Add a note somewhere that this chunk header's meaning between
                // spice and non-spice. For spice it's current chunk application of which we are
                // witnessing, and for non-spice it's chunk_header of the chunk following the one
                // application of which we are witnessing.
                chunk_header.clone(),
                main_transition,
                source_receipt_proofs,
                // (Could also be derived from iterating through the receipts, but
                // that defeats the purpose of this check being a debugging
                // mechanism.)
                applied_receipts_hash,
                chunk.to_transactions().to_vec(),
                implicit_transitions,
            );

            // FIXME(spice): Do conditionally based on config (same as in client).
            self.chain_store.save_latest_chunk_state_witness(&state_witness)?;

            // FIXME(spice): No need to always validate your own witnes, but good for debugging.
            match self.core_processor.validate_state_witness(
                state_witness.clone(),
                &self.chain_store,
                self.epoch_manager.as_ref(),
                self.runtime_adapter.as_ref(),
            ) {
                Ok(_) => {
                    println!("state witness validation success!");
                }
                Err(err) => {
                    println!("Failed state witness validation: {err:?}");
                    // FIXME(spice): don't panic
                    panic!("Failed state witness validation: {err:?}");
                }
            };

            // FIXME(spice): If we are one of the validators bypass witness validation and endorse the
            // chunk immediately.
            self.send_witness_to_chunk_validators(
                state_witness,
                contract_updates,
                main_transition_shard_id,
            );
        }

        Ok(())
    }

    fn send_witness_to_chunk_validators(
        &mut self,
        state_witness: ChunkStateWitness,
        contract_updates: ContractUpdates,
        main_transition_shard_id: ShardId,
    ) {
        self.partial_witness_adapter.send(DistributeStateWitnessRequest {
            state_witness,
            contract_updates,
            main_transition_shard_id,
        });
    }

    fn send_outgoing_receipts(
        &mut self,
        block_hash: CryptoHash,
        receipt_proofs: Vec<ReceiptProof>,
    ) {
        tracing::debug!(target: "chunk_executor", %block_hash, ?receipt_proofs, "sending outoging receipts");
        self.network_adapter.send(PeerManagerMessageRequest::NetworkRequests(
            NetworkRequests::SpiceIncomingReceipts { block_hash, receipt_proofs },
        ));
    }

    fn get_update_shard_job(
        &self,
        me: Option<&AccountId>,
        cached_shard_update_key: CachedShardUpdateKey,
        block: ApplyChunkBlockContext,
        chunk_headers: &Chunks,
        shard_index: ShardIndex,
        prev_block: &Block,
        mode: ApplyChunksMode,
        incoming_receipts: Option<&Vec<ReceiptProof>>,
        storage_context: StorageContext,
    ) -> Result<Option<UpdateShardJob>, Error> {
        let prev_block_hash = prev_block.hash();
        let block_height = block.height;
        let _span =
            tracing::debug_span!(target: "chunk_executor", "get_update_shard_job", ?prev_block_hash, block_height)
                .entered();

        let epoch_id = self.epoch_manager.get_epoch_id_from_prev_block(prev_block_hash)?;
        let shard_layout = self.epoch_manager.get_shard_layout(&epoch_id)?;
        let shard_id = shard_layout.get_shard_id(shard_index)?;
        let shard_context =
            self.get_shard_context(me, prev_block_hash, &epoch_id, shard_id, mode)?;

        if !shard_context.should_apply_chunk {
            return Ok(None);
        }

        let chunk_header = chunk_headers.get(shard_index).ok_or(Error::InvalidShardId(shard_id))?;
        let is_new_chunk = chunk_header.is_new_chunk(block_height);

        let shard_update_reason = if is_new_chunk {
            let chunk = get_chunk_clone_from_header(&self.chain_store, chunk_header)?;
            let tx_valid_list =
                self.chain_store.compute_transaction_validity(prev_block.header(), &chunk);
            let receipts = collect_receipts(incoming_receipts.unwrap());

            let shard_uid = &shard_context.shard_uid;
            let chunk_extra = self.chain_store.get_chunk_extra(prev_block_hash, shard_uid)?;

            let prev_chunk_header =
                Chain::get_prev_chunk_header(self.epoch_manager.as_ref(), &prev_block, shard_id)
                    .unwrap();
            let prev_outgoing_receipts = self.chain_store.get_outgoing_receipts_for_shard(
                self.epoch_manager.as_ref(),
                *prev_block.hash(),
                shard_id,
                prev_chunk_header.height_included(),
            )?;
            let prev_outgoing_receipts_root =
                self.calculate_receipts_root(&shard_layout, &prev_outgoing_receipts)?;
            let chunk_header = chunk_header
                .clone()
                .into_spice_chunk_execution_header(&chunk_extra, prev_outgoing_receipts_root);

            let new_chunk_data = NewChunkData {
                chunk_header,
                transactions: chunk.into_transactions(),
                transaction_validity_check_results: tx_valid_list,
                receipts,
                block,
                storage_context,
            };
            ShardUpdateReason::NewChunk(new_chunk_data)
        } else {
            ShardUpdateReason::OldChunk(OldChunkData {
                block,
                prev_chunk_extra: ChunkExtra::clone(
                    self.chain_store
                        .get_chunk_extra(prev_block_hash, &shard_context.shard_uid)?
                        .as_ref(),
                ),
                storage_context,
            })
        };

        let runtime = self.runtime_adapter.clone();
        Ok(Some((
            shard_id,
            cached_shard_update_key,
            Box::new(move |parent_span| -> Result<ShardUpdateResult, Error> {
                Ok(process_shard_update(
                    parent_span,
                    runtime.as_ref(),
                    shard_update_reason,
                    shard_context,
                )?)
            }),
        )))
    }

    fn get_shard_context(
        &self,
        me: Option<&AccountId>,
        prev_hash: &CryptoHash,
        epoch_id: &EpochId,
        shard_id: ShardId,
        mode: ApplyChunksMode,
    ) -> Result<ShardContext, Error> {
        let cares_about_shard_this_epoch =
            self.shard_tracker.cares_about_shard(me, prev_hash, shard_id, true);
        let cares_about_shard_next_epoch =
            self.shard_tracker.will_care_about_shard(me, prev_hash, shard_id, true);
        let cared_about_shard_prev_epoch =
            self.shard_tracker.cared_about_shard_in_prev_epoch(me, prev_hash, shard_id, true);
        let should_apply_chunk = get_should_apply_chunk(
            mode,
            cares_about_shard_this_epoch,
            cares_about_shard_next_epoch,
            cared_about_shard_prev_epoch,
        );
        let shard_uid = shard_id_to_uid(self.epoch_manager.as_ref(), shard_id, epoch_id)?;
        Ok(ShardContext { shard_uid, should_apply_chunk })
    }

    fn chain_update(&mut self) -> ChainUpdate {
        ChainUpdate::new(
            &mut self.chain_store,
            self.epoch_manager.clone(),
            self.runtime_adapter.clone(),
            // Since we don't produce blocks, this argument is irrelevant.
            DoomslugThresholdMode::NoApprovals,
        )
    }

    fn have_all_receipts(
        &mut self,
        me: Option<&AccountId>,
        block_hash: &CryptoHash,
    ) -> Result<bool, Error> {
        let Some(block_receipts) = self.block_receipts_cache.get(block_hash) else {
            return Ok(false);
        };
        let epoch_id = self.epoch_manager.get_epoch_id(block_hash)?;
        let shard_ids = self.epoch_manager.shard_ids(&epoch_id)?;
        let shards_to_receipts: HashMap<(ShardId, ShardId), &ReceiptProof> = block_receipts
            .iter()
            .map(|proof| ((proof.1.from_shard_id, proof.1.to_shard_id), proof))
            .collect();

        let is_me = true;
        for to_shard_id in &shard_ids {
            if !self.shard_tracker.cares_about_shard_this_or_next_epoch(
                me,
                &block_hash,
                *to_shard_id,
                is_me,
            ) {
                continue;
            }
            for from_shard_id in &shard_ids {
                if !shards_to_receipts.contains_key(&(*from_shard_id, *to_shard_id)) {
                    tracing::debug!(target: "chunk_executor", %block_hash, ?from_shard_id, ?to_shard_id, "still missing some incoming receipts");
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    #[instrument(target = "chunk_executor", level = "debug", skip_all, fields(%block_hash, ?me))]
    fn try_save_incoming_receipts(
        &mut self,
        me: Option<&AccountId>,
        block_hash: &CryptoHash,
    ) -> Result<(), Error> {
        let have_all_receipts = match self.have_all_receipts(me, block_hash) {
            Ok(v) => v,
            Err(err) => {
                tracing::error!(target: "chunk_executor", ?err, ?block_hash, "failed to check if have all receipts");
                return Err(err);
            }
        };
        if !have_all_receipts {
            return Ok(());
        }

        let block_receipts = self.block_receipts_cache.pop(&block_hash).unwrap();

        let receipt_proofs: HashMap<ShardId, Vec<ReceiptProof>> =
            // We cannot filter out correctly when receiving receipts since we may receive them
            // before we know about the corresponding block and can decide which receipts we care
            // about.
            block_receipts.into_iter()
            // FIXME(spice): right now we are recording all receipts we receive. We would later
            // need to make sure not to record duplicates.
            // FIXME(spice): when receiving receipts validate them against previous execution
            // result.
            .unique_by(|proof| (proof.1.from_shard_id, proof.1.to_shard_id))
            .filter(|proof| {
                let is_me = true;
                self.shard_tracker.cares_about_shard_this_or_next_epoch(
                    me,
                    &block_hash,
                    proof.1.to_shard_id,
                    is_me,
                )
            }).fold(HashMap::new(), |mut acc, proof| {
                acc.entry(proof.1.to_shard_id).or_default().push(proof);
                acc
            });

        let mut chain_update = self.chain_update();
        for (to_shard_id, mut proofs) in receipt_proofs {
            // FIXME(spice): Don't relay on sorting here, but for example use order of chunks in a
            // block. Has to be consistent with what is done in witness validation so that receipt
            // order is the same. Order also shouldn't depend on the order in which we receive
            // incoming receipts.
            proofs.sort_by_key(|proof| proof.1.from_shard_id);
            let receipts_shuffle_salt = block_hash;
            shuffle_receipt_proofs(&mut proofs, receipts_shuffle_salt);

            tracing::debug!(target: "chunk_executor", %block_hash, ?to_shard_id, ?proofs, "saving incoming receipts");
            // FIXME(spice): Consired using a separate col for storage incoming receipts,
            // since meaning of the key may be different (prev hash instead of current
            // hash).
            chain_update.save_incoming_receipt(&block_hash, to_shard_id, Arc::new(proofs));
        }
        chain_update.commit()?;
        Ok(())
    }
}

// FIXME(spice): Handling of state witnesses and endorsements likely should happen in a separate
// agent.
impl Handler<ChunkStateWitnessMessage> for ChunkExecutorActor {
    // FIXME(spice): decide if this is required. Not sure what it is.
    // #[perf]
    fn handle(&mut self, msg: ChunkStateWitnessMessage) {
        let ChunkStateWitnessMessage { witness, raw_witness_size } = msg;
        let Some(signer) = self.validator_signer.get() else {
            tracing::error!(target: "spice_core", ?witness, "Received a chunk state witness but this is not a validator node.");
            return;
        };
        if let Err(err) = self.process_chunk_state_witness(witness, raw_witness_size, signer) {
            tracing::error!(target: "client", ?err, "Error processing chunk state witness");
        }
    }
}

impl ChunkExecutorActor {
    pub fn process_chunk_state_witness(
        &mut self,
        witness: ChunkStateWitness,
        raw_witness_size: ChunkStateWitnessSize,
        signer: Arc<ValidatorSigner>,
    ) -> Result<(), Error> {
        tracing::debug!(
            target: "spice_core",
            chunk_hash=?witness.chunk_header.chunk_hash(),
            shard_id=?witness.chunk_header.shard_id(),
            "process_chunk_state_witness",
        );

        // Send the acknowledgement for the state witness back to the chunk producer.
        // This is currently used for network roundtrip time measurement, so we do not need to
        // wait for validation to finish.
        self.send_state_witness_ack(&witness, &signer);

        // FIXME(spice): based on client config - should likely optionally save latest state witness.
        // if self.config.save_latest_witnesses {
        //     self.chain.chain_store.save_latest_chunk_state_witness(&witness)?;
        // }
        let chunk_hash = witness.chunk_header.chunk_hash();

        match self.core_processor.validate_state_witness_and_send_endorsements(
            witness.clone(),
            &self.chain_store,
            self.epoch_manager.as_ref(),
            self.runtime_adapter.as_ref(),
            &self.network_adapter.clone().into_sender(),
            &signer,
        ) {
            Ok(_) => {
                tracing::info!(target: "adhoc", ?chunk_hash, "validated state witness successfully");
                Ok(())
            }
            // FIXME(spice): Check block and execution results explicitly instead of relying on
            // error which may happen for unrelated reasons.
            Err(Error::DBNotFoundErr(err)) => {
                tracing::info!(target: "adhoc", ?chunk_hash, ?err, "saving orphaned state witness; there're either not enough execution results or relevant block isn't available yet");
                self.handle_orphan_state_witness(witness, raw_witness_size)
            }
            Err(err) => Err(err),
        }
    }

    fn process_ready_orphan_state_witnesses(
        &mut self,
        // FIXME(spice): don't need the whole block, only prev_hash
        new_block: &Block,
        signer: Arc<ValidatorSigner>,
    ) -> Result<(), Error> {
        let prev_hash = new_block.header().prev_hash();
        tracing::debug!(target: "adhoc", ?prev_hash, block_hash=?new_block.hash(), "processing orphan state witnesses");
        let ready_witnesses = self
            .orphan_witness_pool
            // in spice chunk_header belongs to the block we want to process, not to the next one.
            .take_state_witnesses_waiting_for_block(prev_hash);
        for witness in ready_witnesses {
            if !self.core_processor.do_all_execution_results_exist(prev_hash, &self.chain_store)? {
                tracing::info!(target: "adhoc", ?prev_hash, "block is available, but still not enough execution results for witness execution");
                // We don't care about size at this point.
                self.orphan_witness_pool.add_orphan_state_witness(witness, 0);
                continue;
            }
            self.core_processor.validate_state_witness_and_send_endorsements(
                witness.clone(),
                &self.chain_store,
                self.epoch_manager.as_ref(),
                self.runtime_adapter.as_ref(),
                &self.network_adapter.clone().into_sender(),
                &signer,
            )?;
        }
        Ok(())
    }

    // FIXME(spice): Reuse, if possible, parts from Client's orphan_witness_handling.rs to do some
    // validations before writing witness to the pool.
    // FIMXE(spice): May be a good idea in addition to check that witness is for a block we know about.
    pub fn handle_orphan_state_witness(
        &mut self,
        witness: ChunkStateWitness,
        witness_size: usize,
    ) -> Result<(), Error> {
        self.orphan_witness_pool.add_orphan_state_witness(witness, witness_size);
        Ok(())
    }

    // FIXME(spice): Dedup with what's in chunk_validators/mod.rs
    fn send_state_witness_ack(&self, witness: &ChunkStateWitness, signer: &Arc<ValidatorSigner>) {
        // In production PartialWitnessActor does not forward a state witness to the chunk producer that
        // produced the witness. However some tests bypass PartialWitnessActor, thus when a chunk producer
        // receives its own state witness, we log a warning instead of panicking.
        // TODO: Make sure all tests run with "test_features" and panic for non-test builds.
        if signer.validator_id() == &witness.chunk_producer {
            tracing::warn!(
                "Validator {:?} received state witness from itself. Witness={:?}",
                signer.validator_id(),
                witness
            );
            return;
        }
        self.network_adapter.send(PeerManagerMessageRequest::NetworkRequests(
            NetworkRequests::ChunkStateWitnessAck(
                witness.chunk_producer.clone(),
                ChunkStateWitnessAck::new(witness),
            ),
        ));
    }
}
