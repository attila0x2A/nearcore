use std::sync::Arc;

use borsh::{BorshDeserialize, BorshSerialize};
use near_primitives_core::hash::CryptoHash;
use near_primitives_core::types::ShardId;
use near_schema_checker_lib::ProtocolSchema;

use crate::sharding::ChunkHash;
use crate::types::chunk_extra::ChunkExtra;

// FIXME(spice): This structure is kind of hacky. Would probably make sense to use it for implementation,
// but not sure if for storage (even if versioned) since it may contain a lot of redundant
// information.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize, ProtocolSchema)]
pub struct ExecutionResult {
    // FIXME(spice): Not sure if this is required, but for now it's convenient.
    pub block_hash: CryptoHash,
    pub chunk_hash: ChunkHash,
    pub shard_id: ShardId,
    // FIXME(spice): Arc probably would have to be removed
    pub chunk_extra: Arc<ChunkExtra>,
    pub outgoing_receipts_root: CryptoHash,
}
