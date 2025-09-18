use near_async::messaging::Sender;
use near_async::{MultiSend, MultiSendMessage, MultiSenderFrom};
use near_primitives::hash::CryptoHash;
use near_primitives::merkle::MerklePath;
use near_primitives::network::PeerId;
use near_primitives::types::{AccountId, MerkleHash, ShardId};

#[derive(actix::Message, Debug, Clone)]
#[rtype(result = "()")]
pub struct SpiceIncomingPartialData {
    pub data: SpicePartialData,
    pub sender: PeerId,
}

#[derive(actix::Message, Debug, Clone)]
#[rtype(result = "()")]
pub struct RequestSpiceData {
    pub data_id: SpiceDataIdentifier,
    // FIXME: Should use route back I've seen in some other places instead?
    pub requester: AccountId,
}

#[derive(Clone, MultiSend, MultiSenderFrom, MultiSendMessage)]
pub struct SpiceDataDistributorSenderForNetwork {
    pub incoming: Sender<SpiceIncomingPartialData>,
    pub request: Sender<RequestSpiceData>,
}

#[derive(borsh::BorshSerialize, borsh::BorshDeserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpiceDataIdentifier {
    ReceiptProof { block_hash: CryptoHash, from_shard_id: ShardId, to_shard_id: ShardId },
    Witness { block_hash: CryptoHash, shard_id: ShardId },
}

impl SpiceDataIdentifier {
    pub fn block_hash(&self) -> &CryptoHash {
        match self {
            SpiceDataIdentifier::ReceiptProof { block_hash, .. } => block_hash,
            SpiceDataIdentifier::Witness { block_hash, .. } => block_hash,
        }
    }
}

#[derive(borsh::BorshSerialize, borsh::BorshDeserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpiceDataCommitment {
    pub hash: CryptoHash,
    pub root: MerkleHash,
    pub encoded_length: u64,
}

#[derive(borsh::BorshSerialize, borsh::BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct SpiceDataPart {
    pub part_ord: u64,
    pub part: Box<[u8]>,
    pub merkle_proof: MerklePath,
}

// TODO(spice): Version this struct since it is sent over the network.
#[derive(borsh::BorshSerialize, borsh::BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct SpicePartialData {
    // We include id to allow finding recipients and producers when receiving the data.
    pub id: SpiceDataIdentifier,
    pub commitment: SpiceDataCommitment,
    pub parts: Vec<SpiceDataPart>,
    // FIXME: If this works include signature as well and restructure to have signed_partial_data or
    // something and signature as top-level fields.
    pub sender: AccountId,
}
