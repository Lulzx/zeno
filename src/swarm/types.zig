//! Core types for the Zeno swarm platform.
//! All extern structs are GPU/C-ABI compatible.

/// Configuration for a swarm simulation.
pub const SwarmConfig = extern struct {
    /// Number of agents in the swarm.
    num_agents: u32 = 0,
    /// Communication range for neighbor detection.
    communication_range: f32 = 10.0,
    /// Maximum neighbors per agent for graph construction.
    max_neighbors: u32 = 32,
    /// Maximum bytes per message payload.
    max_message_bytes: u32 = 48,
    /// Maximum messages each agent can send per step.
    max_messages_per_step: u32 = 4,
    /// Spatial grid cell size (should be >= communication_range).
    grid_cell_size: f32 = 10.0,
    /// Random seed.
    seed: u64 = 42,
    /// Whether to run physics (disable for pure communication tests).
    enable_physics: bool = true,
    /// Padding for alignment.
    _pad: [7]u8 = .{0} ** 7,
};

/// Per-agent state (extern, GPU-compatible, 32 bytes).
pub const AgentState = extern struct {
    /// Unique agent identifier within the env.
    agent_id: u32,
    /// Team identifier (for multi-team scenarios).
    team_id: u32,
    /// Agent status (0=inactive, 1=active, 2=dead).
    status: u32,
    /// Bitfield flags for agent capabilities.
    flags: u32,
    /// Per-agent local state (policy-defined usage).
    local_state: [4]f32 = .{ 0, 0, 0, 0 },
};

/// Message slot for inter-agent communication (64 bytes, cache-line aligned).
pub const MessageSlot = extern struct {
    /// Sender agent ID.
    sender_id: u32,
    /// Receiver agent ID (0xFFFFFFFF = broadcast to neighbors).
    receiver_id: u32,
    /// Application-defined message type tag.
    message_type: u32,
    /// Payload length in bytes (max 48).
    payload_len: u32,
    /// Message payload.
    payload: [48]u8 = .{0} ** 48,

    pub const BROADCAST: u32 = 0xFFFFFFFF;
};

/// Aggregated metrics for the swarm.
pub const SwarmMetrics = extern struct {
    /// Fraction of possible neighbor connections that exist.
    connectivity_ratio: f32 = 0,
    /// Number of disconnected components (1 = fully connected).
    fragmentation_score: f32 = 0,
    /// Number of physics collisions this step.
    collision_count: u32 = 0,
    /// Number of messages delivered this step.
    message_count: u32 = 0,
    /// Total bytes sent this step.
    bytes_sent: u32 = 0,
    /// Total neighbor edges this step.
    total_edges: u32 = 0,
    /// Average neighbors per agent.
    avg_neighbors: f32 = 0,
    /// Padding.
    _pad: u32 = 0,
};
