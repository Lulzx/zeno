//! Physics constants and configuration parameters.

const std = @import("std");

/// Default physics constants.
pub const DEFAULT_GRAVITY: [3]f32 = .{ 0.0, 0.0, -9.81 };
pub const DEFAULT_TIMESTEP: f32 = 0.002; // 500 Hz
pub const DEFAULT_CONTACT_ITERATIONS: u32 = 4;
pub const DEFAULT_POSITION_ITERATIONS: u32 = 4;
pub const DEFAULT_VELOCITY_ITERATIONS: u32 = 1;

/// Numerical tolerances.
pub const EPSILON: f32 = 1e-6;
pub const ANGULAR_EPSILON: f32 = 1e-8;
pub const PENETRATION_SLOP: f32 = 0.005; // Allow small penetration
pub const BAUMGARTE_FACTOR: f32 = 0.2; // Constraint stabilization factor

/// Contact parameters.
pub const DEFAULT_RESTITUTION: f32 = 0.0;
pub const DEFAULT_FRICTION: f32 = 1.0;
pub const MAX_CONTACTS_PER_PAIR: u32 = 4;
pub const DEFAULT_MAX_CONTACTS_PER_ENV: u32 = 64;

/// Joint parameters.
pub const DEFAULT_JOINT_DAMPING: f32 = 0.0;
pub const DEFAULT_JOINT_STIFFNESS: f32 = 0.0;
pub const DEFAULT_JOINT_ARMATURE: f32 = 0.0;

/// Simulation limits.
pub const MAX_BODIES_PER_ENV: u32 = 256;
pub const MAX_JOINTS_PER_ENV: u32 = 256;
pub const MAX_GEOMS_PER_ENV: u32 = 512;
pub const MAX_ACTUATORS_PER_ENV: u32 = 64;
pub const MAX_SENSORS_PER_ENV: u32 = 128;

/// Memory alignment for SIMD operations.
pub const SIMD_ALIGNMENT: usize = 16;
pub const CACHE_LINE_SIZE: usize = 64;

/// Performance tuning.
pub const DEFAULT_THREADGROUP_SIZE: u32 = 256;
pub const DEFAULT_MAX_ENVS: u32 = 16384;

/// Physics configuration.
pub const PhysicsConfig = struct {
    gravity: [3]f32 = DEFAULT_GRAVITY,
    timestep: f32 = DEFAULT_TIMESTEP,
    contact_iterations: u32 = DEFAULT_CONTACT_ITERATIONS,
    position_iterations: u32 = DEFAULT_POSITION_ITERATIONS,
    velocity_iterations: u32 = DEFAULT_VELOCITY_ITERATIONS,
    max_contacts_per_env: u32 = DEFAULT_MAX_CONTACTS_PER_ENV,
    restitution: f32 = DEFAULT_RESTITUTION,
    friction: f32 = DEFAULT_FRICTION,
    enable_gyroscopic_forces: bool = false,
    enable_warm_starting: bool = true,
    baumgarte_factor: f32 = BAUMGARTE_FACTOR,
    penetration_slop: f32 = PENETRATION_SLOP,

    /// Validate configuration parameters.
    pub fn validate(self: *const PhysicsConfig) bool {
        if (self.timestep <= 0.0 or self.timestep > 0.1) return false;
        if (self.contact_iterations == 0) return false;
        if (self.friction < 0.0) return false;
        if (self.restitution < 0.0 or self.restitution > 1.0) return false;
        return true;
    }

    /// Create config optimized for speed (less accurate).
    pub fn fast() PhysicsConfig {
        return .{
            .contact_iterations = 2,
            .position_iterations = 2,
            .velocity_iterations = 1,
            .enable_gyroscopic_forces = false,
            .enable_warm_starting = false,
        };
    }

    /// Create config optimized for accuracy (slower).
    pub fn accurate() PhysicsConfig {
        return .{
            .contact_iterations = 8,
            .position_iterations = 8,
            .velocity_iterations = 4,
            .enable_gyroscopic_forces = true,
            .enable_warm_starting = true,
        };
    }
};

/// Material properties.
pub const Material = struct {
    friction: f32 = DEFAULT_FRICTION,
    restitution: f32 = DEFAULT_RESTITUTION,
    density: f32 = 1000.0, // kg/m³ (water)

    pub const DEFAULT = Material{};

    pub const RUBBER = Material{
        .friction = 1.0,
        .restitution = 0.8,
        .density = 1100.0,
    };

    pub const METAL = Material{
        .friction = 0.4,
        .restitution = 0.3,
        .density = 7800.0,
    };

    pub const WOOD = Material{
        .friction = 0.5,
        .restitution = 0.4,
        .density = 600.0,
    };

    pub const ICE = Material{
        .friction = 0.05,
        .restitution = 0.1,
        .density = 917.0,
    };

    /// Combine materials for contact (geometric mean for friction, max for restitution).
    pub fn combine(a: Material, b: Material) Material {
        return .{
            .friction = @sqrt(a.friction * b.friction),
            .restitution = @max(a.restitution, b.restitution),
            .density = (a.density + b.density) * 0.5,
        };
    }
};

/// Solver parameters for PBD/XPBD.
pub const SolverParams = struct {
    /// Compliance (inverse stiffness) for position constraints.
    position_compliance: f32 = 0.0,
    /// Damping ratio for velocity constraints.
    damping: f32 = 0.0,
    /// Relaxation factor for Gauss-Seidel iteration.
    sor_factor: f32 = 1.0,
    /// Use XPBD (extended position-based dynamics).
    use_xpbd: bool = true,
    /// Substep count for better stability.
    substeps: u32 = 1,

    /// Create solver params optimized for RL throughput.
    pub fn forRL() SolverParams {
        return .{
            .position_compliance = 0.0,
            .damping = 0.005,
            .sor_factor = 1.0,
            .use_xpbd = true,
            .substeps = 1,
        };
    }

    /// Create solver params optimized for accuracy.
    pub fn forAccuracy() SolverParams {
        return .{
            .position_compliance = 0.0,
            .damping = 0.001,
            .sor_factor = 1.0,
            .use_xpbd = true,
            .substeps = 4,
        };
    }
};

/// XPBD-specific configuration.
pub const XPBDConfig = struct {
    /// Number of constraint solver iterations.
    iterations: u32 = 4,
    /// Contact compliance (0 = rigid, >0 = soft).
    contact_compliance: f32 = 0.0,
    /// Joint compliance.
    joint_compliance: f32 = 0.0,
    /// Enable warm starting from previous frame.
    warm_start: bool = true,
    /// Solver relaxation (1.0 = standard).
    relaxation: f32 = 1.0,
    /// Velocity damping factor.
    velocity_damping: f32 = 0.0,

    /// Default config for RL training.
    pub const RL = XPBDConfig{
        .iterations = 4,
        .contact_compliance = 1e-9,
        .joint_compliance = 0.0,
        .warm_start = true,
        .relaxation = 1.0,
        .velocity_damping = 0.005,
    };

    /// High accuracy config.
    pub const ACCURATE = XPBDConfig{
        .iterations = 8,
        .contact_compliance = 0.0,
        .joint_compliance = 0.0,
        .warm_start = true,
        .relaxation = 1.0,
        .velocity_damping = 0.001,
    };
};

/// Broadphase configuration.
pub const BroadphaseConfig = struct {
    /// Cell size for spatial hashing.
    cell_size: f32 = 1.0,
    /// Number of cells in each dimension.
    grid_size: u32 = 64,
    /// Maximum pairs to check per frame.
    max_pairs: u32 = 4096,
    /// Use hierarchical grid.
    hierarchical: bool = false,
};

/// Debug/profiling options.
pub const DebugConfig = struct {
    /// Enable profiling markers.
    enable_profiling: bool = false,
    /// Validate state after each step.
    validate_state: bool = false,
    /// Log constraint violations.
    log_violations: bool = false,
    /// Maximum allowed velocity (clamp if exceeded).
    max_velocity: f32 = 100.0,
    /// Maximum allowed angular velocity.
    max_angular_velocity: f32 = 100.0,
};
