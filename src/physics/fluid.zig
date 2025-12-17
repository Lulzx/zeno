//! Smoothed Particle Hydrodynamics (SPH) Fluid Simulation
//! Implements incompressible fluid with pressure, viscosity, and surface tension.

const std = @import("std");
const constants = @import("constants.zig");

/// SPH kernel radius multiplier (h = particle_radius * KERNEL_MULT).
const KERNEL_MULT: f32 = 4.0;

/// Fluid particle state.
pub const FluidParticle = struct {
    /// Position in world space.
    position: [3]f32,
    /// Velocity.
    velocity: [3]f32,
    /// Predicted position.
    predicted: [3]f32,
    /// Density at particle.
    density: f32 = 0,
    /// Pressure at particle.
    pressure: f32 = 0,
    /// Acceleration from forces.
    acceleration: [3]f32 = .{ 0, 0, 0 },
    /// Particle mass.
    mass: f32 = 1.0,
    /// Cell index for spatial hashing.
    cell_id: u32 = 0,
};

/// Fluid simulation parameters.
pub const FluidParams = struct {
    /// Rest density (kg/m³).
    rest_density: f32 = 1000.0,
    /// Particle mass.
    particle_mass: f32 = 0.02,
    /// Particle radius.
    particle_radius: f32 = 0.05,
    /// Gas constant for pressure (stiffness).
    gas_constant: f32 = 2000.0,
    /// Viscosity coefficient.
    viscosity: f32 = 0.01,
    /// Surface tension coefficient.
    surface_tension: f32 = 0.0728,
    /// Gravity.
    gravity: [3]f32 = .{ 0, 0, -9.81 },
    /// Time step.
    dt: f32 = 0.002,
    /// Boundary friction.
    boundary_friction: f32 = 0.5,
    /// Boundary restitution.
    boundary_restitution: f32 = 0.3,
};

/// GPU-optimized particle data.
pub const FluidParticleGPU = extern struct {
    /// Position xyz + density.
    pos_density: [4]f32 align(16),
    /// Velocity xyz + pressure.
    vel_pressure: [4]f32 align(16),
    /// Acceleration xyz + mass.
    accel_mass: [4]f32 align(16),
    /// Predicted position xyz + cell_id.
    predicted_cell: [4]f32 align(16),
};

/// Spatial hash grid for neighbor search.
pub const SpatialHash = struct {
    allocator: std.mem.Allocator,

    /// Cell size.
    cell_size: f32,
    /// Hash table: cell -> particle indices.
    cells: std.AutoHashMap(u32, std.ArrayListUnmanaged(u32)),
    /// Particle cell assignments.
    particle_cells: std.ArrayListUnmanaged(u32),

    pub fn init(allocator: std.mem.Allocator, cell_size: f32) SpatialHash {
        return .{
            .allocator = allocator,
            .cell_size = cell_size,
            .cells = std.AutoHashMap(u32, std.ArrayListUnmanaged(u32)).init(allocator),
            .particle_cells = .{},
        };
    }

    pub fn deinit(self: *SpatialHash) void {
        var it = self.cells.valueIterator();
        while (it.next()) |list| {
            list.deinit(self.allocator);
        }
        self.cells.deinit();
        self.particle_cells.deinit(self.allocator);
    }

    /// Hash 3D position to cell ID.
    pub fn hash(self: *const SpatialHash, pos: [3]f32) u32 {
        const ix: i32 = @intFromFloat(@floor(pos[0] / self.cell_size));
        const iy: i32 = @intFromFloat(@floor(pos[1] / self.cell_size));
        const iz: i32 = @intFromFloat(@floor(pos[2] / self.cell_size));

        // Morton encoding for spatial coherence
        const ux: u32 = @bitCast(ix);
        const uy: u32 = @bitCast(iy);
        const uz: u32 = @bitCast(iz);

        return ((ux *% 73856093) ^ (uy *% 19349663) ^ (uz *% 83492791));
    }

    /// Clear and rebuild spatial hash.
    pub fn rebuild(self: *SpatialHash, particles: []const FluidParticle) !void {
        // Clear existing cells
        var it = self.cells.valueIterator();
        while (it.next()) |list| {
            list.clearRetainingCapacity();
        }

        // Resize particle cells array
        try self.particle_cells.resize(self.allocator, particles.len);

        // Assign particles to cells
        for (particles, 0..) |p, i| {
            const cell_id = self.hash(p.position);
            self.particle_cells.items[i] = cell_id;

            const gop = try self.cells.getOrPut(cell_id);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{};
            }
            try gop.value_ptr.append(self.allocator, @intCast(i));
        }
    }

    /// Get particles in neighboring cells.
    pub fn getNeighbors(self: *const SpatialHash, pos: [3]f32, result: *std.ArrayListUnmanaged(u32)) !void {
        result.clearRetainingCapacity();

        // Check 27 neighboring cells (3x3x3)
        const offsets = [_]i32{ -1, 0, 1 };

        for (offsets) |dz| {
            for (offsets) |dy| {
                for (offsets) |dx| {
                    const neighbor_pos: [3]f32 = .{
                        pos[0] + @as(f32, @floatFromInt(dx)) * self.cell_size,
                        pos[1] + @as(f32, @floatFromInt(dy)) * self.cell_size,
                        pos[2] + @as(f32, @floatFromInt(dz)) * self.cell_size,
                    };

                    const cell_id = self.hash(neighbor_pos);
                    if (self.cells.get(cell_id)) |cell_particles| {
                        for (cell_particles.items) |idx| {
                            try result.append(self.allocator, idx);
                        }
                    }
                }
            }
        }
    }
};

/// SPH Fluid simulation.
pub const FluidSimulation = struct {
    allocator: std.mem.Allocator,

    /// Fluid particles.
    particles: std.ArrayListUnmanaged(FluidParticle),
    /// Simulation parameters.
    params: FluidParams,
    /// Spatial hash for neighbor search.
    spatial_hash: SpatialHash,
    /// Kernel radius.
    h: f32,
    /// Precomputed kernel constants.
    poly6_const: f32,
    spiky_const: f32,
    viscosity_const: f32,
    /// Neighbor list (reused).
    neighbors: std.ArrayListUnmanaged(u32),
    /// Boundary box.
    boundary_min: [3]f32 = .{ -5, -5, 0 },
    boundary_max: [3]f32 = .{ 5, 5, 10 },

    pub fn init(allocator: std.mem.Allocator, params: FluidParams) FluidSimulation {
        const h = params.particle_radius * KERNEL_MULT;
        const h2 = h * h;
        const h3 = h2 * h;
        const h6 = h3 * h3;
        const h9 = h6 * h3;

        return .{
            .allocator = allocator,
            .particles = .{},
            .params = params,
            .spatial_hash = SpatialHash.init(allocator, h),
            .h = h,
            .poly6_const = 315.0 / (64.0 * std.math.pi * h9),
            .spiky_const = -45.0 / (std.math.pi * h6),
            .viscosity_const = 45.0 / (std.math.pi * h6),
            .neighbors = .{},
        };
    }

    pub fn deinit(self: *FluidSimulation) void {
        self.particles.deinit(self.allocator);
        self.spatial_hash.deinit();
        self.neighbors.deinit(self.allocator);
    }

    /// Add a particle.
    pub fn addParticle(self: *FluidSimulation, pos: [3]f32, vel: [3]f32) !void {
        try self.particles.append(self.allocator, .{
            .position = pos,
            .velocity = vel,
            .predicted = pos,
            .mass = self.params.particle_mass,
        });
    }

    /// Create a block of fluid particles.
    pub fn createBlock(self: *FluidSimulation, min: [3]f32, max: [3]f32) !void {
        const spacing = self.params.particle_radius * 2.0;

        var z = min[2];
        while (z < max[2]) : (z += spacing) {
            var y = min[1];
            while (y < max[1]) : (y += spacing) {
                var x = min[0];
                while (x < max[0]) : (x += spacing) {
                    // Add small jitter to prevent grid artifacts
                    const jitter: [3]f32 = .{
                        (@as(f32, @floatFromInt(@mod(@as(u32, @bitCast(@as(i32, @intFromFloat(x * 1000)))), 100))) - 50.0) * 0.001,
                        (@as(f32, @floatFromInt(@mod(@as(u32, @bitCast(@as(i32, @intFromFloat(y * 1000)))), 100))) - 50.0) * 0.001,
                        (@as(f32, @floatFromInt(@mod(@as(u32, @bitCast(@as(i32, @intFromFloat(z * 1000)))), 100))) - 50.0) * 0.001,
                    };

                    try self.addParticle(.{
                        x + jitter[0],
                        y + jitter[1],
                        z + jitter[2],
                    }, .{ 0, 0, 0 });
                }
            }
        }
    }

    /// Set simulation boundary.
    pub fn setBoundary(self: *FluidSimulation, min: [3]f32, max: [3]f32) void {
        self.boundary_min = min;
        self.boundary_max = max;
    }

    /// Simulation step.
    pub fn step(self: *FluidSimulation) !void {
        if (self.particles.items.len == 0) return;

        // Rebuild spatial hash
        try self.spatial_hash.rebuild(self.particles.items);

        // Compute density and pressure
        try self.computeDensityPressure();

        // Compute forces
        try self.computeForces();

        // Integrate
        self.integrate();

        // Handle boundary collisions
        self.handleBoundaries();
    }

    fn computeDensityPressure(self: *FluidSimulation) !void {
        const h2 = self.h * self.h;

        for (self.particles.items, 0..) |*pi, i| {
            try self.spatial_hash.getNeighbors(pi.position, &self.neighbors);

            var density: f32 = 0;

            for (self.neighbors.items) |j| {
                const pj = &self.particles.items[j];
                const r = sub3(pi.position, pj.position);
                const r2 = dot3(r, r);

                if (r2 < h2) {
                    // Poly6 kernel
                    const diff = h2 - r2;
                    density += pj.mass * self.poly6_const * diff * diff * diff;
                }
            }

            pi.density = @max(density, self.params.rest_density * 0.01);

            // Tait equation for pressure
            const ratio = pi.density / self.params.rest_density;
            pi.pressure = self.params.gas_constant * (ratio - 1.0);

            _ = i;
        }
    }

    fn computeForces(self: *FluidSimulation) !void {
        for (self.particles.items) |*pi| {
            try self.spatial_hash.getNeighbors(pi.position, &self.neighbors);

            var pressure_force: [3]f32 = .{ 0, 0, 0 };
            var viscosity_force: [3]f32 = .{ 0, 0, 0 };

            for (self.neighbors.items) |j| {
                const pj = &self.particles.items[j];

                const r = sub3(pi.position, pj.position);
                const r_len = @sqrt(dot3(r, r));

                if (r_len > constants.EPSILON and r_len < self.h) {
                    const r_norm = scale3(r, 1.0 / r_len);

                    // Pressure force (Spiky kernel gradient)
                    const h_r = self.h - r_len;
                    const pressure_mag = -pj.mass * (pi.pressure + pj.pressure) / (2.0 * pj.density) * self.spiky_const * h_r * h_r;

                    pressure_force[0] += pressure_mag * r_norm[0];
                    pressure_force[1] += pressure_mag * r_norm[1];
                    pressure_force[2] += pressure_mag * r_norm[2];

                    // Viscosity force (Laplacian kernel)
                    const viscosity_mag = self.params.viscosity * pj.mass / pj.density * self.viscosity_const * (self.h - r_len);

                    const vel_diff = sub3(pj.velocity, pi.velocity);
                    viscosity_force[0] += viscosity_mag * vel_diff[0];
                    viscosity_force[1] += viscosity_mag * vel_diff[1];
                    viscosity_force[2] += viscosity_mag * vel_diff[2];
                }
            }

            // Total acceleration
            const inv_density = 1.0 / pi.density;
            pi.acceleration = .{
                (pressure_force[0] + viscosity_force[0]) * inv_density + self.params.gravity[0],
                (pressure_force[1] + viscosity_force[1]) * inv_density + self.params.gravity[1],
                (pressure_force[2] + viscosity_force[2]) * inv_density + self.params.gravity[2],
            };
        }
    }

    fn integrate(self: *FluidSimulation) void {
        const dt = self.params.dt;

        for (self.particles.items) |*p| {
            // Semi-implicit Euler
            p.velocity[0] += p.acceleration[0] * dt;
            p.velocity[1] += p.acceleration[1] * dt;
            p.velocity[2] += p.acceleration[2] * dt;

            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
        }
    }

    fn handleBoundaries(self: *FluidSimulation) void {
        const margin = self.params.particle_radius;
        const restitution = self.params.boundary_restitution;
        const friction = self.params.boundary_friction;

        for (self.particles.items) |*p| {
            // X bounds
            if (p.position[0] < self.boundary_min[0] + margin) {
                p.position[0] = self.boundary_min[0] + margin;
                p.velocity[0] *= -restitution;
                p.velocity[1] *= friction;
                p.velocity[2] *= friction;
            } else if (p.position[0] > self.boundary_max[0] - margin) {
                p.position[0] = self.boundary_max[0] - margin;
                p.velocity[0] *= -restitution;
                p.velocity[1] *= friction;
                p.velocity[2] *= friction;
            }

            // Y bounds
            if (p.position[1] < self.boundary_min[1] + margin) {
                p.position[1] = self.boundary_min[1] + margin;
                p.velocity[1] *= -restitution;
                p.velocity[0] *= friction;
                p.velocity[2] *= friction;
            } else if (p.position[1] > self.boundary_max[1] - margin) {
                p.position[1] = self.boundary_max[1] - margin;
                p.velocity[1] *= -restitution;
                p.velocity[0] *= friction;
                p.velocity[2] *= friction;
            }

            // Z bounds (floor is most important)
            if (p.position[2] < self.boundary_min[2] + margin) {
                p.position[2] = self.boundary_min[2] + margin;
                p.velocity[2] *= -restitution;
                p.velocity[0] *= friction;
                p.velocity[1] *= friction;
            } else if (p.position[2] > self.boundary_max[2] - margin) {
                p.position[2] = self.boundary_max[2] - margin;
                p.velocity[2] *= -restitution;
                p.velocity[0] *= friction;
                p.velocity[1] *= friction;
            }
        }
    }

    /// Get particle count.
    pub fn particleCount(self: *const FluidSimulation) usize {
        return self.particles.items.len;
    }

    /// Get particle positions for rendering.
    pub fn getPositions(self: *const FluidSimulation) []const FluidParticle {
        return self.particles.items;
    }
};

// Vector helpers

fn sub3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn dot3(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn scale3(v: [3]f32, s: f32) [3]f32 {
    return .{ v[0] * s, v[1] * s, v[2] * s };
}

// Tests

test "create fluid block" {
    const allocator = std.testing.allocator;

    var sim = FluidSimulation.init(allocator, .{});
    defer sim.deinit();

    try sim.createBlock(.{ -0.5, -0.5, 0.5 }, .{ 0.5, 0.5, 1.5 });

    try std.testing.expect(sim.particleCount() > 0);
}

test "fluid step" {
    const allocator = std.testing.allocator;

    var sim = FluidSimulation.init(allocator, .{
        .particle_radius = 0.1,
        .dt = 0.01,
    });
    defer sim.deinit();

    // Add a few particles
    try sim.addParticle(.{ 0, 0, 1 }, .{ 0, 0, 0 });
    try sim.addParticle(.{ 0.1, 0, 1 }, .{ 0, 0, 0 });
    try sim.addParticle(.{ 0, 0.1, 1 }, .{ 0, 0, 0 });

    const initial_z = sim.particles.items[0].position[2];

    // Step simulation
    for (0..10) |_| {
        try sim.step();
    }

    // Particles should fall due to gravity
    try std.testing.expect(sim.particles.items[0].position[2] < initial_z);
}
