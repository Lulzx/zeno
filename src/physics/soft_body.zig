//! Soft Body Physics using Position-Based Dynamics (PBD)
//! Implements deformable bodies with distance and volume constraints.

const std = @import("std");
const constants = @import("constants.zig");

/// Soft body particle state.
pub const Particle = struct {
    /// Position in world space.
    position: [3]f32,
    /// Velocity.
    velocity: [3]f32,
    /// Predicted position (used during constraint solving).
    predicted: [3]f32,
    /// Inverse mass (0 = fixed).
    inv_mass: f32,
    /// Rest position (for shape matching).
    rest_position: [3]f32,
};

/// Distance constraint between two particles.
pub const DistanceConstraint = struct {
    /// First particle index.
    p1: u32,
    /// Second particle index.
    p2: u32,
    /// Rest length.
    rest_length: f32,
    /// Stiffness (0-1).
    stiffness: f32 = 1.0,
};

/// Volume constraint for a tetrahedron.
pub const VolumeConstraint = struct {
    /// Four particle indices forming tetrahedron.
    particles: [4]u32,
    /// Rest volume.
    rest_volume: f32,
    /// Stiffness (0-1).
    stiffness: f32 = 1.0,
};

/// Bending constraint for angle preservation.
pub const BendingConstraint = struct {
    /// Four particles: two shared edge + two opposite vertices.
    particles: [4]u32,
    /// Rest dihedral angle.
    rest_angle: f32,
    /// Stiffness.
    stiffness: f32 = 0.5,
};

/// Soft body definition.
pub const SoftBodyDef = struct {
    /// Number of particles.
    num_particles: u32 = 0,
    /// Particle mass.
    mass: f32 = 1.0,
    /// Stiffness for distance constraints (0-1).
    stiffness: f32 = 0.9,
    /// Volume stiffness (0-1).
    volume_stiffness: f32 = 1.0,
    /// Damping coefficient.
    damping: f32 = 0.01,
    /// Pressure (for inflatable bodies).
    pressure: f32 = 0.0,
    /// Friction coefficient.
    friction: f32 = 0.5,
    /// Collision margin.
    margin: f32 = 0.01,
};

/// GPU-optimized particle data (SoA layout).
pub const ParticleGPU = extern struct {
    /// Position xyz + inv_mass.
    pos_invmass: [4]f32 align(16),
    /// Velocity xyz + padding.
    vel: [4]f32 align(16),
    /// Predicted position xyz + padding.
    predicted: [4]f32 align(16),
    /// Rest position xyz + padding.
    rest_pos: [4]f32 align(16),
};

/// GPU-optimized distance constraint.
pub const DistanceConstraintGPU = extern struct {
    /// Particle indices (p1, p2) + rest_length + stiffness.
    data: [4]u32 align(16),

    pub fn fromConstraint(c: DistanceConstraint) DistanceConstraintGPU {
        return .{
            .data = .{
                c.p1,
                c.p2,
                @bitCast(c.rest_length),
                @bitCast(c.stiffness),
            },
        };
    }
};

/// GPU-optimized volume constraint.
pub const VolumeConstraintGPU = extern struct {
    /// Four particle indices.
    particles: [4]u32 align(16),
    /// Rest volume + stiffness + padding.
    params: [4]f32 align(16),

    pub fn fromConstraint(c: VolumeConstraint) VolumeConstraintGPU {
        return .{
            .particles = c.particles,
            .params = .{ c.rest_volume, c.stiffness, 0, 0 },
        };
    }
};

/// Soft body instance.
pub const SoftBody = struct {
    allocator: std.mem.Allocator,

    /// Particles.
    particles: std.ArrayListUnmanaged(Particle),
    /// Distance constraints.
    distance_constraints: std.ArrayListUnmanaged(DistanceConstraint),
    /// Volume constraints.
    volume_constraints: std.ArrayListUnmanaged(VolumeConstraint),
    /// Bending constraints.
    bending_constraints: std.ArrayListUnmanaged(BendingConstraint),

    /// Definition parameters.
    def: SoftBodyDef,

    /// Center of mass.
    center_of_mass: [3]f32 = .{ 0, 0, 0 },
    /// Total mass.
    total_mass: f32 = 0,

    pub fn init(allocator: std.mem.Allocator, def: SoftBodyDef) SoftBody {
        return .{
            .allocator = allocator,
            .particles = .{},
            .distance_constraints = .{},
            .volume_constraints = .{},
            .bending_constraints = .{},
            .def = def,
        };
    }

    pub fn deinit(self: *SoftBody) void {
        self.particles.deinit(self.allocator);
        self.distance_constraints.deinit(self.allocator);
        self.volume_constraints.deinit(self.allocator);
        self.bending_constraints.deinit(self.allocator);
    }

    /// Add a particle and return its index.
    pub fn addParticle(self: *SoftBody, pos: [3]f32, fixed: bool) !u32 {
        const idx: u32 = @intCast(self.particles.items.len);
        const inv_mass = if (fixed) 0.0 else 1.0 / self.def.mass;

        try self.particles.append(self.allocator, .{
            .position = pos,
            .velocity = .{ 0, 0, 0 },
            .predicted = pos,
            .inv_mass = inv_mass,
            .rest_position = pos,
        });

        return idx;
    }

    /// Add distance constraint between two particles.
    pub fn addDistanceConstraint(self: *SoftBody, p1: u32, p2: u32) !void {
        const pos1 = self.particles.items[p1].position;
        const pos2 = self.particles.items[p2].position;
        const rest_length = distance(pos1, pos2);

        try self.distance_constraints.append(self.allocator, .{
            .p1 = p1,
            .p2 = p2,
            .rest_length = rest_length,
            .stiffness = self.def.stiffness,
        });
    }

    /// Add volume constraint for tetrahedron.
    pub fn addVolumeConstraint(self: *SoftBody, p0: u32, p1: u32, p2: u32, p3: u32) !void {
        const particles = [4]u32{ p0, p1, p2, p3 };
        const rest_volume = computeTetraVolume(
            self.particles.items[p0].position,
            self.particles.items[p1].position,
            self.particles.items[p2].position,
            self.particles.items[p3].position,
        );

        try self.volume_constraints.append(self.allocator, .{
            .particles = particles,
            .rest_volume = rest_volume,
            .stiffness = self.def.volume_stiffness,
        });
    }

    /// Create a soft body cube.
    pub fn createCube(allocator: std.mem.Allocator, center: [3]f32, size: f32, resolution: u32, def: SoftBodyDef) !SoftBody {
        var body = SoftBody.init(allocator, def);
        errdefer body.deinit();

        const step = size / @as(f32, @floatFromInt(resolution - 1));
        const half = size * 0.5;

        // Create particle grid
        var particle_grid = std.AutoHashMap([3]u32, u32).init(allocator);
        defer particle_grid.deinit();

        for (0..resolution) |iz| {
            for (0..resolution) |iy| {
                for (0..resolution) |ix| {
                    const pos: [3]f32 = .{
                        center[0] - half + @as(f32, @floatFromInt(ix)) * step,
                        center[1] - half + @as(f32, @floatFromInt(iy)) * step,
                        center[2] - half + @as(f32, @floatFromInt(iz)) * step,
                    };

                    const idx = try body.addParticle(pos, false);
                    try particle_grid.put(.{
                        @intCast(ix),
                        @intCast(iy),
                        @intCast(iz),
                    }, idx);
                }
            }
        }

        // Create distance constraints along edges
        for (0..resolution) |iz| {
            for (0..resolution) |iy| {
                for (0..resolution) |ix| {
                    const idx = particle_grid.get(.{
                        @intCast(ix),
                        @intCast(iy),
                        @intCast(iz),
                    }).?;

                    // X edge
                    if (ix + 1 < resolution) {
                        const neighbor = particle_grid.get(.{
                            @intCast(ix + 1),
                            @intCast(iy),
                            @intCast(iz),
                        }).?;
                        try body.addDistanceConstraint(idx, neighbor);
                    }

                    // Y edge
                    if (iy + 1 < resolution) {
                        const neighbor = particle_grid.get(.{
                            @intCast(ix),
                            @intCast(iy + 1),
                            @intCast(iz),
                        }).?;
                        try body.addDistanceConstraint(idx, neighbor);
                    }

                    // Z edge
                    if (iz + 1 < resolution) {
                        const neighbor = particle_grid.get(.{
                            @intCast(ix),
                            @intCast(iy),
                            @intCast(iz + 1),
                        }).?;
                        try body.addDistanceConstraint(idx, neighbor);
                    }

                    // Diagonal constraints for stability
                    if (ix + 1 < resolution and iy + 1 < resolution) {
                        const diag = particle_grid.get(.{
                            @intCast(ix + 1),
                            @intCast(iy + 1),
                            @intCast(iz),
                        }).?;
                        try body.addDistanceConstraint(idx, diag);
                    }

                    if (ix + 1 < resolution and iz + 1 < resolution) {
                        const diag = particle_grid.get(.{
                            @intCast(ix + 1),
                            @intCast(iy),
                            @intCast(iz + 1),
                        }).?;
                        try body.addDistanceConstraint(idx, diag);
                    }

                    if (iy + 1 < resolution and iz + 1 < resolution) {
                        const diag = particle_grid.get(.{
                            @intCast(ix),
                            @intCast(iy + 1),
                            @intCast(iz + 1),
                        }).?;
                        try body.addDistanceConstraint(idx, diag);
                    }
                }
            }
        }

        // Create volume constraints (tetrahedra from cubes)
        for (0..resolution - 1) |iz| {
            for (0..resolution - 1) |iy| {
                for (0..resolution - 1) |ix| {
                    // Get cube corners
                    const c000 = particle_grid.get(.{ @intCast(ix), @intCast(iy), @intCast(iz) }).?;
                    const c100 = particle_grid.get(.{ @intCast(ix + 1), @intCast(iy), @intCast(iz) }).?;
                    const c010 = particle_grid.get(.{ @intCast(ix), @intCast(iy + 1), @intCast(iz) }).?;
                    const c110 = particle_grid.get(.{ @intCast(ix + 1), @intCast(iy + 1), @intCast(iz) }).?;
                    const c001 = particle_grid.get(.{ @intCast(ix), @intCast(iy), @intCast(iz + 1) }).?;
                    const c101 = particle_grid.get(.{ @intCast(ix + 1), @intCast(iy), @intCast(iz + 1) }).?;
                    const c011 = particle_grid.get(.{ @intCast(ix), @intCast(iy + 1), @intCast(iz + 1) }).?;
                    const c111 = particle_grid.get(.{ @intCast(ix + 1), @intCast(iy + 1), @intCast(iz + 1) }).?;

                    // Split cube into 5 tetrahedra
                    try body.addVolumeConstraint(c000, c100, c010, c001);
                    try body.addVolumeConstraint(c100, c110, c010, c111);
                    try body.addVolumeConstraint(c001, c101, c100, c111);
                    try body.addVolumeConstraint(c001, c011, c010, c111);
                    try body.addVolumeConstraint(c100, c010, c001, c111);
                }
            }
        }

        body.computeCenterOfMass();
        return body;
    }

    /// Create a soft body cloth.
    pub fn createCloth(allocator: std.mem.Allocator, corner: [3]f32, width: f32, height: f32, res_x: u32, res_y: u32, def: SoftBodyDef) !SoftBody {
        var body = SoftBody.init(allocator, def);
        errdefer body.deinit();

        const step_x = width / @as(f32, @floatFromInt(res_x - 1));
        const step_y = height / @as(f32, @floatFromInt(res_y - 1));

        // Create particle grid
        var indices = try allocator.alloc(u32, res_x * res_y);
        defer allocator.free(indices);

        for (0..res_y) |iy| {
            for (0..res_x) |ix| {
                const pos: [3]f32 = .{
                    corner[0] + @as(f32, @floatFromInt(ix)) * step_x,
                    corner[1] + @as(f32, @floatFromInt(iy)) * step_y,
                    corner[2],
                };

                // Fix top corners
                const fixed = (iy == res_y - 1) and (ix == 0 or ix == res_x - 1);
                indices[iy * res_x + ix] = try body.addParticle(pos, fixed);
            }
        }

        // Structural constraints (edges)
        for (0..res_y) |iy| {
            for (0..res_x) |ix| {
                const idx = indices[iy * res_x + ix];

                if (ix + 1 < res_x) {
                    try body.addDistanceConstraint(idx, indices[iy * res_x + ix + 1]);
                }
                if (iy + 1 < res_y) {
                    try body.addDistanceConstraint(idx, indices[(iy + 1) * res_x + ix]);
                }
            }
        }

        // Shear constraints (diagonals)
        for (0..res_y - 1) |iy| {
            for (0..res_x - 1) |ix| {
                const idx = indices[iy * res_x + ix];
                try body.addDistanceConstraint(idx, indices[(iy + 1) * res_x + ix + 1]);
                try body.addDistanceConstraint(indices[iy * res_x + ix + 1], indices[(iy + 1) * res_x + ix]);
            }
        }

        // Bending constraints (skip one)
        for (0..res_y) |iy| {
            for (0..res_x) |ix| {
                const idx = indices[iy * res_x + ix];

                if (ix + 2 < res_x) {
                    try body.addDistanceConstraint(idx, indices[iy * res_x + ix + 2]);
                }
                if (iy + 2 < res_y) {
                    try body.addDistanceConstraint(idx, indices[(iy + 2) * res_x + ix]);
                }
            }
        }

        body.computeCenterOfMass();
        return body;
    }

    /// Compute center of mass.
    pub fn computeCenterOfMass(self: *SoftBody) void {
        var com: [3]f32 = .{ 0, 0, 0 };
        var total_mass: f32 = 0;

        for (self.particles.items) |p| {
            if (p.inv_mass > 0) {
                const m = 1.0 / p.inv_mass;
                com[0] += p.position[0] * m;
                com[1] += p.position[1] * m;
                com[2] += p.position[2] * m;
                total_mass += m;
            }
        }

        if (total_mass > 0) {
            self.center_of_mass = .{
                com[0] / total_mass,
                com[1] / total_mass,
                com[2] / total_mass,
            };
            self.total_mass = total_mass;
        }
    }

    /// CPU simulation step.
    pub fn step(self: *SoftBody, dt: f32, gravity: [3]f32, iterations: u32) void {
        // Apply external forces and predict positions
        for (self.particles.items) |*p| {
            if (p.inv_mass > 0) {
                // Apply gravity
                p.velocity[0] += gravity[0] * dt;
                p.velocity[1] += gravity[1] * dt;
                p.velocity[2] += gravity[2] * dt;

                // Apply damping
                p.velocity[0] *= 1.0 - self.def.damping;
                p.velocity[1] *= 1.0 - self.def.damping;
                p.velocity[2] *= 1.0 - self.def.damping;

                // Predict position
                p.predicted[0] = p.position[0] + p.velocity[0] * dt;
                p.predicted[1] = p.position[1] + p.velocity[1] * dt;
                p.predicted[2] = p.position[2] + p.velocity[2] * dt;
            } else {
                p.predicted = p.position;
            }
        }

        // Solve constraints
        for (0..iterations) |_| {
            // Distance constraints
            for (self.distance_constraints.items) |c| {
                self.solveDistanceConstraint(c);
            }

            // Volume constraints
            for (self.volume_constraints.items) |c| {
                self.solveVolumeConstraint(c);
            }
        }

        // Update velocities and positions
        const inv_dt = 1.0 / dt;
        for (self.particles.items) |*p| {
            if (p.inv_mass > 0) {
                p.velocity[0] = (p.predicted[0] - p.position[0]) * inv_dt;
                p.velocity[1] = (p.predicted[1] - p.position[1]) * inv_dt;
                p.velocity[2] = (p.predicted[2] - p.position[2]) * inv_dt;
                p.position = p.predicted;
            }
        }

        // Ground collision
        for (self.particles.items) |*p| {
            if (p.position[2] < self.def.margin) {
                p.position[2] = self.def.margin;
                p.velocity[2] = 0;
                // Friction
                p.velocity[0] *= 1.0 - self.def.friction;
                p.velocity[1] *= 1.0 - self.def.friction;
            }
        }

        self.computeCenterOfMass();
    }

    fn solveDistanceConstraint(self: *SoftBody, c: DistanceConstraint) void {
        var p1 = &self.particles.items[c.p1];
        var p2 = &self.particles.items[c.p2];

        const w1 = p1.inv_mass;
        const w2 = p2.inv_mass;
        const w_sum = w1 + w2;

        if (w_sum < constants.EPSILON) return;

        const diff: [3]f32 = .{
            p2.predicted[0] - p1.predicted[0],
            p2.predicted[1] - p1.predicted[1],
            p2.predicted[2] - p1.predicted[2],
        };

        const dist = @sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist < constants.EPSILON) return;

        const error_val = dist - c.rest_length;
        const correction = c.stiffness * error_val / (dist * w_sum);

        p1.predicted[0] += diff[0] * correction * w1;
        p1.predicted[1] += diff[1] * correction * w1;
        p1.predicted[2] += diff[2] * correction * w1;

        p2.predicted[0] -= diff[0] * correction * w2;
        p2.predicted[1] -= diff[1] * correction * w2;
        p2.predicted[2] -= diff[2] * correction * w2;
    }

    fn solveVolumeConstraint(self: *SoftBody, c: VolumeConstraint) void {
        var p0 = &self.particles.items[c.particles[0]];
        var p1 = &self.particles.items[c.particles[1]];
        var p2 = &self.particles.items[c.particles[2]];
        var p3 = &self.particles.items[c.particles[3]];

        const volume = computeTetraVolume(p0.predicted, p1.predicted, p2.predicted, p3.predicted);
        const error_val = volume - c.rest_volume;

        if (@abs(error_val) < constants.EPSILON) return;

        // Compute gradients
        const g0 = cross3(sub3(p1.predicted, p3.predicted), sub3(p2.predicted, p3.predicted));
        const g1 = cross3(sub3(p2.predicted, p3.predicted), sub3(p0.predicted, p3.predicted));
        const g2 = cross3(sub3(p0.predicted, p3.predicted), sub3(p1.predicted, p3.predicted));
        const g3: [3]f32 = .{
            -(g0[0] + g1[0] + g2[0]),
            -(g0[1] + g1[1] + g2[1]),
            -(g0[2] + g1[2] + g2[2]),
        };

        const w0 = p0.inv_mass;
        const w1 = p1.inv_mass;
        const w2 = p2.inv_mass;
        const w3 = p3.inv_mass;

        const denom = w0 * dot3(g0, g0) + w1 * dot3(g1, g1) + w2 * dot3(g2, g2) + w3 * dot3(g3, g3);

        if (denom < constants.EPSILON) return;

        const lambda = c.stiffness * error_val / (6.0 * denom);

        p0.predicted[0] -= lambda * w0 * g0[0];
        p0.predicted[1] -= lambda * w0 * g0[1];
        p0.predicted[2] -= lambda * w0 * g0[2];

        p1.predicted[0] -= lambda * w1 * g1[0];
        p1.predicted[1] -= lambda * w1 * g1[1];
        p1.predicted[2] -= lambda * w1 * g1[2];

        p2.predicted[0] -= lambda * w2 * g2[0];
        p2.predicted[1] -= lambda * w2 * g2[1];
        p2.predicted[2] -= lambda * w2 * g2[2];

        p3.predicted[0] -= lambda * w3 * g3[0];
        p3.predicted[1] -= lambda * w3 * g3[1];
        p3.predicted[2] -= lambda * w3 * g3[2];
    }
};

// Helper functions

fn distance(a: [3]f32, b: [3]f32) f32 {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dz = b[2] - a[2];
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

fn computeTetraVolume(p0: [3]f32, p1: [3]f32, p2: [3]f32, p3: [3]f32) f32 {
    const a = sub3(p1, p0);
    const b = sub3(p2, p0);
    const c = sub3(p3, p0);
    return @abs(dot3(a, cross3(b, c))) / 6.0;
}

fn sub3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn cross3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn dot3(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Tests

test "create soft body cube" {
    const allocator = std.testing.allocator;

    var body = try SoftBody.createCube(
        allocator,
        .{ 0, 0, 1 },
        1.0,
        3,
        .{ .stiffness = 0.9, .volume_stiffness = 1.0 },
    );
    defer body.deinit();

    try std.testing.expect(body.particles.items.len == 27); // 3x3x3
    try std.testing.expect(body.distance_constraints.items.len > 0);
    try std.testing.expect(body.volume_constraints.items.len > 0);
}

test "create soft body cloth" {
    const allocator = std.testing.allocator;

    var body = try SoftBody.createCloth(
        allocator,
        .{ -1, -1, 2 },
        2.0,
        2.0,
        10,
        10,
        .{ .stiffness = 0.9 },
    );
    defer body.deinit();

    try std.testing.expect(body.particles.items.len == 100); // 10x10
    try std.testing.expect(body.distance_constraints.items.len > 0);
}

test "soft body step" {
    const allocator = std.testing.allocator;

    var body = try SoftBody.createCube(
        allocator,
        .{ 0, 0, 2 },
        0.5,
        2,
        .{ .stiffness = 0.9 },
    );
    defer body.deinit();

    const initial_z = body.center_of_mass[2];

    // Step simulation
    for (0..100) |_| {
        body.step(0.01, .{ 0, 0, -9.81 }, 4);
    }

    // Should have fallen due to gravity
    try std.testing.expect(body.center_of_mass[2] < initial_z);
}
