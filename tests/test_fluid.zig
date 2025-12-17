//! Tests for SPH fluid simulation.

const std = @import("std");
const testing = std.testing;

const fluid = @import("zeno").physics.fluid;
const FluidParticle = fluid.FluidParticle;
const FluidParams = fluid.FluidParams;
const SpatialHash = fluid.SpatialHash;

test "fluid particle creation" {
    const particle = FluidParticle{
        .position = .{ 0, 0, 0 },
        .velocity = .{ 0, 0, 0 },
        .predicted = .{ 0, 0, 0 },
        .density = 1000.0,
        .pressure = 0.0,
        .acceleration = .{ 0, 0, -9.81 },
        .mass = 0.02,
        .cell_id = 0,
    };

    try testing.expect(particle.density == 1000.0);
    try testing.expect(particle.mass == 0.02);
}

test "fluid particle defaults" {
    const particle = FluidParticle{
        .position = .{ 1, 2, 3 },
        .velocity = .{ 0, 0, 0 },
        .predicted = .{ 1, 2, 3 },
    };

    try testing.expect(particle.density == 0);
    try testing.expect(particle.pressure == 0);
    try testing.expect(particle.mass == 1.0);
}

test "fluid params creation" {
    const params = FluidParams{
        .rest_density = 1000.0,
        .particle_mass = 0.02,
        .particle_radius = 0.05,
        .gas_constant = 2000.0,
        .viscosity = 0.01,
        .surface_tension = 0.0728,
    };

    try testing.expect(params.rest_density == 1000.0);
    try testing.expect(params.viscosity == 0.01);
    try testing.expect(params.surface_tension == 0.0728);
}

test "fluid params defaults" {
    const params = FluidParams{};

    try testing.expect(params.rest_density == 1000.0);
    try testing.expect(params.particle_radius == 0.05);
    try testing.expect(params.gas_constant == 2000.0);
    try testing.expect(params.dt == 0.002);
}

test "water parameters" {
    // Typical water simulation parameters
    const params = FluidParams{
        .rest_density = 1000.0, // kg/m³
        .particle_mass = 0.02,
        .particle_radius = 0.05,
        .gas_constant = 2000.0,
        .viscosity = 0.001, // Pa·s (water at 20°C)
        .surface_tension = 0.0728, // N/m
        .gravity = .{ 0, 0, -9.81 },
    };

    try testing.expect(params.rest_density == 1000.0);
    try testing.expect(params.viscosity == 0.001);
}

test "honey parameters" {
    // Viscous fluid (honey-like)
    const params = FluidParams{
        .rest_density = 1400.0, // kg/m³
        .particle_mass = 0.05,
        .viscosity = 2.0, // Pa·s (honey)
        .gas_constant = 1000.0,
    };

    try testing.expect(params.viscosity == 2.0);
    try testing.expect(params.rest_density == 1400.0);
}

test "spatial hash creation" {
    var hash = SpatialHash.init(testing.allocator, 0.1);
    defer hash.deinit();

    try testing.expect(hash.cell_size == 0.1);
}

test "spatial hash function" {
    var hash = SpatialHash.init(testing.allocator, 0.1);
    defer hash.deinit();

    // Same position should hash to same cell
    const pos1: [3]f32 = .{ 0.05, 0.05, 0.05 };
    const pos2: [3]f32 = .{ 0.06, 0.04, 0.05 };

    const h1 = hash.hash(pos1);
    const h2 = hash.hash(pos2);

    try testing.expect(h1 == h2);
}

test "spatial hash different cells" {
    var hash = SpatialHash.init(testing.allocator, 0.1);
    defer hash.deinit();

    // Positions in different cells should hash differently
    const pos1: [3]f32 = .{ 0.05, 0.05, 0.05 };
    const pos2: [3]f32 = .{ 0.15, 0.05, 0.05 };

    const h1 = hash.hash(pos1);
    const h2 = hash.hash(pos2);

    try testing.expect(h1 != h2);
}

test "boundary parameters" {
    const params = FluidParams{
        .boundary_friction = 0.5,
        .boundary_restitution = 0.3,
    };

    try testing.expect(params.boundary_friction >= 0.0);
    try testing.expect(params.boundary_friction <= 1.0);
    try testing.expect(params.boundary_restitution >= 0.0);
    try testing.expect(params.boundary_restitution <= 1.0);
}

test "gravity direction" {
    const params = FluidParams{
        .gravity = .{ 0, 0, -9.81 },
    };

    try testing.expect(params.gravity[2] < 0); // Pointing down
}
