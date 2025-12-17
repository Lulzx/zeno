//! Tests for soft body (PBD) simulation.

const std = @import("std");
const testing = std.testing;

const soft_body = @import("zeno").physics.soft_body;
const Particle = soft_body.Particle;
const DistanceConstraint = soft_body.DistanceConstraint;
const VolumeConstraint = soft_body.VolumeConstraint;
const BendingConstraint = soft_body.BendingConstraint;
const SoftBodyDef = soft_body.SoftBodyDef;

test "particle creation" {
    const particle = Particle{
        .position = .{ 0, 0, 0 },
        .velocity = .{ 0, 0, 0 },
        .predicted = .{ 0, 0, 0 },
        .inv_mass = 1.0,
        .rest_position = .{ 0, 0, 0 },
    };

    try testing.expect(particle.inv_mass == 1.0);
}

test "distance constraint creation" {
    const constraint = DistanceConstraint{
        .p1 = 0,
        .p2 = 1,
        .rest_length = 0.1,
        .stiffness = 0.9,
    };

    try testing.expect(constraint.p1 == 0);
    try testing.expect(constraint.p2 == 1);
    try testing.expect(constraint.rest_length == 0.1);
    try testing.expect(constraint.stiffness == 0.9);
}

test "distance constraint default stiffness" {
    const constraint = DistanceConstraint{
        .p1 = 0,
        .p2 = 1,
        .rest_length = 1.0,
    };

    try testing.expect(constraint.stiffness == 1.0);
}

test "volume constraint creation" {
    const constraint = VolumeConstraint{
        .particles = .{ 0, 1, 2, 3 },
        .rest_volume = 1.0,
        .stiffness = 0.8,
    };

    try testing.expect(constraint.particles[0] == 0);
    try testing.expect(constraint.particles[3] == 3);
    try testing.expect(constraint.rest_volume == 1.0);
}

test "bending constraint creation" {
    const constraint = BendingConstraint{
        .particles = .{ 0, 1, 2, 3 },
        .rest_angle = 0.0,
        .stiffness = 0.3,
    };

    try testing.expect(constraint.rest_angle == 0.0);
    try testing.expect(constraint.stiffness == 0.3);
}

test "soft body def creation" {
    const def = SoftBodyDef{
        .num_particles = 100,
        .mass = 0.5,
        .stiffness = 0.9,
        .volume_stiffness = 0.95,
        .damping = 0.02,
    };

    try testing.expect(def.num_particles == 100);
    try testing.expect(def.mass == 0.5);
    try testing.expect(def.stiffness == 0.9);
}

test "soft body def defaults" {
    const def = SoftBodyDef{};

    try testing.expect(def.num_particles == 0);
    try testing.expect(def.mass == 1.0);
    try testing.expect(def.stiffness == 0.9);
    try testing.expect(def.volume_stiffness == 1.0);
    try testing.expect(def.damping == 0.01);
}

test "cloth parameters" {
    // Typical cloth parameters
    const def = SoftBodyDef{
        .num_particles = 625, // 25x25 grid
        .stiffness = 0.8,
        .damping = 0.01,
        .friction = 0.3,
        .margin = 0.005,
    };

    try testing.expect(def.num_particles == 625);
    try testing.expect(def.friction == 0.3);
}

test "jelly parameters" {
    // Soft volumetric body
    const def = SoftBodyDef{
        .num_particles = 1000,
        .stiffness = 0.95,
        .volume_stiffness = 0.9,
        .damping = 0.05,
        .pressure = 0.0,
    };

    try testing.expect(def.volume_stiffness == 0.9);
    try testing.expect(def.pressure == 0.0);
}

test "inflatable parameters" {
    // Inflatable soft body
    const def = SoftBodyDef{
        .num_particles = 500,
        .stiffness = 0.5,
        .pressure = 100.0, // Internal pressure
    };

    try testing.expect(def.pressure == 100.0);
}

test "stiffness in valid range" {
    const constraint = DistanceConstraint{
        .p1 = 0,
        .p2 = 1,
        .rest_length = 1.0,
        .stiffness = 0.5,
    };

    // PBD stiffness should be in [0, 1]
    try testing.expect(constraint.stiffness >= 0.0);
    try testing.expect(constraint.stiffness <= 1.0);
}

test "fixed particle" {
    // Fixed particle has inv_mass = 0
    const fixed = Particle{
        .position = .{ 0, 0, 0 },
        .velocity = .{ 0, 0, 0 },
        .predicted = .{ 0, 0, 0 },
        .inv_mass = 0.0, // Fixed!
        .rest_position = .{ 0, 0, 0 },
    };

    try testing.expect(fixed.inv_mass == 0.0);
}
