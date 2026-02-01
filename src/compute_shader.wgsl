// Uniform buffers
struct Params {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
    inject_sources_flag: u32,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    viewport: vec2<f32>,
    _pad0: vec2<f32>
}
@group(0) @binding(0)
var<uniform> params: Params;

// Texture bindings
@group(1) @binding(0)
var velocity_field_texture_read: texture_3d<f32>;
@group(1) @binding(1)
var velocity_feild_texture_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2)
var density_scalar_field_texture_read: texture_3d<f32>;
@group(1) @binding(3)
var density_scalar_field_texture_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(4)
var field_sampler: sampler;

// Constants and structs for density and force source calculation.
const center = vec3<f32>(32.0, 16.0, 32.0);
const radius: f32 = 4.0;
const radius2: f32 = radius * radius;
const peak: f32 = 1.0;
const strength: f32 = 1.0;
const eps: f32 = 1e-6;
struct DensityAndForce {
    density: f32,
    force: vec3<f32>,
}

@compute
@workgroup_size(4, 4, 4)
fn main (
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Global invocation id corresponds to the index of a voxel in the simulation grid.
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.depth) {
        // In case of out of bounds.
        return;
    }

    // Add sources to the scalar fields.
    let density_and_forces = compute_sources(gid);
    let density_source = density_and_forces.density;
    let force_source = density_and_forces.force;

    advect_density(gid);
    advect_velocity(gid);

    // For now we are going to assume that the fluid is invisid. No diffusion
    // TODO: Investigate diffusion for fire simulation (might not be necessary).

    // Add forces to the velocity field.

    // Solve for possion pressure equation.

    // Substract pressure gradient to achieve stability.
}

fn advect_density(gid: vec3<u32>) {
    let uvw = voxel_center_uvw(gid);
    let vel = textureSampleLevel(velocity_field_texture_read, field_sampler, uvw, 0.0).xyz;
    let uvw_back = backtrace(uvw, vel);

    let backtraced_density = textureSampleLevel(density_scalar_field_texture_read, field_sampler, uvw_back, 0.0).x;

    textureStore(
        density_scalar_field_texture_write,
        vec3<i32>(gid),
        vec4<f32>(backtraced_density, 0.0, 0.0, 0.0)
    );
}

fn advect_velocity(gid: vec3<u32>) {
    let uvw = voxel_center_uvw(gid);
    let vel = textureSampleLevel(velocity_field_texture_read, field_sampler, uvw, 0.0).xyz;
    let uvw_back = backtrace(uvw, vel);

    // store velocity in RGB (A unused)
    let backtraced_velocity = textureSampleLevel(velocity_field_texture_read, field_sampler, uvw_back, 0.0).xyz;

    textureStore(
        velocity_feild_texture_write,
        vec3<i32>(gid),
        vec4<f32>(backtraced_velocity, 0.0)
    );
}

// Returns the center of the voxel indexed at gid.
fn voxel_center_uvw(gid: vec3<u32>) -> vec3<f32> {
    let w = f32(params.width);
    let h = f32(params.height);
    let d = f32(params.depth);
    return vec3<f32>(
        (f32(gid.x) + 0.5) / w,
        (f32(gid.y) + 0.5) / h,
        (f32(gid.z) + 0.5) / d
    );
}

// Returns the uvw backtraced by the given velocity scaled by the simulation timestep.
fn backtrace(uvw: vec3<f32>, velocity: vec3<f32>) -> vec3<f32> {
    let w = f32(params.width);
    let h = f32(params.height);
    let d = f32(params.depth);

    // Velocity is in cells per second so convert to texture coordinates per second, which are in range of [0, 1]
    let vel_uvw = vec3<f32>(velocity.x / w, velocity.y / h, velocity.z / d);
    return uvw - params.dt * vel_uvw;
}

// Computes the density and force source at the given voxel. Adds a blob of density and forces to puff it out.
// Will return a non-zero value when params.inject_source_flag is on.
fn compute_sources(gid: vec3<u32>) -> DensityAndForce {
    var output: DensityAndForce;
    output.density = 0.0;
    output.force = vec3<f32>(0.0);

    if (params.inject_sources_flag == u0) {
        return
    }

    let coord = vec3<i32>(gid);

    let position = vec3<f32>(gid) + vec3<f32>(0.5);
    let distance_from_center = position - center;
    let distance_from_center2 = dot(distance_from_center, distance_from_center);

    // Position is too far from center, no source.
    if (distance_from_center2 > radius2) {
        return output;
    }

    let sigma = max(radius * 0.35, 1e-6);
    let sigma2 = sigma * sigma;
    let density_to_add = peak * exp(-distance_from_center2 / (2.0 * sigma2));
    output.density = density_to_add;

    let dir = distance_from_center / max(sqrt(distance_from_center2), eps); // normalized
    let force = dir * (strength * add);
    output.force = force;

    return output;
}
