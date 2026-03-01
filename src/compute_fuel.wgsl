// Uniform buffers
struct Params {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    viewport: vec2<f32>,
    _pad0: vec2<f32>
}
@group(0) @binding(0)
var<uniform> params: Params;

// Texture bindings
@group(1) @binding(0)
var scalar_field_read: texture_3d<f32>;
@group(1) @binding(1)
var scalar_field_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2)
var field_sampler: sampler;

// Takes on values [0, 1]
const gamma_fuel: f32 = 0.005;

/**
 * Fuel is stored in the first (x) channel.
 */

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

    let current_fuel = get_fuel(gid);
    let decay = pow(1.0 - gamma_fuel, params.dt);
    let new_fuel = current_fuel * decay;

    textureStore(
        scalar_field_write,
        vec3<i32>(gid),
        vec4<f32>(new_fuel, get_temperature(gid), 0.0, 0.0)
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

fn get_fuel(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).x;
}

fn get_temperature(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).y;
}
