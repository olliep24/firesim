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

// The temperature at which the fuel burns.
const T_burn: f32 = 500.0;

/**
 * Temperature is stored in the second (y) channel.
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

    let current_temperature = get_current_temperature(gid);
    let heating = get_heating(gid);
    let cooling = get_cooling(gid);

    textureStore(
        scalar_field_write,
        vec3<i32>(gid),
        vec4<f32>(get_fuel(gid), current_temperature + heating + cooling, 0.0, 0.0)
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

fn get_current_temperature(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).y;
}

fn get_fuel(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).x;
}

fn get_heating(index: vec3<u32>) -> f32 {
    let fuel = get_fuel(index);
    return fuel * T_burn;
}

fn get_cooling(index: vec3<u32>) -> f32 {
    // TODO: Revisit the Stefan-Boltzmann Law
    return 0.0;
}
