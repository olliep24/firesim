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


const COOLING: f32 = 3000.0;
const BURN_TEMPERATURE: f32 = 1700.0;

// Temperature normalization: T_sim ∈ [0,1], where T_sim = 1.0 corresponds to T_MAX_KELVIN.
// Must match T_MAX_KELVIN in render_shader.wgsl. TODO: Update.
// const BURN_TEMPERATURE: f32 = 2000.0;

/**
 * Temperature is stored in the second (y) channel.
 * Temperature is set by source injection (add_source.wgsl) and decays via
 * Stefan-Boltzmann radiative cooling. There is no per-frame heating from smoke.
 */

@compute
@workgroup_size(4, 4, 4)
fn main (
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.depth) {
        return;
    }

    let current_temperature = get_current_temperature(gid);
    let cooling = get_cooling(current_temperature);
    let cooled_temperature = current_temperature + cooling;

    let fuel = get_fuel(gid);
    let fuel_temperature = fuel * BURN_TEMPERATURE;

    let new_temperature = max(cooled_temperature, fuel_temperature);
    textureStore(
        scalar_field_write,
        vec3<i32>(gid),
        vec4<f32>(get_smoke(gid), new_temperature, fuel, 0.0)
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

fn get_smoke(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).x;
}

// Stefan-Boltzmann radiative coolin (aproximation).
fn get_cooling(T: f32) -> f32 {
    return -params.dt * COOLING * pow(T / BURN_TEMPERATURE, 4.0);
}

fn get_fuel(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).z;
}
