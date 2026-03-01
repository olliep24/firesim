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

// Energy released per unit of fuel per second. Temperature is normalized to [0, 1].
// Equilibrium temperature: T_eq = (T_burn / COOLING_RATE)^0.25
// With T_burn=3.0 and COOLING_RATE=5.0: T_eq = 0.88 → 1760 K (visible orange)
const T_burn: f32 = 3.0;

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

    let new_temperature = clamp(current_temperature + heating + cooling, 0.0, 1.0);
    textureStore(
        scalar_field_write,
        vec3<i32>(gid),
        vec4<f32>(get_fuel(gid), new_temperature, 0.0, 0.0)
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
    return fuel * T_burn * params.dt;
}

// Stefan-Boltzmann cooling: dT/dt = -COOLING_RATE * T^4
//
// Both heating and cooling are now dt-scaled (rates in per-second units),
// so they reach equilibrium at: T_eq^4 = (fuel * T_burn) / COOLING_RATE
//
// With COOLING_RATE = 2.5 and full fuel (1.0):
//   T_eq = (1.0 / 2.5)^0.25 ≈ 0.80  →  1600 K  →  orange-yellow
// With fuel = 0.2:
//   T_eq ≈ 0.54  →  1080 K  →  dark red
// Raise COOLING_RATE to make the flame cooler/redder; lower it to make it hotter/whiter.
const COOLING_RATE: f32 = 5.0;

fn get_cooling(index: vec3<u32>) -> f32 {
    let T = get_current_temperature(index);
    return -COOLING_RATE * T * T * T * T * params.dt;
}
