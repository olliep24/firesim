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

// Temperature normalization: T_sim ∈ [0,1], where T_sim = 1.0 corresponds to T_MAX_KELVIN.
// Must match T_MAX_KELVIN in render_shader.wgsl. TODO: Update.
const BURN_TEMPERATURE: f32 = 2000.0;

// Stefan-Boltzmann constant (NIST 2018 CODATA)
const SIGMA: f32 = 5.6704e-8;   // W · m⁻² · K⁻⁴

// Emissivity of soot-laden fire gas (dimensionless, 0..1).
// 1.0 = perfect blackbody; soot-laden combustion gas ≈ 0.85.
const EMISSIVITY: f32 = 0.85;

// Thermal mass per unit area: ρ × c_v × L  [J · m⁻² · K⁻¹]
// At 750: T(1s) = 0.73 → 1460 K (orange), T(3s) = 0.56 → 1120 K (red).
// Increase to slow the yellow→red gradient; decrease to speed it up.
const THERMAL_MASS: f32 = 50.0;

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
    let cooling = get_cooling(gid);
    let cooled_temperature = current_temperature + cooling;

    let fuel = get_fuel(gid);
    let fuel_temperature = fuel * BURN_TEMPERATURE;

    let new_temperature = max(cooled_temperature, fuel_temperature);
    textureStore(
        scalar_field_write,
        vec3<i32>(gid),
        vec4<f32>(get_smoke(gid), new_temperature, 0.0, 0.0)
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

// Stefan-Boltzmann radiative cooling: dT_K/dt = −(ε·σ / MC) · T_K⁴
//
// Converting to normalized simulation temperature (T_sim = T_K / T_MAX_KELVIN):
//   dT_sim/dt = −(ε·σ·T_MAX_KELVIN³ / MC) · T_sim⁴
fn get_cooling(index: vec3<u32>) -> f32 {
    let T_sim = get_current_temperature(index);
    let T_K = T_sim;
    // dT_K/dt = -(ε·σ / MC) · T_K⁴
    let dTK_dt = -(EMISSIVITY * SIGMA / THERMAL_MASS) * T_K * T_K * T_K * T_K;
    // Convert back to normalized units and apply timestep
    return dTK_dt * params.dt;
}

fn get_fuel(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(scalar_field_read, field_sampler, uvw, 0.0).z;
}
