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

@group(1) @binding(0)
var scalar_source: texture_storage_3d<rgba16float, write>;

const center:  vec3<f32> = vec3<f32>(64.0, 32.0, 64.0);
const radius:  f32 = 24.0;
const radius2: f32 = radius * radius;
const peak:    f32 = 1.0;

// Units of fuel injected at the Gaussian peak.
const INJECTION_RATE: f32 = 1.0;

// Smoke density injected as a fixed multiple of temperature injection.
// Smoke equilibrium: S_eq = INJECTION_RATE * SMOKE_SCALE / ln(2)
// At 15.0: S_eq(center) ≈ 6.5 → clearly opaque.
// Because smoke = temp * SMOKE_SCALE at every voxel, both channels follow
// the same Gaussian profile → rendered opacity and color scale together
// → consistent radial color gradient.
const SMOKE_SCALE: f32 = 1.0;

/* Adds smoke density (x) and temperature (y) to the source texture. */
@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec3<i32>(gid);
    let position = vec3<f32>(gid) + vec3<f32>(0.5);
    let d = position - center;
    let dist2 = dot(d, d);

    if dist2 > radius2 { return; }

    let sigma  = max(radius * 0.35, 1e-6);
    let sigma2 = sigma * sigma;

    let gaussian = peak * exp(-dist2 / (2.0 * sigma2));
    let fuel = gaussian * INJECTION_RATE;

    textureStore(
        scalar_source,
        coord,
        vec4<f32>(0.0, 0.0, fuel, 0.0)
    );
}
