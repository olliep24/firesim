// Uniform buffers
struct Params {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    viewport: vec2<f32>,
    elapsed_time: f32,
    _pad0: f32,
}
@group(0) @binding(0)
var<uniform> params: Params;

@group(1) @binding(0)
var scalar_source: texture_storage_3d<rgba16float, write>;

const center: vec3<f32> = vec3<f32>(64.0, 32.0, 64.0);
const radius: f32 = 24.0;
const radius2: f32 = radius * radius;
const peak: f32 = 1.0;

// Units of fuel injected at the Gaussian peak.
const INJECTION_RATE: f32 = 1.5;

// Spatial frequency of noise features (smaller = larger blobs).
const NOISE_SCALE: f32 = 0.08;
// How strongly noise modulates fuel (0 = no effect, 1 = fuel can reach zero at noise troughs).
const NOISE_AMPLITUDE: f32 = 0.5;
// How fast the noise pattern animates (units/second).
const NOISE_SPEED: f32 = 1;

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash3(i + vec3(0.0, 0.0, 0.0)), hash3(i + vec3(1.0, 0.0, 0.0)), u.x),
            mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), u.x),
            mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), u.x), u.y),
        u.z
    );
}

// 3-octave FBM, returns value in [-1, 1].
fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 3; i++) {
        v += a * (value_noise(p * freq) * 2.0 - 1.0);
        a *= 0.5;
        freq *= 2.0;
    }
    return v;
}

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

    let noise_p = position * NOISE_SCALE + vec3(0.0, 0.0, params.elapsed_time * NOISE_SPEED);
    let noise = fbm(noise_p);
    let fuel = gaussian * INJECTION_RATE * max(0.0, 1.0 + NOISE_AMPLITUDE * noise);

    textureStore(
        scalar_source,
        coord,
        vec4<f32>(0.0, 0.0, fuel, 0.0)
    );
}
