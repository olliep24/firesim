// Uniform buffers
struct CameraUniform {
    camera_pos: vec3<f32>,
    _pad0: f32,

    camera_forward: vec3<f32>,
    _pad1: f32,

    camera_right: vec3<f32>,
    _pad2: f32,

    camera_up: vec3<f32>,
    _pad3: f32,

    tan_half_fovy: f32,
    aspect: f32,
    _pad4: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

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
@group(1) @binding(0)
var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Cover entire screen with a triangle
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

// Texture bindings: x = smoke density, y = temperature (normalized [0,1])
@group(2) @binding(0)
var density_scalar_field: texture_3d<f32>;
@group(2) @binding(1)
var field_sampler: sampler;

/* Blackbody radiation helpers */

// First radiation constant: c₁ = 2hc² = 3.74177419e-16 W·m², expressed in nm units
// (λ⁵ is in nm⁵, so c₁ is scaled by (10⁹)⁵ / (10⁹)² ... net factor 10⁴⁵ × 10⁻¹⁶ = 10²⁹)
const C1: f32 = 3.74177e29;   // W · sr⁻¹ · m⁻² · nm⁻⁵

// Second radiation constant: c₂ = hc/k
const C2: f32 = 14387769.0;   // nm · K

// Camera exposure: converts physical spectral radiance [W·sr⁻¹·m⁻²] to display range.
// At 2000 K, the integrated Planck Y ≈ 5e12 W·sr⁻¹·m⁻², so 1e-13 maps peak fire to
// ~0.5 pre-tone-mapping. Tune: too dark → increase; blown out → decrease.
const EXPOSURE: f32 = 1e-13;

// Grey color for cool smoke scattering ambient light.
// Increase for brighter/lighter smoke; decrease for darker smoke.
const SMOKE_COLOR: vec3<f32> = vec3<f32>(0.35, 0.35, 0.35);

// Extinction coefficient for smoke. Higher = denser/more opaque smoke.
const SIGMA_SMOKE: f32 = 0.1;

// Planck spectral radiance B(λ, T) = c₁ / (λ⁵ · (exp(c₂/λT) − 1))
// Returns 0 when the exponent would overflow f32 (very low T or short λ).
fn planck(lambda_nm: f32, T_K: f32) -> f32 {
    let x = C2 / (lambda_nm * T_K);
    if x > 85.0 { return 0.0; }
    return C1 / (pow(lambda_nm, 5.0) * (exp(x) - 1.0));
}

// CIE 1931 color matching functions approximated by sums of Gaussians.
// Coefficients from Wyman, Sloan & Shirley (2013).
fn gauss(lambda: f32, mu: f32, sigma: f32) -> f32 {
    let t = (lambda - mu) / sigma;
    return exp(-0.5 * t * t);
}

fn cie_x(l: f32) -> f32 {
    return 1.056 * gauss(l, 599.8, 37.9)
         + 0.362 * gauss(l, 442.0, 16.0)
         - 0.065 * gauss(l, 501.1, 20.4);
}

fn cie_y(l: f32) -> f32 {
    return 0.821 * gauss(l, 568.8, 46.4)
         + 0.286 * gauss(l, 530.9, 21.7);
}

fn cie_z(l: f32) -> f32 {
    return 1.217 * gauss(l, 437.0, 17.0)
         + 0.681 * gauss(l, 459.0, 27.0);
}

// Numerically integrate Planck × CIE CMFs over 380–720 nm (35 steps × 10 nm).
// Returns physical radiance scaled by EXPOSURE for display.
fn blackbody_xyz(T_K: f32) -> vec3<f32> {
    var xyz = vec3<f32>(0.0);
    for (var i = 0u; i < 35u; i++) {
        let l = 380.0 + f32(i) * 10.0;
        let p = planck(l, T_K);
        xyz += vec3<f32>(p * cie_x(l), p * cie_y(l), p * cie_z(l));
    }
    return xyz * EXPOSURE;
}

// CIE XYZ → linear sRGB using the standard D65 whitepoint matrix.
// In WGSL mat3x3 is column-major, so each vec3 argument is a column.
fn xyz_to_linear_srgb(xyz: vec3<f32>) -> vec3<f32> {
    return mat3x3<f32>(
        vec3<f32>( 3.2406, -0.9689,  0.0557),
        vec3<f32>(-1.5372,  1.8758, -0.2040),
        vec3<f32>(-0.4986,  0.0415,  1.0570)
    ) * xyz;
}

// Reinhard tone mapping followed by gamma correction (γ = 2.2).
fn tone_map(c: vec3<f32>) -> vec3<f32> {
    let mapped = c / (c + vec3<f32>(1.0));
    return pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

// Convert simulation temperature (Kelvin) to a display RGB color.
fn blackbody_color(temperature: f32) -> vec3<f32> {
    if temperature < 300.0 { return vec3<f32>(0.0); }
    let xyz = blackbody_xyz(temperature);
    return tone_map(xyz_to_linear_srgb(xyz));
}

@fragment
fn fs_main(@builtin(position) frag_clip_position: vec4<f32>) -> @location(0) vec4<f32> {
    // Frag (pixel) coordinates normalized to 0..1
    let uv = frag_clip_position.xy / params.viewport;

    // Convert to NDC. We don't care about z coordinate because we're doing ray construction.
    let ndc = vec2<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0
    );

    // Construct ray
    let ro = camera.camera_pos;
    let rd = normalize(
        camera.camera_forward +
        ndc.x * camera.aspect * camera.tan_half_fovy * camera.camera_right +
        ndc.y * camera.tan_half_fovy * camera.camera_up
    );

    let bmin = params.box_min.xyz;
    let bmax = params.box_max.xyz;

    let hit = intersect_aabb(ro, rd, bmin, bmax);
    let t_enter = max(hit.x, 0.0);
    let t_exit = hit.y;

    if (t_exit <= t_enter) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let steps: u32 = 64u;
    let len = t_exit - t_enter;
    let ds = len / f32(steps);

    var t = t_enter;
    var accum_color = vec3<f32>(0.0);
    var accum_alpha = 0.0;

    for (var i: u32 = 0u; i < steps; i = i + 1u) {
        let p = ro + rd * (t + 0.5 * ds);
        let uvw = (p - bmin) / (bmax - bmin);

        let s = textureSampleLevel(density_scalar_field, field_sampler, uvw, 0.0);
        let smoke = s.x;
        let temp = s.y;

        // Beer-Lambert extinction: alpha contribution from smoke density this step
        let smoke_alpha = 1.0 - exp(-smoke * SIGMA_SMOKE * ds);

        // Emission from blackbody radiation at this temperature
        let emit_color = blackbody_color(temp);

        // Front-to-back compositing: smoke scattering + fire emission
        accum_color += (1.0 - accum_alpha) * (SMOKE_COLOR * smoke_alpha + emit_color);
        accum_alpha += (1.0 - accum_alpha) * smoke_alpha;

        if (accum_alpha > 0.99) { break; }

        t = t + ds;
    }

    return vec4<f32>(accum_color, 1.0);
}

fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / rd;
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;

    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);

    let t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
    let t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);
    return vec2<f32>(t_enter, t_exit);
}
