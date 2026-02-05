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

// Texture bindings
@group(2) @binding(0)
var density_scalar_field: texture_3d<f32>;
@group(2) @binding(1)
var field_sampler: sampler;

// Fragment shader
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
        // Miss
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let steps: u32 = 64u;
    let len = t_exit - t_enter;
    let ds = len / f32(steps);

    var t = t_enter;
    var accum_alpha = 0.0;

    for (var i: u32 = 0u; i < steps; i = i + 1u) {
        let p = ro + rd * (t + 0.5 * ds);

        let uvw = (p - bmin) / (bmax - bmin); // should be 0..1 in-box
        let d = textureSampleLevel(density_scalar_field, field_sampler, uvw, 0.0).x;

        // Convert density -> per-step opacity (Beer-Lambert)
        let sigma_t = 8.0 * d;                 // tune constant
        let a = 1.0 - exp(-sigma_t * ds);

        // Front-to-back alpha accumulation (no color yet)
        accum_alpha = accum_alpha + (1.0 - accum_alpha) * a;

        if (accum_alpha > 0.99) { break; }

        t = t + ds;
    }

    return vec4<f32>(accum_alpha, accum_alpha, accum_alpha, 1.0);
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

