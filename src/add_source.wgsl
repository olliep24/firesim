@group(0) @binding(0)
var scalar_source: texture_storage_3d<rgba16float, write>;

// Location to add scalar values
// TODO: Maybe add configuration params as input. Maybe through command line?
const center = vec3<f32>(64.0, 32.0, 64.0);
const radius: f32 = 24.0;
const radius2: f32 = radius * radius;
const peak: f32 = 1.0;
const strength: f32 = 10.0;
const eps: f32 = 1e-6;
const up = vec4<f32>(0.0, 1.0, 0.0, 0.0);

/* Adds sources to the scalar texture. Will overwrite entire texture */
@compute
@workgroup_size(4, 4, 4)
fn main (
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let coord = vec3<i32>(gid);

    let position = vec3<f32>(gid) + vec3<f32>(0.5);
    let distance_from_center = position - center;
    let distance_from_center2 = dot(distance_from_center, distance_from_center);

    // Position is too far from center, pass.
    if (distance_from_center2 > radius2) { return; }

    let sigma = max(radius * 0.35, 1e-6);
    let sigma2 = sigma * sigma;
    let scalar_to_add = peak * exp(-distance_from_center2 / (2.0 * sigma2));

    // X = smoke density, Y = temperature (source is the hot combustion zone).
    // Both are injected at equal intensity so the fire base is immediately at max temperature.
    // TODO: consider changing to rate per second, then multiply by dt.
    textureStore(
        scalar_source,
        coord,
        vec4<f32>(scalar_to_add, scalar_to_add, 0.0, 0.0)
    );
}
