@group(0) @binding(0)
var force_sources: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1)
var density_sources: texture_storage_3d<rgba16float, write>;

// Location to add dye density and velocity.
// TODO: Maybe add configuration params as input. Maybe through command line?
const center = vec3<f32>(32.0, 16.0, 32.0);
const radius: f32 = 4.0;
const radius2: f32 = radius * radius;
const peak: f32 = 1.0;
const strength: f32 = 1.0;
const eps: f32 = 1e-6;

/* Adds sources to the density texture and the force texture. Will overwrite entire texture */
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
    let density_to_add = peak * exp(-distance_from_center2 / (2.0 * sigma2));

    // Add the dye
    textureStore(
        density_sources,
        coord,
        vec4<f32>(density_to_add, 0.0, 0.0, 0.0)
    );

    // Add an outward puff of force. Has an x, y, and z component.
    let dir = distance_from_center / max(sqrt(distance_from_center2), eps); // normalized
    let force = dir * (strength * add);
    textureStore(
        force_sources,
        coord,
        vec4<f32>(force, 0.0)
    );
}