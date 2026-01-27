// Uniform buffers
struct Params {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
}
@group(0) @binding(0)
var<uniform> params: Params;

// Texture bindings
@group(1) @binding(0)
var velocity_field_texture: texture_3d<f32>;
@group(1) @binding(1)
var density_scalar_field_texture_read: texture_3d<f32>;
@group(1) @binding(2)
var density_scalar_field_texture_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(3)
var field_sampler: sampler;

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

    let w = f32(params.width);
    let h = f32(params.height);
    let d = f32(params.depth);

    // Current voxel center in normalized texture coordinates [0,1]
    let uvw = vec3<f32>(
        // Use +0.5 so we treat each voxel as centered sampling.
        (f32(gid.x) + 0.5) / w,
        (f32(gid.y) + 0.5) / h,
        (f32(gid.z) + 0.5) / d
    );

    // Trilinear interpolation using texture sampler.
    let velocity = textureSampleLevel(velocity_field_texture, field_sampler, uvw, 0.0).xyz;

    // Backtrace in normalized coordinates.
    // TODO: Come back and scale velocity to world coordinates. Currently in cells / sec.
    let velocity_uvw = vec3<f32>(
        (velocity.x) / w,
        (velocity.y) / h,
        (velocity.z) / d
    );

    let uvw_backtrace = uvw - params.dt * velocity_uvw;

    // Semi-Lagrangian: sample previous scalar field at the backtraced position.
    // Trilinear interpolation using texture sampler.
    // Sampler is clamped to edge, so no out of bounds issues.
    // TODO: Investigate bounding conditions.
    let backtraced_center = textureSampleLevel(
        density_scalar_field_texture_read,
        field_sampler,
        uvw_backtrace,
        0.0
    ).x;

    textureStore(
        density_scalar_field_texture_write,
        vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)),
        vec4<f32>(backtraced_center, 0.0, 0.0, 0.0)
    );
}
