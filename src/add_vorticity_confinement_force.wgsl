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
var velocity_vector_field_read: texture_3d<f32>;
@group(1) @binding(1)
var velocity_vector_field_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2)
var curl: texture_3d<f32>;
@group(1) @binding(3)
var field_sampler: sampler;

const confinement_constant: f32 = 0.100;

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

    // skip outer boundary cells
    // TODO: Revisit boundary conditions
    if (
        gid.x == 0u || gid.x >= params.width  - 1u ||
        gid.y == 0u || gid.y >= params.height - 1u ||
        gid.z == 0u || gid.z >= params.depth  - 1u
    ) {
        return;
    }

    let uvw = voxel_center_uvw(gid);
    let velocity = textureSampleLevel(velocity_vector_field_read, field_sampler, uvw, 0.0).xyz;

    let curl_magnitude_gradient = get_curl_magnitude_gradient(gid);

    let curl_magnitude_gradient_magnitude = length(curl_magnitude_gradient);
    if curl_magnitude_gradient_magnitude < 0.0001 {
        textureStore(velocity_vector_field_write, vec3<i32>(gid), vec4<f32>(velocity, 0.0));
        return;
    }

    // Normalized location vector that points to the highest nearby vorticity concentration
    let N = normalize(curl_magnitude_gradient);

    // Compute confined vorticity vector
    let curl_value = get_curl(gid);
    let f = confinement_constant * cross(N, curl_value);

    textureStore(
        velocity_vector_field_write,
        vec3<i32>(gid),
        vec4<f32>(velocity + f, 0.0)
    );
}

fn get_curl_magnitude_gradient(gid: vec3<u32>) -> vec3<f32> {
    let right_curl = get_curl_magnitude(vec3<u32>(gid.x + 1, gid.y, gid.z));
    let left_curl = get_curl_magnitude(vec3<u32>(gid.x - 1, gid.y, gid.z));
    let x_finite_partial = (right_curl - left_curl) * 0.5;

    let up_curl = get_curl_magnitude(vec3<u32>(gid.x, gid.y + 1, gid.z));
    let down_curl = get_curl_magnitude(vec3<u32>(gid.x, gid.y - 1, gid.z));
    let y_finite_partial = (up_curl - down_curl) * 0.5;

    let front_curl = get_curl_magnitude(vec3<u32>(gid.x, gid.y, gid.z + 1));
    let back_curl = get_curl_magnitude(vec3<u32>(gid.x, gid.y, gid.z - 1));
    let z_finite_partial = (front_curl - back_curl) * 0.5;

    return vec3<f32>(x_finite_partial, y_finite_partial, z_finite_partial);
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

fn get_curl(index: vec3<u32>) -> vec3<f32> {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(curl, field_sampler, uvw, 0.0).xyz;
}

fn get_curl_magnitude(index: vec3<u32>) -> f32 {
    return length(get_curl(index));
}
