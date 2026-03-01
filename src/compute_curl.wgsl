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
var curl: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2)
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

    // skip outer boundary cells
    // TODO: Revisit boundary conditions
    if (
        gid.x == 0u || gid.x >= params.width  - 1u ||
        gid.y == 0u || gid.y >= params.height - 1u ||
        gid.z == 0u || gid.z >= params.depth  - 1u
    ) {
        return;
    }

    let curl_value = get_curl(gid);

    textureStore(
        curl,
        vec3<i32>(gid),
        vec4<f32>(curl_value, 0.0)
    );
}

fn get_curl(gid: vec3<u32>) -> vec3<f32> {
    let right_velocity = get_velocity(vec3<u32>(gid.x + 1, gid.y, gid.z));
    let left_velocity = get_velocity(vec3<u32>(gid.x - 1, gid.y, gid.z));

    let up_velocity = get_velocity(vec3<u32>(gid.x, gid.y + 1, gid.z));
    let down_velocity = get_velocity(vec3<u32>(gid.x, gid.y - 1, gid.z));

    let front_velocity = get_velocity(vec3<u32>(gid.x, gid.y, gid.z + 1));
    let back_velocity = get_velocity(vec3<u32>(gid.x, gid.y, gid.z - 1));

    let curl_x = (up_velocity.z - down_velocity.z - front_velocity.y + back_velocity.y) * 0.5;
    let curl_y = (front_velocity.x - back_velocity.x - right_velocity.z + left_velocity.z) * 0.5;
    let curl_z = (right_velocity.y - left_velocity.y - up_velocity.x + down_velocity.x) * 0.5;

    return vec3<f32>(curl_x, curl_y, curl_z);
}

fn get_velocity(index: vec3<u32>) -> vec3<f32> {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(velocity_vector_field_read, field_sampler, uvw, 0.0).xyz;
}
