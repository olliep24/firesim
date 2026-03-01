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
var velocity_vector_field_read: texture_3d<f32>;
@group(1) @binding(1)
var velocity_vector_field_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2)
var scalar_field_read: texture_3d<f32>;

const up = vec3<f32>(0.0, 1.0, 0.0);
const beta = 0.01;

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.depth) {
        return;
    }
    let coord = vec3<i32>(gid);

    let velocity = textureLoad(velocity_vector_field_read, coord, 0).xyz;
    let temperature = textureLoad(scalar_field_read, coord, 0).y;

    let buoyancy_force = beta * temperature * params.dt * up;

    textureStore(
        velocity_vector_field_write,
        vec3<i32>(gid),
        vec4<f32>(velocity + buoyancy_force, 0.0)
    );
}
