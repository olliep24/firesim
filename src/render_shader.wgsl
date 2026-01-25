// Uniform buffers
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Vertex shader
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    // Billboard the instance - always facing the camera.
    let instance_center_world = vec3<f32>(model_matrix[3][0], model_matrix[3][1], model_matrix[3][2]);

    let camera_right_world = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let camera_up_world = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    let vertex_position_world =
        instance_center_world
        + camera_right_world * model.position.x
        + camera_up_world * model.position.y;

    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = camera.proj * camera.view * vec4<f32>(vertex_position_world, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
