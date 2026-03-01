@group(0) @binding(0)
var scalar_source: texture_storage_3d<rgba16float, write>;

const zero = vec4<f32>(0.0);

@compute
@workgroup_size(4, 4, 4)
fn main (
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    textureStore(
        scalar_source,
        vec3<i32>(gid),
        zero
    );
}