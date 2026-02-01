@group(0) @binding(0)
var force_sources: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1)
var density_sources: texture_storage_3d<rgba16float, write>;

const zero = vec4<f32>(0.0);

@compute
@workgroup_size(4, 4, 4)
fn main (
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Zero out sources after we added them to the simulation.
    textureStore(
        force_sources,
        vec3<i32>(gid),
        zero
    );

    textureStore(
        density_sources,
        vec3<i32>(gid),
        zero
    );
}