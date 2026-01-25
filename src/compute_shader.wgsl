@group(0) @binding(1)
var velocity_field_texture: texture_3d<f32>;
@group(0) @binding(2)
var particle_center_scalar_field_texture_read: texture_3d<f32>;
@group(0) @binding(3)
var particle_center_scalar_field_texture_write: texture_storage_3d<r16float, write>;
@group(0) @binding(4)
var field_sampler: sampler;

@compute
@workgroup_size(64)
fn main (
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {

}