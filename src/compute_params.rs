/// Struct to contain read-only params for the compute pipeline.
/// Should be passed to the shader via a uniform buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeParams {
    pub dt: f32,
}

impl ComputeParams {
    pub fn new(dt: instant::Duration) -> ComputeParams {
        Self {
            dt: dt.as_secs_f32(),
        }
    }
}
