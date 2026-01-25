use crate::config::GRID_DIMENSION_LENGTH;

/// Struct to contain read-only params for the compute pipeline.
/// Should be passed to the shader via a uniform buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeParams {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
}

impl ComputeParams {
    pub fn new(dt: instant::Duration) -> ComputeParams {
        Self {
            dt: dt.as_secs_f32(),
            width: GRID_DIMENSION_LENGTH,
            height: GRID_DIMENSION_LENGTH,
            depth: GRID_DIMENSION_LENGTH,
        }
    }
}
