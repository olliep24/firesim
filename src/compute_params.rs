use std::time::Duration;
use crate::config::GRID_DIMENSION_LENGTH;

/// Struct to contain read-only params for the compute pipeline.
/// Should be passed to the shader via a uniform buffer.
/// TODO: Rename.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeParams {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
    box_min: [f32; 4], // xyz + padding
    box_max: [f32; 4], // xyz + padding
}

impl ComputeParams {
    pub fn new(box_min: [f32; 4], box_max: [f32; 4]) -> Self {
        Self {
            dt: Duration::new(0, 0).as_secs_f32(),
            width: GRID_DIMENSION_LENGTH,
            height: GRID_DIMENSION_LENGTH,
            depth: GRID_DIMENSION_LENGTH,
            box_min,
            box_max,
        }
    }

    pub fn update_dt(&mut self, dt: Duration) {
        self.dt = dt.as_secs_f32();
    }
}
