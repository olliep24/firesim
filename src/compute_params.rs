use std::time::Duration;
use crate::config::GRID_DIMENSION_LENGTH;

/// Struct to contain read-only params for the compute pipeline.
/// Should be passed to the shader via a uniform buffer.
/// TODO: Rename.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeParams {
    // TODO: Make a grid struct to contain all the information about the grid.
    // It shouldn't be stored here. This is only the means of transportation.
    dt: f32,
    /// Number of voxels along the X axis of the simulation grid.
    width: u32,
    /// Number of voxels along the Y axis of the simulation grid.
    height: u32,
    /// Number of voxels along the Z axis of the simulation grid.
    depth: u32,
    /// Minimum point in world space for the simulation grid.
    /// xyz + padding.
    box_min: [f32; 4],
    /// Maximum point in world space for the simulation grid.
    /// // xyz + padding.
    box_max: [f32; 4],
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
