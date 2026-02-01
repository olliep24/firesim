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
    /// Value to indicate how strong to inject sources (should be a value between 0 and 1.0).
    /// Could be represented as a boolean, but minimizing branching on the GPU is ideal.
    inject_sources_strength: f32,
    /// Minimum point in world space for the simulation grid.
    /// xyz + padding.
    box_min: [f32; 4],
    /// Maximum point in world space for the simulation grid.
    /// xyz + padding.
    box_max: [f32; 4],
    /// Number of pixels [width, height]
    viewport: [f32; 2],
    _pad0: [f32; 2],
}

impl ComputeParams {
    pub fn new(box_min: [f32; 4], box_max: [f32; 4], config: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            dt: Duration::new(0, 0).as_secs_f32(),
            width: GRID_DIMENSION_LENGTH,
            height: GRID_DIMENSION_LENGTH,
            depth: GRID_DIMENSION_LENGTH,
            inject_sources_strength: 0.0,
            box_min,
            box_max,
            viewport: [config.width as f32, config.height as f32],
            _pad0: [0.0; 2],
        }
    }

    pub fn update_dt(&mut self, dt: Duration, ) {
        self.dt = dt.as_secs_f32();
    }

    pub fn update_viewport(&mut self, config: &wgpu::SurfaceConfiguration) {
        self.viewport = [config.width as f32, config.height as f32];
    }

    pub fn set_inject_sources_strength(&mut self, strength: f32) {
        self.inject_sources_strength = strength;
    }

    pub fn inject_sources_strength(&self) -> f32 {
        self.inject_sources_strength
    }
}
