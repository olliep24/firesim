pub const GRID_DIMENSION_LENGTH: u32 = 64;
/* Grid will be a cube and have GRID_SIZE x GRID_SIZE x GRID_SIZE voxels. */
pub const GRID_DIMENSIONS: wgpu::Extent3d = wgpu::Extent3d {
    width: GRID_DIMENSION_LENGTH,
    height: GRID_DIMENSION_LENGTH,
    depth_or_array_layers: GRID_DIMENSION_LENGTH,
};
pub const GRID_VOXEL_SIDE_LENGTH: f32 = 0.025;
pub const VELOCITY_SCALE: f32 = 15.0;
