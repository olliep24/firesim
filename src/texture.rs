/**
Each channel (RBGA) in the texture will be a 16-bit float.
TODO: My current machine allows this will the texture usages I need, but add check for this.
*/
const CHANNEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GRID_SIZE: u32 = 64;
/* Grid will be a cube and have GRID_SIZE x GRID_SIZE x GRID_SIZE voxels. */
const GRID_DIMENSION: wgpu::Extent3d = wgpu::Extent3d {
    width: GRID_SIZE,
    height: GRID_SIZE,
    depth_or_array_layers: GRID_SIZE,
};

pub struct Texture {
    #[allow(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Creates a texture, texture view, and sample for the compute pipeline.
    /// The returned texture represents 3D grid for the simulation, indexed by u, v, and w.
    /// Depending on the inputted format, the color channels and their precision can be used for
    /// compute pipelines.
    ///
    /// For example a format of Rgba16Float will have 4 channels at every combination of
    /// u, v, and w. Each channel is represented by a 16-bit float. This would be useful for a 3D
    /// vector field.
    ///
    /// Depending on the number of channels need and their precision, use the appropriate format for
    /// memory efficiency.
    pub fn create_compute_texture(device: &wgpu::Device, format: wgpu::TextureFormat, label: &str) -> Self {
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size: GRID_DIMENSION,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("density_linear_clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            // Setting these to linear will do trilinear interpolation semi-Lagrangian advection.
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self { texture, view, sampler }
    }

    pub fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }
        );

        Self { texture, view, sampler }
    }
}
