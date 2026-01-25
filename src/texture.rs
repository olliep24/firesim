use half::f16;
use crate::config::{GRID_DIMENSIONS, GRID_DIMENSION_LENGTH};

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
    /// memory efficiency. Although, the format may not be available on your machine for the texture
    /// usages.
    pub fn create_compute_texture(device: &wgpu::Device, format: wgpu::TextureFormat, label: Option<&str>) -> Self {
        let desc = wgpu::TextureDescriptor {
            label,
            size: GRID_DIMENSIONS,
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

    /// Writes a tornado velocity vector field to the given texture with a rgba16f format.
    /// The velocity's x, y, and z components will be written to the texture's r, g, and b channels
    /// respectively.
    pub fn write_velocity_3d_rgba16f_tornado(
        &self,
        queue: &wgpu::Queue
    ) {
        // TODO: Assert that the format is rgba16f.

        let width = GRID_DIMENSION_LENGTH;
        let height = GRID_DIMENSION_LENGTH;
        let depth = GRID_DIMENSION_LENGTH;

        // RGBA16F = 4 channels * 2 bytes = 8 bytes per voxel
        let bytes_per_voxel: usize = 8;
        let voxel_count = (width as usize) * (height as usize) * (depth as usize);
        let mut data = vec![0u8; voxel_count * bytes_per_voxel];

        // Map integer index to [-1, 1] at voxel center.
        // e.g. for i in (0, 1,..., 63) maps to voxel center scaled within (-1, 1).
        let to_unit = |i: u32, n: u32| -> f32 {
            let fi = i as f32 + 0.5;
            let fn_ = n as f32;
            (fi / fn_) * 2.0 - 1.0
        };

        let eps: f32 = 1e-6;

        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let px = to_unit(x, width);
                    let pz = to_unit(z, depth);

                    // Tangent around Y axis: (pz, 0, -px) normalized
                    let r2 = px * px + pz * pz;

                    let (vx, vy, vz) = if r2 < eps {
                        // On the axis: direction undefined; set to zero (or choose a fixed direction).
                        (0.0, 0.0, 0.0)
                    } else {
                        let inv_r = 1.0 / r2.sqrt();
                        (pz * inv_r, 0.0, -px * inv_r)
                    };

                    let r16 = f16::from_f32(vx).to_bits();
                    let g16 = f16::from_f32(vy).to_bits();
                    let b16 = f16::from_f32(vz).to_bits();
                    let a16 = f16::from_f32(0.0).to_bits();

                    let i = (x as usize)
                        + (width as usize) * ((y as usize) + (height as usize) * (z as usize));
                    let base = i * bytes_per_voxel;

                    data[base + 0..base + 2].copy_from_slice(&r16.to_le_bytes());
                    data[base + 2..base + 4].copy_from_slice(&g16.to_le_bytes());
                    data[base + 4..base + 6].copy_from_slice(&b16.to_le_bytes());
                    data[base + 6..base + 8].copy_from_slice(&a16.to_le_bytes());
                }
            }
        }

        // 8 bytes per texel
        let bytes_per_row = width * 8;
        let rows_per_image = height;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
        );
    }
}
