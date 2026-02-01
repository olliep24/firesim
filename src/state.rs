use std::sync::Arc;
use wgpu::{Device, Queue, Surface, SurfaceConfiguration};
use wgpu::util::DeviceExt;
use winit::event::ElementState;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::window::Window;

use crate::camera::{Camera, CameraController, CameraUniform, Projection};
use crate::texture::Texture;
use crate::compute_params::ComputeParams;
use crate::config::{GRID_DIMENSION_LENGTH, GRID_VOXEL_SIDE_LENGTH};

/**
Each channel (RBGA) in the texture will be a 16-bit float.
TODO: My current machine allows this will the texture usages I need, but add check for this.
*/
const VECTOR_FIELD_CHANNEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SCALAR_FIELD_CHANNEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct State {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    is_surface_configured: bool,
    depth_texture: Texture,
    camera: Camera,
    pub camera_controller: CameraController,
    projection: Projection,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    density_texture_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    compute_params: ComputeParams,
    compute_params_bind_group: wgpu::BindGroup,
    compute_params_buffer: wgpu::Buffer,
    use_a_to_b: bool,
    compute_bind_group_a_to_b: wgpu::BindGroup,
    compute_bind_group_b_to_a: wgpu::BindGroup,
    add_input_pipeline: wgpu::ComputePipeline,
    add_input_bind_group_a: wgpu::BindGroup,
    add_input_bind_group_b: wgpu::BindGroup,
    pending_input: bool,
    pub mouse_pressed: bool,
    pub window: Arc<Window>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        // TODO: Look into this comment to see if want anything other than sRGB
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render_shader.wgsl").into()),
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        // TODO: Move these to constants
        let camera = Camera::new((0.6125, 1.25, 2.5), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let camera_controller = CameraController::new(1.0, 0.2);
        let projection = Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);

        let mut camera_uniform = CameraUniform::default();
        camera_uniform.update(&camera, &projection);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("Camera Bind Group Layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("Camera Bind Group"),
        });

        let box_min = [0.0, 0.0, 0.0, 0.0];

        let extent = GRID_DIMENSION_LENGTH as f32 * GRID_VOXEL_SIDE_LENGTH;
        let box_max = [extent, extent, extent, 0.0];

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into()),
        });

        let compute_params = ComputeParams::new(box_min, box_max, &config);

        let compute_params_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Compute Parameters Buffer"),
                contents: bytemuck::cast_slice(&[compute_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let compute_params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Pipeline Bind Group Layout"),
            entries: &[
                // 0. Uniform buffer for compute params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        });

        let compute_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Params Bind Group"),
            layout: &compute_params_bind_group_layout,
            entries: &[
                // binding 0: Compute params
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_params_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Pipeline Bind Group Layout"),
            entries: &[
                // 0. Velocity vector field texture read.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 1. Velocity vector field texture write.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SCALAR_FIELD_CHANNEL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // 2. Density scalar field texture input
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 3. Density scalar field texture output
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SCALAR_FIELD_CHANNEL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // 4. Sampler for density texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }
            ]
        });

        // TODO: Add note on why we're using a texture here instead of a buffer.
        let velocity_vector_field_texture_a = Texture::create_compute_texture(
            &device,
            VECTOR_FIELD_CHANNEL_FORMAT,
            Some("Velocity Field Texture")
        );

        let velocity_vector_field_texture_b = Texture::create_compute_texture(
            &device,
            VECTOR_FIELD_CHANNEL_FORMAT,
            Some("Velocity Field Texture")
        );

        // Set static velocity field.
        velocity_vector_field_texture_a.write_velocity_3d_rgba16f_tornado(&queue);

        let density_scalar_field_texture_a = Texture::create_compute_texture(
            &device,
            SCALAR_FIELD_CHANNEL_FORMAT,
            Some("Density Scalar Field Texture A")
        );

        let density_scalar_field_texture_b = Texture::create_compute_texture(
            &device,
            SCALAR_FIELD_CHANNEL_FORMAT,
            Some("Density Scalar Field Texture B")
        );

        density_scalar_field_texture_a.write_density_blob_rgba16f(
            &queue,
            [16.0, 16.0, 16.0],
            4.0,
            1.0
        );

        // Create two bind groups to ping pong between, controlled by use_a_to_b flag.
        let compute_bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A to B"),
            layout: &compute_bind_group_layout,
            entries: &[
                // binding 0: Velocity vector field read
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&velocity_vector_field_texture_a.view)
                },
                // binding 1: Velocity vector field write
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&velocity_vector_field_texture_b.view)
                },
                // binding 2: Density scalar field read
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_a.view)
                },
                // binding 3: Density scalar field write
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_b.view)
                },
                // binding 4: Sampler. Will work for scalar or velocity field.
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&density_scalar_field_texture_a.sampler)
                },
            ],
        });

        let compute_bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A to B"),
            layout: &compute_bind_group_layout,
            entries: &[
                // binding 0: Velocity vector field read
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&velocity_vector_field_texture_b.view)
                },
                // binding 1: Velocity vector field write
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&velocity_vector_field_texture_a.view)
                },
                // binding 2: Density scalar field read
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_b.view)
                },
                // binding 3: Density scalar field write
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_a.view)
                },
                // binding 4: Sampler. Will work for scalar or velocity field.
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&density_scalar_field_texture_a.sampler)
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &compute_params_bind_group_layout,
                    &compute_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            // Will default to @compute
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let density_texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Texture Bind Group Layout"),
            entries: &[
                // 0. Density scalar field texture input
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // 1. Sampler for density texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }
            ]
        });

        let density_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &density_texture_bind_group_layout,
            entries: &[
                // binding 0: Density scalar field read
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_a.view)
                },
                // binding 1: Sampler for density scalar field (either a or b work)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&density_scalar_field_texture_a.sampler)
                },
            ],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &compute_params_bind_group_layout,
                    &density_texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let add_input_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Add Input Bind Group Layout"),
            entries: &[
                // velocity storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // density storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ]
        });

        let add_input_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add Input Bind Group"),
            layout: &add_input_bind_group_layout,
            entries: &[
                // binding 0: Velocity vector field a
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&velocity_vector_field_texture_a.view)
                },
                // binding 1: Density scalar field a
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_a.view)
                },
            ],
        });

        let add_input_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add Input Bind Group"),
            layout: &add_input_bind_group_layout,
            entries: &[
                // binding 0: Velocity vector field b
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&velocity_vector_field_texture_b.view)
                },
                // binding 1: Density scalar field b
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&density_scalar_field_texture_b.view)
                },
            ],
        });

        let add_input_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Add Input Pipeline Layout"),
                bind_group_layouts: &[
                    &add_input_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let add_input_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Input Pipeline"),
            layout: Some(&add_input_pipeline_layout),
            module: &compute_shader,
            // Will default to @compute
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            depth_texture,
            camera,
            camera_controller,
            projection,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            render_pipeline,
            density_texture_bind_group,
            compute_pipeline,
            compute_params,
            compute_params_bind_group,
            compute_params_buffer,
            use_a_to_b: true,
            compute_bind_group_a_to_b,
            compute_bind_group_b_to_a,
            add_input_pipeline,
            add_input_bind_group_a,
            add_input_bind_group_b,
            pending_input: false,
            mouse_pressed: false,
            window,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.projection.resize(width, height);
            self.depth_texture = Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.compute_params.update_viewport(&self.config);
            self.is_surface_configured = true;
        }
    }

    pub fn update(&mut self, dt: instant::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update(&self.camera, &self.projection);
        /*
        Potential to optimize:
        We can create a separate buffer and copy its contents to our camera_buffer. The new buffer
        is known as a staging buffer. This method is usually how it's done as it allows the contents
        of the main buffer (in this case, camera_buffer) to be accessible only by the GPU. The GPU
        can do some speed optimizations, which it couldn't if we could access the buffer via
        the CPU.
         */
        // TODO: Make this a fixed timestep.
        self.compute_params.update_dt(dt);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
        self.queue.write_buffer(&self.compute_params_buffer, 0, bytemuck::cast_slice(&[self.compute_params]));
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, key_state: ElementState) {
        if code == KeyCode::Escape && key_state.is_pressed() {
            event_loop.exit();
        } else if code == KeyCode::Space && key_state.is_pressed() {
            self.compute_params.set_inject_sources_strength(1.0);
            self.queue.write_buffer(&self.compute_params_buffer, 0, bytemuck::cast_slice(&[self.compute_params]));
        } else {
            self.camera_controller.process_keyboard(code, key_state);
        }
    }

    pub fn handle_mouse_click(&mut self, mouse_state: ElementState) {
        self.mouse_pressed = mouse_state.is_pressed();
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // We can't render unless the surface is configured
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Add sources to density and forces if present.
        if self.pending_input {
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                compute_pass.set_pipeline(&self.add_input_pipeline);

                let bind_group = if self.use_a_to_b {
                    &self.add_input_bind_group_a
                } else {
                    &self.add_input_bind_group_b
                };

                compute_pass.set_bind_group(0, bind_group, &[]);

                // We specified 4 threads per dimension in the compute shader.
                let num_dispatches_per_dimension = GRID_DIMENSION_LENGTH / 4;
                compute_pass.dispatch_workgroups(
                    num_dispatches_per_dimension,
                    num_dispatches_per_dimension,
                    num_dispatches_per_dimension
                );
            }

            self.pending_input = false;
        }

        // Simulate
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.compute_pipeline);

            compute_pass.set_bind_group(0, &self.compute_params_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                if self.use_a_to_b { &self.compute_bind_group_a_to_b } else { &self.compute_bind_group_b_to_a },
                &[]
            );

            // We specified 4 threads per dimension in the compute shader.
            let num_dispatches_per_dimension = GRID_DIMENSION_LENGTH / 4;
            compute_pass.dispatch_workgroups(
                num_dispatches_per_dimension,
                num_dispatches_per_dimension,
                num_dispatches_per_dimension
            );

            // Ping pong between textures.
            self.use_a_to_b = !self.use_a_to_b;
        }

        if self.compute_params.inject_sources_strength() > 0.0 {
            // If we added input, then
            self.compute_params.set_inject_sources_strength(0.0);
            self.queue.write_buffer(&self.compute_params_buffer, 0, bytemuck::cast_slice(&[self.compute_params]));
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.compute_params_bind_group, &[]);
            render_pass.set_bind_group(2, &self.density_texture_bind_group, &[]);

            // Full screen triangle, no vertex/index buffer.
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}