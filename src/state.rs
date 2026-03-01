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
use crate::compute_step::ComputeStep;
use crate::config::{GRID_DIMENSION_LENGTH, GRID_VOXEL_SIDE_LENGTH};
use crate::ping_pong::PingPong;

/**
Each channel (RBGA) in the texture will be a 16-bit float.
The 16-bit float channel is filterable (needed for interpolation) but the 32-bit float channel
is not.
TODO: My current machine allows this will the texture usages I need, but add check for this.
TODO: Make just one format.
*/
const CHANNEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NUMBER_DISPATCHES_PER_DIMENSION: u32 = GRID_DIMENSION_LENGTH / 4;
const WORKGROUPS: (u32, u32, u32) = (
    NUMBER_DISPATCHES_PER_DIMENSION,
    NUMBER_DISPATCHES_PER_DIMENSION,
    NUMBER_DISPATCHES_PER_DIMENSION
);
const JACOBI_ITERATIONS: u32 = 20;

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
    density_texture_bind_group_layout: wgpu::BindGroupLayout,
    compute_params: ComputeParams,
    compute_params_bind_group: wgpu::BindGroup,
    compute_params_buffer: wgpu::Buffer,
    add_source_pipeline: wgpu::ComputePipeline,
    remove_source_pipeline: wgpu::ComputePipeline,
    add_source_bind_group: wgpu::BindGroup,
    scalar_field_ping_pong: PingPong,
    velocity_vector_field_ping_pong: PingPong,
    scalar_source_texture: Texture,
    advect_scalars_compute_step: ComputeStep,
    advect_velocity_compute_step: ComputeStep,
    add_forces_to_velocity_compute_step: ComputeStep,
    compute_divergence_bind_group_layout: wgpu::BindGroupLayout,
    compute_divergence_pipeline: wgpu::ComputePipeline,
    divergence_texture: Texture,
    pressure_ping_pong: PingPong,
    compute_pressure_compute_step: ComputeStep,
    subtract_pressure_gradient_compute_step: ComputeStep,
    compute_curl_bind_group_layout: wgpu::BindGroupLayout,
    compute_curl_pipeline: wgpu::ComputePipeline,
    curl_texture: Texture,
    add_vorticity_confinement_force_compute_step: ComputeStep,
    compute_temperature_compute_step: ComputeStep,
    compute_fuel_compute_step: ComputeStep,
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

        // TODO: Add note on why we're using a texture here instead of a buffer.
        let scalar_field_texture_a = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Scalar Field Texture A")
        );

        let scalar_field_texture_b = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Scalar Field Texture B")
        );

        let scalar_field_ping_pong = PingPong::new(
            scalar_field_texture_a,
            scalar_field_texture_b,
        );

        let scalar_source_texture = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Scalar Source Texture")
        );

        let velocity_vector_field_texture_a = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Velocity Field Texture A")
        );

        let velocity_vector_field_texture_b = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Velocity Field Texture B")
        );

        let velocity_vector_field_ping_pong = PingPong::new(
            velocity_vector_field_texture_a,
            velocity_vector_field_texture_b,
        );

        // TODO: Rename
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

        let add_source_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Add Source Bind Group Layout"),
            entries: &[
                //  Source texture write.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: CHANNEL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ]
        });

        let add_source_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add Source Bind Group"),
            layout: &add_source_bind_group_layout,
            entries: &[
                // binding 0: Scalar field
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scalar_source_texture.view)
                },
            ],
        });

        let add_source_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Add Source Pipeline Layout"),
                bind_group_layouts: &[
                    &add_source_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let add_source_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Source Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("add_source.wgsl").into()),
        });

        let add_source_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Source Pipeline"),
            layout: Some(&add_source_pipeline_layout),
            module: &add_source_shader,
            // Will default to @compute
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let remove_source_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Remove Source Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("remove_source.wgsl").into()),
        });

        let remove_source_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Source Pipeline"),
            layout: Some(&add_source_pipeline_layout),
            module: &remove_source_shader,
            // Will default to @compute
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create advect scalars compute step
        let advect_scalars_compute_step = create_advect_scalars_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let advect_velocity_compute_step = create_advect_velocity_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let add_forces_to_velocity_compute_step = create_add_forces_to_velocity_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let compute_divergence_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Divergence Bind Group Layout"),
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
                // 1. Divergence texture write.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: CHANNEL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // 2. Sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }
            ]
        });

        let compute_divergence_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Divergence Pipeline Layout"),
                bind_group_layouts: &[
                    &compute_params_bind_group_layout,
                    &compute_divergence_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_divergence_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Divergence Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute_divergence.wgsl").into()),
        });

        let compute_divergence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Divergence Pipeline"),
            layout: Some(&compute_divergence_pipeline_layout),
            module: &compute_divergence_shader,
            // Will default to @compute
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let divergence_texture = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Divergence Texture")
        );

        let pressure_texture_a = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Pressure Texture A")
        );

        let pressure_texture_b = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Pressure Texture B")
        );

        let pressure_ping_pong = PingPong::new(
            pressure_texture_a,
            pressure_texture_b,
        );

        let compute_pressure_compute_step = create_compute_pressure_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let subtract_pressure_gradient_compute_step = create_subtract_pressure_gradient_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let compute_curl_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Curl Bind Group Layout"),
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
                // 1. Curl texture write.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: CHANNEL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // 2. Sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }
            ]
        });

        let compute_curl_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Curl Pipeline Layout"),
                bind_group_layouts: &[
                    &compute_params_bind_group_layout,
                    &compute_curl_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_curl_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Curl Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute_curl.wgsl").into()),
        });

        let compute_curl_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Curl Pipeline"),
            layout: Some(&compute_curl_pipeline_layout),
            module: &compute_curl_shader,
            // Will default to @compute
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let curl_texture = Texture::create_compute_texture(
            &device,
            CHANNEL_FORMAT,
            Some("Curl Texture")
        );

        let add_vorticity_confinement_force_compute_step = create_add_vorticity_confinement_force_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let compute_temperature_compute_step = create_compute_temperature_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

        let compute_fuel_compute_step = create_compute_fuel_compute_step(
            &device,
            &compute_params_bind_group_layout
        );

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
            density_texture_bind_group_layout,
            compute_params,
            compute_params_bind_group,
            compute_params_buffer,
            add_source_pipeline,
            remove_source_pipeline,
            add_source_bind_group,
            scalar_field_ping_pong,
            velocity_vector_field_ping_pong,
            scalar_source_texture,
            advect_scalars_compute_step,
            advect_velocity_compute_step,
            add_forces_to_velocity_compute_step,
            compute_divergence_bind_group_layout,
            compute_divergence_pipeline,
            divergence_texture,
            compute_pressure_compute_step,
            pressure_ping_pong,
            subtract_pressure_gradient_compute_step,
            compute_curl_bind_group_layout,
            compute_curl_pipeline,
            curl_texture,
            add_vorticity_confinement_force_compute_step,
            compute_temperature_compute_step,
            compute_fuel_compute_step,
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
        } else if code == KeyCode::KeyF && key_state.is_pressed() {
            self.pending_input = true;
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

        /* Add Sources if Present */
        if self.pending_input {
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                compute_pass.set_pipeline(&self.add_source_pipeline);

                compute_pass.set_bind_group(0, &self.add_source_bind_group, &[]);

                compute_pass.dispatch_workgroups(
                    NUMBER_DISPATCHES_PER_DIMENSION,
                    NUMBER_DISPATCHES_PER_DIMENSION,
                    NUMBER_DISPATCHES_PER_DIMENSION
                );
            }
        }

        /* Simulation Steps */

        // Advect scalars
        let (read_texture, write_texture) = self.scalar_field_ping_pong.get_read_and_write();
        let textures_read_only: [&wgpu::TextureView; 2] = [
            self.velocity_vector_field_ping_pong.get_read(),
            &self.scalar_source_texture.view
        ];

        self.advect_scalars_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &textures_read_only,
            Some(self.scalar_field_ping_pong.get_sampler()),
            WORKGROUPS
        );

        self.scalar_field_ping_pong.swap();

        // Compute temperature
        let (read_texture, write_texture) = self.scalar_field_ping_pong.get_read_and_write();

        self.compute_temperature_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &[],
            Some(self.scalar_field_ping_pong.get_sampler()),
            WORKGROUPS
        );

        self.scalar_field_ping_pong.swap();

        // Compute fuel usage
        let (read_texture, write_texture) = self.scalar_field_ping_pong.get_read_and_write();

        self.compute_fuel_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &[],
            Some(self.scalar_field_ping_pong.get_sampler()),
            WORKGROUPS
        );

        self.scalar_field_ping_pong.swap();

        // Advect velocity
        let (read_texture, write_texture) = self.velocity_vector_field_ping_pong.get_read_and_write();

        self.advect_velocity_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &[],
            Some(self.velocity_vector_field_ping_pong.get_sampler()),
            WORKGROUPS
        );

        self.velocity_vector_field_ping_pong.swap();

        // Add forces to velocity
        let (read_texture, write_texture) = self.velocity_vector_field_ping_pong.get_read_and_write();
        let textures_read_only: [&wgpu::TextureView; 1] = [self.scalar_field_ping_pong.get_read()];

        self.add_forces_to_velocity_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &textures_read_only,
            None,
            WORKGROUPS
        );

        self.velocity_vector_field_ping_pong.swap();

        // Vorticity Confinement
        // Compute curl
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.compute_curl_pipeline);

            let compute_curl_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Curl Group"),
                layout: &self.compute_curl_bind_group_layout,
                entries: &[
                    // binding 0: Velocity vector field read
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.velocity_vector_field_ping_pong.get_read())
                    },
                    // binding 1: Divergence scalar field write
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.curl_texture.view)
                    },
                    // binding 2: Sample
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.velocity_vector_field_ping_pong.get_sampler())
                    }
                ],
            });

            compute_pass.set_bind_group(0, &self.compute_params_bind_group, &[]);
            compute_pass.set_bind_group(1, &compute_curl_bind_group, &[]);

            compute_pass.dispatch_workgroups(
                NUMBER_DISPATCHES_PER_DIMENSION,
                NUMBER_DISPATCHES_PER_DIMENSION,
                NUMBER_DISPATCHES_PER_DIMENSION
            );
        }

        // Add vorticity confinement force
        let (read_texture, write_texture) = self.velocity_vector_field_ping_pong.get_read_and_write();
        let textures_read_only: [&wgpu::TextureView; 1] = [&self.curl_texture.view];

        self.add_vorticity_confinement_force_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &textures_read_only,
            Some(self.velocity_vector_field_ping_pong.get_sampler()),
            WORKGROUPS
        );

        self.velocity_vector_field_ping_pong.swap();

        // Projection
        // Compute divergence
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.compute_divergence_pipeline);

            let compute_divergence_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Divergence Group"),
                layout: &self.compute_divergence_bind_group_layout,
                entries: &[
                    // binding 0: Velocity vector field read
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.velocity_vector_field_ping_pong.get_read())
                    },
                    // binding 1: Divergence scalar field write
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.divergence_texture.view)
                    },
                    // binding 2: Sample
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.velocity_vector_field_ping_pong.get_sampler())
                    }
                ],
            });

            compute_pass.set_bind_group(0, &self.compute_params_bind_group, &[]);
            compute_pass.set_bind_group(1, &compute_divergence_bind_group, &[]);

            compute_pass.dispatch_workgroups(
                NUMBER_DISPATCHES_PER_DIMENSION,
                NUMBER_DISPATCHES_PER_DIMENSION,
                NUMBER_DISPATCHES_PER_DIMENSION
            );
        }

        // Compute pressure via Jacobi method
        for _ in 0..JACOBI_ITERATIONS {
            let (read_texture, write_texture) = self.pressure_ping_pong.get_read_and_write();
            let textures_read_only: [&wgpu::TextureView; 1] = [&self.divergence_texture.view];

            self.compute_pressure_compute_step.dispatch(
                &self.device,
                &mut encoder,
                &self.compute_params_bind_group,
                read_texture,
                write_texture,
                &textures_read_only,
                Some(self.pressure_ping_pong.get_sampler()),
                WORKGROUPS
            );

            self.pressure_ping_pong.swap();
        }

        // Subtract pressure gradient from the velocity field.
        let (read_texture, write_texture) = self.velocity_vector_field_ping_pong.get_read_and_write();
        let textures_read_only: [&wgpu::TextureView; 1] = [&self.pressure_ping_pong.get_read()];

        self.subtract_pressure_gradient_compute_step.dispatch(
            &self.device,
            &mut encoder,
            &self.compute_params_bind_group,
            read_texture,
            write_texture,
            &textures_read_only,
            Some(self.velocity_vector_field_ping_pong.get_sampler()),
            WORKGROUPS
        );

        self.velocity_vector_field_ping_pong.swap();

        /* Remove Sources if Present */

        if self.pending_input {
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                compute_pass.set_pipeline(&self.remove_source_pipeline);

                compute_pass.set_bind_group(0, &self.add_source_bind_group, &[]);

                compute_pass.dispatch_workgroups(
                    NUMBER_DISPATCHES_PER_DIMENSION,
                    NUMBER_DISPATCHES_PER_DIMENSION,
                    NUMBER_DISPATCHES_PER_DIMENSION
                );
            }

            // We are done adding the sources.
            self.pending_input = false;
        }

        /* Render simulation result */

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
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
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

            let density_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture Bind Group"),
                layout: &self.density_texture_bind_group_layout,
                entries: &[
                    // binding 0: Density scalar field read
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.scalar_field_ping_pong.get_read())
                    },
                    // binding 1: Sampler for density scalar field (either a or b work)
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.scalar_field_ping_pong.get_sampler())
                    },
                ],
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.compute_params_bind_group, &[]);
            render_pass.set_bind_group(2, &density_texture_bind_group, &[]);

            // Full screen triangle, no vertex/index buffer.
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// TODO: Find a better way to organize this code.
/* Helper functions to create each compute step */

fn create_advect_scalars_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let advect_scalars_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Advect Scalars Bind Group Layout"),
        entries: &[
            // 0. Scalar field texture read.
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
            // 1. Scalar field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Velocity vector field texture read.
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
            // 3. Density source texture input.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D3,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // 4. Sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let advect_scalars_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Advect Scalars Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &advect_scalars_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let advect_scalars_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Advect Scalars Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("advect_scalars.wgsl").into()),
    });

    let advect_scalars_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Advect Scalars Pipeline"),
        layout: Some(&advect_scalars_pipeline_layout),
        module: &advect_scalars_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Advect Scalars Compute Step",
        advect_scalars_pipeline,
        advect_scalars_bind_group_layout,
    )
}

fn create_advect_velocity_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let advect_velocity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Advect Velocity Bind Group Layout"),
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
            // 1. Vector velocity field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let advect_velocity_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Advect Velocity Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &advect_velocity_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let advect_velocity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Advect Velocity Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("advect_velocity.wgsl").into()),
    });

    let advect_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Advect Velocity Pipeline"),
        layout: Some(&advect_velocity_pipeline_layout),
        module: &advect_velocity_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Advect Velocity Compute Step",
        advect_velocity_pipeline,
        advect_velocity_bind_group_layout,
    )
}

fn create_add_forces_to_velocity_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let add_forces_to_velocity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Add Forces to Velocity Bind Group Layout"),
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
            // 1. Vector velocity field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Scalar texture read.
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
        ]
    });

    let add_forces_to_velocity_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Add Forces to Velocity Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &add_forces_to_velocity_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let add_forces_to_velocity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Add Forces to Velocity Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("add_forces_to_velocity.wgsl").into()),
    });

    let add_forces_to_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Add Forces to Velocity Pipeline"),
        layout: Some(&add_forces_to_velocity_pipeline_layout),
        module: &add_forces_to_velocity_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Add Forces to Velocity Compute Step",
        add_forces_to_velocity_pipeline,
        add_forces_to_velocity_bind_group_layout,
    )
}

fn create_compute_pressure_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let compute_pressure_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Pressure Bind Group Layout"),
        entries: &[
            // 0. Pressure texture read.
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
            // 1. Pressure texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Divergence texture read.
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
            // 3. Sampler,
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let compute_pressure_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pressure Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &compute_pressure_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let compute_pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Pressure Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("compute_pressure.wgsl").into()),
    });

    let compute_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Add Forces to Velocity Pipeline"),
        layout: Some(&compute_pressure_pipeline_layout),
        module: &compute_pressure_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Compute Pressure Compute Step",
        compute_pressure_pipeline,
        compute_pressure_bind_group_layout,
    )
}

fn create_subtract_pressure_gradient_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let subtract_pressure_gradient_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Subtract Pressure Gradient Bind Group Layout"),
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
            // 1. Vector velocity field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Pressure texture,
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
            // 3. Sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let subtract_pressure_gradient_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Subtract Gradient Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &subtract_pressure_gradient_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let subtract_pressure_gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Subtract Pressure Gradient Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("subtract_pressure_gradient.wgsl").into()),
    });

    let subtract_pressure_gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Subtract Pressure Gradient Pipeline"),
        layout: Some(&subtract_pressure_gradient_pipeline_layout),
        module: &subtract_pressure_gradient_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Subtract Pressure Gradient Compute Step",
        subtract_pressure_gradient_pipeline,
        subtract_pressure_gradient_bind_group_layout,
    )
}

fn create_add_vorticity_confinement_force_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let add_vorticity_confinement_force_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Add Vorticity Confinement Force Bind Group Layout"),
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
            // 1. Vector velocity field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Curl texture,
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
            // 3. Sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let add_vorticity_confinement_force_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Add Vorticity Confinement Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &add_vorticity_confinement_force_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let add_vorticity_confinement_force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Add Vorticity Confinement Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("add_vorticity_confinement_force.wgsl").into()),
    });

    let subtract_pressure_gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Subtract Pressure Gradient Pipeline"),
        layout: Some(&add_vorticity_confinement_force_pipeline_layout),
        module: &add_vorticity_confinement_force_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Subtract Pressure Gradient Compute Step",
        subtract_pressure_gradient_pipeline,
        add_vorticity_confinement_force_bind_group_layout,
    )
}

fn create_compute_temperature_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let compute_temperature_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Temperature Bind Group Layout"),
        entries: &[
            // 0. Scalar field texture read.
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
            // 1. Scalar field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let compute_temperature_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Temperature Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &compute_temperature_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let compute_temperature_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Temperature Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("compute_temperature.wgsl").into()),
    });

    let compute_temperature_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Temperature Pipeline"),
        layout: Some(&compute_temperature_pipeline_layout),
        module: &compute_temperature_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Advect Scalars Compute Step",
        compute_temperature_pipeline,
        compute_temperature_bind_group_layout,
    )
}

fn create_compute_fuel_compute_step(device: &Device, compute_params_bind_group_layout: &wgpu::BindGroupLayout) -> ComputeStep {
    let compute_fuel_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Fuel Bind Group Layout"),
        entries: &[
            // 0. Scalar field texture read.
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
            // 1. Scalar field texture write.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: CHANNEL_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
            // 2. Sampler.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ]
    });

    let compute_fuel_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Fuel Pipeline Layout"),
            bind_group_layouts: &[
                compute_params_bind_group_layout,
                &compute_fuel_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let compute_fuel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Fuel Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("compute_fuel.wgsl").into()),
    });

    let compute_fuel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Temperature Pipeline"),
        layout: Some(&compute_fuel_pipeline_layout),
        module: &compute_fuel_shader,
        // Will default to @compute
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    ComputeStep::new(
        "Advect Scalars Compute Step",
        compute_fuel_pipeline,
        compute_fuel_bind_group_layout,
    )
}
