use std::sync::Arc;
use std::time::Duration;
use wgpu::{Device, Queue, Surface, SurfaceConfiguration};
use wgpu::util::DeviceExt;
use winit::event::ElementState;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::window::Window;

use crate::vertex::Vertex;
use crate::camera::{Camera, CameraController, CameraUniform, Projection};
use crate::instance::{Instance, InstanceRaw};
use crate::texture::Texture;
use crate::compute_params::ComputeParams;
use crate::config::{GRID_DIMENSION_LENGTH, GRID_VOXEL_SIDE_LENGTH};

/**
Each channel (RBGA) in the texture will be a 16-bit float.
TODO: My current machine allows this will the texture usages I need, but add check for this.
*/
const VECTOR_FIELD_CHANNEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SCALAR_FIELD_CHANNEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const SQUARE_SCALE: f32 = 0.01;

const VERTICES: &[Vertex] = &[
    // bottom-left
    Vertex { position: [-SQUARE_SCALE, -SQUARE_SCALE, 0.0], color: [SQUARE_SCALE, 0.0, SQUARE_SCALE] },
    // bottom-right
    Vertex { position: [SQUARE_SCALE, -SQUARE_SCALE, 0.0], color: [SQUARE_SCALE, 0.0, SQUARE_SCALE] },
    // top-right
    Vertex { position: [SQUARE_SCALE, SQUARE_SCALE, 0.0], color: [SQUARE_SCALE, 0.0, SQUARE_SCALE] },
    // top-left
    Vertex { position: [-SQUARE_SCALE, SQUARE_SCALE, 0.0], color: [SQUARE_SCALE, 0.0, SQUARE_SCALE] },
];

const INDICES: &[u16] = &[
    0, 1, 2,
    0, 2, 3,
];

const NUM_INSTANCES_PER_VOXEL_SIDE: u32 = 2;

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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    compute_params_buffer: wgpu::Buffer,
    use_a_to_b: bool,
    compute_bind_group_a_to_b: wgpu::BindGroup,
    compute_bind_group_b_to_a: wgpu::BindGroup,
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

        let camera = Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let camera_controller = CameraController::new(4.0, 0.4);
        let projection = Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

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
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let num_indices = INDICES.len() as u32;
        let instance_displacement = GRID_VOXEL_SIDE_LENGTH / NUM_INSTANCES_PER_VOXEL_SIDE as f32;

        let instances = (0..GRID_DIMENSION_LENGTH).flat_map(|x| {
            (0..GRID_DIMENSION_LENGTH).flat_map(move |y| {
                (0..GRID_DIMENSION_LENGTH).flat_map(move |z| {
                    // Create billboards in each voxel.
                    (0..NUM_INSTANCES_PER_VOXEL_SIDE).flat_map(move |i| {
                        (0..NUM_INSTANCES_PER_VOXEL_SIDE).flat_map(move |j| {
                            (0..NUM_INSTANCES_PER_VOXEL_SIDE).map(move |k| {
                                let x_position = x as f32 * GRID_VOXEL_SIDE_LENGTH + instance_displacement * i as f32;
                                let y_position = y as f32 * GRID_VOXEL_SIDE_LENGTH + instance_displacement * j as f32;
                                let z_position = z as f32 * GRID_VOXEL_SIDE_LENGTH + instance_displacement * k as f32;

                                let position = cgmath::Vector3 {
                                    x: x_position,
                                    y: y_position,
                                    z: z_position,
                                };

                                Instance {
                                    position
                                }
                            })
                        })
                    })
                })
            })
        }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::desc(), InstanceRaw::desc()
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
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

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into()),
        });

        let compute_params = ComputeParams::new(Duration::new(0, 0));

        let compute_params_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Compute Parameters Buffer"),
                contents: bytemuck::cast_slice(&[compute_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Pipeline Bind Group Layout"),
            entries: &[
                // 0. Uniform buffer for compute params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1. Velocity vector field texture.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
        let velocity_field_texture = Texture::create_compute_texture(
            &device,
            VECTOR_FIELD_CHANNEL_FORMAT,
            Some("Velocity Field Texture")
        );

        // Set static velocity field.
        velocity_field_texture.write_velocity_3d_rgba16f_tornado(&queue);

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

        // Create two bind groups to ping pong between, controlled by use_a_to_b flag.
        let compute_bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A to B"),
            layout: &compute_bind_group_layout,
            entries: &[
                // binding 0: Compute params
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_params_buffer.as_entire_binding(),
                },
                // binding 1: Velocity field read
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&velocity_field_texture.view)
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
                // binding 4: Sampler for density scalar field (either a or b work)
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
                // binding 0: Compute params
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compute_params_buffer.as_entire_binding(),
                },
                // binding 1: Velocity field read
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&velocity_field_texture.view)
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
                // binding 4: Sampler for density scalar field (either a or b work)
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&density_scalar_field_texture_b.sampler)
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
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
            vertex_buffer,
            index_buffer,
            num_indices,
            instances,
            instance_buffer,
            compute_pipeline,
            compute_params_buffer,
            use_a_to_b: true,
            compute_bind_group_a_to_b,
            compute_bind_group_b_to_a,
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
            self.is_surface_configured = true;
        }
    }

    pub fn update(&mut self, dt: instant::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        /*
        Potential to optimize:
        We can create a separate buffer and copy its contents to our camera_buffer. The new buffer
        is known as a staging buffer. This method is usually how it's done as it allows the contents
        of the main buffer (in this case, camera_buffer) to be accessible only by the GPU. The GPU
        can do some speed optimizations, which it couldn't if we could access the buffer via
        the CPU.
         */
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
        self.queue.write_buffer(&self.compute_params_buffer, 0, bytemuck::cast_slice(&[ComputeParams::new(dt)]));
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, key_state: ElementState) {
        if code == KeyCode::Escape && key_state.is_pressed() {
            event_loop.exit();
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

        {
            // TODO: Figure out number of dispatches and work groups.
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.compute_pipeline);

            compute_pass.set_bind_group(
                0,
                if self.use_a_to_b { &self.compute_bind_group_a_to_b } else { &self.compute_bind_group_b_to_a },
                &[]
            );
            // Ping pong between textures.
            self.use_a_to_b = !self.use_a_to_b;

            // We specified 4 threads per dimension in the compute shader.
            let num_dispatches_per_dimension = GRID_DIMENSION_LENGTH / 4;
            compute_pass.dispatch_workgroups(
                num_dispatches_per_dimension,
                num_dispatches_per_dimension,
                num_dispatches_per_dimension
            );
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
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0,0..self.instances.len() as _);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}