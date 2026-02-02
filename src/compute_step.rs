/// Struct to contain each computation step in the simulation.
/// It owns the compute pipeline and bind group layout for the computation step.
///
/// Each computation step can be broken down into reading and writing to a texture. This is managed
/// through ping-ponging. Each step can also take in a varying number of read only textures.
///
/// The bind group that this struct owns will be set at bind group 1.
pub struct ComputeStep {
    label: &'static str,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputeStep {
    pub fn new(label: &'static str, compute_pipeline: wgpu::ComputePipeline, bind_group_layout: wgpu::BindGroupLayout) -> Self {
        Self {
            label,
            compute_pipeline,
            bind_group_layout,
        }
    }

    /// Dispatches workers and completes the compute shader that this struct represents.
    ///
    /// The read texture will be binded at index 0.
    /// The write texture will be binded at index 1.
    /// The read only textures will be binded in the order they were given.
    /// The sampler, if provided, will be binded at the last index.
    ///
    /// The bind group layout created on this struct's creation needs to have the layout that will
    /// be binded base on this functions input.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        compute_params_bind_group: &wgpu::BindGroup,
        texture_read: &wgpu::TextureView,
        texture_write: &wgpu::TextureView,
        textures_read_only: &[&wgpu::TextureView],
        sampler: Option<&wgpu::Sampler>,
        workgroups: (u32, u32, u32),
    ) {
        // Build entries in the order WGSL expects:
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(texture_read),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(texture_write),
        });

        for (i, v) in textures_read_only.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            });
        }

        if let Some(s) = sampler {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + textures_read_only.len() as u32,
                resource: wgpu::BindingResource::Sampler(s),
            });
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.label),
            layout: &self.bind_group_layout,
            entries: &entries,
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, compute_params_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
}