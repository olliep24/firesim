use crate::texture::Texture;

pub struct PingPong {
    texture_a: Texture,
    texture_b: Texture,
    a_to_b: bool,
}

impl PingPong {
    pub fn new(texture_a: Texture, texture_b: Texture) -> Self {
        Self {
            texture_a,
            texture_b,
            a_to_b: true,
        }
    }

    pub fn get_read_and_write(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        let read_and_write = if self.a_to_b {
            (&self.texture_a.view, &self.texture_b.view)
        } else {
            (&self.texture_b.view, &self.texture_a.view)
        };

        read_and_write
    }

    pub fn get_read(&self) -> &wgpu::TextureView {
        let read = if self.a_to_b {
            // If we were to read from a and write to b, then a has the most up-to-date data.
            &self.texture_a.view
        } else {
            // If we were to read from b and write to a, then a has the most up-to-date data.
            &self.texture_b.view
        };

        read
    }

    pub fn swap(&mut self) {
        self.a_to_b = !self.a_to_b;
    }

    pub fn get_sampler(&self) -> &wgpu::Sampler {
        &self.texture_a.sampler
    }
}