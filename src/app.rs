use std::sync::Arc;
use instant::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::Window;
use crate::state::State;

#[derive(Default)]
pub struct App {
    state: Option<State>,
    last_render_time: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());
        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        let last_render_time = *self.last_render_time.get_or_insert_with(Instant::now);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now - last_render_time;
                self.last_render_time = Some(now);
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => state.handle_key(event_loop, code, key_state),
            WindowEvent::MouseInput { button: MouseButton::Left, state: mouse_state, ..} => {
                state.handle_mouse_click(mouse_state);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            DeviceEvent::MouseMotion { delta } => {
                if state.mouse_pressed {
                    state.camera_controller.handle_mouse(delta.0, delta.1);
                }
            },
            DeviceEvent::MouseWheel { delta } => {
                state.camera_controller.handle_mouse_scroll(&delta);
            }
            _ => {}
        }
    }
}