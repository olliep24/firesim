use cgmath::*;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use instant::Duration;
use std::f32::consts::FRAC_PI_2;
use winit::keyboard::KeyCode;

/*
The coordinate system in Wgpu is based on DirectX and Metal's coordinate systems. That means that
in normalized device coordinates (opens new window), the x-axis and y-axis are in the range of
-1.0 to +1.0, and the z-axis is 0.0 to +1.0. The cgmath crate (as well as most game math crates)
is built for OpenGL's coordinate system. This matrix will scale and translate our scene from
OpenGL's coordinate system to WGPU's.
 */
#[rustfmt::skip]
const _OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::from_cols(
    Vector4::new(1.0, 0.0, 0.0, 0.0),
    Vector4::new(0.0, 1.0, 0.0, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 1.0),
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug)]
pub struct Camera {
    position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    pub fn new<
        V: Into<Point3<f32>>,
        Y: Into<Rad<f32>>,
        P: Into<Rad<f32>>,
    >(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    /// Calculates the view matrix for the camera.
    pub fn _calc_view_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(
                cos_pitch * cos_yaw,
                sin_pitch,
                cos_pitch * sin_yaw
            ).normalize(),
            Vector3::unit_y(),
        )
    }

    /// Calculates the forward vector of the camera.
    fn calc_forward(&self) -> Vector3<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Vector3::new(
            cos_pitch * cos_yaw,
            sin_pitch,
            cos_pitch * sin_yaw
        ).normalize()
    }

    /// Calculates the up vector of the camera (just <0, 1, 0>).
    fn calc_up(&self) -> Vector3<f32> {
        Vector3::unit_y()
    }

    /// Calculates the forward vector of the camera.
    fn calc_right(&self) -> Vector3<f32> {
        self.calc_forward().cross(self.calc_up())
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    _znear: f32,
    _zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(
        width: u32,
        height: u32,
        fovy: F,
        _znear: f32,
        _zfar: f32,
    ) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            _znear,
            _zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    /// Calculates the projection matrix.
    pub fn _calc_matrix(&self) -> Matrix4<f32> {
        _OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self._znear, self._zfar)
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    // Padding is required to satisfy 16-byte alignment rules for uniform buffers.
    camera_position: [f32; 3],
    _pad0: f32,

    camera_forward: [f32; 3],
    _pad1: f32,

    camera_right: [f32; 3],
    _pad2: f32,

    camera_up: [f32; 3],
    _pad3: f32,

    tan_half_fovy: f32,
    aspect: f32,
    _pad4: [f32; 2],
}

impl CameraUniform {
    /// Updates the camera uniform given a camera and projection.
    pub fn update(&mut self, camera: &Camera, projection: &Projection) {
        self.camera_position = camera.position.into();
        self.camera_forward = camera.calc_forward().into();
        self.camera_right = camera.calc_right().into();
        self.camera_up = camera.calc_up().into();
        self.tan_half_fovy = (projection.fovy.0 * 0.5).tan();
        self.aspect = projection.aspect;
    }
}

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) {
        let amount = if state == ElementState::Pressed { 1.0 } else { 0.0 };

        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
            }
            KeyCode::Space => {
                self.amount_up = amount;
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.amount_down = amount;
            }
            _ => {},
        }
    }

    pub fn handle_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn handle_mouse_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition {
                                             y: scroll,
                                             ..
                                         }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non-cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}
