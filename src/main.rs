mod app;
mod state;
mod camera;
mod texture;
mod compute_params;
mod config;
mod compute_step;
mod ping_pong;

use winit::event_loop::{ControlFlow, EventLoop};
use crate::app::{run, App};

fn main() {
    run().unwrap();
}
