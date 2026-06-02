# firesim

A real-time 3D fire and smoke simulator running entirely on the GPU, written in Rust with [wgpu](https://github.com/gfx-rs/wgpu). Compiles to native desktop and WebAssembly.

![firesim demo](assets/firesim.gif)
## Overview

The simulation uses an Eulerian fluid solver on a 128³ voxel grid. Each frame runs a sequence of WGSL compute shaders that evolve a scalar density/temperature field and a velocity field, then renders the result via volume ray marching.

## Simulation Pipeline

Each frame executes the following stages in order:

1. **Add source** — injects smoke density into the scalar field (toggled with `F`)
2. **Advect scalars** — moves smoke density through the velocity field using semi-Lagrangian advection
3. **Compute temperature** — derives temperature from density (stored in the `y` channel of the scalar texture)
4. **Decay smoke** — attenuates density over time
5. **Advect velocity** — self-advects the velocity field
6. **Add forces** — applies buoyancy: hot voxels receive an upward impulse proportional to temperature
7. **Vorticity confinement** — computes the curl of the velocity field, then injects a corrective force to restore turbulent detail lost to numerical dissipation
8. **Projection** — enforces incompressibility:
   - Compute divergence of the velocity field
   - Solve for pressure via 20 Jacobi iterations (ping-pong buffers)
   - Subtract the pressure gradient from velocity

## Rendering

A full-screen triangle is drawn and the fragment shader ray-marches 64 steps through an axis-aligned bounding box:

- **Smoke** — Beer-Lambert extinction using the accumulated density; composited front-to-back
- **Fire** — physically-based blackbody radiation: Planck's law integrated against CIE 1931 color matching functions, converted XYZ → linear sRGB, then Reinhard tone-mapped

## Implementation Notes

- All fields are stored as `Rgba16Float` 3D textures; ping-pong double-buffering avoids read/write hazards
- Each compute stage is wrapped in a `ComputeStep` that manages its pipeline and bind group
- Supports both native (Vulkan/Metal/DX12) and WebAssembly (WebGL) backends

## Controls

| Key                                | Action |
|------------------------------------|--------|
| `F`                                | Toggle smoke/fire injection |
| `WASD` / `SPACE` / `SHIFT` / mouse | Orbit camera |
| `Escape`                           | Quit |

## Building

```sh
# Native
cargo run

# WebAssembly
wasm-pack build --target web
```
