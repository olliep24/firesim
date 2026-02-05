// Uniform buffers
struct Params {
    dt: f32,
    width: u32,
    height: u32,
    depth: u32,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    viewport: vec2<f32>,
    _pad0: vec2<f32>
}
@group(0) @binding(0)
var<uniform> params: Params;

@group(1) @binding(0)
var pressure_read: texture_3d<f32>;
@group(1) @binding(0)
var pressure_write: texture_storage_3d<rgba16float, write>;
@group(1) @binding(1)
var divergence: texture_3d<f32>;
@group(1) @binding(2)
var field_sampler: sampler;

@compute
@workgroup_size(4, 4, 4)
fn main (
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Global invocation id corresponds to the index of a voxel in the simulation grid.
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.depth) {
        // In case of out of bounds.
        return;
    }

    // skip outer boundary cells
    // TODO: Revisit boundary conditions
    if (
        gid.x == 0u || gid.x >= params.width  - 1u ||
        gid.y == 0u || gid.y >= params.height - 1u ||
        gid.z == 0u || gid.z >= params.depth  - 1u
    ) {
        return;
    }

    let pressure_estimate = get_pressure_estimate(gid);
    textureStore(
        pressure_write,
        vec3<i32>(gid),
        vec4<f32>(pressure_estimate, 0.0, 0.0, 0.0)
    );
}

// Returns the center of the voxel indexed at gid.
fn voxel_center_uvw(gid: vec3<u32>) -> vec3<f32> {
    let w = f32(params.width);
    let h = f32(params.height);
    let d = f32(params.depth);
    return vec3<f32>(
        (f32(gid.x) + 0.5) / w,
        (f32(gid.y) + 0.5) / h,
        (f32(gid.z) + 0.5) / d
    );
}

// Computes the pressure estimate at the given gid for this iteration of the jacobi method.
// This is solving the Poisson-pressure equation.
// This assumes that the cells are one unit apart.
fn get_pressure_estimate(gid: vec3<u32>) {
    let right_pressure = get_pressure(vec3<u32>(gid.x + 1, gid.y, gid.z));
    let left_pressure = get_pressure(vec3<u32>(gid.x - 1, gid.y, gid.z));

    let up_pressure = get_pressure(vec3<u32>(gid.x, gid.y + 1, gid.z));
    let down_pressure = get_pressure(vec3<u32>(gid.x, gid.y - 1, gid.z));

    let front_pressure = get_pressure(vec3<u32>(gid.x, gid.y, gid.z + 1));
    let back_pressure = get_pressure(vec3<u32>(gid.x, gid.y, gid.z - 1));

    let divergence = get_divergence(gid);

    return (right_pressure + left_pressure + up_pressure +
            down_pressure + front_pressure + back_pressure - divergence) / 6;
}

fn get_pressure(index: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(index);
    return textureSampleLevel(pressure_read, field_sampler, uvw, 0.0).x;
}

fn get_divergence(gid: vec3<u32>) -> f32 {
    let uvw = voxel_center_uvw(gid);
    return textureSampleLevel(divergence, field_sampler, uvw, 0.0).x;
}
