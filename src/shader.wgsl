// Even tho we dont need the globals in this example,
// I still like to define them
struct Globals {
    current_time: f32,
    rng_state: u32,
};
@group(0)
@binding(0)
var<uniform> globals: Globals;


// Should be the same as in the simulations.rs file. Otherwise the array
// flattening will not work
const MAX_INITIAL_CONDITIONS: u32 = 32;
const MAX_ITERATIONS: u32 = 10000;
const MAX_OUTPUT: u32 = MAX_ITERATIONS * MAX_INITIAL_CONDITIONS * 4;


// The simulation holds the initial conditions 
// and the simulation parameters
struct InitialCondition{
    initial_state: vec4<f32>,
    params: vec4<f32>,
}
struct Simulation{
    initial_conditions: array<InitialCondition, MAX_INITIAL_CONDITIONS>,
    dt: f32,
    max_t: f32,
};
@group(0)
@binding(1)
var<storage, read_write> simulation: Simulation;


// The output array is a flat array that maps the output (t,sp,x,y,z)
// of the simulation to a 1D array
@group(0)
@binding(2)
var<storage, read_write> output: array<f32, MAX_OUTPUT>;


@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get initial condition for this thread
    let initial_condition = simulation.initial_conditions[global_id.x];
    let initial_state = initial_condition.initial_state;
    var state = initial_state;

    sigma = initial_condition.params.x;
    rho = initial_condition.params.y;
    beta = initial_condition.params.z;

    // Run the simulation
    var n_iterations = simulation.max_t / simulation.dt;
    var i = 0u;
    while (i < u32(n_iterations)) {
        state = runge_kutta(state, simulation.dt , 1u);
        write_point_to_output(i, global_id.x, state);
        i = i + 1u;
    }
}


/* -------------------------------------------------------------------------- */
/*                               Output mapping                               */
/* -------------------------------------------------------------------------- */

fn write_point_to_output(iteration: u32, starting_point: u32, v: vec4<f32>) {
    let idx = output_index(iteration, starting_point, 0u);
    output[idx] = v.x;
    output[idx + 1u] = v.y;
    output[idx + 2u] = v.z;
    output[idx + 3u] = v.w;
}

fn output_index(iteration: u32, stating_point: u32, component: u32) -> u32 {
    return component + 4u * (stating_point + MAX_INITIAL_CONDITIONS * iteration);
}


/* -------------------------------------------------------------------------- */
/*                                 Runge Kutta                                */
/* -------------------------------------------------------------------------- */
// Implement a simple Runge-Kutta integrator for a 3D system of ODEs
//  dy/dt = f(t,y)

var<workgroup> sigma: f32;
var<workgroup> rho: f32;
var<workgroup> beta: f32;

fn f(t:f32, y: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sigma * (y.y - y.x),
        rho * y.x -y.y - y.x * y.z,
        y.x * y.y - beta * y.z
    );
}


/// Performs multiple steps of the Runge-Kutta integration method.
///
/// # Parameters
/// - `t`: The current time.
/// - `y`: The current state vector (3D vector).
/// - `h`: The step size for the integration.
/// - `n`: The number of steps to perform.
///
/// # Returns
/// The state vector after performing `n` steps of the Runge-Kutta method.
fn runge_kutta(y: vec4<f32>, h: f32, n: u32) -> vec4<f32> {
    var y_n = y.yzw;
    var t = y.x;
    for (var i = 0u; i < n; i = i + 1u) {
        y_n = runge_kutta_step(t, y_n, h);
        t = t + h;
    }
    return vec4<f32>(t, y_n);
}


/*
Butcher table:
0
1/2 	1/2
1/2 	0    	1/2
1 	    0 	      0 	1 	
	    1/6 	1/3 	1/3 	1/6 
*/
fn runge_kutta_step(t: f32, y: vec3<f32>, h: f32) -> vec3<f32> {
    var k1 = f(t, y);
    var k2 = f(t + h / 2.0, y + h / 2.0 * k1);
    var k3 = f(t + h / 2.0, y + h / 2.0 * k2);
    var k4 = f(t + h, y + h * k3);
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

