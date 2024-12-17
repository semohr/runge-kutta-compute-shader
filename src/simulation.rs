use std::default;

const MAX_INITIAL_CONDITIONS: usize = 32;

/// State vector consists of 4 floats.
/// t, x, y, z
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct State(pub f32, pub f32, pub f32, pub f32);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub sigma: f32,
    pub rho: f32,
    pub beta: f32,
    _padding: u32,
}

impl Params {
    pub fn new(sigma: f32, rho: f32, beta: f32) -> Self {
        Self {
            sigma,
            rho,
            beta,
            _padding: 0,
        }
    }
}

impl default::Default for State {
    fn default() -> Self {
        Self(0.0, 0.0, 0.0, 0.0)
    }
}

/// Initial conditions for a simulation.
///
/// The initial state of the system, and rk4 parameters for the simulation.
///
/// The simulation will record the state of the system at each time step.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InitialCondition {
    pub initial_state: State, // 16 bytes
    pub params: Params,       // 16 bytes
}

impl InitialCondition {
    pub fn new(initial_state: [f32; 4], sigma: f32, rho: f32, beta: f32) -> Self {
        Self {
            initial_state: State(
                initial_state[0],
                initial_state[1],
                initial_state[2],
                initial_state[3],
            ),
            params: Params::new(sigma, rho, beta),
        }
    }
}

/// A simulation that holds multiple initial conditions.
///
/// This struct is used to manage and run simulations based on a set of initial
/// conditions. Each initial condition defines the starting state of the system
/// and the parameters for the simulation, such as time step and maximum time.
///
/// The simulation can handle up to `MAX_INITIAL_CONDITIONS` initial conditions.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Simulation {
    pub initial_conditions: [InitialCondition; MAX_INITIAL_CONDITIONS], // 32 bytes
    // The time step for each recorded position.
    dt: f32, // 4 bytes
    // The maximum t value for the simulation.
    max_t: f32,         // 4 bytes
    _padding: [u32; 2], // 8 bytes
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            initial_conditions: [InitialCondition::new(
                [0.0, 0.0, 0.0, 0.0], // initial_state: State::default(
                10.0,                 // sigma: 10.0,
                28.0,                 // rho: 28.0,
                8.0 / 3.0,            // beta: 8.0 / 3.0,
            ); MAX_INITIAL_CONDITIONS],
            dt: 0.01,
            max_t: 100.0,
            _padding: [0; 2],
        }
    }

    pub fn from_initial_conditions(
        initial_conditions: &[InitialCondition],
        dt: f32,
        num_steps: u32,
    ) -> Self {
        let mut simulation = Self::new();
        simulation.dt = dt;
        simulation.max_t = num_steps as f32 * dt;
        for (i, ic) in initial_conditions.iter().enumerate() {
            simulation.initial_conditions[i] = *ic;
        }
        simulation
    }

    pub fn num_iterations(&self) -> usize {
        (self.max_t / self.dt).ceil() as usize
    }
}

// Has to be a power of 2.
const MAX_ITERATIONS: usize = 10000;

#[repr(C)]
pub struct Output {
    pub position: Vec<State>,
}

impl Output {
    pub fn new() -> Self {
        Self { position: vec![] }
    }

    pub fn get_state(&self, t: usize, i: usize) -> State {
        self.position[t * MAX_INITIAL_CONDITIONS + i]
    }

    pub fn max_mem_size() -> usize {
        std::mem::size_of::<State>() * Self::max_size()
    }
    pub fn max_size() -> usize {
        MAX_ITERATIONS * MAX_INITIAL_CONDITIONS
    }
}
