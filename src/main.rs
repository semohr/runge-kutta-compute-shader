use simulation::{InitialCondition, Output, Simulation, State};
use std::fs::File;
use std::io::prelude::*;
use std::iter::Iterator;

mod simulation;

fn main() {
    // Setup logging for the application
    // for wasm target log via console_log crate
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");

        crate::utils::add_web_nothing_to_see_msg();
        wasm_bindgen_futures::spawn_local(run());
    }
}

async fn run() {
    // Define some initial conditions for the simulation
    let sigma_values = vec![10.0];
    let rho_values = vec![28.0];
    let beta_values = vec![2.667];
    let mut initial_conditions: Vec<InitialCondition> = vec![];

    for sigma in sigma_values.iter() {
        for rho in rho_values.iter() {
            for beta in beta_values.iter() {
                initial_conditions.push(InitialCondition::new(
                    [0.0, 0.0, 1.0, 1.05],
                    sigma.clone(),
                    rho.clone(),
                    beta.clone(),
                ));
            }
        }
    }

    log::info!("Initial conditions defined ({})!", initial_conditions.len());

    // Define the simulation
    let simulation = Simulation::from_initial_conditions(&initial_conditions, 0.01, 10000);

    // Engage!
    let context = WgpuContext::new().await;
    let results = compute(simulation, &context).await;

    // Write all defined iterations to output for each starting position to a file
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut file = File::create("output.txt").unwrap();
        file.write_all(b"initial_condition, time, x, y, z\n")
            .unwrap();

        for (i, ic) in initial_conditions.iter().enumerate() {
            file.write_all(
                format!(
                    "{}, {}, {}, {}, {}\n",
                    i,
                    ic.initial_state.0,
                    ic.initial_state.1,
                    ic.initial_state.2,
                    ic.initial_state.3
                )
                .as_bytes(),
            )
            .unwrap();

            for t in 0..simulation.num_iterations() {
                let position = results.get_state(t, i);
                file.write_all(
                    format!(
                        "{}, {}, {}, {}, {}\n",
                        i, position.0, position.1, position.2, position.3
                    )
                    .as_bytes(),
                )
                .unwrap();
            }
        }
        log::info!("Results written to output.txt.");
    }
}

fn generate_rand_u32() -> u32 {
    let mut bytes = [0u8; 4];
    getrandom::getrandom(&mut bytes[..]).unwrap();
    u32::from_le_bytes(bytes)
}

async fn compute(simulation: Simulation, context: &WgpuContext) -> Output {
    log::info!("Starting compute");
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f32();
    let state = generate_rand_u32();

    context.queue.write_buffer(
        &context.globals_buffer,
        0,
        bytemuck::cast_slice(&[Globals { time, state }]),
    );
    log::info!("Wrote globals buffer.");

    context.queue.write_buffer(
        &context.simulation_buffer,
        0,
        bytemuck::cast_slice(&[simulation]),
    );
    log::info!("Wrote simulation buffer.");

    let mut output = Output::new();
    context.queue.write_buffer(
        &context.output_storage_buffer,
        0,
        bytemuck::cast_slice(&vec![State::default(); Output::max_size()]),
    );

    let mut command_encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.pipeline);
        compute_pass.set_bind_group(0, &context.bind_group, &[]);
        // Run every initial condition in parallel.
        let num_groups = simulation.initial_conditions.len();
        compute_pass.dispatch_workgroups(num_groups as u32, 1, 1);
        log::info!("Dispatching {num_groups} workgroups.");
    } // We finish the compute pass by dropping it.

    // Entire storage buffer -> staging buffer.
    command_encoder.copy_buffer_to_buffer(
        &context.output_storage_buffer,
        0,
        &context.output_staging_buffer,
        0,
        context.output_storage_buffer.size(),
    );
    context.queue.submit(Some(command_encoder.finish()));

    log::info!("Submitted commands.");
    let buffer_slice = context.output_staging_buffer.slice(..);
    // Now things get complicated. WebGPU, for safety reasons, only allows either the GPU
    // or CPU to access a buffer's contents at a time. We need to "map" the buffer which means
    // flipping ownership of the buffer over to the CPU and making access legal. We do this
    // with `BufferSlice::map_async`.
    //
    // The problem is that map_async is not an async function so we can't await it. What
    // we need to do instead is pass in a closure that will be executed when the slice is
    // either mapped or the mapping has failed.
    //
    // The problem with this is that we don't have a reliable way to wait in the main
    // code for the buffer to be mapped and even worse, calling get_mapped_range or
    // get_mapped_range_mut prematurely will cause a panic, not return an error.
    //
    // Using channels solves this as awaiting the receiving of a message from
    // the passed closure will force the outside code to wait. It also doesn't hurt
    // if the closure finishes before the outside code catches up as the message is
    // buffered and receiving will just pick that up.
    //
    // It may also be worth noting that although on native, the usage of asynchronous
    // channels is wholly unnecessary, for the sake of portability to WASM (std channels
    // don't work on WASM,) we'll use async channels that work on both native and WASM.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());

    // In order for the mapping to be completed, one of three things must happen.
    // One of those can be calling `Device::poll`. This isn't necessary on the web as devices
    // are polled automatically but natively, we need to make sure this happens manually.
    // `Maintain::Wait` will cause the thread to wait on native but not on WebGpu.
    context
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    log::info!("Device polled.");

    // Now we await the receiving and panic if anything went wrong because we're lazy.
    receiver.recv_async().await.unwrap().unwrap();

    // NOW we can call get_mapped_range.
    {
        let view = buffer_slice.get_mapped_range();
        output.position = bytemuck::cast_slice(&view).to_vec();
    }
    log::info!("Results written to local buffer.");

    // count all values that are not zero
    let count = output.position.iter().filter(|x| x.3 != 0.0).count();
    log::info!("Counted {} non-zero values.", count);

    // And finally, we unmap the buffer.
    context.output_staging_buffer.unmap();
    return output;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    time: f32,
    state: u32,
}

/// A convenient way to hold together all the useful wgpu stuff together.
struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    globals_buffer: wgpu::Buffer,
    simulation_buffer: wgpu::Buffer,
    output_staging_buffer: wgpu::Buffer,
    output_storage_buffer: wgpu::Buffer,
}

impl WgpuContext {
    async fn new() -> WgpuContext {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        // Our shader, kindly compiled with Naga.
        let descriptor = wgpu::include_wgsl!("shader.wgsl");
        let shader = device.create_shader_module(descriptor);

        // Create globals buffer that holds uniform data (for now only time)
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Globals buffer"),
            size: std::mem::size_of::<Globals>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,

            mapped_at_creation: false,
        });

        // This is where the GPU will read from and write to.
        let simulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Simulation buffer"),
            size: std::mem::size_of::<simulation::Simulation>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output storage buffer"),
            size: Output::max_mem_size() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // For portability reasons, WebGPU draws a distinction between memory that is
        // accessible by the CPU and memory that is accessible by the GPU. Only
        // buffers accessible by the CPU can be mapped and accessed by the CPU and
        // only buffers visible to the GPU can be used in shaders. In order to get
        // data from the GPU, we need to use CommandEncoder::copy_buffer_to_buffer
        // (which we will later) to copy the buffer modified by the GPU into a
        // mappable, CPU-accessible buffer which we'll create here.
        let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output staging buffer"),
            size: Output::max_mem_size() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // This can be though of as the function signature for our CPU-GPU function.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        // Going to have this be None just to be safe.
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // This ties actual resources stored in the GPU to our metaphorical function
        // through the binding slots we defined above.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: simulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        WgpuContext {
            device,
            queue,
            pipeline,
            bind_group,
            simulation_buffer,
            output_storage_buffer,
            output_staging_buffer,
            globals_buffer,
        }
    }
}
