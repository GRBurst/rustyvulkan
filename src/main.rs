//TODO:
// Rotation von GameObject implementieren -> soll auch Kamera korrekt drehen
// Andere Bewegungsrichtungen von GameObject implementieren -> move_forward fertig 
// Relation zwischen Kamera und gO muss geprüft werden

mod core;
mod game;
mod math;
mod memory;
mod platform;
mod renderer;
mod resources;
mod scene;
mod debug;

use crate::{
    platform::input::InputSystem,
    platform::window::{WindowSystem, WindowConfig},
    renderer::*,
    renderer::buffer,
    renderer::render_loop::RenderLoop,
    resources::*,
    scene::*,
    scene::camera::Camera,
    debug::{
        setup_debug_messenger, 
        ENABLE_VALIDATION_LAYERS,
        get_layer_names_and_pointers,
        check_validation_layer_support
    }
};
use ash::{vk, Device, Entry, Instance};
use ash::ext::debug_utils;
use ash::khr::{surface, swapchain};
use ash::vk::Handle;

use std::{
    ffi::{CStr, CString},
    io::Cursor,
    mem::{size_of, align_of},
    path::Path,
    ptr,
    sync::Arc,
};

use cgmath::{Deg, Matrix4, Vector3};

use winit::{
    event::{Event, WindowEvent}, 
    event_loop::{ControlFlow, EventLoop},
    keyboard::KeyCode, 
    window::Window,
    raw_window_handle::HasDisplayHandle,
};

// Constants moved to the top of the file for better organization
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

fn main() {
    env_logger::init();

    // Create the event loop
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    
    // Create window system
    let window_system = WindowSystem::new(
        WindowConfig {
            title: "Vulkan tutorial with Ash".to_string(),
            width: WIDTH,
            height: HEIGHT,
            resizable: true,
        },
        &event_loop
    );
    
    let mut app = VulkanApp::new(window_system);
    
    // Initialize the render loop
    app.init_render_loop();
    
    let mut dirty_swapchain = false;

    // Run the event loop
    event_loop
        .run(move |event, elwt| {
            // Set the control flow to poll mode
            elwt.set_control_flow(ControlFlow::Poll);
            
            match event {
                Event::NewEvents(_) => {
                    // Input system will handle resetting input states
                }
                Event::AboutToWait => {
                    // Draw a frame
                    let resized = app.draw_frame();
                    if resized || dirty_swapchain {
                        app.recreate_swapchain();
                        dirty_swapchain = false;
                    }
                    
                    // Reset input states after processing
                    app.input_system.reset_frame_state();
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => {
                        app.window_system.record_resize(size.width, size.height);
                        dirty_swapchain = true;
                    },
                    // Let the input system handle all input-related events
                    _ => {
                        app.input_system.process_event(&event, app.window_system.window());
                    }
                },
                Event::LoopExiting => app.wait_gpu_idle(),
                _ => {}
            }
        })
        .unwrap();
}

struct VulkanApp {
    window_system: WindowSystem,
    
    // New input system
    input_system: InputSystem,

    // Vulkan context and core resources
    vk_context: VkContext,
    queue_families_indices: QueueFamiliesIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    
    // New render loop for handling rendering
    render_loop: Option<RenderLoop>,
    
    // Legacy fields - will eventually be moved to RenderLoop
    swapchain: swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,
    msaa_samples: vk::SampleCountFlags,
    color_texture: texture::Texture,
    depth_format: vk::Format,
    depth_texture: texture::Texture,
    texture: texture::Texture,
    render_objects: Vec<RenderObject>,
    game_objects: Vec<GameObject>,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_memories: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,
}

impl VulkanApp {
    fn new(window_system: WindowSystem) -> Self {
        log::debug!("Creating application.");

        let window = window_system.window();
        let entry = unsafe { Entry::load().expect("Failed to create entry.") };
        let instance = Self::create_instance(&entry, window);

        // Create surface using the window system
        let (surface, surface_khr) = window_system.create_surface(&entry, &instance);

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let (physical_device, queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);

        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &instance,
                physical_device,
                queue_families_indices,
            );

        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let vk_context = VkContext::new(
            entry,
            instance,
            debug_utils,
            debug_report_callback.unwrap().1,
            surface,
            surface_khr,
            physical_device,
            device.clone(),
        );

        let dimensions = window_system.get_dimensions();
        let (swapchain, swapchain_khr, properties, images) =
            Self::create_swapchain_and_images(&vk_context, queue_families_indices, dimensions);
        let swapchain_image_views =
            Self::create_swapchain_image_views(vk_context.device(), &images, properties);

        let msaa_samples = vk_context.get_max_usable_sample_count();
        let depth_format = texture::find_depth_format(&vk_context);

        let render_pass =
            Self::create_render_pass(vk_context.device(), properties, msaa_samples, depth_format);
        let descriptor_set_layout = Self::create_descriptor_set_layout(vk_context.device());
        let (pipeline, layout) = pipeline::create_pipeline(
            vk_context.device(),
            properties,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
        );

        let command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_families_indices,
            vk::CommandPoolCreateFlags::empty(),
        );
        let transient_command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_families_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        let color_texture = texture::create_color_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            properties,
            msaa_samples,
        );

        let depth_texture = texture::create_depth_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            depth_format,
            properties.extent,
            msaa_samples,
        );

        let texture = texture::Texture::load_from_file(
            &vk_context,
            command_pool,
            graphics_queue,
            "images/chalet.jpg",
        );

        let game_objects = Self::create_game_objects(&vk_context, command_pool, graphics_queue);

        let (uniform_buffers, uniform_buffer_memories) =
            buffer::create_uniform_buffers(&vk_context, images.len());

        let swapchain_framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        let descriptor_pool = Self::create_descriptor_pool(vk_context.device(), images.len() as u32);

        let descriptor_sets = Self::create_descriptor_sets(
            vk_context.device(),
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            texture,
        );

        // Create command buffers
        let command_buffers = Self::create_and_register_command_buffers(
            vk_context.device(),
            command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            &game_objects,
            layout,
            &descriptor_sets,
            pipeline,
        );

        let mut app = Self {
            window_system,
            
            // Initialize input system
            input_system: InputSystem::new(),
            
            vk_context,
            queue_families_indices,
            graphics_queue,
            present_queue,
            render_loop: None,
            swapchain,
            swapchain_khr,
            swapchain_properties: properties,
            images,
            swapchain_image_views,
            render_pass,
            descriptor_set_layout,
            pipeline_layout: layout,
            pipeline,
            swapchain_framebuffers,
            command_pool,
            transient_command_pool,
            msaa_samples,
            color_texture,
            depth_format,
            depth_texture,
            texture,
            render_objects: Vec::new(),
            game_objects,
            uniform_buffers,
            uniform_buffer_memories,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            in_flight_frames: InFlightFrames::new(Vec::new()), // Temporary empty value
        };

        // Now create and set the in_flight_frames using the device from vk_context
        app.in_flight_frames = Self::create_sync_objects(app.vk_context.device());

        app
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().expect("Failed to get display handle").as_raw())
                .unwrap();
        let mut extension_names = extension_names.to_vec();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(debug_utils::NAME.as_ptr());
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };
        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);
        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    /// Pick the first suitable physical device.
    ///
    /// # Requirements
    /// - At least one queue family with one queue supportting graphics.
    /// - At least one queue family with one queue supporting presentation to `surface_khr`.
    /// - Swapchain extension support.
    ///
    /// # Returns
    ///
    /// A tuple containing the physical device and the queue families indices.
    fn pick_physical_device(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (device, queue_families_indices)
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let extention_support = Self::check_device_extension_support(instance, device);
        let is_swapchain_adequate = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extention_support
            && is_swapchain_adequate
            && features.sampler_anisotropy == vk::TRUE
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extentions = Self::get_required_device_extensions();

        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        for required in required_extentions.iter() {
            let found = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            });

            if !found {
                return false;
            }
        }

        true
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [swapchain::NAME]
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn get_required_device_extensions() -> [&'static CStr; 2] {
        [swapchain::NAME, ash::khr::portability_subset::NAME]
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    ///
    /// #Returns
    ///
    /// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
    fn find_queue_families(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap()
            };
            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }

    /// Create the logical device to interact with `device`, a graphics queue
    /// and a presentation queue.
    ///
    /// # Returns
    ///
    /// Return a tuple containing the logical device, the graphics queue and the presentation queue.
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (Device, vk::Queue, vk::Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features);

        // Build device and queues
        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device.")
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        (device, graphics_queue, present_queue)
    }

    /// Create the swapchain with optimal settings possible with
    /// `device`.
    ///
    /// # Returns
    ///
    /// A tuple containing the swapchain loader and the actual swapchain.
    fn create_swapchain_and_images(
        vk_context: &VkContext,
        queue_families_indices: QueueFamiliesIndices,
        dimensions: [u32; 2],
    ) -> (
        swapchain::Device,
        vk::SwapchainKHR,
        SwapchainProperties,
        Vec<vk::Image>,
    ) {
        let details = SwapchainSupportDetails::new(
            vk_context.physical_device(),
            vk_context.surface(),
            vk_context.surface_khr(),
        );
        let properties = details.get_ideal_swapchain_properties(dimensions);

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;
        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let graphics = queue_families_indices.graphics_index;
        let present = queue_families_indices.present_index;
        let families_indices = [graphics, present];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(vk_context.surface_khr())
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if graphics != present {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
            // .old_swapchain() We don't have an old swapchain but can't pass null
        };

        let swapchain = swapchain::Device::new(vk_context.instance(), vk_context.device());
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, properties, images)
    }

    /// Create one image view for each image of the swapchain.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                Self::create_image_view(
                    device,
                    *image,
                    1,
                    swapchain_properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_image_view(
        device: &Device,
        image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        texture::create_image_view(device, image, mip_levels, format, aspect_mask)
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachement_desc = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let resolve_attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let attachment_descs = [
            color_attachment_desc,
            depth_attachement_desc,
            resolve_attachment_desc,
        ];

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment_refs = [resolve_attachment_ref];

        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .resolve_attachments(&resolve_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref);
        let subpass_descs = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
        let subpass_deps = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps);

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let ubo_binding = buffer::UniformBufferObject::get_descriptor_set_layout_binding();
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [ubo_binding, sampler_binding];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    /// Create a descriptor pool to allocate the descriptor sets.
    fn create_descriptor_pool(device: &Device, size: u32) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(size);

        unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() }
    }

    /// Create one descriptor set for each uniform buffer.
    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        texture: Texture,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| layout)
            .collect::<Vec<_>>();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter())
            .for_each(|(set, buffer)| {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .offset(0)
                    .range(size_of::<buffer::UniformBufferObject>() as vk::DeviceSize);
                let buffer_infos = [buffer_info];

                let image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.view)
                    .sampler(texture.sampler.unwrap());
                let image_infos = [image_info];

                let ubo_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos);
                let sampler_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos);

                let descriptor_writes = [ubo_descriptor_write, sampler_descriptor_write];

                unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }
            });

        descriptor_sets
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        color_texture: Texture,
        depth_texture: Texture,
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        image_views
            .iter()
            .map(|view| [color_texture.view, depth_texture.view, *view])
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<_>>()
    }

    fn create_command_pool(
        device: &Device,
        queue_families_indices: QueueFamiliesIndices,
        create_flags: vk::CommandPoolCreateFlags,
    ) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_families_indices.graphics_index)
            .flags(create_flags);

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
        game_objects: &[GameObject],
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        graphics_pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _);

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                // .inheritance_info() null since it's a primary command buffer
                unsafe {
                    device
                        .begin_command_buffer(buffer, &command_buffer_begin_info)
                        .unwrap()
                };
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];
                let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
                    })
                    .clear_values(&clear_values);

                unsafe {
                    device.cmd_begin_render_pass(
                        buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    )
                };
            }

            // Bind pipeline
            unsafe {
                device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline)
            };

            // Draw each object
            for game_object in game_objects {
                if let Some(render_object) = &game_object.render_object {
                    // Bind vertex buffer
                    let vertex_buffers = [render_object.vertex_buffer];
                    let offsets = [0];
                    unsafe { 
                        device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &offsets);
                        device.cmd_bind_index_buffer(
                            buffer,
                            render_object.index_buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                        
                        // Bind descriptor set
                        let null = [];
                        device.cmd_bind_descriptor_sets(
                            buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout,
                            0,
                            &descriptor_sets[i..=i],
                            &null,
                        );

                        // Draw
                        device.cmd_draw_indexed(
                            buffer,
                            render_object.index_count as _,
                            1,
                            0,
                            0,
                            0,
                        );
                    };
                }
            }

            // End render pass
            unsafe { device.cmd_end_render_pass(buffer) };

            // End command buffer
            unsafe { device.end_command_buffer(buffer).unwrap() };
        });

        buffers
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects)
        }

        InFlightFrames::new(sync_objects_vec)
    }

    pub fn wait_gpu_idle(&self) {
        // If the render loop is initialized, use it, otherwise fall back to legacy code
        if let Some(render_loop) = &self.render_loop {
            render_loop.wait_gpu_idle();
        } else {
            unsafe { self.vk_context.device().device_wait_idle().unwrap() };
        }
    }

    pub fn draw_frame(&mut self) -> bool {
        // If render_loop is available, use it
        if let Some(render_loop) = &mut self.render_loop {
            return render_loop.draw_frame();
        }
        
        // Otherwise, use the legacy implementation
        // This will be removed once render_loop is fully implemented
        
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.vk_context
                .device()
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap()
        };

        let result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };
        
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return true;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };
        
        unsafe { self.vk_context.device().reset_fences(&wait_fences).unwrap() };

        self.update_uniform_buffers(image_index);

        let device = self.vk_context.device();
        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];
        
        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(self.graphics_queue, &submit_infos, in_flight_fence)
                    .unwrap()
            };
        }
        
        let swapchains = [self.swapchain_khr];
        let images_indices = [image_index];
        
        {
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices);
            // .results() null since we only have one swapchain
            let result = unsafe {
                self.swapchain
                    .queue_present(self.present_queue, &present_info)
            };
            match result {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return true;
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        false
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    pub fn recreate_swapchain(&mut self) {
        // If render_loop is available, use it
        if let Some(render_loop) = &mut self.render_loop {
            render_loop.recreate_swapchain();
            return;
        }
        
        // Otherwise, use the legacy implementation
        // This will be removed once render_loop is fully implemented
        
        self.wait_gpu_idle();
        self.cleanup_swapchain();

        // Get dimensions from window_system
        let dimensions = self.window_system.take_resize_dimensions()
            .unwrap_or_else(|| self.window_system.get_dimensions());
            
        let (swapchain, swapchain_khr, properties, images) = Self::create_swapchain_and_images(
            &self.vk_context,
            self.queue_families_indices,
            dimensions,
        );
        let swapchain_image_views =
            Self::create_swapchain_image_views(self.vk_context.device(), &images, properties);

        let render_pass = Self::create_render_pass(
            self.vk_context.device(),
            properties,
            self.msaa_samples,
            self.depth_format,
        );
        let (pipeline, layout) = pipeline::create_pipeline(
            self.vk_context.device(),
            properties,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
        );

        let color_texture = texture::create_color_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            properties,
            self.msaa_samples,
        );
        let depth_texture = texture::create_depth_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            self.depth_format,
            properties.extent,
            self.msaa_samples,
        );

        // We'll reuse the existing texture instead of reloading it from disk
        // This prevents flickering by avoiding the texture reload process
        
        // Create new uniform buffers sized appropriately for the new swapchain
        let (uniform_buffers, uniform_buffer_memories) =
            buffer::create_uniform_buffers(&self.vk_context, images.len());

        // Important: Recreate descriptor sets to bind the new uniform buffers
        // This was missing and likely caused the flickering
        let descriptor_sets = Self::create_descriptor_sets(
            self.vk_context.device(),
            self.descriptor_pool,
            self.descriptor_set_layout,
            &uniform_buffers,
            self.texture, // Reuse existing texture
        );

        let swapchain_framebuffers = Self::create_framebuffers(
            self.vk_context.device(),
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        // We'll reuse the existing game objects instead of recreating them
        // This preserves game state and prevents model reloading
        let command_buffers = Self::create_and_register_command_buffers(
            self.vk_context.device(),
            self.command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            &self.game_objects, // Use existing game objects
            layout,
            &descriptor_sets,
            pipeline,
        );

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.images = images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.pipeline_layout = layout;
        self.pipeline = pipeline;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        // We don't reassign the texture as we're reusing it
        // self.texture = texture;
        // We don't clear game objects as we're preserving them
        // self.render_objects = Vec::new();  
        // self.game_objects = game_objects;  
        self.uniform_buffers = uniform_buffers;
        self.uniform_buffer_memories = uniform_buffer_memories;
        self.descriptor_sets = descriptor_sets;
        self.command_buffers = command_buffers;
    }

    /// Clean up the swapchain and all resources that depend on it.
    /// 
    /// This method uses a defensive approach to resource cleanup:
    /// 1. It avoids cleaning up game object resources to preserve them during swapchain recreation
    /// 2. It checks for null/zero handles before destroying resources to prevent double-free issues
    /// 3. It nullifies handles after destruction to mark them as cleaned up
    fn cleanup_swapchain(&mut self) {
        let device = self.vk_context.device();
        unsafe {
            // We preserve game object resources during swapchain recreation
            // to prevent flickering and maintain game state
            
            // Clean up render resources
            self.depth_texture.destroy(device);
            self.color_texture.destroy(device);
            self.swapchain_framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None));
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            
            // Clean up pipeline resources with null checks
            if self.pipeline.as_raw() != 0 {
                device.destroy_pipeline(self.pipeline, None);
                // Zero out the handle to prevent double-free
                self.pipeline = vk::Pipeline::null();
            }
            
            if self.pipeline_layout.as_raw() != 0 {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
                // Zero out the handle to prevent double-free
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            
            if self.render_pass.as_raw() != 0 {
                device.destroy_render_pass(self.render_pass, None);
                // Zero out the handle to prevent double-free
                self.render_pass = vk::RenderPass::null();
            }
            
            // Clean up image views with null checks
            self.swapchain_image_views
                .iter_mut()
                .for_each(|v| {
                    if v.as_raw() != 0 {
                        device.destroy_image_view(*v, None);
                        *v = vk::ImageView::null();
                    }
                });
                
            // Clean up swapchain
            if self.swapchain_khr.as_raw() != 0 {
                self.swapchain.destroy_swapchain(self.swapchain_khr, None);
                self.swapchain_khr = vk::SwapchainKHR::null();
            }
            
            // Clean up uniform buffers and their memory
            // These will be recreated with the new swapchain
            if !self.uniform_buffers.is_empty() {
                self.uniform_buffer_memories
                    .iter_mut()
                    .for_each(|m| {
                        if m.as_raw() != 0 {
                            device.free_memory(*m, None);
                            *m = vk::DeviceMemory::null();
                        }
                    });
                    
                self.uniform_buffers
                    .iter_mut()
                    .for_each(|b| {
                        if b.as_raw() != 0 {
                            device.destroy_buffer(*b, None);
                            *b = vk::Buffer::null();
                        }
                    });
            }
            
            // Reset descriptor pool instead of destroying it
            // This allows reusing the pool for new descriptor sets
            if !self.descriptor_sets.is_empty() {
                device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                    .expect("Failed to reset descriptor pool!");
                // Clear the descriptor sets list to avoid stale references
                self.descriptor_sets.clear();
            }
        }
    }

    fn update_uniform_buffers(&mut self, current_image: u32) {
        if let Some(player) = self.game_objects.get_mut(0) {
            // Store camera movement directions based on input
            let mut forward_movement = 0.0;
            let mut right_movement = 0.0;
            let mut up_movement = 0.0;
            let mut rotation = None;
            
            // Check keyboard input
            if self.input_system.is_key_pressed(KeyCode::KeyW) {
                forward_movement += 0.03;
            }
            if self.input_system.is_key_pressed(KeyCode::KeyS) {
                forward_movement -= 0.03;
            }
            if self.input_system.is_key_pressed(KeyCode::KeyA) {
                right_movement -= 0.03;
            }
            if self.input_system.is_key_pressed(KeyCode::KeyD) {
                right_movement += 0.03;
            }
            
            // Get mouse movement
            if let Some(delta) = self.input_system.take_mouse_delta() {
                let sensitivity = 0.1;
                rotation = Some((-delta.y * sensitivity, -delta.x * sensitivity));
            }
            
            // Handle mouse wheel for zoom
            if let Some(wheel_delta) = self.input_system.get_wheel_delta() {
                up_movement = wheel_delta * 0.3;
            }
            
            // Now apply movements to camera and player
            if let Some(camera) = player.camera.as_mut() {
                // Apply rotation if any
                if let Some((x_rot, y_rot)) = rotation {
                    camera.rotate_camera(x_rot, y_rot);
                }
                
                // Apply right/left movement
                if right_movement != 0.0 {
                    camera.move_camera(camera.get_right() * right_movement);
                }
                
                // Apply up/down movement (from mouse wheel)
                if up_movement != 0.0 {
                    camera.move_camera(Vector3::new(0.0, 0.0, up_movement));
                }
                
                // Handle backward movement directly on camera
                if forward_movement < 0.0 {
                    camera.move_camera(camera.get_view_direction() * forward_movement);
                }
                
                // Set up UBO for rendering
                let ubo = buffer::UniformBufferObject {
                    model: Matrix4::from_angle_x(Deg(0.0)),
                    view: camera.look_to(camera.get_view_direction()),
                    proj: camera.get_projection_matrix(
                        self.swapchain_properties.extent.width as f32
                            / self.swapchain_properties.extent.height as f32
                    ),
                };
                
                let ubos = [ubo];

                let buffer_mem = self.uniform_buffer_memories[current_image as usize];
                let size = size_of::<buffer::UniformBufferObject>() as vk::DeviceSize;
                unsafe {
                    let device = self.vk_context.device();
                    let data_ptr = device
                        .map_memory(buffer_mem, 0, size, vk::MemoryMapFlags::empty())
                        .unwrap();
                    let mut align = ash::util::Align::new(data_ptr, align_of::<Matrix4<f32>>() as _, size);
                    align.copy_from_slice(&ubos);
                    device.unmap_memory(buffer_mem);
                }
            }
            
            // Handle forward movement on the player (after camera operations are complete)
            if forward_movement > 0.0 {
                player.move_forward(forward_movement);
            }
        }
    }

    /// Creates game objects for the scene
    fn create_game_objects(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Vec<GameObject> {
        let mut game_objects = Vec::new();
        
        // Add player game object with camera (no render object)
        let camera = Camera::default();
        let player = GameObject::new_with_camera(Some(camera));
        game_objects.push(player);

        // Add plane game object (with render object, no camera)
        let plane_render_object = {
            let plane_model = Model::load("models/plane.obj");
            let (vertex_buffer, vertex_buffer_memory) = buffer::create_vertex_buffer(
                vk_context,
                command_pool,
                graphics_queue,
                &plane_model.vertices,
            );
            let (index_buffer, index_buffer_memory) = buffer::create_index_buffer(
                vk_context,
                command_pool,
                graphics_queue,
                &plane_model.indices,
            );
            let index_count = plane_model.indices.len();
            RenderObject {
                model: plane_model,
                vertex_buffer,
                vertex_buffer_memory,
                index_buffer,
                index_buffer_memory,
                index_count,
            }
        };
        let plane_go = GameObject::new_with_render_object(plane_render_object);
        game_objects.push(plane_go);

        // Add teapot game object (with render object, no camera)
        let teapot_render_object = {
            let teapot_model = Model::load("models/teapot.obj");
            let (vertex_buffer, vertex_buffer_memory) = buffer::create_vertex_buffer(
                vk_context,
                command_pool,
                graphics_queue,
                &teapot_model.vertices,
            );
            let (index_buffer, index_buffer_memory) = buffer::create_index_buffer(
                vk_context,
                command_pool,
                graphics_queue,
                &teapot_model.indices,
            );
            let index_count = teapot_model.indices.len();
            RenderObject {
                model: teapot_model,
                vertex_buffer,
                vertex_buffer_memory,
                index_buffer,
                index_buffer_memory,
                index_count,
            }
        };
        
        let teapot_go = GameObject::new_with_render_object(teapot_render_object);
        game_objects.push(teapot_go);
        
        game_objects
    }

    // Initialize the render loop to handle rendering
    pub fn init_render_loop(&mut self) {
        if self.render_loop.is_none() {
            self.render_loop = Some(RenderLoop::new(
                Arc::new(self.vk_context.clone()),
                (self.queue_families_indices.graphics_index, self.queue_families_indices.present_index),
                self.graphics_queue,
                self.present_queue,
                self.command_pool,
                self.msaa_samples,
                self.window_system.clone(),
                self.input_system.clone(),
            ));
        }
    }
}

impl Drop for VulkanApp {
    /// Clean up all Vulkan resources during application shutdown.
    /// 
    /// This method ensures proper resource cleanup by:
    /// 1. First cleaning up swapchain-dependent resources
    /// 2. Using null checks to prevent double-free issues
    /// 3. Checking if handles are valid (non-zero) before destroying them
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        self.wait_gpu_idle();
        
        // First clean up all resources that depend on the swapchain
        self.cleanup_swapchain();

        let device = self.vk_context.device();
        
        // Clean up game object resources with null checks
        unsafe {
            for game_object in &self.game_objects {
                if let Some(render_object) = &game_object.render_object {
                    // Check for null/zero handle before destroying
                    if render_object.vertex_buffer.as_raw() != 0 {
                        device.destroy_buffer(render_object.vertex_buffer, None);
                    }
                    
                    if render_object.vertex_buffer_memory.as_raw() != 0 {
                        device.free_memory(render_object.vertex_buffer_memory, None);
                    }
                    
                    if render_object.index_buffer.as_raw() != 0 {
                        device.destroy_buffer(render_object.index_buffer, None);
                    }
                    
                    if render_object.index_buffer_memory.as_raw() != 0 {
                        device.free_memory(render_object.index_buffer_memory, None);
                    }
                }
            }
        }
        
        // Clean up synchronization objects
        self.in_flight_frames.destroy(device);
        
        unsafe {
            // Clean up descriptor resources
            if self.descriptor_pool.as_raw() != 0 {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            
            if self.descriptor_set_layout.as_raw() != 0 {
                device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            
            // Clean up uniform buffers with null checks
            for buffer in &self.uniform_buffers {
                if buffer.as_raw() != 0 {
                    device.destroy_buffer(*buffer, None);
                }
            }
            
            for memory in &self.uniform_buffer_memories {
                if memory.as_raw() != 0 {
                    device.free_memory(*memory, None);
                }
            }
                
            // Clean up texture resources
            self.texture.destroy(device);
            
            // Clean up command pools
            if self.transient_command_pool.as_raw() != 0 {
                device.destroy_command_pool(self.transient_command_pool, None);
            }
            
            if self.command_pool.as_raw() != 0 {
                device.destroy_command_pool(self.command_pool, None);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct QueueFamiliesIndices {
    graphics_index: u32,
    present_index: u32,
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}
