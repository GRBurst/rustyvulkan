use ash::{
    vk,
    Device,
    Entry,
    Instance,
    ext::debug_utils,
    khr::{surface, swapchain as khr_swapchain},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;
use std::{ffi::CString, mem::{size_of, offset_of}};
use ash::vk::make_api_version;

use crate::gameobject::Vertex;
use super::{
    context::VkContext,
    swapchain::{SwapchainProperties, SwapchainSupportDetails},
};

const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

pub struct VulkanRenderer {
    context: VkContext,
    swapchain_support: SwapchainSupportDetails,
    swapchain_loader: khr_swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

// Add constants for synchronization
const MAX_FRAMES_IN_FLIGHT: usize = 2;

// Add vertex data
const VERTICES: &[Vertex] = &[
    Vertex { pos: [-0.5, -0.5, 0.0], color: [1.0, 0.0, 0.0], coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
    Vertex { pos: [0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0], coords: [1.0, 0.0], normal: [0.0, 0.0, 1.0] },
    Vertex { pos: [0.0, 0.5, 0.0], color: [0.0, 0.0, 1.0], coords: [0.5, 1.0], normal: [0.0, 0.0, 1.0] },
];

impl VulkanRenderer {
    pub fn new(window: &Window) -> Self {
        // Initialize Vulkan
        let entry = unsafe { Entry::load().expect("Failed to load Vulkan") };
        let instance = Self::create_instance(&entry, window);
        let context = Self::create_context(&entry, &instance, window);
        
        // Create swapchain
        let (swapchain_support, swapchain_khr, properties, images) = 
            Self::create_swapchain(&context, window.inner_size().width, window.inner_size().height);

        let swapchain_loader = khr_swapchain::Device::new(context.instance(), context.device());

        // Create image views
        let image_views = Self::create_image_views(&context, &images, properties);

        // Create render pass
        let render_pass = Self::create_render_pass(&context, properties);

        // Create graphics pipeline
        let (descriptor_set_layout, pipeline_layout, pipeline) = 
            Self::create_graphics_pipeline(&context, properties, render_pass);

        // Create framebuffers
        let framebuffers = Self::create_framebuffers(&context, &image_views, render_pass, properties);

        // Create command pool and buffers
        let command_pool = Self::create_command_pool(&context);
        let command_buffers = Self::create_command_buffers(&context, command_pool, &framebuffers);

        let mut renderer = Self {
            context,
            swapchain_support,
            swapchain_loader,
            swapchain_khr,
            swapchain_properties: properties,
            images,
            image_views,
            render_pass,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            vertex_buffer: vk::Buffer::null(),
            vertex_buffer_memory: vk::DeviceMemory::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            current_frame: 0,
        };

        // Create vertex buffer
        renderer.create_vertex_buffer();

        // Create descriptor pool and sets
        renderer.create_descriptor_pool();
        renderer.create_descriptor_sets();

        // Create synchronization objects
        renderer.create_sync_objects();

        renderer
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(make_api_version(0, 0, 1, 0))
            .api_version(make_api_version(0, 1, 0, 0));

        let extension_names = ash_window::enumerate_required_extensions(
            window.display_handle().unwrap().as_raw()
        ).unwrap();

        let mut extension_names = extension_names.to_vec();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        unsafe { entry.create_instance(&create_info, None).unwrap() }
    }

    fn create_context(entry: &Entry, instance: &Instance, window: &Window) -> VkContext {
        // Create debug messenger if validation layers are enabled
        let (debug_utils, debug_callback) = if ENABLE_VALIDATION_LAYERS {
            let debug_utils = debug_utils::Instance::new(entry, instance);
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_callback = unsafe {
                debug_utils
                    .create_debug_utils_messenger(&debug_info, None)
                    .expect("Failed to create debug callback")
            };
            (debug_utils, debug_callback)
        } else {
            (
                debug_utils::Instance::new(entry, instance),
                vk::DebugUtilsMessengerEXT::null(),
            )
        };

        // Create surface
        let surface = surface::Instance::new(entry, instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().into(),
                None,
            )
            .expect("Failed to create surface")
        };

        // Select physical device
        let physical_device = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
                .into_iter()
                .find(|&device| {
                    let properties = instance.get_physical_device_properties(device);
                    let features = instance.get_physical_device_features(device);
                    
                    // Check if device supports graphics operations
                    let queue_families = instance.get_physical_device_queue_family_properties(device);
                    let has_graphics = queue_families.iter().any(|family| {
                        family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    });

                    // Check surface support
                    let surface_support = queue_families
                        .iter()
                        .enumerate()
                        .any(|(index, _)| unsafe {
                            surface
                                .get_physical_device_surface_support(
                                    device,
                                    index as u32,
                                    surface_khr,
                                )
                                .unwrap_or(false)
                        });

                    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                        && features.geometry_shader == 1
                        && has_graphics
                        && surface_support
                })
                .expect("Failed to find suitable GPU")
        };

        // Create logical device
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .find(|(_index, family)| {
                    family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(index, _)| index as u32)
                .expect("Could not find graphics queue family")
        };

        let device_extensions = vec![khr_swapchain::NAME.as_ptr()];
        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        // Create VkContext using the existing new implementation
        VkContext::new(
            entry.clone(),
            instance.clone(),
            debug_utils,
            debug_callback,
            surface,
            surface_khr,
            physical_device,
            device,
        )
    }

    fn create_swapchain(
        context: &VkContext,
        width: u32,
        height: u32,
    ) -> (SwapchainSupportDetails, vk::SwapchainKHR, SwapchainProperties, Vec<vk::Image>) {
        // Get swapchain support details
        let swapchain_support = SwapchainSupportDetails::new(
            context.physical_device(),
            context.surface(),
            context.surface_khr(),
        );

        // Get ideal swapchain properties
        let properties = swapchain_support.get_ideal_swapchain_properties([width, height]);

        // Create swapchain
        let swapchain_loader = khr_swapchain::Device::new(context.instance(), context.device());
        
        // Choose image count
        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count > 0 {
            image_count = image_count.min(swapchain_support.capabilities.max_image_count);
        }

        // Create swapchain info
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(context.surface_khr())
            .min_image_count(image_count)
            .image_format(properties.format.format)
            .image_color_space(properties.format.color_space)
            .image_extent(properties.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(properties.present_mode)
            .clipped(true);

        // Create swapchain
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("Failed to create swapchain")
        };

        // Get swapchain images
        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        (swapchain_support, swapchain, properties, images)
    }

    fn create_image_views(
        context: &VkContext,
        images: &[vk::Image],
        properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(properties.format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe {
                    context
                        .device()
                        .create_image_view(&create_info, None)
                        .expect("Failed to create image view")
                }
            })
            .collect()
    }

    fn create_render_pass(
        context: &VkContext,
        properties: SwapchainProperties,
    ) -> vk::RenderPass {
        // Create color attachment
        let color_attachment = vk::AttachmentDescription::default()
            .format(properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        // Create color attachment reference
        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        // Create subpass
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        // Create subpass dependency
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        // Create render pass
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe {
            context
                .device()
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_graphics_pipeline(
        context: &VkContext,
        properties: SwapchainProperties,
        render_pass: vk::RenderPass,
    ) -> (vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline) {
        // Read shader files
        let vert_shader_code = std::fs::read("assets/shaders/shader.vert.spv")
            .expect("Failed to read vertex shader");
        let frag_shader_code = std::fs::read("assets/shaders/shader.frag.spv")
            .expect("Failed to read fragment shader");

        // Create shader modules
        let vert_shader_module = Self::create_shader_module(context, &vert_shader_code);
        let frag_shader_module = Self::create_shader_module(context, &frag_shader_code);

        // Create shader stages
        let entry_point_name = unsafe { CString::from_vec_unchecked(b"main".to_vec()) };
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(entry_point_name.as_c_str()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(entry_point_name.as_c_str()),
        ];

        // Vertex input state
        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let binding_descriptions = [binding_description];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        // Input assembly state
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport and scissor state
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(properties.extent.width as f32)
            .height(properties.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(properties.extent);

        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        // Rasterization state
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        // Multisampling state
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Color blend state
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blend_attachments = [color_blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        // Pipeline layout
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_bindings = [ubo_layout_binding, sampler_layout_binding];
        let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&layout_bindings);

        let descriptor_set_layout = unsafe {
            context.device().create_descriptor_set_layout(&descriptor_layout_info, None)
                .expect("Failed to create descriptor set layout")
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts);

        let pipeline_layout = unsafe {
            context.device().create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        // Create graphics pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            context.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
            .expect("Failed to create graphics pipeline")[0]
        };

        // Cleanup shader modules
        unsafe {
            context.device().destroy_shader_module(vert_shader_module, None);
            context.device().destroy_shader_module(frag_shader_module, None);
        }

        (descriptor_set_layout, pipeline_layout, pipeline)
    }

    fn create_shader_module(context: &VkContext, code: &[u8]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(unsafe { std::slice::from_raw_parts(
                code.as_ptr() as *const u32,
                code.len() / 4,
            )});

        unsafe {
            context.device().create_shader_module(&create_info, None)
                .expect("Failed to create shader module")
        }
    }

    fn create_framebuffers(
        context: &VkContext,
        image_views: &[vk::ImageView],
        render_pass: vk::RenderPass,
        properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        image_views
            .iter()
            .map(|&image_view| {
                let attachments = [image_view];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(properties.extent.width)
                    .height(properties.extent.height)
                    .layers(1);

                unsafe {
                    context.device().create_framebuffer(&create_info, None)
                        .expect("Failed to create framebuffer")
                }
            })
            .collect()
    }

    fn create_command_pool(context: &VkContext) -> vk::CommandPool {
        let queue_family_indices = unsafe {
            context.instance()
                .get_physical_device_queue_family_properties(context.physical_device())
                .iter()
                .enumerate()
                .find(|(_, properties)| {
                    properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(index, _)| index as u32)
                .expect("Failed to find graphics queue family")
        };

        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices);

        unsafe {
            context.device().create_command_pool(&create_info, None)
                .expect("Failed to create command pool")
        }
    }

    fn create_command_buffers(
        context: &VkContext,
        command_pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);

        unsafe {
            context.device().allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffers")
        }
    }

    pub fn draw_frame(&mut self) -> bool {
        // Wait for the previous frame
        unsafe {
            self.context.device()
                .wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)
                .expect("Failed to wait for fence");
        }

        // Get next image from swapchain
        let (image_index, is_suboptimal) = unsafe {
            match self.swapchain_loader.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok((index, suboptimal)) => (index, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return true,
                Err(e) => panic!("Failed to acquire next image: {}", e),
            }
        };

        // Reset the fence for this frame
        unsafe {
            self.context.device()
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
                .expect("Failed to reset fence");
        }

        // Begin command buffer recording
        let command_buffer = self.command_buffers[image_index as usize];
        let begin_info = vk::CommandBufferBeginInfo::default();
        
        unsafe {
            self.context.device()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin recording command buffer");
        }

        // Begin render pass
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_properties.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            let device = self.context.device();
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            // Bind the graphics pipeline
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            // Bind vertex buffer
            let vertex_buffers = [self.vertex_buffer];
            let offsets = [0];
            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

            // Draw call
            device.cmd_draw(command_buffer, VERTICES.len() as u32, 1, 0, 0);

            // End render pass
            device.cmd_end_render_pass(command_buffer);

            // End command buffer recording
            device.end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");

            // Submit command buffer
            let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
            let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(&command_buffer))
                .signal_semaphores(&signal_semaphores);

            device.queue_submit(
                device.get_device_queue(0, 0),
                &[submit_info],
                self.in_flight_fences[self.current_frame],
            ).expect("Failed to submit draw command buffer");

            // Present the image
            let swapchains = [self.swapchain_khr];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self.swapchain_loader.queue_present(
                device.get_device_queue(0, 0),
                &present_info,
            ) {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return true,
                Err(e) => panic!("Failed to present queue: {}", e),
            }
        }

        // Update current frame
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        is_suboptimal
    }

    pub fn wait_device_idle(&self) {
        unsafe {
            self.context.device().device_wait_idle().unwrap();
        }
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        // Wait for the device to be idle before recreating
        unsafe { self.context.device().device_wait_idle().unwrap() };

        // Clean up old resources
        unsafe {
            let device = self.context.device();
            
            for framebuffer in &self.framebuffers {
                device.destroy_framebuffer(*framebuffer, None);
            }
            
            for view in &self.image_views {
                device.destroy_image_view(*view, None);
            }
            
            self.swapchain_loader.destroy_swapchain(self.swapchain_khr, None);
        }

        // Recreate swapchain and dependent resources
        let (swapchain_support, swapchain_khr, properties, images) = 
            Self::create_swapchain(&self.context, width, height);
        
        let image_views = Self::create_image_views(&self.context, &images, properties);
        let framebuffers = Self::create_framebuffers(
            &self.context, 
            &image_views, 
            self.render_pass,
            properties
        );

        // Update struct fields
        self.swapchain_support = swapchain_support;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.images = images;
        self.image_views = image_views;
        self.framebuffers = framebuffers;
    }

    fn find_memory_type(
        context: &VkContext,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties = context.get_mem_properties();
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type");
    }

    fn create_buffer(
        context: &VkContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            context
                .device()
                .create_buffer(&buffer_info, None)
                .expect("Failed to create buffer")
        };

        let mem_requirements = unsafe {
            context.device().get_buffer_memory_requirements(buffer)
        };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(Self::find_memory_type(
                context,
                mem_requirements.memory_type_bits,
                properties,
            ));

        let buffer_memory = unsafe {
            context
                .device()
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory")
        };

        unsafe {
            context
                .device()
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind buffer memory");
        }

        (buffer, buffer_memory)
    }

    fn create_vertex_buffer(&mut self) {
        let buffer_size = (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64;

        // Create staging buffer
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            &self.context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        // Copy vertex data to staging buffer
        unsafe {
            let data_ptr = self
                .context
                .device()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory") as *mut Vertex;
            data_ptr.copy_from_nonoverlapping(VERTICES.as_ptr(), VERTICES.len());
            self.context.device().unmap_memory(staging_buffer_memory);
        }

        // Create vertex buffer
        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            &self.context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // Copy data from staging buffer to vertex buffer
        self.copy_buffer(staging_buffer, vertex_buffer, buffer_size);

        // Cleanup staging buffer
        unsafe {
            let device = self.context.device();
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        self.vertex_buffer = vertex_buffer;
        self.vertex_buffer_memory = vertex_buffer_memory;
    }

    fn copy_buffer(&self, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize) {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            self.context
                .device()
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate command buffer")[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            let device = self.context.device();
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer");

            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);

            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]);

            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&command_buffer));

            device
                .queue_submit(device.get_device_queue(0, 0), &[submit_info], vk::Fence::null())
                .expect("Failed to submit copy command buffer");
            device.queue_wait_idle(device.get_device_queue(0, 0))
                .expect("Failed to wait for queue idle");

            device.free_command_buffers(self.command_pool, &[command_buffer]);
        }
    }

    fn create_descriptor_pool(&mut self) {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        self.descriptor_pool = unsafe {
            self.context
                .device()
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };
    }

    fn create_descriptor_sets(&mut self) {
        let layouts = vec![self.descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        self.descriptor_sets = unsafe {
            self.context
                .device()
                .allocate_descriptor_sets(&alloc_info)
                .expect("Failed to allocate descriptor sets")
        };
    }

    fn create_sync_objects(&mut self) {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let device = self.context.device();
                
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_info, None)
                    .expect("Failed to create image available semaphore");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_info, None)
                    .expect("Failed to create render finished semaphore");
                let in_flight_fence = device
                    .create_fence(&fence_info, None)
                    .expect("Failed to create in-flight fence");

                self.image_available_semaphores.push(image_available_semaphore);
                self.render_finished_semaphores.push(render_finished_semaphore);
                self.in_flight_fences.push(in_flight_fence);
            }
        }
    }

    fn cleanup_sync_objects(&mut self) {
        unsafe {
            let device = self.context.device();
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                device.destroy_semaphore(self.image_available_semaphores[i], None);
                device.destroy_semaphore(self.render_finished_semaphores[i], None);
                device.destroy_fence(self.in_flight_fences[i], None);
            }
        }
    }

    fn cleanup_vertex_buffer(&mut self) {
        unsafe {
            let device = self.context.device();
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);
        }
    }

    fn cleanup_descriptor_pool(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            // Wait for the device to finish all operations
            self.context.device().device_wait_idle().unwrap();

            // Clean up synchronization objects
            self.cleanup_sync_objects();

            // Clean up vertex buffer
            self.cleanup_vertex_buffer();

            // Clean up descriptor pool
            self.cleanup_descriptor_pool();

            // Clean up framebuffers
            for framebuffer in &self.framebuffers {
                self.context.device().destroy_framebuffer(*framebuffer, None);
            }

            // Clean up graphics pipeline
            self.context.device().destroy_pipeline(self.pipeline, None);
            self.context.device().destroy_pipeline_layout(self.pipeline_layout, None);

            // Clean up render pass
            self.context.device().destroy_render_pass(self.render_pass, None);

            // Clean up swapchain image views
            for image_view in &self.image_views {
                self.context.device().destroy_image_view(*image_view, None);
            }

            // Clean up swapchain
            self.swapchain_loader.destroy_swapchain(self.swapchain_khr, None);

            // Clean up command pool
            self.context.device().destroy_command_pool(self.command_pool, None);
        }
    }
}

extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message = unsafe { std::ffi::CStr::from_ptr(callback_data.p_message) };

    println!(
        "Validation Layer: {:?} {:?} {:?}",
        message_severity, message_type, message
    );

    vk::FALSE
} 