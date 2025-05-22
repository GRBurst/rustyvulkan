// Render loop management for Vulkan applications
// Handles frame rendering, synchronization, and swapchain recreation

use std::mem::{size_of, align_of};
use std::sync::Arc;
use std::collections::HashSet;

use ash::{vk, Device};
use ash::vk::Handle;
use ash::khr::swapchain;
use cgmath::{Deg, Matrix4, Vector3};

use crate::{
    platform::input::InputSystem,
    platform::window::WindowSystem,
    renderer::{
        buffer,
        context::VkContext,
        pipeline,
        SwapchainProperties,
        SwapchainSupportDetails
    },
    resources::texture,
    scene::GameObject,
    scene::camera::Camera,
};

use winit::keyboard::KeyCode;

/// Constants for frame synchronization
pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

/// Synchronization objects for a single frame
#[derive(Copy, Clone)]
pub struct SyncObjects {
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub fence: vk::Fence,
}

impl SyncObjects {
    /// Destroys the synchronization objects
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

/// Manages frame synchronization objects
pub struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    /// Creates a new InFlightFrames instance
    pub fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    /// Destroys all synchronization objects
    pub fn destroy(&self, device: &Device) {
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

/// Creates synchronization objects for frame rendering
pub fn create_sync_objects(device: &Device) -> InFlightFrames {
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

/// Main render loop manager
pub struct RenderLoop {
    // Core Vulkan context
    vk_context: Arc<VkContext>,
    
    // Queue indices and handles
    queue_families_indices: (u32, u32), // (graphics_index, present_index)
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    
    // Swapchain and related resources
    swapchain: swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    
    // Rendering resources
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    
    // MSAA resources
    msaa_samples: vk::SampleCountFlags,
    color_texture: texture::Texture,
    depth_format: vk::Format,
    depth_texture: texture::Texture,
    
    // Command resources
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    
    // Descriptor resources
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool, 
    descriptor_sets: Vec<vk::DescriptorSet>,
    
    // Uniform buffer resources
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_memories: Vec<vk::DeviceMemory>,
    
    // Game resources
    game_objects: Vec<GameObject>,
    texture: texture::Texture,
    
    // Frame synchronization
    in_flight_frames: InFlightFrames,
    
    // External systems - using references instead of Arc
    window_system: WindowSystem,
    input_system: InputSystem,
}

impl RenderLoop {
    /// Creates a new render loop manager that handles the frame rendering pipeline
    ///
    /// This method initializes all rendering-related resources including the swapchain,
    /// render pass, graphics pipeline, framebuffers, and synchronization objects.
    /// It follows the same initialization sequence as the original VulkanApp,
    /// ensuring compatibility with the existing code.
    pub fn new(
        vk_context: Arc<VkContext>,
        queue_families_indices: (u32, u32), // (graphics_index, present_index)
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        command_pool: vk::CommandPool,
        msaa_samples: vk::SampleCountFlags,
        window_system: WindowSystem,
        input_system: InputSystem,
    ) -> Self {
        let dimensions = window_system.get_dimensions();
        
        // Create swapchain and related resources
        let (swapchain, swapchain_khr, properties, images) = 
            Self::create_swapchain(&vk_context, queue_families_indices, dimensions);
            
        let swapchain_image_views = 
            Self::create_swapchain_image_views(vk_context.device(), &images, properties);
            
        // Create render pass
        let depth_format = texture::find_depth_format(&vk_context);
        let render_pass = Self::create_render_pass(
            vk_context.device(),
            properties,
            msaa_samples,
            depth_format
        );
        
        // Create descriptor set layout
        let descriptor_set_layout = Self::create_descriptor_set_layout(vk_context.device());
        
        // Create pipeline
        let (pipeline, pipeline_layout) = pipeline::create_pipeline(
            vk_context.device(),
            properties,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
        );
        
        // Create MSAA color and depth textures
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
        
        // Load texture
        let texture = texture::Texture::load_from_file(
            &vk_context,
            command_pool,
            graphics_queue,
            "images/chalet.jpg",
        );
        
        // Create game objects
        let game_objects = Self::create_game_objects(&vk_context, command_pool, graphics_queue);
        
        // Create uniform buffers
        let (uniform_buffers, uniform_buffer_memories) =
            buffer::create_uniform_buffers(&vk_context, images.len());
            
        // Create descriptor pool and sets
        let descriptor_pool = Self::create_descriptor_pool(vk_context.device(), images.len() as u32);
        
        let descriptor_sets = Self::create_descriptor_sets(
            vk_context.device(),
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            texture.clone(),
        );
        
        // Create framebuffers
        let framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain_image_views,
            color_texture.clone(),
            depth_texture.clone(),
            render_pass,
            properties,
        );
        
        // Create command buffers
        let command_buffers = Self::create_command_buffers(
            vk_context.device(), 
            command_pool,
            &framebuffers,
            render_pass,
            properties,
            game_objects.as_slice(),
            pipeline_layout,
            &descriptor_sets,
            pipeline,
        );
        
        // Create synchronization objects
        let in_flight_frames = create_sync_objects(vk_context.device());
        
        Self {
            vk_context,
            queue_families_indices,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_khr,
            swapchain_properties: properties,
            images,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            msaa_samples,
            color_texture,
            depth_format,
            depth_texture,
            command_pool,
            command_buffers,
            descriptor_set_layout,
            descriptor_pool, 
            descriptor_sets,
            uniform_buffers,
            uniform_buffer_memories,
            game_objects,
            texture,
            in_flight_frames,
            window_system,
            input_system,
        }
    }
    
    /// Waits for the GPU to finish execution of all commands
    pub fn wait_gpu_idle(&self) {
        unsafe {
            self.vk_context
                .device()
                .device_wait_idle()
                .expect("Failed to wait for device idle!");
        }
    }
    
    /// Renders a new frame
    ///
    /// Returns true if swapchain needs to be recreated
    pub fn draw_frame(&mut self) -> bool {
        // Get current sync objects
        let sync_objects = match self.in_flight_frames.next() {
            Some(objects) => objects,
            None => return false,
        };
        
        let device = self.vk_context.device();
        
        // Wait for fence
        unsafe {
            let fences = [sync_objects.fence];
            device
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("Failed to wait for fence!");
            device
                .reset_fences(&fences)
                .expect("Failed to reset fence!");
        }
        
        // Acquire next image
        let result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                sync_objects.image_available_semaphore,
                vk::Fence::null(),
            )
        };
        
        // Handle swapchain recreation if needed
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return true,
            Err(error) => panic!("Failed to acquire next image: {:?}", error),
        };
        
        // Update uniform buffers
        self.update_uniform_buffers(image_index as u32);
        
        // Submit command buffer
        let command_buffers = [self.command_buffers[image_index as usize]];
        let wait_semaphores = [sync_objects.image_available_semaphore];
        let signal_semaphores = [sync_objects.render_finished_semaphore];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
            
        unsafe {
            device
                .queue_submit(self.graphics_queue, &[submit_info], sync_objects.fence)
                .expect("Failed to submit draw command buffer!");
        }
        
        // Present
        let swapchains = [self.swapchain_khr];
        let image_indices = [image_index];
        
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
            
        let result = unsafe {
            self.swapchain
                .queue_present(self.present_queue, &present_info)
        };
        
        // Check if swapchain recreation is needed
        match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(error) => panic!("Failed to present queue: {:?}", error),
            _ => false,
        }
    }
    
    /// Recreates the swapchain and all dependent resources
    pub fn recreate_swapchain(&mut self) {
        // Wait for the device to be idle before recreating resources
        self.wait_gpu_idle();
        
        // Clean up old swapchain resources
        self.cleanup_swapchain();
        
        // Get new dimensions from window system
        let dimensions = self.window_system.take_resize_dimensions()
            .unwrap_or_else(|| self.window_system.get_dimensions());
            
        // Create new swapchain
        let (swapchain, swapchain_khr, properties, images) = Self::create_swapchain(
            &self.vk_context, 
            self.queue_families_indices,
            dimensions,
        );
        
        // Create new swapchain image views
        let swapchain_image_views = Self::create_swapchain_image_views(
            self.vk_context.device(),
            &images,
            properties,
        );
        
        // Create new render pass
        let render_pass = Self::create_render_pass(
            self.vk_context.device(),
            properties,
            self.msaa_samples,
            self.depth_format,
        );
        
        // Create new graphics pipeline
        let (pipeline, layout) = pipeline::create_pipeline(
            self.vk_context.device(),
            properties,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
        );
        
        // Create new MSAA resources
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
        // Create new uniform buffers sized appropriately for the new swapchain
        let (uniform_buffers, uniform_buffer_memories) =
            buffer::create_uniform_buffers(&self.vk_context, images.len());
        
        // Reset descriptor pool and recreate descriptor sets
        unsafe {
            self.vk_context.device().reset_descriptor_pool(
                self.descriptor_pool,
                vk::DescriptorPoolResetFlags::empty(),
            ).expect("Failed to reset descriptor pool!");
        }
        
        // Create new descriptor sets
        let descriptor_sets = Self::create_descriptor_sets(
            self.vk_context.device(),
            self.descriptor_pool,
            self.descriptor_set_layout,
            &uniform_buffers,
            self.texture.clone(),
        );
        
        // Create new framebuffers
        let framebuffers = Self::create_framebuffers(
            self.vk_context.device(),
            &swapchain_image_views,
            color_texture.clone(),
            depth_texture.clone(),
            render_pass,
            properties,
        );
        
        // Create new command buffers
        let command_buffers = Self::create_command_buffers(
            self.vk_context.device(),
            self.command_pool,
            &framebuffers,
            render_pass,
            properties,
            &self.game_objects,
            layout,
            &descriptor_sets,
            pipeline,
        );
        
        // Update object fields with new resources
        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.images = images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.pipeline_layout = layout;
        self.pipeline = pipeline;
        self.framebuffers = framebuffers;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.uniform_buffers = uniform_buffers;
        self.uniform_buffer_memories = uniform_buffer_memories;
        self.descriptor_sets = descriptor_sets;
        self.command_buffers = command_buffers;
    }
    
    /// Updates the uniform buffers for the current frame
    fn update_uniform_buffers(&mut self, current_image: u32) {
        use crate::scene::camera::Camera;
        
        // Find the player (with camera) from game objects
        let player = self.game_objects.iter_mut().find(|obj| obj.camera.is_some());
        
        if let Some(game_object) = player {
            if let Some(camera) = &mut game_object.camera {
                // Process input for camera movement
                self.input_system.apply_camera_movement(
                    |direction| {
                        camera.move_camera(direction);
                    },
                    0.1,
                );
                
                // Update camera position based on player movement
                let view_direction = camera.get_view_direction();
                let right = camera.get_right();
                let up = camera.get_up();
                
                // Log player and camera positions for debugging
                log::info!(
                    "Player Position: ({:.2}, {:.2}, {:.2})",
                    game_object.transform.position.x,
                    game_object.transform.position.y,
                    game_object.transform.position.z
                );
                
                log::info!(
                    "Camera Position: ({:.2}, {:.2}, {:.2})",
                    camera.transform.position.x,
                    camera.transform.position.y,
                    camera.transform.position.z
                );
                
                // Create view and projection matrices
                let view = camera.look_to(view_direction);
                let aspect = self.swapchain_properties.extent.width as f32
                    / self.swapchain_properties.extent.height as f32;
                let proj = camera.get_projection_matrix(aspect);
                
                // Create and update the uniform buffer object
                let ubo = buffer::UniformBufferObject {
                    model: Matrix4::from_angle_y(Deg(0.0)),
                    view,
                    proj,
                };
                
                let device = self.vk_context.device();
                let memory_size = std::mem::size_of::<buffer::UniformBufferObject>() as u64;
                
                unsafe {
                    // Map memory
                    let data_ptr = device
                        .map_memory(
                            self.uniform_buffer_memories[current_image as usize],
                            0,
                            memory_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to map memory!");
                    
                    // Copy data
                    std::ptr::copy_nonoverlapping(
                        &ubo as *const buffer::UniformBufferObject as *const u8,
                        data_ptr as *mut u8,
                        memory_size as usize,
                    );
                    
                    // Unmap memory
                    device.unmap_memory(self.uniform_buffer_memories[current_image as usize]);
                }
            }
        }
    }
    
    /// Cleans up swapchain and all dependent resources
    fn cleanup_swapchain(&mut self) {
        let device = self.vk_context.device();
        
        unsafe {
            // Clean up depth and color resources
            self.depth_texture.destroy(device);
            self.color_texture.destroy(device);
            
            // Clean up framebuffers
            for framebuffer in &self.framebuffers {
                device.destroy_framebuffer(*framebuffer, None);
            }
            
            // Clean up command buffers
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            
            // Clean up pipeline resources
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            
            // Clean up swapchain image views
            for image_view in &self.swapchain_image_views {
                device.destroy_image_view(*image_view, None);
            }
            
            // Clean up swapchain
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            
            // Clean up uniform buffers
            if !self.uniform_buffers.is_empty() {
                for memory in &self.uniform_buffer_memories {
                    device.free_memory(*memory, None);
                }
                
                for buffer in &self.uniform_buffers {
                    device.destroy_buffer(*buffer, None);
                }
            }
            
            // We don't destroy descriptor sets here - they're cleaned up when the pool is reset
        }
    }

    // Helper methods for swapchain recreation
    
    fn create_swapchain(
        vk_context: &VkContext,
        queue_families_indices: (u32, u32),
        dimensions: [u32; 2],
    ) -> (
        swapchain::Device,
        vk::SwapchainKHR,
        SwapchainProperties,
        Vec<vk::Image>,
    ) {
        let device = vk_context.device();
        let physical_device = vk_context.physical_device();
        let surface = vk_context.surface();
        let surface_khr = vk_context.surface_khr();
        
        // Get swapchain support details
        let support_details = unsafe {
            let capabilities = surface
                .get_physical_device_surface_capabilities(physical_device, surface_khr)
                .expect("Failed to get surface capabilities");
                
            let formats = surface
                .get_physical_device_surface_formats(physical_device, surface_khr)
                .expect("Failed to get surface formats");
                
            let present_modes = surface
                .get_physical_device_surface_present_modes(physical_device, surface_khr)
                .expect("Failed to get surface present modes");
                
            SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        };
        
        // Choose surface format
        let surface_format = unsafe {
            support_details
                .formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::B8G8R8A8_SRGB
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(&support_details.formats[0])
                .clone()
        };
        
        // Choose present mode
        let present_mode = unsafe {
            if support_details
                .present_modes
                .contains(&vk::PresentModeKHR::MAILBOX)
            {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            }
        };
        
        // Choose extent
        let extent = unsafe {
            if support_details.capabilities.current_extent.width != u32::MAX {
                support_details.capabilities.current_extent
            } else {
                let min = support_details.capabilities.min_image_extent;
                let max = support_details.capabilities.max_image_extent;
                
                vk::Extent2D {
                    width: dimensions[0].clamp(min.width, max.width),
                    height: dimensions[1].clamp(min.height, max.height),
                }
            }
        };
        
        // Choose image count
        let mut image_count = support_details.capabilities.min_image_count + 1;
        if support_details.capabilities.max_image_count > 0
            && image_count > support_details.capabilities.max_image_count
        {
            image_count = support_details.capabilities.max_image_count;
        }
        
        // Create swapchain properties
        let properties = SwapchainProperties {
            format: surface_format.format,
            extent,
        };
        
        // Set up queue family indices
        let (graphics_index, present_index) = queue_families_indices;
        let queue_family_indices = [graphics_index, present_index];
        let is_same_queue = graphics_index == present_index;
        
        // Create swapchain
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_khr)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(if is_same_queue {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            })
            .queue_family_indices(if is_same_queue {
                &[] // Empty slice for exclusive mode
            } else {
                &queue_family_indices // Both indices for concurrent mode
            })
            .pre_transform(support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);
        
        let swapchain_ext = swapchain::Device::new(vk_context.instance(), device);
        
        let swapchain_khr = unsafe {
            swapchain_ext
                .create_swapchain(&create_info, None)
                .expect("Failed to create swapchain!")
        };
        
        let images = unsafe {
            swapchain_ext
                .get_swapchain_images(swapchain_khr)
                .expect("Failed to get swapchain images!")
        };
        
        (swapchain_ext, swapchain_khr, properties, images)
    }
    
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());
        
        for &image in swapchain_images {
            let image_view = Self::create_image_view(
                device,
                image,
                1,
                swapchain_properties.format,
                vk::ImageAspectFlags::COLOR,
            );
            swapchain_image_views.push(image_view);
        }
        
        swapchain_image_views
    }
    
    /// Creates an image view for the specified image
    fn create_image_view(
        device: &Device,
        image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        let components = vk::ComponentMapping::default()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);
        
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(components)
            .subresource_range(subresource_range);
        
        unsafe {
            device
                .create_image_view(&create_info, None)
                .expect("Failed to create image view!")
        }
    }
    
    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        // Color attachment
        let color_attachment = vk::AttachmentDescription::default()
            .format(swapchain_properties.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            
        // Color attachment reference
        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            
        // Depth attachment
        let depth_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            
        // Depth attachment reference
        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            
        // Resolve attachment (for MSAA)
        let resolve_attachment = vk::AttachmentDescription::default()
            .format(swapchain_properties.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
            
        // Resolve attachment reference
        let resolve_attachment_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            
        // Subpass
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_attachment_ref])
            .resolve_attachments(&[resolve_attachment_ref])
            .depth_stencil_attachment(&depth_attachment_ref);
            
        // Subpass dependency
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
            
        // Create render pass
        let attachments = [color_attachment, depth_attachment, resolve_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency];
        
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
            
        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create render pass!")
        }
    }
    
    fn create_pipeline(
        &self,
        properties: SwapchainProperties,
        render_pass: vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        // To be implemented
        todo!("Implement create_pipeline")
    }
    
    fn create_color_texture(
        &self,
        _properties: SwapchainProperties,
    ) -> texture::Texture {
        // To be implemented
        todo!("Implement create_color_texture")
    }
    
    fn create_depth_texture(
        &self,
        _properties: SwapchainProperties,
    ) -> texture::Texture {
        // To be implemented
        todo!("Implement create_depth_texture")
    }
    
    fn create_descriptor_sets(
        &self,
        _uniform_buffers: &[vk::Buffer],
    ) -> Vec<vk::DescriptorSet> {
        // To be implemented
        todo!("Implement create_descriptor_sets")
    }
    
    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        color_texture: texture::Texture,
        depth_texture: texture::Texture,
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = Vec::with_capacity(image_views.len());
        
        for &image_view in image_views {
            let attachments = [
                color_texture.view, // Color attachment (MSAA)
                depth_texture.view, // Depth attachment
                image_view,         // Resolve attachment
            ];
            
            let framebuffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_properties.extent.width)
                .height(swapchain_properties.extent.height)
                .layers(1);
                
            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create framebuffer!")
            };
            
            framebuffers.push(framebuffer);
        }
        
        framebuffers
    }
    
    fn create_command_buffers(
        device: &Device, 
        command_pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        properties: SwapchainProperties,
        game_objects: &[GameObject],
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);
            
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate command buffers!")
        };
        
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let framebuffer = framebuffers[i];
            
            // Begin command buffer
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::empty());
                
            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin command buffer!");
            }
            
            // Begin render pass
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
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ];
            
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: properties.extent,
                })
                .clear_values(&clear_values);
                
            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                
                // Bind pipeline
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );
                
                // Record draw commands for each game object with a render object
                for game_object in game_objects {
                    if let Some(render_object) = &game_object.render_object {
                        // Bind vertex and index buffers
                        let vertex_buffers = [render_object.vertex_buffer];
                        let offsets = [0];
                        
                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &vertex_buffers,
                            &offsets,
                        );
                        
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            render_object.index_buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                        
                        // Bind descriptor sets
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout,
                            0,
                            &descriptor_sets[i..=i],
                            &[],
                        );
                        
                        // Draw indexed
                        device.cmd_draw_indexed(
                            command_buffer,
                            render_object.index_count as u32,
                            1,
                            0,
                            0,
                            0,
                        );
                    }
                }
                
                // End render pass
                device.cmd_end_render_pass(command_buffer);
                
                // End command buffer
                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to end command buffer!");
            }
        }
        
        command_buffers
    }

    // Create helper methods that are used during initialization
    
    /// Creates a descriptor set layout for binding uniform buffers and textures
    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let ubo_binding = buffer::UniformBufferObject::get_descriptor_set_layout_binding();
        
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
            
        let bindings = [ubo_binding, sampler_binding];
        
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
            
        unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create descriptor set layout!")
        }
    }
    
    /// Creates a descriptor pool for allocating descriptor sets
    fn create_descriptor_pool(device: &Device, size: u32) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(size);
            
        let sampler_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(size);
            
        let pool_sizes = [ubo_pool_size, sampler_pool_size];
        
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(size);
            
        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool!")
        }
    }
    
    /// Creates descriptor sets for binding resources to the shaders
    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        texture: texture::Texture,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = vec![layout; uniform_buffers.len()];
        
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
            
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };
        
        for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(size_of::<buffer::UniformBufferObject>() as u64);
                
            let buffer_infos = [buffer_info];
            
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(texture.sampler.unwrap());
                
            let image_infos = [image_info];
            
            let buffer_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos);
                
            let image_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos);
                
            unsafe {
                device.update_descriptor_sets(&[buffer_write, image_write], &[]);
            }
        }
        
        descriptor_sets
    }
    
    /// Creates game objects for the scene
    fn create_game_objects(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Vec<GameObject> {
        use crate::scene::{GameObject, Model, RenderObject, camera::Camera};
        
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
}

impl Drop for RenderLoop {
    fn drop(&mut self) {
        log::debug!("Dropping render loop.");
        self.wait_gpu_idle();
        
        // First clean up all resources that depend on the swapchain
        self.cleanup_swapchain();

        let device = self.vk_context.device();
        
        // Clean up game object resources with validation
        // We need to track which resources have already been cleaned up to avoid double-free issues
        let mut cleaned_buffers = std::collections::HashSet::new();
        let mut cleaned_memories = std::collections::HashSet::new();
        
        unsafe {
            for game_object in &self.game_objects {
                if let Some(render_object) = &game_object.render_object {
                    // Only clean up resources that haven't been cleaned up already
                    // Use buffer_to_usize and memory_to_usize to get identifiers for the HashSet
                    let vertex_buffer_handle = buffer_to_usize(render_object.vertex_buffer);
                    let vertex_memory_handle = memory_to_usize(render_object.vertex_buffer_memory);
                    let index_buffer_handle = buffer_to_usize(render_object.index_buffer);
                    let index_memory_handle = memory_to_usize(render_object.index_buffer_memory);
                    
                    if !cleaned_buffers.contains(&vertex_buffer_handle) {
                        device.destroy_buffer(render_object.vertex_buffer, None);
                        cleaned_buffers.insert(vertex_buffer_handle);
                    }
                    
                    if !cleaned_memories.contains(&vertex_memory_handle) {
                        device.free_memory(render_object.vertex_buffer_memory, None);
                        cleaned_memories.insert(vertex_memory_handle);
                    }
                    
                    if !cleaned_buffers.contains(&index_buffer_handle) {
                        device.destroy_buffer(render_object.index_buffer, None);
                        cleaned_buffers.insert(index_buffer_handle);
                    }
                    
                    if !cleaned_memories.contains(&index_memory_handle) {
                        device.free_memory(render_object.index_buffer_memory, None);
                        cleaned_memories.insert(index_memory_handle);
                    }
                }
            }
        }
        
        // Clean up synchronization objects
        self.in_flight_frames.destroy(device);
        
        unsafe {
            // Clean up descriptor resources
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            
            // Clean up uniform buffers with validation
            for memory in &self.uniform_buffer_memories {
                let memory_handle = memory_to_usize(*memory);
                if !cleaned_memories.contains(&memory_handle) {
                    device.free_memory(*memory, None);
                }
            }
            
            for buffer in &self.uniform_buffers {
                let buffer_handle = buffer_to_usize(*buffer);
                if !cleaned_buffers.contains(&buffer_handle) {
                    device.destroy_buffer(*buffer, None);
                }
            }
                
            // Clean up texture resources
            self.texture.destroy(device);
            
            // Clean up command pool
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

// Helper functions to convert Vulkan handles to usize for HashSet usage
fn buffer_to_usize(buffer: vk::Buffer) -> usize {
    // A buffer handle is just an integer ID or pointer, so we can convert it to usize
    buffer.as_raw() as usize
}

fn memory_to_usize(memory: vk::DeviceMemory) -> usize {
    // A device memory handle is just an integer ID or pointer, so we can convert it to usize
    memory.as_raw() as usize
} 