use ash::{
    vk,
    Device,
    Entry,
    Instance,
    ext::debug_utils,
    khr::{surface, swapchain as khr_swapchain},
};
use raw_window_handle::HasDisplayHandle;
use winit::window::Window;
use std::ffi::CString;
use ash::vk::make_api_version;

use super::{
    context::VkContext,
    swapchain::{SwapchainProperties, SwapchainSupportDetails},
};

const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

pub struct VulkanRenderer {
    context: VkContext,
    swapchain_loader: SwapchainSupportDetails,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl VulkanRenderer {
    pub fn new(window: &Window) -> Self {
        // Initialize Vulkan
        let entry = unsafe { Entry::load().expect("Failed to load Vulkan") };
        let instance = Self::create_instance(&entry, window);
        let context = Self::create_context(&entry, &instance, window);
        
        // Create swapchain
        let (swapchain, swapchain_khr, properties, images) = 
            Self::create_swapchain(&context, window.inner_size().width, window.inner_size().height);
        
        // Create image views
        let image_views = Self::create_image_views(&context, &images, properties);

        // Create render pass
        let render_pass = Self::create_render_pass(&context, properties);

        // Create graphics pipeline
        let (pipeline_layout, pipeline) = Self::create_graphics_pipeline(&context, properties, render_pass);

        // Create framebuffers
        let framebuffers = Self::create_framebuffers(&context, &image_views, render_pass, properties);

        // Create command pool and buffers
        let command_pool = Self::create_command_pool(&context);
        let command_buffers = Self::create_command_buffers(&context, command_pool, &framebuffers);

        Self {
            context,
            swapchain_loader: swapchain,
            swapchain_khr,
            swapchain_properties: properties,
            images,
            image_views,
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers,
        }
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
        todo!()
    }

    fn create_swapchain(
        context: &VkContext,
        width: u32,
        height: u32,
    ) -> (SwapchainSupportDetails, vk::SwapchainKHR, SwapchainProperties, Vec<vk::Image>) {
        todo!()
    }

    fn create_image_views(
        context: &VkContext,
        images: &[vk::Image],
        properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        todo!()
    }

    fn create_render_pass(
        context: &VkContext,
        properties: SwapchainProperties,
    ) -> vk::RenderPass {
        todo!()
    }

    fn create_graphics_pipeline(
        context: &VkContext,
        properties: SwapchainProperties,
        render_pass: vk::RenderPass,
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        todo!()
    }

    fn create_framebuffers(
        context: &VkContext,
        image_views: &[vk::ImageView],
        render_pass: vk::RenderPass,
        properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        todo!()
    }

    fn create_command_pool(context: &VkContext) -> vk::CommandPool {
        todo!()
    }

    fn create_command_buffers(
        context: &VkContext,
        command_pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
    ) -> Vec<vk::CommandBuffer> {
        todo!()
    }

    pub fn draw_frame(&mut self) -> bool {
        todo!()
    }

    pub fn wait_device_idle(&self) {
        unsafe {
            self.context.device().device_wait_idle().unwrap();
        }
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        todo!()
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            device.device_wait_idle().unwrap();

            // Cleanup in reverse order of creation
            device.destroy_command_pool(self.command_pool, None);
            
            for framebuffer in &self.framebuffers {
                device.destroy_framebuffer(*framebuffer, None);
            }

            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);

            for view in &self.image_views {
                device.destroy_image_view(*view, None);
            }

            //self.swapchain_loader.destroy_swapchain(self.swapchain_khr, None);
        }
    }
} 