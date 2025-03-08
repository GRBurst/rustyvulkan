use ash::{Device, Entry, Instance};
use ash::ext::debug_utils;
use ash::vk::DebugUtilsMessengerEXT;
use ash::vk::SurfaceKHR;
use ash::khr::surface;
use ash::vk;

#[derive(Clone)]
pub struct VkContext {
    entry: Entry,
    instance: Instance,
    debug_utils: debug_utils::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
    surface: surface::Instance,
    surface_khr: SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    mem_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VkContext {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> &surface::Instance {
        &self.surface
    }

    pub fn surface_khr(&self) -> vk::SurfaceKHR {
        self.surface_khr
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    pub fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        self.mem_properties
    }

    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, *candidate)
            };
            (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    pub fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    pub fn new(
        entry: Entry,
        instance: Instance,
        debug_utils: debug_utils::Instance,
        debug_callback: vk::DebugUtilsMessengerEXT,
        surface: surface::Instance,
        surface_khr: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        device: Device,
    ) -> Self {
        VkContext {
            entry,
            instance,
            debug_utils,
            debug_callback,
            surface,
            surface_khr,
            physical_device,
            device,
            mem_properties: vk::PhysicalDeviceMemoryProperties::default(),
        }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.debug_utils.destroy_debug_utils_messenger(self.debug_callback, None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.instance.destroy_instance(None);
        }
    }
}
 