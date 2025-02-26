use ash::khr::surface;
use ash::vk;

#[derive(Clone, Copy)]
pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        device: vk::PhysicalDevice,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
        unsafe {
            let capabilities = surface
                .get_physical_device_surface_capabilities(device, surface_khr)
                .unwrap();
            let formats = surface
                .get_physical_device_surface_formats(device, surface_khr)
                .unwrap();
            let present_modes = surface
                .get_physical_device_surface_present_modes(device, surface_khr)
                .unwrap();

            Self {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    pub fn get_ideal_swapchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
    ) -> SwapchainProperties {
        let format = self.choose_surface_format();
        let present_mode = self.choose_present_mode();
        let extent = self.choose_extent(preferred_dimensions);

        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    fn choose_surface_format(&self) -> vk::SurfaceFormatKHR {
        if self.formats.len() == 1 && self.formats[0].format == vk::Format::UNDEFINED {
            return vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }

        self.formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(self.formats[0])
    }

    fn choose_present_mode(&self) -> vk::PresentModeKHR {
        if self.present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if self.present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        }
    }

    fn choose_extent(&self, preferred_dimensions: [u32; 2]) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent
        } else {
            let min = self.capabilities.min_image_extent;
            let max = self.capabilities.max_image_extent;
            let width = preferred_dimensions[0].clamp(min.width, max.width);
            let height = preferred_dimensions[1].clamp(min.height, max.height);

            vk::Extent2D { width, height }
        }
    }
} 