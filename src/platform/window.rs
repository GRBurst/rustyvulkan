// Window management system for Vulkan applications
// Handles window creation, resize events, and surface creation

use winit::{
    dpi::PhysicalSize,
    event_loop::{EventLoop},
    window::{Window, WindowBuilder},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
};
use ash::{
    vk,
    Entry, Instance,
    khr::surface,
};

/// Configuration options for creating a window
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Vulkan Application".to_string(),
            width: 800,
            height: 600,
            resizable: true,
        }
    }
}

/// Manages window creation and lifecycle
pub struct WindowSystem {
    window: Window,
    resize_dimensions: Option<[u32; 2]>,
}

impl WindowSystem {
    /// Creates a new window with the provided configuration and event loop
    pub fn new(config: WindowConfig, event_loop: &EventLoop<()>) -> Self {
        // Create the window
        let window = WindowBuilder::new()
            .with_title(config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .with_resizable(config.resizable)
            .build(event_loop)
            .expect("Failed to create window");
            
        // Create the window system
        Self {
            window,
            resize_dimensions: None,
        }
    }
    
    /// Returns a reference to the window
    pub fn window(&self) -> &Window {
        &self.window
    }
    
    /// Creates a Vulkan surface for this window
    pub fn create_surface(&self, entry: &Entry, instance: &Instance) -> (surface::Instance, vk::SurfaceKHR) {
        // Create the surface
        let surface_khr = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                self.window.display_handle().expect("Failed to get display handle").as_raw(),
                self.window.window_handle().expect("Failed to get window handle").as_raw(),
                None,
            )
            .unwrap()
        };
        
        let surface_instance = surface::Instance::new(entry, instance);
        
        (surface_instance, surface_khr)
    }
    
    /// Records a resize event
    pub fn record_resize(&mut self, width: u32, height: u32) {
        self.resize_dimensions = Some([width, height]);
    }
    
    /// Gets and clears the recorded resize dimensions
    pub fn take_resize_dimensions(&mut self) -> Option<[u32; 2]> {
        self.resize_dimensions.take()
    }
    
    /// Returns the current dimensions of the window
    pub fn get_dimensions(&self) -> [u32; 2] {
        let size = self.window.inner_size();
        [size.width, size.height]
    }
    
    /// Returns the title of the window
    pub fn title(&self) -> String {
        self.window.title().to_string()
    }
}
