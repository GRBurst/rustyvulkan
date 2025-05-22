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
use std::sync::Arc;

/// Configuration options for creating a window
#[derive(Clone, Debug)]
pub struct WindowConfig {
    /// Title of the window
    pub title: String,
    /// Width of the window in pixels
    pub width: u32,
    /// Height of the window in pixels
    pub height: u32,
    pub resizable: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Rusty Vulkan".to_string(),
            width: 1280,
            height: 720,
            resizable: true,
        }
    }
}

/// Manages window creation and lifecycle
/// 
/// Provides a clean interface for creating and manipulating the application window,
/// handling window events, and accessing window properties.
pub struct WindowSystem {
    window: Arc<Window>,
    config: WindowConfig,
    resize_dimensions: Option<[u32; 2]>,
}

impl Clone for WindowSystem {
    fn clone(&self) -> Self {
        Self {
            window: Arc::clone(&self.window),
            config: self.config.clone(),
            resize_dimensions: self.resize_dimensions.clone(),
        }
    }
}

impl WindowSystem {
    /// Creates a new window system with the given configuration
    pub fn new(config: WindowConfig, event_loop: &EventLoop<()>) -> Self {
        let window = WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .with_resizable(config.resizable)
            .build(event_loop)
            .expect("Failed to create window");

        Self {
            window: Arc::new(window),
            config,
            resize_dimensions: None,
        }
    }
    
    /// Returns a reference to the underlying window
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

    /// Returns the window's dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }
}
