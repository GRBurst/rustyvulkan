use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Window as WinitWindow, WindowBuilder},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct Window {
    event_loop: EventLoop<()>,
    window: WinitWindow,
    dimensions: PhysicalSize<u32>,
}

impl Window {
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        let event_loop = EventLoop::new().unwrap();
        let dimensions = PhysicalSize::new(width, height);
        
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(dimensions)
            .build(&event_loop)
            .unwrap();

        Self {
            event_loop,
            window,
            dimensions,
        }
    }

    pub fn get_native_window(&self) -> &WinitWindow {
        &self.window
    }

    pub fn get_dimensions(&self) -> PhysicalSize<u32> {
        self.dimensions
    }
}

// Implement window handle traits
impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        self.window.window_handle()
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        self.window.display_handle()
    }
}
