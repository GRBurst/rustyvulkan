//TODO:
// Rotation von GameObject implementieren -> soll auch Kamera korrekt drehen
// Andere Bewegungsrichtungen von GameObject implementieren -> move_forward fertig 
// Relation zwischen Kamera und gO muss geprüft werden

mod core;
mod scene;
mod gameobject;
mod debug;
mod fs;
mod math;
mod platform;
mod renderer;  // Add renderer module

use crate::{
    scene::*,
    gameobject::{GameObject, RenderObject, Model, Vertex},
    platform::InputManager,
    renderer::{VulkanRenderer, SwapchainProperties},  // Update this line
};

use winit::{
    dpi::PhysicalSize,
    event::{Event, MouseScrollDelta, WindowEvent, KeyEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

fn main() {
    env_logger::init();

    let config = core::EngineConfig::default();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let window = WindowBuilder::new()
        .with_title("Vulkan tutorial with Ash")
        .with_inner_size(PhysicalSize::new(config.width, config.height))
        .build(&event_loop)
        .unwrap();

    let mut renderer = VulkanRenderer::new(&window);  // Use our new renderer
    let mut input_manager = InputManager::new();      // Use our new input manager
    let mut dirty_swapchain = false;

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::NewEvents(_) => {
                    // reset input states on new frame
                    {
                        input_manager.is_cursor_captured = false;
                        input_manager.wheel_delta = None;
                        input_manager.mouse_delta = None;
                        input_manager.key_press = None;
                    }
                }
                //TODO: refactor to Redraw Request?
                Event::AboutToWait => {
                    // update input state after accumulating event
                  
                    // render
                    {
                        if dirty_swapchain {
                            let size = window.inner_size();
                            if size.width > 0 && size.height > 0 {
                                renderer.recreate_swapchain(size.width, size.height);
                            } else {
                                return;
                            }
                        }
                        dirty_swapchain = renderer.draw_frame();
                    }
                    // Reset input states after processing
                    input_manager.wheel_delta = None;
                    input_manager.mouse_delta = None;
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized { .. } => dirty_swapchain = true,
                    WindowEvent::Focused(focused) => {
                        if focused {
                            // Capture and hide cursor when window gains focus
                            window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                .or_else(|_e| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
                                .unwrap();
                            //window.set_cursor_visible(false);
                            input_manager.is_cursor_captured = true;
                        } else {
                            // Release cursor when window loses focus
                            window.set_cursor_grab(winit::window::CursorGrabMode::None).unwrap();
                           // window.set_cursor_visible(true);
                            input_manager.is_cursor_captured = false;
                        }
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        if let Some((last_x, last_y)) = input_manager.last_cursor_position {
                            // Only calculate delta if we're in captured mode
                            if input_manager.is_cursor_captured {
                                input_manager.mouse_delta = Some([
                                    (position.x - last_x) as i32,
                                    (position.y - last_y) as i32
                                ]);
                            }
                        }
                        input_manager.last_cursor_position = Some((position.x, position.y));
                        
                        // Only center the cursor after recording the delta
                        if input_manager.is_cursor_captured {
                            let window_size = window.inner_size();
                            let center = winit::dpi::PhysicalPosition::new(
                                window_size.width as f64 / 2.0,
                                window_size.height as f64 / 2.0
                            );
                            // Only center if we've moved significantly from center to reduce events
                            if (position.x - center.x).abs() > 2.0 || 
                               (position.y - center.y).abs() > 2.0 {
                                window.set_cursor_position(center).unwrap();
                                // Update last position to center to avoid jumps
                                input_manager.last_cursor_position = Some((center.x, center.y));
                            }
                        }
                    },
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, v_lines),
                        ..
                    } => {
                        input_manager.wheel_delta = Some(v_lines);
                    }
                    WindowEvent::KeyboardInput { 
                        event: KeyEvent{
                            physical_key: PhysicalKey::Code(KeyCode::KeyW),
                            state,
                            ..
                        }, .. 
                    } => {
                        if state.is_pressed(){
                            input_manager.key_press = Some(KeyCode::KeyW);
                        }
                        else{
                            input_manager.key_press = None;
                        }
                        input_manager.pressed_key_w = input_manager.key_press;
                        
                    },
                    WindowEvent::KeyboardInput { 
                        event: KeyEvent{
                            physical_key: PhysicalKey::Code(KeyCode::KeyA),
                            state,
                            ..
                        }, .. 
                    } => {
                        if state.is_pressed(){
                            input_manager.key_press = Some(KeyCode::KeyA);
                        }
                        else{
                            input_manager.key_press = None;
                        }
                        input_manager.pressed_key_a = input_manager.key_press;
                        
                    },
                    WindowEvent::KeyboardInput { 
                        event: KeyEvent{
                            physical_key: PhysicalKey::Code(KeyCode::KeyS),
                            state,
                            ..
                        }, .. 
                    } => {
                        if state.is_pressed(){
                            input_manager.key_press = Some(KeyCode::KeyS);
                        }
                        else{
                            input_manager.key_press = None;
                        }
                        input_manager.pressed_key_s = input_manager.key_press;
                        
                    },
                    WindowEvent::KeyboardInput { 
                        event: KeyEvent{
                            physical_key: PhysicalKey::Code(KeyCode::KeyD),
                            state,
                            ..
                        }, .. 
                    } => {
                        if state.is_pressed(){
                            input_manager.key_press = Some(KeyCode::KeyD);
                        }
                        else{
                            input_manager.key_press = None;
                        }
                        input_manager.pressed_key_d = input_manager.key_press;
                        
                    },
                    _ => (),
                },
                Event::LoopExiting => renderer.wait_device_idle(),
                _ => (),
            }
        })
        .unwrap();
}
