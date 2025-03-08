use std::collections::HashMap;

use cgmath::{Vector2, Vector3};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

/// Manages input state and provides access to input events
///
/// This struct keeps track of keyboard, mouse and other input events,
/// providing a clean abstraction for game logic to respond to user input.
/// It handles cursor capturing, mouse movement tracking, and keyboard state.
#[derive(Default)]
pub struct InputSystem {
    state: InputState,
    last_cursor_position: Option<(f64, f64)>,
}

/// Represents the current state of all inputs
#[derive(Default)]
struct InputState {
    is_cursor_captured: bool,
    mouse_delta: Option<[i32; 2]>,
    wheel_delta: Option<f32>,
    key_states: HashMap<KeyCode, bool>,
}

impl InputSystem {
    /// Creates a new input system
    pub fn new() -> Self {
        Self {
            state: InputState::default(),
            last_cursor_position: None,
        }
    }

    /// Processes a window event and updates the input state
    pub fn process_event(&mut self, event: &WindowEvent, window: &Window) {
        match event {
            WindowEvent::Focused(focused) => {
                self.handle_focus_change(*focused, window);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(*position, window);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(*delta);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
            }
            _ => {}
        }
    }

    /// Handles window focus changes
    fn handle_focus_change(&mut self, focused: bool, window: &Window) {
        if focused {
            // Capture and hide cursor when window gains focus
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked));
            // Uncomment to hide cursor
            // window.set_cursor_visible(false);
            self.state.is_cursor_captured = true;
        } else {
            // Release cursor when window loses focus
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
            // Uncomment to show cursor
            // window.set_cursor_visible(true);
            self.state.is_cursor_captured = false;
        }
    }

    /// Handles cursor movement events
    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>, window: &Window) {
        if let Some((last_x, last_y)) = self.last_cursor_position {
            // Calculate delta movement
            if self.state.is_cursor_captured {
                let dx = (position.x - last_x) as i32;
                let dy = (position.y - last_y) as i32;
                self.state.mouse_delta = Some([dx, dy]);
            }
        }

        // Update last cursor position
        self.last_cursor_position = Some((position.x, position.y));

        // Only center the cursor after recording the delta
        if self.state.is_cursor_captured {
            // Get the inner size of the window
            let window_size = window.inner_size();
            let center = PhysicalPosition::new(
                window_size.width as f64 / 2.0,
                window_size.height as f64 / 2.0,
            );

            // Only center if we've moved significantly from center to reduce events
            if (position.x - center.x).abs() > 10.0 || (position.y - center.y).abs() > 10.0 {
                // Set the cursor position back to the center
                let _ = window.set_cursor_position(center);
                self.last_cursor_position = Some((center.x, center.y));
            }
        }
    }

    /// Handles mouse wheel events
    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, v_lines) => {
                self.state.wheel_delta = Some(v_lines);
            }
            MouseScrollDelta::PixelDelta(pixels) => {
                // Convert pixel delta to lines (approximate)
                self.state.wheel_delta = Some(pixels.y as f32 / 120.0);
            }
        }
    }

    /// Handles keyboard input events
    fn handle_keyboard_input(&mut self, event: &KeyEvent) {
        if let PhysicalKey::Code(key_code) = event.physical_key {
            let is_pressed = event.state == ElementState::Pressed;
            self.state.key_states.insert(key_code, is_pressed);
        }
    }

    /// Checks if a specific key is currently pressed
    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        *self.state.key_states.get(&key_code).unwrap_or(&false)
    }

    /// Gets the current mouse delta movement
    ///
    /// Returns the mouse movement delta since the last frame, and
    /// resets it to None.
    pub fn take_mouse_delta(&mut self) -> Option<Vector2<f32>> {
        self.state.mouse_delta.take().map(|[x, y]| Vector2::new(x as f32, y as f32))
    }

    /// Gets the current wheel delta movement
    pub fn get_wheel_delta(&self) -> Option<f32> {
        self.state.wheel_delta
    }

    /// Resets all input states for the next frame
    pub fn reset_frame_state(&mut self) {
        // Keep key presses, but reset deltas
        self.state.mouse_delta = None;
        self.state.wheel_delta = None;
    }

    /// Updates camera movement based on current input state
    ///
    /// Takes the current input state and applies it to camera movement
    /// for first-person camera controls.
    pub fn apply_camera_movement<F>(&mut self, mut move_camera_fn: F, speed: f32)
    where
        F: FnMut(Vector3<f32>),
    {
        // Handle mouse look input
        if let Some(delta) = self.take_mouse_delta() {
            let sensitivity = 0.1;
            let rotation = Vector3::new(
                -delta.y * sensitivity,
                -delta.x * sensitivity,
                0.0,
            );
            move_camera_fn(rotation);
        }

        // Handle mouse wheel for zoom
        if let Some(wheel_delta) = self.get_wheel_delta() {
            move_camera_fn(Vector3::new(0.0, 0.0, wheel_delta * 0.3));
        }

        // Handle WASD movement
        let mut movement = Vector3::new(0.0, 0.0, 0.0);
        
        if self.is_key_pressed(KeyCode::KeyW) {
            movement.z += speed;
        }
        if self.is_key_pressed(KeyCode::KeyS) {
            movement.z -= speed;
        }
        if self.is_key_pressed(KeyCode::KeyA) {
            movement.x -= speed;
        }
        if self.is_key_pressed(KeyCode::KeyD) {
            movement.x += speed;
        }
        
        // Apply movement if any key is pressed
        if movement.x != 0.0 || movement.y != 0.0 || movement.z != 0.0 {
            move_camera_fn(movement);
        }
    }

    /// Returns whether the cursor is currently captured
    pub fn is_cursor_captured(&self) -> bool {
        self.state.is_cursor_captured
    }
}
