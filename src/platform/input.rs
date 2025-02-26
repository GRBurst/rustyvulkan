use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct InputManager {
    pub is_cursor_captured: bool,
    pub mouse_delta: Option<[i32; 2]>,
    pub wheel_delta: Option<f32>,
    pub pressed_keys: Vec<KeyCode>,
    pub last_cursor_position: Option<(f64, f64)>,
    pub key_press: Option<KeyCode>,
    pub pressed_key_w: Option<KeyCode>,
    pub pressed_key_a: Option<KeyCode>,
    pub pressed_key_s: Option<KeyCode>,
    pub pressed_key_d: Option<KeyCode>,
}

pub enum InputEvent {
    MouseMove(f64, f64),
    MouseWheel(f32),
    KeyPress(KeyCode),
    KeyRelease(KeyCode),
    CursorCapture(bool),
}

impl InputManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::MouseMove(x, y) => {
                if let Some((last_x, last_y)) = self.last_cursor_position {
                    if self.is_cursor_captured {
                        self.mouse_delta = Some([
                            (x - last_x) as i32,
                            (y - last_y) as i32,
                        ]);
                    }
                }
                self.last_cursor_position = Some((x, y));
            },
            InputEvent::MouseWheel(delta) => {
                self.wheel_delta = Some(delta);
            },
            InputEvent::KeyPress(key) => {
                if !self.pressed_keys.contains(&key) {
                    self.pressed_keys.push(key);
                }
            },
            InputEvent::KeyRelease(key) => {
                self.pressed_keys.retain(|&k| k != key);
            },
            InputEvent::CursorCapture(captured) => {
                self.is_cursor_captured = captured;
            },
        }
    }

    pub fn clear_frame_state(&mut self) {
        self.mouse_delta = None;
        self.wheel_delta = None;
    }
}
