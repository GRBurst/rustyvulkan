pub struct EngineConfig {
    pub width: u32,
    pub height: u32,
    pub max_frames_in_flight: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            max_frames_in_flight: 2,
        }
    }
}
