use cgmath::{Matrix4, Point3, Vector2, Vector3, Rad, InnerSpace};
use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub struct Camera {
    pos: Point3<f32>,
    near_plane: f32,
    far_plane: f32,
    horizontal_angle: f32,
    vertical_angle: f32,
}

impl Camera {

    pub fn get_right(&self) -> Vector3<f32> {
        Vector3::new(
                (self.horizontal_angle - PI/2.0).sin(),
                0.0,
                (self.horizontal_angle - PI/2.0).cos(),
            ).normalize()
    }

    pub fn get_view_direction(&self) -> Vector3<f32> {
        Vector3::new(
            self.vertical_angle.cos() * self.horizontal_angle.sin(),
            self.vertical_angle.sin(),
            self.vertical_angle.cos() * self.horizontal_angle.cos(),
            ).normalize()
    }

    pub fn get_up(&self) -> Vector3<f32> {
        self.get_right().cross(self.get_view_direction()).normalize()
    }

    pub fn look_at(&self, view_pos: Vector3<f32>) -> Matrix4<f32> {
        Matrix4::look_to_rh(self.pos, view_pos, self.get_up())
    }

    pub fn move_camera(&mut self, dist: Vector3<f32>) {
        self.pos += dist;
    }

    pub fn move_forward(&mut self, dist: f32) {
        self.move_camera(dist * self.get_view_direction());
    }

    pub fn move_backward(&mut self, dist: f32) {
        self.move_camera(-dist * self.get_view_direction());
    }

    pub fn rotate(&mut self, degree: Vector2<Rad<f32>>) {
        self.horizontal_angle = (self.horizontal_angle + degree.x.0) % PI;
        self.vertical_angle = (self.vertical_angle + degree.y.0) % PI;
    }

}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            pos: Point3::new(0.0, 1.0, 0.0),
            near_plane: 0.1,
            far_plane: 100.0,
            horizontal_angle: PI,
            vertical_angle: 0.0,
        }
    }
}
