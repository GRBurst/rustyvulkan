use crate::math::{Transform, matrix::perspective};

use cgmath::{Matrix4, Point3, Vector2, Vector3, Deg, Rad, InnerSpace, Quaternion};
use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub struct Camera {
    pub transform: Transform<f32>,
    h_fov: Deg<f32>,
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

    pub fn look_at(&self, view_point: Point3<f32>) -> Matrix4<f32> {
        Matrix4::look_at_rh(self.transform.position, view_point, self.get_up())
    }

    pub fn look_to(&self, view_dir: Vector3<f32>) -> Matrix4<f32> {
        Matrix4::look_to_rh(self.transform.position, view_dir, self.get_up())
    }

    pub fn move_camera(&mut self, dist: Vector3<f32>) {
        self.transform.position += dist;
    }

    pub fn move_forward(&mut self, dist: f32) {
        self.move_camera(dist * self.get_view_direction());
        println!("Camera: {}, {}, {}", self.transform.position.x, self.transform.position.y, self.transform.position.z);
    }

    pub fn move_backward(&mut self, dist: f32) {
        self.move_camera(dist * self.get_view_direction());
        println!("Camera: {}, {}, {}", self.transform.position.x, self.transform.position.y, self.transform.position.z);
    }

    pub fn rotate(&mut self, degree: Vector2<Rad<f32>>) {
        self.horizontal_angle = (self.horizontal_angle - degree.x.0) % (2.0*PI);
        self.vertical_angle = (self.vertical_angle + degree.y.0) % (2.0*PI);
    }

    /// Rotates the camera based on x and y inputs (in degrees)
    ///
    /// This is a convenience method that converts the input values to radians
    /// and calls the rotate method.
    pub fn rotate_camera(&mut self, x_degrees: f32, y_degrees: f32) {
        // Convert to radians
        let x_rad = Rad(x_degrees * PI / 180.0);
        let y_rad = Rad(y_degrees * PI / 180.0);
        self.rotate(Vector2::new(x_rad, y_rad));
    }

    pub fn get_projection_matrix(&self, aspect: f32) -> Matrix4<f32> {
        perspective(self.h_fov, aspect, self.near_plane, self.far_plane)
    }
    
    pub fn print(&self) {
        //println!("Camera: {}, {}, {}", self.transform.position.x, self.transform.position.y, self.transform.position.z);
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            transform: Transform::new(Point3::new(0.0, 1.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            h_fov: Deg(45.0),
            near_plane: 0.1,
            far_plane: 1000.0,
            horizontal_angle: PI,
            vertical_angle: 0.0,
        }
    }
}
