use crate::math::clamp;
use cgmath::{Deg, Euler, Matrix4, Point3, Quaternion, Vector3, Vector4};

#[derive(Clone, Copy)]
pub struct Camera {
    pos: Vector3<f32>,
    orientation: Quaternion<f32>,
}

impl Camera {
    // pub fn rotate(&mut self, theta: f32, phi: f32) {
    //     // self.theta += theta;
    //     // self.phi = self.phi + phi;
    //     let phi = self.phi + phi;
    //     self.phi = clamp(phi, 10.0_f32.to_radians(), 170.0_f32.to_radians());
    //     // println!("Theta: {}, Phi: {}, R: {}", self.theta, self.phi, self.r);
    //     // println!("Theta: {}", self.theta);
    //     // println!("Phi: {}", self.phi);
    // }

    // pub fn forward(&mut self, r: f32) {
    //     self.r -= r;
    //     println!("Theta: {}, Phi: {}, R: {}", self.theta, self.phi, self.r);
    // }

    // pub fn move(&mut self, direction: Vector3<f32>) {
    //     self.pos += v
    // }

    // pub fn forward(&mut self, steps: f32) -> Matrix4<f32> {
    //     let rot4 = Matrix4::from(self.orientation);
    //     let trans4 = Matrix4::from_cols(
    //         Vector4::unit_x(),
    //         Vector4::unit_y(),
    //         Vector4::unit_z(),
    //         self.pos.extend(1.0),
    //     );
    //     return trans4 * rot4;
    // }
    //

    pub fn get_direction(&mut self) -> Vector3<f32> {
        let rot4 = Matrix4::from(self.orientation);

    }


}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            pos: Vector3::new(0.0, 0.0, 0.0),
            orientation: Quaternion::from(Euler{
                x: Deg(0.0),
                y: Deg(45.0),
                z: Deg(0.0),
            }),
        }
    }
}
    // pub fn position(&self) -> Point3<f32> {
    //     let new_pos = Point3::new(
    //         self.r * self.phi.sin() * self.theta.sin(),
    //         self.r * self.phi.cos(),
    //         self.r * self.phi.sin() * self.theta.cos(),
    //     );
    //     // println!("Theta: {}, Phi: {}, R: {}", self.theta, self.phi, self.r);
    //     new_pos
    // }
