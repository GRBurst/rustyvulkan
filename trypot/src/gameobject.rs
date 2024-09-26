use crate::{camera::*, math, transform::{self, *}};

use cgmath::{Deg, InnerSpace, Matrix4, Point3, Quaternion, Rad, Vector2, Vector3};


#[derive(Clone, Copy)]
pub struct GameObject {
    pub camera: Option<Camera>,
    pub transform: Transform<f32>,
    //pub mesh: Option<Mesh>
}

impl GameObject {
    /// Construct a new quaternion from one scalar component and three
    /// imaginary components.
    #[inline]
    pub const fn default() -> GameObject {
        GameObject { transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), camera: None}
    }

    pub const fn new_empty() -> GameObject {
        GameObject { transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), camera: None}
    }

    pub const fn new_with_camera( cam: Option<Camera>) -> Self {
        GameObject {
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            camera: cam
        }
    }

    pub fn move_by(&mut self, x: f32, y: f32, z: f32) {
        self.transform.move_by(x, y, z);
        self.camera = self.camera.map(|mut cam| {
            cam.move_forward(z);
            cam
        });
    }

    pub fn print(&self)
    {
        println!("X: {} Y: {} Z: {}", self.transform.position.x, self.transform.position.y, self.transform.position.z )
    }
}
