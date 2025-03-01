

use cgmath::{Euler, Point3, Quaternion, Rad, BaseFloat};

#[derive(Clone, Copy)]
pub struct Transform<S:BaseFloat> {
    pub position: Point3<S>,
    pub rotation: Quaternion<S>
}

impl<S:BaseFloat> Transform<S> {
    /// Construct a new quaternion from one scalar component and three
    /// imaginary components.
    #[inline]
    pub const fn new(position:Point3<S>, rotation: Quaternion<S>) -> Transform<S> {
        Transform {position: position, rotation: rotation}
    }

    pub fn move_to_pos(&mut self, new_pos: Point3<S>) {
        self.position.x = new_pos.x;
        self.position.y = new_pos.y;
        self.position.z = new_pos.z;
    }

    pub fn move_by_p3(&mut self, new_pos: Point3<S>) {
        self.position.x += new_pos.x;
        self.position.y += new_pos.y;
        self.position.z += new_pos.z;
    }

    pub fn move_by(&mut self, x: S, y: S, z: S) {
        self.position.x += x;
        self.position.y += y;
        self.position.z += z;
    }

    pub fn add_rotation_p3(&mut self, rot: Point3<S>) {
        let mut bla = Euler::from(self.rotation);
        let blu = Euler::new(Rad(rot.x), Rad(rot.y), Rad(rot.z));
        bla.x += blu.x;
        bla.y += blu.y;
        bla.z += blu.z;

        self.rotation = Quaternion::from(bla);
        
    }

    pub fn add_rotation(&mut self, x: S, y: S, z: S) {
        let mut bla = Euler::from(self.rotation);
        let blu = Euler::new(Rad(x), Rad(y), Rad(z));
        bla.x += blu.x;
        bla.y += blu.y;
        bla.z += blu.z;

        self.rotation = Quaternion::from(bla);
        
    }
    
}


    
