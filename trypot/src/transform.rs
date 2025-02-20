use cgmath::{Euler, Point3, Quaternion, Rad, BaseFloat, Vector3};

#[derive(Clone, Copy)]
pub struct Transform<S:BaseFloat> {
    pub position: Point3<S>,
    pub rotation: Quaternion<S>,
    pub scale: Vector3<S>
}

impl<S:BaseFloat> Transform<S> {
    /// Construct a new quaternion from one scalar component and three
    /// imaginary components.
    #[inline]
    pub const fn new(position:Point3<S>, rotation: Quaternion<S>, scale: Vector3<S>) -> Transform<S> {
        Transform {position: position, rotation: rotation, scale: scale}
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
    
    pub fn get_local_position(&self) -> Vector3<S> {
        Vector3::new(
            self.position.x,
            self.position.y,
            self.position.z,
        )
    }
}

impl<S: BaseFloat> Default for Transform<S> {
    fn default() -> Self {
        Self {
            position: Point3::new(S::zero(), S::zero(), S::zero()),
            rotation: Quaternion::new(S::one(), S::zero(), S::zero(), S::zero()),
            scale: Vector3::new(S::one(), S::one(), S::one()),
        }
    }
}


    