use crate::math;

use cgmath::{Deg, InnerSpace, Matrix4, Point3, Quaternion, Rad, Vector2, Vector3};

#[derive(Clone, Copy)]
pub struct Transform<S> {
    pub position: Point3<S>,
    pub rotation: Quaternion<S>
}

impl<S> Transform<S> {
    /// Construct a new quaternion from one scalar component and three
    /// imaginary components.
    #[inline]
    pub const fn new(position:Point3<S>, rotation: Quaternion<S>) -> Transform<S> {
        Transform {position: position, rotation: rotation}
    }
}