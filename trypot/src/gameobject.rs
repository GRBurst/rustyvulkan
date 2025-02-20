use crate::{camera::*, transform::*, fs};
use ash::vk;
use cgmath::{Matrix4, Point3, Vector3, Quaternion, Rotation3};
use std::sync::{Arc, Weak};

use std::mem::{size_of, offset_of};

#[derive(Clone)]
pub struct GameObject {
    pub transform: Transform<f32>,
    camera: Option<Arc<Camera>>,
    pub render_object: Option<RenderObject>,
    pub model_matrix: Option<Matrix4<f32>>,
    pub parent: Option<Weak<GameObject>>,
    pub children: Vec<Arc<GameObject>>,
}

impl GameObject {
    pub const fn default() -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)), 
            camera: None,
            render_object: None,
            model_matrix: None,
            parent: None,
            children: Vec::new(),
        }
    }

    pub const fn new_empty() -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), 
                                      Quaternion::new(1.0, 0.0, 0.0, 0.0), 
                                         Vector3::new(1.0, 1.0, 1.0)), 
            camera: None,
            render_object: None,
            model_matrix: None,
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn new_with_camera(cam: Option<Arc<Camera>>) -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), 
                                      Quaternion::new(1.0, 0.0, 0.0, 0.0), 
                                         Vector3::new(1.0,1.0, 1.0)), 
            camera: cam,
            render_object: None,
            model_matrix: None,
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn new_with_render_object(render_object: RenderObject) -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), 
                                      Quaternion::new(1.0, 0.0, 0.0, 0.0),
                                         Vector3::new(1.0, 1.0, 1.0)), 
            camera: None,
            render_object: Some(render_object),
            model_matrix: None,
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn new_with_camera_and_render_object(cam: Option<Arc<Camera>>, render_object: RenderObject) -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), 
                                      Quaternion::new(1.0, 0.0, 0.0, 0.0),
                                         Vector3::new(1.0,1.0, 1.0)), 
            camera: cam,
            render_object: Some(render_object),
            model_matrix: None,
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn print(&self) {
        println!("X: {} Y: {} Z: {}", self.transform.position.x, self.transform.position.y, self.transform.position.z )
    }

    pub fn add_camera(&mut self, camera_transform: Transform<f32>) {
        // Create a temporary Arc of self for the weak reference
        let this = GameObject {
            transform: self.transform.clone(),
            camera: None,
            render_object: self.render_object.clone(),
            model_matrix: self.model_matrix.clone(),
            parent: self.parent.clone(),
            children: self.children.clone(),
        };
        
        let this_ref = Arc::new(this);
        let weak_ref = Arc::downgrade(&this_ref);
        
        self.camera = Some(Arc::new(Camera::new(camera_transform, weak_ref)));
    }

    pub fn move_forward(&mut self, amount: f32) {
        let forward = Vector3::new(0.0, 0.0, -1.0);
        let world_forward = self.transform.rotation * forward;
        
        // Move the GameObject
        self.transform.position += world_forward * amount;
        if let Some(ref camera) = self.camera {
            println!("Camera: {}, {}, {}", camera.transform.position.x, camera.transform.position.y, camera.transform.position.z);
        }
        println!("Player: {}, {}, {}", self.transform.position.x, self.transform.position.y, self.transform.position.z);
    }

    pub fn move_by(&mut self, x: f32, y: f32, z: f32) {
        self.transform.position.x += x;
        self.transform.position.y += y;
        self.transform.position.z += z;

        if let Some(mut camera) = self.camera.as_mut() {
            //camera.transform.move_by(x, y, z);
        }
    }

    pub fn world_matrix(&self) -> Matrix4<f32> {
        //let local = 
        self.model_matrix.unwrap()
        /*
        if let Some(parent) = &self.parent {
            parent.upgrade().map_or(local, |p| p.world_matrix() * local)
        } else {
            local
        }*/
    }

    pub fn rotate(&mut self, pitch: f32, yaw: f32, roll: f32) {
        // Create rotation quaternions for each axis
        let pitch_rotation = Quaternion::from_angle_x(cgmath::Rad(pitch));
        let yaw_rotation = Quaternion::from_angle_y(cgmath::Rad(yaw));
        let roll_rotation = Quaternion::from_angle_z(cgmath::Rad(roll));

        // Combine rotations (order: yaw * pitch * roll)
        let rotation = yaw_rotation * pitch_rotation * roll_rotation;
        
        // Apply the rotation to the current rotation
        self.transform.rotation = rotation * self.transform.rotation;

        // If there's a camera, rotate it around the GameObject's pivot
        if let Some(camera) = self.camera.as_mut() {
            // Get the relative position of camera to GameObject
            let relative_pos = camera.transform.position - self.transform.position;
            
            // Rotate the relative position
            let rotated_pos = rotation * relative_pos;
            
            // Set the new camera position
            //camera.transform.position = self.transform.position + rotated_pos;
            
            // Update camera rotation
            //camera.transform.rotation = self.transform.rotation;
        }
    }

    pub fn camera(&self) -> Option<&Arc<Camera>> {
        self.camera.as_ref()
    }

    pub fn camera_mut(&mut self) -> Option<&mut Arc<Camera>> {
        self.camera.as_mut()
    }
}

#[derive(Clone)]
pub struct RenderObject {
    pub model: Model,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub index_count: usize,
}

#[derive(Clone)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Model {
    pub fn load(path: &str) -> Self {
        let mut cursor = fs::load(path);
        let (models, _) = tobj::load_obj_buf(
            &mut cursor,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
            |_| Ok((vec![], ahash::AHashMap::new())),
        )
        .unwrap();

        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let coords = mesh.texcoords.as_slice();
        let vertex_count = mesh.positions.len() / 3;
        let normals = mesh.normals.as_slice();

        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let vertex = Vertex {
                pos: [
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2],
                ],
                color: [1.0, 1.0, 1.0],
                coords: [coords[i * 2], coords[i * 2 + 1]],
                normal: [
                    normals[i * 3],
                    normals[i * 3 + 1],
                    normals[i * 3 + 2],
                ],
            };
            vertices.push(vertex);
        }

        Model {
            vertices,
            indices: mesh.indices.clone(),
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as _);
        let color_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as _);
        let coords_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, coords) as _);
        let normals_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, normal) as _);
        [position_desc, color_desc, coords_desc, normals_desc]
    }
}