use crate::{camera::*, transform::*, fs};
use ash::vk;
use cgmath::{Matrix4, Point3, Vector3, Quaternion};
use std::mem::{size_of, offset_of};

#[derive(Clone)]
pub struct GameObject {
    pub camera: Option<Camera>,
    pub transform: Transform<f32>,
    pub render_object: Option<RenderObject>,
    pub model_matrix: Option<Matrix4<f32>>
}

impl GameObject {
    pub const fn default() -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), 
            camera: None,
            render_object: None,
            model_matrix: None
        }
    }

    pub const fn new_empty() -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), 
            camera: None,
            render_object: None,
            model_matrix: None
        }
    }

    pub fn new_with_camera(cam: Option<Camera>) -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), 
            camera: cam,
            render_object: None,
            model_matrix: None
        }
    }

    pub fn new_with_render_object(render_object: RenderObject) -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), 
            camera: None,
            render_object: Some(render_object),
            model_matrix: None
        }
    }

    pub fn new_with_camera_and_render_object(cam: Option<Camera>, render_object: RenderObject) -> GameObject {
        GameObject { 
            transform: Transform::new(Point3::new(0.0, 0.0, 0.0), Quaternion::new(1.0, 0.0, 0.0, 0.0)), 
            camera: cam,
            render_object: Some(render_object),
            model_matrix: None
        }
    }

    pub fn print(&self) {
        println!("X: {} Y: {} Z: {}", self.transform.position.x, self.transform.position.y, self.transform.position.z )
    }

    pub fn move_forward(&mut self, amount: f32)  {
        let forward = Vector3::new(0.0, 0.0, -1.0);
        self.transform.position += self.transform.rotation * forward * amount;
        println!("Player: {}, {}, {}", self.transform.position.x, self.transform.position.y, self.transform.position.z);
        
        if let Some(camera) = self.camera.as_mut() {
            camera.move_camera(self.transform.rotation * forward * amount);
            println!("Camera: {}, {}, {}", camera.transform.position.x, camera.transform.position.y, camera.transform.position.z);
        }
    }

    pub fn move_by(&mut self, x: f32, y: f32, z: f32) {
        self.transform.position.x += x;
        self.transform.position.y += y;
        self.transform.position.z += z;

        
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