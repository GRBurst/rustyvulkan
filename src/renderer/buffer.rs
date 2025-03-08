use std::mem::size_of;

use ash::{vk, Device};
use cgmath::Matrix4;

use crate::scene::gameobject::Vertex;
use crate::renderer::context::VkContext;

/// Module for managing Vulkan buffer resources
///
/// This module provides functions for creating and managing different types of buffers:
/// - Vertex buffers
/// - Index buffers
/// - Uniform buffers
/// - General device-local buffers with data transfer
///
/// Most buffer operations in this module rely on helper functions for buffer creation,
/// memory allocation and command execution that were previously in texture.rs.

/// Creates buffers for storing uniform data
pub fn create_uniform_buffers(
    vk_context: &VkContext,
    count: usize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let buffer_size = size_of::<UniformBufferObject>() as u64;
    let mut uniform_buffers = Vec::with_capacity(count);
    let mut uniform_buffer_memories = Vec::with_capacity(count);

    for _ in 0..count {
        let (buffer, memory, _) = create_buffer(
            vk_context,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        uniform_buffers.push(buffer);
        uniform_buffer_memories.push(memory);
    }

    (uniform_buffers, uniform_buffer_memories)
}

/// Creates a vertex buffer and loads vertex data
pub fn create_vertex_buffer(
    vk_context: &VkContext,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    vertices: &[Vertex],
) -> (vk::Buffer, vk::DeviceMemory) {
    create_device_local_buffer_with_data::<u32, _>(
        vk_context,
        command_pool,
        transfer_queue,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vertices,
    )
}

/// Creates an index buffer and loads index data
pub fn create_index_buffer(
    vk_context: &VkContext,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    indices: &[u32],
) -> (vk::Buffer, vk::DeviceMemory) {
    create_device_local_buffer_with_data::<u16, _>(
        vk_context,
        command_pool,
        transfer_queue,
        vk::BufferUsageFlags::INDEX_BUFFER,
        indices,
    )
}

/// Creates a device-local buffer and fills it with data
///
/// This is a generic function used by both vertex and index buffer creation.
/// It creates a staging buffer in host-visible memory, copies data to it,
/// then transfers that data to a device-local buffer for optimal performance.
pub fn create_device_local_buffer_with_data<A, T: Copy>(
    vk_context: &VkContext,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> (vk::Buffer, vk::DeviceMemory) {
    // Calculate buffer size
    let buffer_size = (size_of::<T>() * data.len()) as u64;

    // Create staging buffer (host visible)
    let (staging_buffer, staging_memory, staging_mem_size) = create_buffer(
        vk_context,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    // Copy data to staging buffer
    unsafe {
        let memory_ptr = vk_context
            .device()
            .map_memory(
                staging_memory,
                0,
                staging_mem_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to map memory!");

        let mut memory_slice = ash::util::Align::new(
            memory_ptr,
            align_of::<A>() as u64,
            staging_mem_size,
        );
        memory_slice.copy_from_slice(data);

        vk_context.device().unmap_memory(staging_memory);
    }

    // Create device local buffer
    let (buffer, memory, _) = create_buffer(
        vk_context,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    // Copy data from staging buffer to device local buffer
    copy_buffer(
        vk_context.device(),
        command_pool,
        transfer_queue,
        staging_buffer,
        buffer,
        buffer_size,
    );

    // Clean up staging buffer
    unsafe {
        vk_context.device().destroy_buffer(staging_buffer, None);
        vk_context.device().free_memory(staging_memory, None);
    }

    (buffer, memory)
}

/// Copies data from one buffer to another using a temporary command buffer
pub fn copy_buffer(
    device: &Device,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) {
    execute_one_time_commands(device, command_pool, transfer_queue, |buffer| {
        let region = vk::BufferCopy::default().size(size);
        let regions = [region];

        unsafe { device.cmd_copy_buffer(buffer, src, dst, &regions) };
    });
}

/// Creates a buffer with the specified parameters
///
/// Creates a Vulkan buffer with the given size, usage flags, and memory
/// property requirements. Allocates and binds memory for the buffer.
pub fn create_buffer(
    vk_context: &VkContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mem_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory, vk::DeviceSize) {
    let device = vk_context.device();

    // Create buffer
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };

    // Get memory requirements
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    // Allocate memory
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(find_memory_type(
            mem_requirements,
            vk_context.get_mem_properties(),
            mem_properties,
        ));

    let memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

    // Bind buffer to memory
    unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };

    (buffer, memory, mem_requirements.size)
}

/// Finds a suitable memory type that satisfies the given requirements
pub fn find_memory_type(
    requirements: vk::MemoryRequirements,
    mem_properties: vk::PhysicalDeviceMemoryProperties,
    required_properties: vk::MemoryPropertyFlags,
) -> u32 {
    // Find a memory type that satisfies both the requirements and the properties
    for i in 0..mem_properties.memory_type_count {
        if requirements.memory_type_bits & (1 << i) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(required_properties)
        {
            return i;
        }
    }
    panic!("Failed to find suitable memory type.")
}

/// Executes a one-time command using a temporary command buffer
fn execute_one_time_commands<F: FnOnce(vk::CommandBuffer)>(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    executor: F,
) {
    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&alloc_info)
            .expect("Failed to allocate command buffers!")[0]
    };

    // Begin command buffer
    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Failed to begin command buffer!");
    }

    // Execute user-provided function
    executor(command_buffer);

    // End command buffer
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("Failed to end command buffer!");
    }

    // Fix temporary value dropped issue
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

    unsafe {
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .expect("Failed to submit command buffer!");
        device
            .queue_wait_idle(queue)
            .expect("Failed to wait for queue idle!");
        device.free_command_buffers(command_pool, &command_buffers);
    }
}

/// Uniform buffer object containing transform matrices for rendering
#[derive(Copy, Clone)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

impl UniformBufferObject {
    /// Creates a descriptor set layout binding for the uniform buffer
    pub fn get_descriptor_set_layout_binding<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
    }
}

/// Helper function to get the correct memory alignment
#[inline]
fn align_of<T>() -> usize {
    std::mem::align_of::<T>()
}

/// Copies a buffer to an image.
pub fn copy_buffer_to_image(
    device: &Device,
    command_pool: vk::CommandPool,
    transition_queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    extent: vk::Extent2D,
) {
    execute_one_time_commands(device, command_pool, transition_queue, |command_buffer| {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });

        let regions = [region];

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    });
} 