use ash::vk;
use super::context::VkContext;
use std::ops::Drop;

pub struct Buffer {
    context: VkContext,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

// RAII wrapper for temporary command buffers
struct TempCommandBuffer {
    context: VkContext,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl TempCommandBuffer {
    fn new(context: &VkContext, command_pool: vk::CommandPool) -> Self {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            context
                .device()
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate command buffer")[0]
        };

        Self {
            context: context.clone(),
            command_pool,
            command_buffer,
        }
    }

    fn begin(&self) {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        
        unsafe {
            self.context.device()
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("Failed to begin command buffer");
        }
    }

    fn end(&self) {
        unsafe {
            self.context.device()
                .end_command_buffer(self.command_buffer)
                .expect("Failed to end command buffer");
        }
    }

    fn submit_and_wait(&self) {
        unsafe {
            let device = self.context.device();
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            device.queue_submit(device.get_device_queue(0, 0), &[submit_info], vk::Fence::null())
                .expect("Failed to submit command buffer");

            device.queue_wait_idle(device.get_device_queue(0, 0))
                .expect("Failed to wait for queue idle");
        }
    }

    fn get_command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

impl Drop for TempCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.device().free_command_buffers(self.command_pool, &[self.command_buffer]);
        }
    }
}

impl Buffer {
    pub fn new(
        context: &VkContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            context
                .device()
                .create_buffer(&buffer_info, None)
                .expect("Failed to create buffer")
        };

        let mem_requirements = unsafe {
            context.device().get_buffer_memory_requirements(buffer)
        };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(Self::find_memory_type(
                context,
                mem_requirements.memory_type_bits,
                properties,
            ));

        let memory = unsafe {
            context
                .device()
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory")
        };

        unsafe {
            context
                .device()
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind buffer memory");
        }

        Self { 
            context: context.clone(),
            buffer,
            memory,
        }
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn copy_to_buffer(
        context: &VkContext,
        command_pool: vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        // Create a temporary command buffer that will be automatically freed
        let temp_cmd = TempCommandBuffer::new(context, command_pool);
        
        // Begin command buffer
        temp_cmd.begin();

        // Record copy command
        unsafe {
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);

            context.device().cmd_copy_buffer(
                temp_cmd.get_command_buffer(),
                src_buffer,
                dst_buffer,
                &[copy_region]
            );
        }

        // End and submit command buffer
        temp_cmd.end();
        temp_cmd.submit_and_wait();
        
        // Wait for device to be idle before dropping the command buffer
        unsafe {
            context.device().device_wait_idle().unwrap();
        }
    }

    fn find_memory_type(
        context: &VkContext,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties = context.get_mem_properties();
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type");
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            // Wait for the device to be idle before cleanup
            self.context.device().device_wait_idle().unwrap();
            
            // Destroy buffer and free memory
            self.context.device().destroy_buffer(self.buffer, None);
            self.context.device().free_memory(self.memory, None);
        }
    }
}
