use ash::{
    Device,
    vk::{self, Format, ImageTiling, ImageUsageFlags, MemoryPropertyFlags, SampleCountFlags, SurfaceKHR},
    khr::swapchain,
    vk::Handle,
};
use std::cmp::max;
use std::ffi::CString;
use std::ptr;

use crate::platform::fs;
use crate::renderer::{SwapchainProperties, VkContext};
use crate::renderer::buffer;

/// Represents a texture in the rendering system, containing all necessary Vulkan resources.
#[derive(Clone, Copy)]
pub struct Texture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: Option<vk::Sampler>,
    pub mip_levels: u32,
    pub format: vk::Format,
}

impl Texture {
    /// Creates a new Texture from the given Vulkan resources.
    pub fn new(
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        sampler: Option<vk::Sampler>,
        mip_levels: u32,
        format: vk::Format,
    ) -> Self {
        Texture {
            image,
            memory,
            view,
            sampler,
            mip_levels,
            format,
        }
    }

    /// Destroys all Vulkan resources associated with this texture.
    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if let Some(sampler) = self.sampler.take() {
                device.destroy_sampler(sampler, None);
            }
            
            if self.view.as_raw() != 0 {
                device.destroy_image_view(self.view, None);
                self.view = vk::ImageView::null();
            }
            
            if self.image.as_raw() != 0 {
                device.destroy_image(self.image, None);
                self.image = vk::Image::null();
            }
            
            if self.memory.as_raw() != 0 {
                device.free_memory(self.memory, None);
                self.memory = vk::DeviceMemory::null();
            }
        }
    }

    /// Loads a texture from the specified image file.
    pub fn load_from_file(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        copy_queue: vk::Queue,
        path: &str,
    ) -> Self {
        log::debug!("Loading texture from file: {}", path);
        let cursor = fs::load(path);
        
        // Load the image using the image crate
        let img = image::load(cursor, image::ImageFormat::Jpeg)
            .expect("Failed to load texture image!")
            .to_rgba8();
        
        let width = img.width();
        let height = img.height();
        let image_size = (width * height * 4) as vk::DeviceSize;
        
        // Calculate the number of mip levels
        let mip_levels = ((max(width, height) as f32).log2().floor() as u32) + 1;
        
        // Create a staging buffer to transfer the image data to the GPU
        let (staging_buffer, staging_buffer_memory, _) = buffer::create_buffer(
            vk_context,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        
        // Map memory and copy the image data to the staging buffer
        unsafe {
            let data_ptr = vk_context
                .device()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory!") as *mut u8;
            
            ptr::copy_nonoverlapping(img.as_ptr(), data_ptr, image_size as usize);
            
            vk_context.device().unmap_memory(staging_buffer_memory);
        }
        
        // Create the image with the appropriate format and mip levels
        let extent = vk::Extent2D { width, height };
        let (image, memory) = create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        );
        
        // Transition the image layout for copy operation
        transition_image_layout(
            vk_context.device(),
            command_pool,
            copy_queue,
            image,
            mip_levels,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        
        // Copy the staging buffer to the image
        buffer::copy_buffer_to_image(
            vk_context.device(),
            command_pool,
            copy_queue,
            staging_buffer,
            image,
            extent,
        );
        
        // Generate mipmaps
        generate_mipmaps(
            vk_context,
            command_pool,
            copy_queue,
            image,
            extent,
            vk::Format::R8G8B8A8_SRGB,
            mip_levels,
        );
        
        // Create image view
        let view = create_image_view(
            vk_context.device(),
            image,
            mip_levels,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
        );
        
        // Create texture sampler
        let sampler = {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(mip_levels as f32);
            
            unsafe {
                vk_context
                    .device()
                    .create_sampler(&info, None)
                    .expect("Failed to create texture sampler!")
            }
        };
        
        // Cleanup staging buffer
        unsafe {
            vk_context.device().destroy_buffer(staging_buffer, None);
            vk_context.device().free_memory(staging_buffer_memory, None);
        }
        
        Texture::new(
            image, 
            memory, 
            view, 
            Some(sampler),
            mip_levels,
            vk::Format::R8G8B8A8_SRGB
        )
    }
}

/// Creates a color texture for MSAA.
pub fn create_color_texture(
    vk_context: &VkContext,
    command_pool: vk::CommandPool,
    transition_queue: vk::Queue,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    // Create the image
    let (image, memory) = create_image(
        vk_context,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        swapchain_properties.extent,
        1,
        msaa_samples,
        swapchain_properties.format.format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
    );
    
    // Create image view
    let view = create_image_view(
        vk_context.device(),
        image,
        1,
        swapchain_properties.format.format,
        vk::ImageAspectFlags::COLOR,
    );
    
    Texture::new(
        image,
        memory,
        view,
        None,
        1,
        swapchain_properties.format.format
    )
}

/// Creates a depth texture.
pub fn create_depth_texture(
    vk_context: &VkContext,
    command_pool: vk::CommandPool,
    transition_queue: vk::Queue,
    format: vk::Format,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    // Create the image
    let (image, memory) = create_image(
        vk_context,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        extent,
        1,
        msaa_samples,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    );
    
    // Create image view
    let view = create_image_view(
        vk_context.device(),
        image,
        1,
        format,
        vk::ImageAspectFlags::DEPTH,
    );
    
    // Transition the image layout for depth usage
    transition_image_layout(
        vk_context.device(),
        command_pool,
        transition_queue,
        image,
        1,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    );
    
    Texture::new(
        image,
        memory,
        view,
        None,
        1,
        format
    )
}

/// Finds a supported depth format.
pub fn find_depth_format(vk_context: &VkContext) -> vk::Format {
    find_supported_format(
        vk_context,
        &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ],
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

/// Checks if a format has a stencil component.
pub fn has_stencil_component(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
}

/// Finds a supported format from a list of candidates.
fn find_supported_format(
    vk_context: &VkContext,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> vk::Format {
    for &format in candidates {
        let properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_format_properties(vk_context.physical_device(), format)
        };
        
        if tiling == vk::ImageTiling::LINEAR
            && properties.linear_tiling_features.contains(features)
        {
            return format;
        } else if tiling == vk::ImageTiling::OPTIMAL
            && properties.optimal_tiling_features.contains(features)
        {
            return format;
        }
    }
    
    panic!("Failed to find supported format!");
}

/// Creates an image with the specified parameters.
pub fn create_image(
    vk_context: &VkContext,
    mem_properties: vk::MemoryPropertyFlags,
    extent: vk::Extent2D,
    mip_levels: u32,
    sample_count: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
) -> (vk::Image, vk::DeviceMemory) {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(sample_count);
    
    let image = unsafe {
        vk_context
            .device()
            .create_image(&image_info, None)
            .expect("Failed to create image!")
    };
    
    let mem_requirements = unsafe {
        vk_context.device().get_image_memory_requirements(image)
    };
    
    let memory_type_index = buffer::find_memory_type(
        mem_requirements,
        vk_context.get_mem_properties(),
        mem_properties,
    );

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);
    
    let memory = unsafe {
        vk_context
            .device()
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate image memory!")
    };
    
    unsafe {
        vk_context
            .device()
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind image memory!");
    };
    
    (image, memory)
}

/// Creates an image view with the specified parameters.
pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    mip_levels: u32,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags,
) -> vk::ImageView {
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        });
    
    unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("Failed to create image view!")
    }
}

/// Transitions an image from one layout to another.
pub fn transition_image_layout(
    device: &Device,
    command_pool: vk::CommandPool,
    transition_queue: vk::Queue,
    image: vk::Image,
    mip_levels: u32,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    execute_one_time_commands(device, command_pool, transition_queue, |command_buffer| {
        let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
            match (old_layout, new_layout) {
                // Undefined -> Transfer Destination
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                // Transfer Destination -> Shader Read Only
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                // Undefined -> Depth Stencil Attachment
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                ),
                _ => panic!("Unsupported layout transition!"),
            };

        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            if has_stencil_component(format) {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        let barriers = [barrier];

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    });
}

/// Generates mipmaps for an image.
pub fn generate_mipmaps(
    vk_context: &VkContext,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    image: vk::Image,
    extent: vk::Extent2D,
    format: vk::Format,
    mip_levels: u32,
) {
    // Check if image format supports linear blitting
    let format_properties = unsafe {
        vk_context
            .instance()
            .get_physical_device_format_properties(vk_context.physical_device(), format)
    };

    if !format_properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        panic!("Texture image format does not support linear blitting!");
    }

    execute_one_time_commands(vk_context.device(), command_pool, transfer_queue, |command_buffer| {
        let mut barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                base_mip_level: 0,
            });

        let mut mip_width = extent.width as i32;
        let mut mip_height = extent.height as i32;

        for i in 1..mip_levels {
            // Transition previous level to SRC_OPTIMAL
            barrier.subresource_range.base_mip_level = i - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                vk_context.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            // Blit from previous level to current level
            let src_offsets = [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width,
                    y: mip_height,
                    z: 1,
                },
            ];

            let next_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let next_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            let dst_offsets = [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: next_mip_width,
                    y: next_mip_height,
                    z: 1,
                },
            ];

            let blit = vk::ImageBlit::default()
                .src_offsets(src_offsets)
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets(dst_offsets)
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe {
                vk_context.device().cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            // Transition previous level to SHADER_READ_ONLY
            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                vk_context.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            mip_width = next_mip_width;
            mip_height = next_mip_height;
        }

        // Transition last mip level
        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            vk_context.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    });
}

/// Executes a one-time command.
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
