use ash::{vk, Device};
use std::ffi::CString;
use std::path::Path;

use crate::platform::fs;
use crate::renderer::SwapchainProperties;
use crate::scene::gameobject;

/// Reads shader data from a file.
/// 
/// # Arguments
/// * `path` - Path to the shader file
/// 
/// # Returns
/// A vector of u32 containing the shader data in SPIR-V format
pub fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
    log::debug!("Loading shader file {}", path.as_ref().to_str().unwrap());
    let mut cursor = fs::load(path);
    ash::util::read_spv(&mut cursor).unwrap()
}

/// Creates a shader module.
/// 
/// # Arguments
/// * `device` - The logical device
/// * `code` - The SPIR-V shader code
/// 
/// # Returns
/// A shader module
pub fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

/// Creates a graphics pipeline.
/// 
/// # Arguments
/// * `device` - The logical device
/// * `swapchain_properties` - Properties of the swapchain
/// * `msaa_samples` - Multisampling count
/// * `render_pass` - The render pass
/// * `descriptor_set_layout` - The descriptor set layout
/// 
/// # Returns
/// A tuple containing the pipeline and its layout
pub fn create_pipeline(
    device: &Device,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let vertex_source = read_shader_from_file("shaders/shader.vert.spv");
    let fragment_source = read_shader_from_file("shaders/shader.frag.spv");

    let vertex_shader_module = create_shader_module(device, &vertex_source);
    let fragment_shader_module = create_shader_module(device, &fragment_source);

    let entry_point_name = CString::new("main").unwrap();
    let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(&entry_point_name);
    let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(&entry_point_name);
    let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

    let vertex_binding_descs = [gameobject::Vertex::get_binding_description()];
    let vertex_attribute_descs = gameobject::Vertex::get_attribute_descriptions();
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding_descs)
        .vertex_attribute_descriptions(&vertex_attribute_descs);

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_properties.extent.width as _,
        height: swapchain_properties.extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let viewports = [viewport];
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_properties.extent,
    };
    let scissors = [scissor];
    let viewport_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples)
        .min_sample_shading(1.0)
        // .sample_mask() // null
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);
    let color_blend_attachments = [color_blend_attachment];

    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let layout = {
        let layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        // .push_constant_range;

        unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
    };

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterizer_info)
        .multisample_state(&multisampling_info)
        .depth_stencil_state(&depth_stencil_info)
        .color_blend_state(&color_blending_info)
        // .dynamic_state() null since don't have any dynamic states
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    // .base_pipeline_handle() null since it is not derived from another
    // .base_pipeline_index(-1) same
    let pipeline_infos = [pipeline_info];

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
            .unwrap()[0]
    };

    unsafe {
        device.destroy_shader_module(vertex_shader_module, None);
        device.destroy_shader_module(fragment_shader_module, None);
    };

    (pipeline, layout)
}
