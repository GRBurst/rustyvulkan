use ash::vk;
use std::ffi::CString;
use crate::gameobject::Vertex;
use super::context::VkContext;
use super::swapchain::SwapchainProperties;

pub struct Pipeline {
    context: VkContext,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl Pipeline {
    pub fn new(
        context: &VkContext,
        properties: SwapchainProperties,
        render_pass: vk::RenderPass,
    ) -> Self {
        // Create shader modules
        let vert_shader_path = "assets/shaders/shader.vert.spv";
        let frag_shader_path = "assets/shaders/shader.frag.spv";

        let vert_shader_code = std::fs::read(vert_shader_path)
            .expect(&format!("Failed to read vertex shader at {}", vert_shader_path));
        let frag_shader_code = std::fs::read(frag_shader_path)
            .expect(&format!("Failed to read fragment shader at {}", frag_shader_path));

        // Verify shader file sizes
        if vert_shader_code.len() % 4 != 0 {
            panic!("Vertex shader size is not a multiple of 4");
        }
        if frag_shader_code.len() % 4 != 0 {
            panic!("Fragment shader size is not a multiple of 4");
        }

        let vert_shader_module = Self::create_shader_module(context, &vert_shader_code);
        let frag_shader_module = Self::create_shader_module(context, &frag_shader_code);

        // Create shader stages
        let entry_point_name = unsafe { CString::from_vec_unchecked(b"main".to_vec()) };
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(entry_point_name.as_c_str()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(entry_point_name.as_c_str()),
        ];

        // Vertex input state
        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let binding_descriptions = [binding_description];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        // Input assembly state
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport and scissor state
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(properties.extent.width as f32)
            .height(properties.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(properties.extent);

        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        // Rasterization state
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        // Multisampling state
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Color blend state
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blend_attachments = [color_blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        // Create descriptor set layout
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_bindings = [ubo_layout_binding, sampler_layout_binding];
        let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&layout_bindings);

        let descriptor_set_layout = unsafe {
            context.device().create_descriptor_set_layout(&descriptor_layout_info, None)
                .expect("Failed to create descriptor set layout")
        };

        // Create pipeline layout
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts);

        let layout = unsafe {
            context.device().create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        // Create graphics pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            context.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
            .expect("Failed to create graphics pipeline")[0]
        };

        // Cleanup shader modules
        unsafe {
            context.device().destroy_shader_module(vert_shader_module, None);
            context.device().destroy_shader_module(frag_shader_module, None);
        }

        Self {
            context: context.clone(),
            layout,
            pipeline,
            descriptor_set_layout,
        }
    }

    pub fn get_pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn get_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn cleanup(&mut self) {
        unsafe {
            let device = self.context.device();
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }

    fn create_shader_module(context: &VkContext, code: &[u8]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(unsafe { std::slice::from_raw_parts(
                code.as_ptr() as *const u32,
                code.len() / 4,
            )});

        unsafe {
            context.device().create_shader_module(&create_info, None)
                .expect("Failed to create shader module")
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.cleanup();
    }
}
