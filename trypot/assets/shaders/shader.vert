#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vCoords;
layout(location = 3) in vec3 vNormals;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoords;
layout(location = 2) out vec3 fragNormals;
layout(location = 3) out vec4 fragPos;
layout(location = 4) out vec4 vLightPos;

vec3 lightPos = vec3(0.0, 1.0, 0.0);
void main() {

    mat4 mv_matrix = ubo.view * ubo.model;
    mat3 norm_matrix = transpose(inverse(mat3(mv_matrix)));

    fragNormals = normalize( norm_matrix * vNormals );
    fragColor = vColor;
    fragCoords = vCoords;
    fragPos = mv_matrix * vec4(vPosition,1.0);
    vLightPos = ubo.model * vec4(lightPos, 1.0);

    gl_Position = ubo.proj * mv_matrix * vec4(vPosition, 1.0);
}
