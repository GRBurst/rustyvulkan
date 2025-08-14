#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragCoords;
layout(location = 2) in vec3 fragNormals;
layout(location = 3) in vec4 fragPos;
layout(location = 4) in vec4 vLightPos;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragCoords)*vec4(fragColor, 1.0);
    vec3 lightDir = normalize(vec3(fragPos) - vec3(vLightPos));
    // vec3 lightDir = normalize(vec3(vLightPos) - vec3(fragPos));
    // float spec = dot(lightDir, fragNormals);
    // outColor = vec4(fragNormals * spec, 1.0);
}
