#version 460

layout(location = 0) in vec3 fragPos;
layout(location = 1) flat in vec3 normal;
layout(location = 2) in vec2 UV;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D color_sampler;

void main() {
    vec4 texture_color = texture(color_sampler, UV);
    texture_color.w = 1;
    outColor = texture_color;
}
