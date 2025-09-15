#version 460

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec2 UV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D color_sampler;

void main() {
    outColor = texture(color_sampler, UV);
}
