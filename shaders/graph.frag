#version 460

layout(location = 0) in float xpos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

void main() {
    outColor = vec4(xpos, 0, 0, 1);
}
