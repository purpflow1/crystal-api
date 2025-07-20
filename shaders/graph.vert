#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;

layout(location = 0) out float xpos;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    xpos = inPosition.x + inNormal.x + inUV.x + inColor.x;
}
