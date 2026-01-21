#version 450

layout(set = 1, binding = 0) readonly buffer ShaderStorageBufferObject {
    mat4 model[];
} ssbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 outUV;

void main() {
    mat4 model = ssbo.model[0];
    gl_Position = model * vec4(inPosition, 1.0);
    outUV = inUV;
}
