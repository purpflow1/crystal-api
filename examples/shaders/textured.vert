#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

layout(set = 1, binding = 0) readonly buffer ShaderStorageBufferObject {
    mat4 model[];
} ssbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec2 outUV;

void main() {
    mat4 model = ssbo.model[gl_InstanceIndex];
    gl_Position = ubo.eye * model * vec4(inPosition, 1.0);
    fragPos = vec3(model * vec4(inPosition, 1.0));
    outUV = inUV;
}
