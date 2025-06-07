#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    float time;
} ubo;

layout(set = 1, binding = 0) readonly buffer ShaderStorageBufferObject {
    mat4 model[];
} ssbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec2 UV;

void main() {
    mat4 model = ssbo.model[gl_InstanceIndex];
    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition, 1.0);
    UV = inUV;
}
