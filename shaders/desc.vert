#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

layout(set = 1, binding = 0) readonly buffer ShaderStorageBufferObject {
    mat4 model[];
} ssbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 outColor;
layout(location = 4) out uint currentIndex;

void main() {
    mat4 model = ssbo.model[gl_InstanceIndex];
    currentIndex = gl_InstanceIndex;
    gl_Position = ubo.eye * model * vec4(inPosition, 1.0);
    fragPos = vec3(model * vec4(inPosition, 1.0));
    outNormal = mat3(transpose(inverse(model))) * inNormal;
    outUV = inUV;
    outColor = inColor;
}
