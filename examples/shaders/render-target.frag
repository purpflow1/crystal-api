#version 460

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec2 UV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 eye;
    float time;
} ubo;

void main() {
    vec3 light_blue = vec3(0.357, 0.808, 0.98);
    vec3 pink = vec3(0.961, 0.663, 0.722);
    vec3 white = vec3(1);

    vec3 result = light_blue;

    if (fragPos.y > 0.5) ;
    else if (fragPos.y > 0.3) result = pink;
    else if (fragPos.y > 0.1) result = white;
    else if (fragPos.y > -0.1) result = pink;

    outColor = vec4(result, 1);
}
