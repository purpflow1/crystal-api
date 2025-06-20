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

layout(set = 1, binding = 1) readonly buffer LightBufferObject {
    vec3 light[60];
} lbo;

layout(set = 1, binding = 2) readonly buffer InfoLightBufferObject {
    uint light_info[3];
} ilbo;

layout(set = 2, binding = 0) uniform sampler2D color_sampler;

void main() {
    vec4 final_color = vec4(0, 0, 0, 1);

    vec3 norm = normalize(normal);

    uint offset = 0;

    for (; offset < ilbo.light_info[0]; offset++) {
        final_color += vec4(lbo.light[offset], 0);
    }

    for (; offset < ilbo.light_info[0] + ilbo.light_info[1]; offset += 2) {}

    for (; offset < ilbo.light_info[0] + ilbo.light_info[1] + ilbo.light_info[2]; offset += 2) {
        vec3 light_color = lbo.light[offset];
        vec3 light_pos = lbo.light[offset + 1];

        vec3 light_dir = normalize(light_pos - fragPos);

        float diff = max(dot(norm, light_dir), 0.0);
        vec3 diffuse = diff * vec3(1, 1, 1);

        final_color += vec4(diffuse, 0);
    }

    vec4 texture_color = texture(color_sampler, UV);
    outColor = texture_color * final_color;
}
