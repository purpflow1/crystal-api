#version 450

layout(location = 0) in vec2 UV;

layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D color_sampler;

void main() {
    vec4 texture_color = texture(color_sampler, UV);
    outColor = texture_color;
}
