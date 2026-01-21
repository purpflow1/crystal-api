#version 460

layout(set = 0, binding = 0) uniform UniformBufferObject {
    vec2 resolution;
} ubo;

layout(location = 0) in vec2 UV;
layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D color_sampler;

const float threshold = 1.0;
const int scale = 5;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec4 blur(sampler2D image, vec2 uv, vec2 resolution) {
    vec2 texelSize = 1.0 / resolution;
    vec4 result = vec4(0.0);
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    for (int x = -scale + 1; x < scale; x++) {
        for (int y = -scale + 1; y < scale; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;

            float weight = weights[abs(x)] * weights[abs(y)];
            result += texture(image, uv + offset) * weight;
        }
    }

    return result;
}

void main() {
    vec4 color = texture(color_sampler, UV);
    color.w = 1;

    float brightness = luminance(color.rgb);
    vec4 clipped_color = (brightness > threshold) ? color : vec4(0.0);
    vec4 blurred = blur(color_sampler, UV, ubo.resolution);

    outColor = clipped_color + blurred;
}
