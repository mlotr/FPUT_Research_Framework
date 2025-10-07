shader_type spatial;

uniform float amplitude : hint_range(0.0, 2.0) = 0.5;
uniform float mesh_width = 10.0;

uniform float wave_positions[128];
uniform float wave_speeds[128];

void vertex() {
    int idx = int((VERTEX.x / mesh_width + 0.5) * 128.0);
    idx = clamp(idx, 0, 127);
    float height = wave_positions[idx] * amplitude;
    VERTEX.z += height;
}

void fragment() {
    int idx = int((VERTEX.x / mesh_width + 0.5) * 128.0);
    idx = clamp(idx, 0, 127);
    float speed = abs(wave_speeds[idx]);
    ALBEDO = vec3(speed * 5.0, 0.2, 1.0 - speed * 5.0);
}
