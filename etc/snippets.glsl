
//////////////////////////////////////////
// helper functions

// encodes a 3D motion vector to a 32-bit uint:
// [3-bit signs x/y/z][10-bit abs x][9-bit abs y][10-bit abs z]
uint encode_3d_motion(in vec3 motion) {
	const float range = 64.0; // [-range, range]
	vec3 signs = sign(motion);
	vec3 cmotion = clamp(abs(motion), 0.0, range);
	cmotion = log2(cmotion + 1.0); // use log2 scaling
	cmotion *= 1.0 / log2(range + 1.0);
	cmotion.xz *= 1024.0; // 2^10
	cmotion.y *= 512.0; // 2^9
	return ((signs.x < 0.0 ? 0x80000000u : 0u) |
			(signs.y < 0.0 ? 0x40000000u : 0u) |
			(signs.z < 0.0 ? 0x20000000u : 0u) |
			(clamp(uint(cmotion.x), 0u, 1023u) << 19u) |
			(clamp(uint(cmotion.y), 0u, 511u) << 10u) |
			(clamp(uint(cmotion.z), 0u, 1023u)));
}

// encodes a 2D motion vector to a 32-bit uint:
// [16-bit y][16-bit x]
uint encode_2d_motion(in vec2 motion) {
	// +/- 2^15 - 1, fit into 16 bits
	vec2 cmotion = clamp(motion * 32767.0, -32767.0, 32767.0);
	// weird bit reinterpretation chain, b/c there is no direct way to interpret an ivec2 as an uvec2
	uvec2 umotion = floatBitsToUint(intBitsToFloat(ivec2(cmotion))) & 0xFFFFu;
	return (umotion.x | (umotion.y << 16u));
}

//////////////////////////////////////////
// scatter

// vs:
uniform mat4 mvm;
uniform mat4 prev_mvm;
in vec3 vertex_pos;
out vec3 motion;
// ...

void main() {
	// get the input vertex position,
	// transform it with the model-view matrix from the previous and current frame,
	// then create the (camera-space) vector from the previous to the current position
	vec4 pos = vec4(vertex_pos, 1.0);
	vec4 prev_pos = prev_mvm * pos;
	vec4 cur_pos = mvm * pos;
	motion = cur_pos.xyz - prev_pos.xyz;
	// ...
}

// fs:
in vec3 motion;
// ...
layout (location = 0) out vec4 frag_color;
layout (location = 1) out uint motion_color;
// ...

void main() {
	motion_color = encode_3d_motion(motion);
	// ...
}

//////////////////////////////////////////
// bidirectional gather

// vs:
uniform mat4 mvpm; // @t
uniform mat4 next_mvpm; // @t+1
uniform mat4 prev_mvpm; // @t-1
in vec3 vertex_pos;
out vec4 motion_prev;
out vec4 motion_now;
out vec4 motion_next;
// ...

void main() {
	// get the input vertex position,
	// transform it with the model-view-projection matrix from the previous, current and next frame,
	// then create the screen-space motion vector
	vec4 pos = vec4(vertex_pos, 1.0);
	motion_prev = prev_mvpm * pos;
	motion_now = mvpm * pos;
	motion_next = next_mvpm * pos;
	// ...
}

// fs:
in vec4 motion_prev;
in vec4 motion_now;
in vec4 motion_next;
// ...
layout (location = 0) out vec4 frag_color;
layout (location = 1) out uint motion_forward;
layout (location = 2) out uint motion_backward;
layout (location = 3) out vec2 motion_depth;
// ...

void main() {
	motion_forward = encode_2d_motion((motion_next.xy / motion_next.w) - (motion_now.xy / motion_now.w));
	motion_backward = encode_2d_motion((motion_prev.xy / motion_prev.w) - (motion_now.xy / motion_now.w));
	motion_depth = vec2((motion_next.z / motion_next.w) - (motion_now.z / motion_now.w),
						(motion_prev.z / motion_prev.w) - (motion_now.z / motion_now.w));
	// ...
}

//////////////////////////////////////////
// forward-only gather

// vs:
uniform mat4 mvpm; // @t
uniform mat4 next_mvpm; // @t+1
in vec3 vertex_pos;
out vec4 motion_now;
out vec4 motion_next;
// ...

void main() {
	// get the input vertex position,
	// transform it with the model-view-projection matrix from the current and next frame,
	// then create the screen-space motion vector
	vec4 pos = vec4(vertex_pos, 1.0);
	motion_now = mvpm * pos;
	motion_next = next_mvpm * pos;
	// ...
}

// fs:
in vec4 motion_now;
in vec4 motion_next;
// ...
layout (location = 0) out vec4 frag_color;
layout (location = 1) out uint motion_forward;
// ...

void main() {
	motion_forward = encode_2d_motion((motion_next.xy / motion_next.w) - (motion_now.xy / motion_now.w));
	// ...
}
