
//////////////////////////////////////////
// helper functions

// encodes a 3D motion vector to a 32-bit uint:
// [3-bit signs x/y/z][10-bit abs x][9-bit abs y][10-bit abs z]
static uint32_t encode_3d_motion(thread const float3& motion) {
	constexpr const float range = 64.0; // [-range, range]
	const float3 signs = sign(motion);
	float3 cmotion = clamp(abs(motion), 0.0, range);
	cmotion = fast::log2(cmotion + 1.0); // use log2 scaling
	cmotion *= 0.16604764621594f; // 1.0 / precise::log2(range + 1.0)
	cmotion.xz *= 1024.0f; // 2^10
	cmotion.y *= 512.0f; // 2^9
	cmotion.xz = clamp(cmotion.xz, 0.0f, 1023.0f);
	cmotion.y = clamp(cmotion.y, 0.0f, 511.0f);
	const uint3 ui_cmotion = (uint3)cmotion;
	return ((signs.x < 0.0f ? 0x80000000u : 0u) |
			(signs.y < 0.0f ? 0x40000000u : 0u) |
			(signs.z < 0.0f ? 0x20000000u : 0u) |
			(ui_cmotion.x << 19u) |
			(ui_cmotion.y << 10u) |
			(ui_cmotion.z));
}

// encodes a 2D motion vector to a 32-bit uint:
// [16-bit y][16-bit x]
static uint32_t encode_2d_motion(thread const float2& motion) {
	// +/- 2^15 - 1, fit into 16 bits
	float2 cmotion = clamp(motion * 32767.0f, -32767.0f, 32767.0f);
	uint2 umotion = as_type<uint2>(int2(cmotion)) & 0xFFFFu;
	return (umotion.x | (umotion.y << 16u));
}

//////////////////////////////////////////
// scatter

struct scatter_uniforms_t {
	matrix_float4x4 mvm;
	matrix_float4x4 prev_mvm;
	/* ... */
};

struct scatter_vs_output {
	float3 motion;
	/* ... */
};

struct scatter_fs_output {
	/* e.g.: half4 color [[color(0)]] */
	uint32_t motion [[color(1)]];
	/* ... */
};

vertex scatter_vs_output scatter_vs(device const packed_float3* in_position [[buffer(0)]],
									constant scatter_uniforms_t& uniforms [[buffer(1)]],
									/* ... */
									const unsigned int vid [[vertex_id]]) {
	scatter_vs_output out;
	
	// get the vertex position for this id,
	// transform it with the model-view matrix from the previous and current frame,
	// then create the (camera-space) vector from the previous to the current position
	float4 pos(in_position[vid], 1.0f);
	float4 prev_pos = uniforms.prev_mvm * pos;
	float4 cur_pos = uniforms.mvm * pos;
	out.motion = cur_pos.xyz - prev_pos.xyz;
	
	return out;
}

fragment scatter_fs_output scatter_fs(const scatter_vs_output in [[stage_in]] /* , ... */) {
	return {
		/* color, ... */
		encode_3d_motion(in.motion)
	}
}

//////////////////////////////////////////
// bidirectional gather

struct gather_uniforms_t {
	matrix_float4x4 mvpm; // @t
	matrix_float4x4 next_mvpm; // @t+1
	matrix_float4x4 prev_mvpm; // @t-1
	/* ... */
};

struct gather_vs_output {
	float4 motion_prev;
	float4 motion_now;
	float4 motion_next;
	/* ... */
};

struct gather_fs_output {
	/* e.g.: half4 color [[color(0)]] */
	uint32_t motion_forward [[color(1)]];
	uint32_t motion_backward [[color(2)]];
	half2 motion_depth [[color(3)]];
	/* ... */
};

vertex gather_vs_output gather_vs(device const packed_float3* in_position [[buffer(0)]],
								  constant gather_uniforms_t& uniforms [[buffer(1)]],
								  /* ... */
								  const unsigned int vid [[vertex_id]]) {
	gather_vs_output out;
	
	// get the vertex position for this id,
	// transform it with the model-view-projection matrix from the previous, current and next frame,
	// then create the screen-space motion vector
	float4 pos(in_position[vid], 1.0f);
	out.motion_prev = uniforms.prev_mvpm * pos;
	out.motion_now = uniforms.mvpm * pos;
	out.motion_next = uniforms.next_mvpm * pos;
	
	return out;
}

fragment gather_fs_output gather_fs(const gather_vs_output in [[stage_in]] /* , ... */) {
	return {
		/* color, ... */
		encode_2d_motion((in.motion_next.xy / in.motion_next.w) - (in.motion_now.xy / in.motion_now.w)),
		encode_2d_motion((in.motion_prev.xy / in.motion_prev.w) - (in.motion_now.xy / in.motion_now.w)),
		half2 {
			(half)((in.motion_next.z / in.motion_next.w) - (in.motion_now.z / in.motion_now.w)),
			(half)((in.motion_prev.z / in.motion_prev.w) - (in.motion_now.z / in.motion_now.w))
		}
	}
}

//////////////////////////////////////////
// forward-only gather

struct gather_fwd_uniforms_t {
	matrix_float4x4 mvpm; // @t
	matrix_float4x4 next_mvpm; // @t+1
	/* ... */
};

struct gather_fwd_vs_output {
	float4 motion_now;
	float4 motion_next;
	/* ... */
};

struct gather_fwd_fs_output {
	/* e.g.: half4 color [[color(0)]] */
	uint32_t motion_forward [[color(1)]];
	/* ... */
};

vertex gather_fwd_vs_output gather_fwd_vs(device const packed_float3* in_position [[buffer(0)]],
										  constant gather_fwd_uniforms_t& uniforms [[buffer(1)]],
										  /* ... */
										  const unsigned int vid [[vertex_id]]) {
	gather_fwd_vs_output out;
	
	// get the vertex position for this id,
	// transform it with the model-view-projection matrix from the current and next frame,
	// then create the screen-space motion vector
	float4 pos(in_position[vid], 1.0f);
	out.motion_now = uniforms.mvpm * pos;
	out.motion_next = uniforms.next_mvpm * pos;
	
	return out;
}

fragment gather_fwd_fs_output gather_fwd_fs(const gather_fwd_vs_output in [[stage_in]] /* , ... */) {
	return {
		/* color, ... */
		encode_2d_motion((in.motion_next.xy / in.motion_next.w) - (in.motion_now.xy / in.motion_now.w)),
	}
}
