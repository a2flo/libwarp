
//////////////////////////////////////////
// helper functions

// encodes a 3D motion vector to a 32-bit uint:
// [3-bit signs x/y/z][10-bit abs x][9-bit abs y][10-bit abs z]
static uint32_t encode_3d_motion(const float3& motion) {
	constexpr const float range = 64.0f; // [-range, range]
	const float3 signs = motion.sign();
	float3 cmotion = (motion.absed().clamp(0.0f, range) + 1.0f).log2();
	// encode x and z with 10-bit, y with 9-bit (2^10, 2^9, 2^10)
	constexpr const auto ce_scale = float3 { 1024.0f, 512.0f, 1024.0f } / math::log2(range + 1.0f);
	cmotion *= ce_scale;
	cmotion.clamp(0.0f, { 1023.0f, 511.0f, 1023.0f });
	const auto ui_cmotion = (uint3)cmotion;
	return ((signs.x < 0.0f ? 0x80000000u : 0u) |
			(signs.y < 0.0f ? 0x40000000u : 0u) |
			(signs.z < 0.0f ? 0x20000000u : 0u) |
			(ui_cmotion.x << 19u) |
			(ui_cmotion.y << 10u) |
			(ui_cmotion.z));
}

// encodes a 2D motion vector to a 32-bit uint:
// [16-bit y][16-bit x]
static uint32_t encode_2d_motion(const float2& motion) {
	// +/- 2^15 - 1, fit into 16 bits
	const auto cmotion = short2((motion * 32767.0f).clamp(-32767.0f, 32767.0f));
	return *(uint32_t*)&cmotion;
}

//////////////////////////////////////////
// scatter

struct scatter_uniforms_t {
	matrix4f mvm;
	matrix4f prev_mvm;
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

vertex auto scatter_vs(buffer<const float3> in_position,
					   param<scatter_uniforms_t> uniforms
					   /* ... */) {
	scatter_vs_output out;
	
	// get the vertex position for this id,
	// transform it with the model-view matrix from the previous and current frame,
	// then create the (camera-space) vector from the previous to the current position
	float4 pos { in_position[vertex_id], 1.0f };
	float4 prev_pos = pos * uniforms.prev_mvm;
	float4 cur_pos = pos * uniforms.mvm;
	out.motion = cur_pos.xyz - prev_pos.xyz;
	
	return out;
}

fragment auto scatter_fs(const scatter_vs_output in [[stage_input]] /* , ... */) {
	return scatter_fs_output {
		/* color, ... */
		encode_3d_motion(in.motion)
	}
}

//////////////////////////////////////////
// bidirectional gather

struct gather_uniforms_t {
	matrix4f mvpm; // @t
	matrix4f next_mvpm; // @t+1
	matrix4f prev_mvpm; // @t-1
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

vertex auto gather_vs(buffer<const float3> in_position,
					  param<gather_uniforms_t> uniforms
					  /* ... */) {
	gather_vs_output out;
	
	// get the vertex position for this id,
	// transform it with the model-view-projection matrix from the previous, current and next frame,
	// then create the screen-space motion vector
	float4 pos { in_position[vertex_id], 1.0f };
	out.motion_prev = pos * uniforms.prev_mvpm;
	out.motion_now = pos * uniforms.mvpm;
	out.motion_next = pos * uniforms.next_mvpm;
	
	return out;
}

fragment auto gather_fs(const gather_vs_output in [[stage_input]] /* , ... */) {
	return gather_fs_output {
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
	matrix4f mvpm; // @t
	matrix4f next_mvpm; // @t+1
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

vertex auto gather_fwd_vs(buffer<const float3> in_position,
						  param<gather_fwd_uniforms_t> uniforms
						  /* ... */) {
	gather_fwd_vs_output out;
	
	// get the vertex position for this id,
	// transform it with the model-view-projection matrix from the current and next frame,
	// then create the screen-space motion vector
	float4 pos { in_position[vertex_id], 1.0f };
	out.motion_now = pos * uniforms.mvpm;
	out.motion_next = pos * uniforms.next_mvpm;
	
	return out;
}

fragment auto gather_fwd_fs(const gather_fwd_vs_output in [[stage_input]] /* , ... */) {
	return gather_fwd_fs_output {
		/* color, ... */
		encode_2d_motion((in.motion_next.xy / in.motion_next.w) - (in.motion_now.xy / in.motion_now.w)),
	}
}
