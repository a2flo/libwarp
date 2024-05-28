/*
 *  libwarp
 *  Copyright (C) 2015 - 2021 Florian Ziesche
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 2 of the License only.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef __LIBWARP_WARP_KERNELS_HPP__
#define __LIBWARP_WARP_KERNELS_HPP__

#include <floor/core/essentials.hpp>

//////////////////////////////////////////
// compile time defines

// LIBWARP_SCREEN_WIDTH: screen width in px
#if !defined(LIBWARP_SCREEN_WIDTH)
#define LIBWARP_SCREEN_WIDTH 1280
#endif

// LIBWARP_SCREEN_HEIGHT: screen height in px
#if !defined(LIBWARP_SCREEN_HEIGHT)
#define LIBWARP_SCREEN_HEIGHT 720
#endif

// LIBWARP_SCREEN_FOV: camera/projection-matrix field of view
#if !defined(LIBWARP_SCREEN_FOV)
#define LIBWARP_SCREEN_FOV 72.0f
#endif

// LIBWARP_NEAR_PLANE: camera near plane distance
#if !defined(LIBWARP_NEAR_PLANE)
#define LIBWARP_NEAR_PLANE 0.5f
#endif

// LIBWARP_FAR_PLANE: camera far plane distance
#if !defined(LIBWARP_FAR_PLANE)
#define LIBWARP_FAR_PLANE 500.0f
#endif

// TILE_SIZE_X: work-group x-size / tile width
#if !defined(TILE_SIZE_X)
#if defined(__WINDOWS__) && defined(FLOOR_COMPUTE_HOST)
// on windows with host-compute: must be limited to 8
#define TILE_SIZE_X 8
#else // everywhere else: always use 32 as the default
#define TILE_SIZE_X 32
#endif
#endif

// TILE_SIZE_Y: work-group y-size / tile height
#if !defined(TILE_SIZE_Y)
#if defined(__WINDOWS__) && defined(FLOOR_COMPUTE_HOST)
// on windows with host-compute: must be limited to 8
#define TILE_SIZE_Y 8
#else // everywhere else: always use 16 as the default
#define TILE_SIZE_Y 16
#endif
#endif

#if defined(FLOOR_COMPUTE_HOST) && !defined(FLOOR_COMPUTE_HOST_DEVICE)
// work-around host (!device) compilation
#define SCREEN_ORIGIN_LEFT_TOP 1
#endif

// always defined externally (since compute backend might not match rendering backend)
#if !defined(SCREEN_ORIGIN_LEFT_BOTTOM) && !defined(SCREEN_ORIGIN_LEFT_TOP)
#error "must either define SCREEN_ORIGIN_LEFT_BOTTOM or SCREEN_ORIGIN_LEFT_TOP"
#endif

// depth buffer types
enum class depth_type {
	// normalized in [0, 1], default for Metal/Vulkan
	normalized,
	// z/w depth
	z_div_w,
	// linear depth [0, far-plane]
	linear,
};

#if !defined(DEFAULT_DEPTH_TYPE)
#define DEFAULT_DEPTH_TYPE depth_type::normalized
#define NATIVE_DEPTH_IMAGE 1
#endif

//////////////////////////////////////////
// all compute code from here
#if defined(FLOOR_COMPUTE)

#if defined(FLOOR_COMPUTE_HOST)
#include <floor/compute/device/common.hpp>
#endif

// if tiles (work-group x/y sizes) overlap the screen size, a check is neccesary to ignore the overlapping work-items
#if (((LIBWARP_SCREEN_WIDTH / TILE_SIZE_X) * TILE_SIZE_X) != LIBWARP_SCREEN_WIDTH) || (((LIBWARP_SCREEN_HEIGHT / TILE_SIZE_Y) * TILE_SIZE_Y) != LIBWARP_SCREEN_HEIGHT)
#define screen_check() if (global_id.x >= LIBWARP_SCREEN_WIDTH || global_id.y >= LIBWARP_SCREEN_HEIGHT) return
#else
#define screen_check() /* nop */
#endif

// choose between native depth buffer image type and single-channel color image type that emulates a depth buffer
#if NATIVE_DEPTH_IMAGE == 1
using depth_image_type = const_image_2d_depth<float>;
#else
using depth_image_type = const_image_2d<float1>;
#endif

namespace warp_camera {
	// sceen size in fp
	static constexpr const float2 screen_size { float(LIBWARP_SCREEN_WIDTH), float(LIBWARP_SCREEN_HEIGHT) };
	// 1 / screen size in fp
	static constexpr const float2 inv_screen_size { 1.0f / screen_size };
	// screen width / height aspect ratio
	static constexpr const float aspect_ratio { screen_size.x / screen_size.y };
	// projection up vector
	static constexpr const float _up_vec { const_math::tan(const_math::deg_to_rad(LIBWARP_SCREEN_FOV) * 0.5f) };
	// projection right vector
	static constexpr const float right_vec { _up_vec * aspect_ratio };
#if defined(SCREEN_ORIGIN_LEFT_BOTTOM)
	static constexpr const float up_vec { _up_vec };
#else // flip up vector for "left top" origin
	static constexpr const float up_vec { -_up_vec };
#endif
	// [near, far] plane, needed for depth correction
	static constexpr const float2 near_far_plane { LIBWARP_NEAR_PLANE, LIBWARP_FAR_PLANE };
	
	// reconstructs a 3D position from a 2D screen coordinate and its associated real world depth
	static float3 reconstruct_position(const uint2& coord, const float& linear_depth) {
		// originally this was: ((float2(coord) + 0.5f) * 2.0f * inv_screen_size - 1.0f) * float2(right_vec, up_vec) * linear_depth
		// -> simplified below (with constexpr terms forced to be computed at compile-time)
		constexpr const auto ce_term_1 = inv_screen_size * float2(right_vec, up_vec);
		constexpr const auto ce_term_2 = 2.0f * ce_term_1;
		constexpr const auto ce_term_3 = ce_term_1 - float2(right_vec, up_vec);
		return {
			(float2(coord) * ce_term_2 + ce_term_3) * linear_depth,
			-linear_depth
		};
	}
	
	// reprojects a 3D position back to 2D
	static float2 reproject_position(const float3& position) {
		constexpr const auto ce_term = float2 { 1.0f / right_vec, 1.0f / up_vec };
		const auto proj_dst_coord = (position.xy * ce_term) / -position.z;
		return ((proj_dst_coord * 0.5f + 0.5f) * screen_size);
	}
	
	// linearizes the input depth value according to the depth type and returns the real world depth value
	template <depth_type type = DEFAULT_DEPTH_TYPE>
	constexpr static float linearize_depth(const float& depth) {
		if (type == depth_type::normalized) {
			// reading from the actual depth buffer which is normalized in [0, 1]
			constexpr const float2 near_far_projection {
				-(near_far_plane.y + near_far_plane.x) / (near_far_plane.x - near_far_plane.y),
				(2.0f * near_far_plane.y * near_far_plane.x) / (near_far_plane.x - near_far_plane.y),
			};
			// special case: clear/full depth, assume this comes from a normalized sky box
			return (depth == 1.0f ? 1.0f : near_far_projection.y / (depth - near_far_projection.x));
		} else if (type == depth_type::z_div_w) {
			// depth is written as z/w in shader -> need to perform a small adjustment to account for near/far plane to get the real world depth
			// (note that this error is almost imperceptible and could just be ignored)
			return depth + near_far_plane.x - (depth * (near_far_plane.x / near_far_plane.y));
		} else if (type == depth_type::linear) {
			// already linear, just pass through
			return depth;
		}
		floor_unreachable();
	}
};

// decodes the encoded input 3D motion vector
// format: [1-bit sign x][1-bit sign y][1-bit sign z][10-bit x][9-bit y][10-bit z]
floor_inline_always static float3 decode_3d_motion(const uint32_t& encoded_motion) {
	// lookup into constant memory + 1 shift is faster than 3 ANDs + 3 cmps/sels
	static constexpr const float3 signs_lookup[] {
		{ 1.0f, 1.0f, 1.0f },
		{ 1.0f, 1.0f, -1.0f },
		{ 1.0f, -1.0f, 1.0f },
		{ 1.0f, -1.0f, -1.0f },
		{ -1.0f, 1.0f, 1.0f },
		{ -1.0f, 1.0f, -1.0f },
		{ -1.0f, -1.0f, 1.0f },
		{ -1.0f, -1.0f, -1.0f },
	};
	const float3 signs = signs_lookup[encoded_motion >> 29u];
	const uint3 shifted_motion {
		(encoded_motion >> 19u) & 0x3FFu,
		(encoded_motion >> 10u) & 0x1FFu,
		encoded_motion & 0x3FFu
	};
	constexpr const float3 adjust {
		const_math::log2(64.0f + 1.0f) / 1024.0f,
		const_math::log2(64.0f + 1.0f) / 512.0f,
		const_math::log2(64.0f + 1.0f) / 1024.0f
	};
	return signs * ((float3(shifted_motion) * adjust).exp2() - 1.0f);
}

// computes the "scattered" destination coordinate of the pixel at 'coord',
// according to it's depth value (which is also returned) and motion vector, as well as the current time delta
floor_inline_always static auto scatter(const int2& coord,
										const float& delta,
										depth_image_type img_depth,
										const_image_2d<uint1> img_motion) {
	// read rendered/input depth and linearize it (linear distance from the camera origin)
	const auto linear_depth = warp_camera::linearize_depth(img_depth.read(coord));
	// get 3d motion for this pixel
	const auto motion = decode_3d_motion(img_motion.read(coord));
	// reconstruct 3D position from depth + camera/screen setup,
	// then predict/compute new 3D position from current motion and time
	const auto new_pos = warp_camera::reconstruct_position(coord, linear_depth) + delta * motion;
	// -> return
	const struct {
		const uint2 coord;
		const float linear_depth;
	} ret {
		// project 3D position back into 2D
		// NOTE: convert directly to uint so that we won't have to check for >= 0
		.coord = warp_camera::reproject_position(new_pos),
		.linear_depth = linear_depth
	};
	return ret;
}

//
kernel_2d() void libwarp_warp_scatter_depth(depth_image_type img_depth,
											const_image_2d<uint1> img_motion,
											// NOTE: depth buffer is technically a float, but since Vulkan can't perform any
											//       fp atomics, we need to reinterpret this as uint32_t data
											//       note that this doesn't change the outcome of atomic_min (for values >= 0)
											buffer<uint32_t> depth_buffer,
											param<float> delta) {
	screen_check();
	
	const auto scattered = scatter(global_id.xy, delta, img_depth, img_motion);
	if (scattered.coord.x < LIBWARP_SCREEN_WIDTH &&
		scattered.coord.y < LIBWARP_SCREEN_HEIGHT) {
		atomic_min(&depth_buffer[scattered.coord.y * LIBWARP_SCREEN_WIDTH + scattered.coord.x],
				   *(const uint32_t*)&scattered.linear_depth);
	}
}
//
kernel_2d() void libwarp_warp_scatter_color(const_image_2d<float> img_color,
											depth_image_type img_depth,
											const_image_2d<uint1> img_motion,
											image_2d<float4, true> img_out_color,
											buffer<const float> depth_buffer,
											param<float> delta) {
	screen_check();
	
	const auto coord = global_id.xy;
	const auto scattered = scatter(coord, delta, img_depth, img_motion);
	if (scattered.coord.x >= LIBWARP_SCREEN_WIDTH ||
		scattered.coord.y >= LIBWARP_SCREEN_HEIGHT) {
		return;
	}
	
	const auto dst_depth = depth_buffer[scattered.coord.y * LIBWARP_SCREEN_WIDTH + scattered.coord.x];
	if (scattered.linear_depth > dst_depth) {
		return;
	}
	
	auto color = img_color.read(coord);
	color.w = 1.0f; // px fixup
	img_out_color.write(scattered.coord, color);
}

// decodes the encoded input 2D motion vector
// format: [16-bit y][16-bit x]
static float2 decode_2d_motion(const uint32_t& encoded_motion) {
	// NOTE: this no longer needs to perform Y * -1 multiplication here
	return unpack_snorm_2x16(encoded_motion) * 0.5f;
}

// gaussian blur helper functions (used in warp_gather_forward)
#define TAP_COUNT 21u
template <uint32_t tap_count>
static constexpr uint32_t find_effective_n() {
	// minimal contribution a fully white pixel must have to affect the blur result
	// (ignoring the fact that 0.5 gets rounded up to 1 and that multiple outer pixels combined can produce values > 1)
	constexpr const auto min_contribution = 1.0L / 255.0L;
	
	// start at the wanted tap count and go up by 2 "taps" if the row is unusable (and no point going beyond 64)
	for (uint32_t count = tap_count; count < 64u; count += 2) {
		// / 2^N for this row
		const long double sum_div = 1.0L / (long double)const_math::pow(2ull, int(count - 1));
		for (uint32_t i = 0u; i <= count; ++i) {
			const auto coeff = const_math::binomial(count - 1u, i);
			// is the coefficient large enough to produce a visible result?
			if ((sum_div * (long double)coeff) > min_contribution) {
				// if so, check how many usable values this row has now (should be >= desired tap count)
				if ((count - i * 2) < tap_count) {
					break;
				}
				return count;
			}
		}
	}
	return 0;
}

// computes the blur coefficients for the specified tap count (at compile-time)
template <uint32_t tap_count>
static constexpr auto compute_coefficients() {
	const_array<float, tap_count> ret {};
	
	// compute binomial coefficients and divide them by 2^(effective tap count - 1)
	// this is basically computing a row in pascal's triangle, using all values (or the middle part) as coefficients
	const auto effective_n = find_effective_n<tap_count>();
	const long double sum_div = 1.0L / (long double)const_math::pow(2ull, int(effective_n - 1));
	for (uint32_t i = 0u, k = (effective_n - tap_count) / 2u; i < tap_count; ++i, ++k) {
		// coefficient_i = (n choose k) / 2^n
		ret[i] = float(sum_div * (long double)const_math::binomial(effective_n - 1, k));
	}
	
	return ret;
}

//! the amount of search iterations we perform in gather kernels
//! TODO: this may be dependent on the screen size, needs more research
static constexpr const uint32_t gather_search_iterations { 6u };

kernel_2d() void libwarp_warp_gather_forward(const_image_2d<float> img_color,
											 const_image_2d<uint1> img_motion,
											 image_2d<float4, true> img_out_color,
											 param<float> delta) {
	screen_check();
	
	// iterate
	const int2 coord { global_id.xy };
	const float2 p_init = (float2(coord) + 0.5f) * warp_camera::inv_screen_size; // start at pixel center (this is p_t+alpha)
	float2 p_fwd = p_init;
	float4 fallback_color;
#pragma unroll
	for (uint32_t i = 0; i < gather_search_iterations; ++i) {
		const auto motion = decode_2d_motion(img_motion.read(p_fwd));
		p_fwd = p_init - delta * motion;
		fallback_color += (1.0f / float(gather_search_iterations)) * img_color.read_linear_repeat_mirrored(p_fwd);
	}
	
#if 0 // just read the sample, ignoring any error
	img_out_color.write(coord, img_color.read_linear(p_fwd));
#else // if screen-space error is too high, compute directional blur
	const auto motion_fwd = decode_2d_motion(img_motion.read(p_fwd));
	const auto err_fwd = ((p_fwd + delta * motion_fwd - p_init).dot() +
						  // account for out-of-bound access (-> large error so any checks will fail)
						  ((p_fwd < 0.0f).any() || (p_fwd > 1.0f).any() ? 1.0e10f : 0.0f));
	
	const float epsilon_1 { 0.00025f };
	const float epsilon_1_sq { epsilon_1 * epsilon_1 };
	float4 color;
	if (err_fwd >= epsilon_1_sq) {
		// compute directional blur in the motion direction of the pixel
		static constexpr const auto coeffs = compute_coefficients<TAP_COUNT>();
		static constexpr const int overlap = TAP_COUNT / 2;
		const auto dir = motion_fwd.normalized();
		
#pragma clang loop unroll_count(TAP_COUNT)
		for (int i = -overlap; i <= overlap; ++i) {
			// TODO: use linear sampling / blur
			color += coeffs[size_t(overlap + i)] * img_color.read(coord + int2(float(i) * dir));
		}
		color = (color + fallback_color) * 0.5f;
	} else {
		color = img_color.read_linear(p_fwd);
	}
	img_out_color.write(coord, color);
#endif
}

kernel_2d() void libwarp_warp_gather(const_image_2d<float> img_color,
									 depth_image_type img_depth,
									 const_image_2d<float> img_color_prev,
									 depth_image_type img_depth_prev,
									 const_image_2d<uint1> img_motion_forward,
									 const_image_2d<uint1> img_motion_backward,
									 // packed <forward depth: fwd t-1 -> t (used here), backward depth: bwd t-1 -> t-2 (unused here)>
									 const_image_2d<float2> img_motion_depth_forward,
									 // packed <forward depth: t+1 -> t (unused here), backward depth: t -> t-1 (used here)>
									 const_image_2d<float2> img_motion_depth_backward,
									 image_2d<float4, true> img_out_color,
									 param<float> delta) {
	screen_check();
	
	// iterate
	const float2 p_init = (float2(global_id.xy) + 0.5f) * warp_camera::inv_screen_size; // start at pixel center (this is p_t+alpha)
	// dual init, opposing init
	float2 p_fwd = p_init + delta * decode_2d_motion(img_motion_backward.read(p_init));
	float2 p_bwd = p_init + (1.0f - delta) * decode_2d_motion(img_motion_forward.read(p_init));
#pragma unroll
	for (uint32_t i = 0; i < gather_search_iterations; ++i) {
		const auto motion = decode_2d_motion(img_motion_forward.read(p_fwd));
		p_fwd = p_init - delta * motion;
	}
#pragma unroll
	for (uint32_t i = 0; i < gather_search_iterations; ++i) {
		const auto motion = decode_2d_motion(img_motion_backward.read(p_bwd));
		p_bwd = p_init - (1.0f - delta) * motion;
	}
	
	// read fwd/bwd color for the found pixel locations
	const auto color_fwd = img_color_prev.read_linear_repeat_mirrored(p_fwd);
	const auto color_bwd = img_color.read_linear_repeat_mirrored(p_bwd);
	
	// read final motion vector + depth (packed)
	const auto motion_fwd = decode_2d_motion(img_motion_forward.read(p_fwd));
	const auto motion_bwd = decode_2d_motion(img_motion_backward.read(p_bwd));
	const auto depth_fwd = img_motion_depth_forward.read(p_fwd).x;
	const auto depth_bwd = img_motion_depth_backward.read(p_bwd).y;
	
	// compute screen space error
	const auto err_fwd = ((p_fwd + delta * motion_fwd - p_init).dot() +
						  // account for out-of-bound access (-> large error so any checks will fail)
						  ((p_fwd < 0.0f).any() || (p_fwd > 1.0f).any() ? 1.0e10f : 0.0f));
	const auto err_bwd = ((p_bwd + (1.0f - delta) * motion_bwd - p_init).dot() +
						  ((p_bwd < 0.0f).any() || (p_bwd > 1.0f).any() ? 1.0e10f : 0.0f));
	// TODO: should have a more tangible epsilon, e.g. max pixel offset -> (max_offset / screen_size).max_element()
	const float epsilon_1 { 0.00025f };
	const float epsilon_1_sq { epsilon_1 * epsilon_1 };
	
	// NOTE: scene depth type is dependent on the renderer (-> use the default), motion depth is always z/w
	// -> need to linearize both to properly add + compare them
	const auto z_fwd = (warp_camera::linearize_depth(img_depth_prev.read(p_fwd)) +
						delta * warp_camera::linearize_depth<depth_type::z_div_w>(depth_fwd));
	const auto z_bwd = (warp_camera::linearize_depth(img_depth.read(p_bwd)) +
						(1.0f - delta) * warp_camera::linearize_depth<depth_type::z_div_w>(depth_bwd));
	const auto depth_diff = abs(z_fwd - z_bwd);
	constexpr const float epsilon_2 { 2.0f }; // aka "max depth difference between fwd and bwd"
	
	// check if fwd/bwd pass the screen-space error check
	const bool fwd_valid = (err_fwd < epsilon_1_sq);
	const bool bwd_valid = (err_bwd < epsilon_1_sq);
	// interpolation between fwd/bwd color and back-projection/forward-projection from the other color frame using the fwd/bwd motion
	const auto proj_color_fwd = color_fwd.interpolated(img_color.read_linear_repeat_mirrored(p_fwd + motion_fwd), delta);
	const auto proj_color_bwd = img_color_prev.read_linear_repeat_mirrored(p_bwd + motion_bwd).interpolated(color_bwd, delta);
	float4 color;
	if (fwd_valid && bwd_valid) {
		if (depth_diff < epsilon_2) {
			// case 1: both fwd and bwd are valid
			color = (err_fwd < err_bwd ? proj_color_fwd : proj_color_bwd);
		} else {
			// case 2: select the one closer to the camera (occlusion)
			if (z_fwd < z_bwd) {
				// depth from other frame
				const auto z_fwd_other = (img_depth.read(p_fwd + motion_fwd) +
										  (1.0f - delta) * img_motion_depth_backward.read(p_fwd + motion_fwd).y);
				color = (abs(z_fwd - z_fwd_other) < epsilon_2 ? proj_color_fwd : color_fwd);
			} else { // bwd < fwd
				const auto z_bwd_other = (img_depth_prev.read(p_bwd + motion_bwd) +
										  delta * img_motion_depth_forward.read(p_bwd + motion_bwd).x);
				color = (abs(z_bwd - z_bwd_other) < epsilon_2 ? proj_color_bwd : color_bwd);
			}
		}
	} else if (fwd_valid) {
		color = color_fwd;
	} else if (bwd_valid) {
		color = color_bwd;
	}
	// case 3 / else: both are invalid -> just do a linear interpolation between the two
	else {
		color = color_fwd.interpolated(color_bwd, delta);
	}
	
	img_out_color.write(global_id.xy, color);
}

kernel_2d() void libwarp_single_px_fixup(image_2d<float4> warp_img) {
	screen_check();
	
	const int2 coord { global_id.xy };
	const auto color = warp_img.read(coord);
	
	// 0 if it hasn't been written (needs fixup), 1 if it has been written
	if (color.w >= 1.0f) {
		return;
	}
	
	// sample pixels around
	const float4 colors[] {
		// read with offset if possible
		warp_img.read_repeat_mirrored(coord, int2 { 0, -1 }),
		warp_img.read_repeat_mirrored(coord, int2 { 1, 0 }),
		warp_img.read_repeat_mirrored(coord, int2 { 0, 1 }),
		warp_img.read_repeat_mirrored(coord, int2 { -1, 0 }),
	};
	
	float3 avg;
	float sum = 0.0f;
#pragma unroll
	for (const auto& col : colors) {
		// .w is 1.0f if valid, 0.0f if not (just multiply and add instead of checking)
		avg += col.xyz * col.w;
		sum += col.w;
	}
	avg *= 1.0f / sum;
	
	// write new averaged color
	warp_img.write(coord, float4 { avg, 1.0f /* pretend this is a valid pixel now */ });
}

kernel_2d() void libwarp_img_clear(image_2d<float4, true> img,
								   param<float4> clear_color) {
	screen_check();
	img.write(global_id.xy, float4 { clear_color.xyz, 0.0f });
}

kernel_2d() void libwarp_debug_depth_output(const_image_2d_depth<float> depth_in,
											image_2d<float4> output) {
	auto depth = warp_camera::linearize_depth(depth_in.read(global_id.xy));
	depth = math::fmod(depth, 1.0f);
	output.write(global_id.xy, { depth, depth, depth, 1.0f });
}

kernel_2d() void libwarp_debug_motion_2d_output(const_image_2d<uint1> motion_in,
												image_2d<float4> output) {
	const auto motion = decode_2d_motion(motion_in.read(global_id.xy));
	output.write(global_id.xy, { math::abs(motion.x), math::abs(motion.y), 0.0f, 1.0f });
}

kernel_2d() void libwarp_debug_motion_3d_output(const_image_2d<uint1> motion_in,
												image_2d<float4> output) {
	const auto encoded_motion = motion_in.read(global_id.xy);
	auto motion = decode_3d_motion(encoded_motion);
	motion /= 64.0f;
	motion.abs();
	output.write(global_id.xy, { motion.x, motion.y, motion.z, 1.0f });
}

kernel_2d() void libwarp_debug_motion_depth_output(const_image_2d<float2> motion_depth_in,
												   image_2d<float4> output) {
	auto motion_depth = motion_depth_in.read(global_id.xy);
	
	static constexpr const float2 near_far_plane { LIBWARP_NEAR_PLANE, LIBWARP_FAR_PLANE };
	motion_depth = motion_depth + near_far_plane.x - (motion_depth * (near_far_plane.x / near_far_plane.y));
	
	motion_depth = (motion_depth % 0.0005f) * 2000.0f;
	output.write(global_id.xy, { motion_depth.x, motion_depth.y, 0.0f, 1.0f });
}

#endif // FLOOR_COMPUTE

#endif // __LIBWARP_WARP_KERNELS_HPP__
