/*
 *  libwarp
 *  Copyright (C) 2015 Florian Ziesche
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

#ifndef __LIBWARP_INTERNAL_HPP__
#define __LIBWARP_INTERNAL_HPP__

#include <libwarp/libwarp.h>
#include <floor/floor/floor.hpp>
#include <floor/threading/thread_base.hpp>

//
enum WARP_KERNEL : uint32_t {
	KERNEL_SCATTER_DEPTH_PASS = 0,
	KERNEL_SCATTER_COLOR_DEPTH_TEST,
	KERNEL_SCATTER_CLEAR,
	KERNEL_SCATTER_FIXUP,
	KERNEL_GATHER_FORWARD_ONLY,
	KERNEL_GATHER_BIDIRECTIONAL,
	__MAX_WARP_KERNEL
};
floor_inline_always static constexpr size_t warp_kernel_count() {
	return (size_t)WARP_KERNEL::__MAX_WARP_KERNEL;
}

struct libwarp_state_struct {
	shared_ptr<compute_context> ctx;
	shared_ptr<compute_device> dev;
	shared_ptr<compute_queue> dev_queue;
	uint2 tile_size { 32, 16 }; // == 512 work-items which should work everywhere
	
	//
	struct camera_setup_program {
		shared_ptr<compute_program> program;
		array<shared_ptr<compute_kernel>, warp_kernel_count()> kernels;
	};
	vector<pair<libwarp_camera_setup, shared_ptr<camera_setup_program>>> programs;
	
	//
	struct {
		shared_ptr<compute_image> color;
		shared_ptr<compute_image> depth;
		shared_ptr<compute_image> motion;
		shared_ptr<compute_image> output;
		shared_ptr<compute_buffer> depth_buffer;
	} scatter;
	struct {
		shared_ptr<compute_image> color;
		shared_ptr<compute_image> motion;
		shared_ptr<compute_image> output;
	} gather_forward;
	struct {
		shared_ptr<compute_image> color[2];
		shared_ptr<compute_image> depth[2];
		shared_ptr<compute_image> motion[4];
		shared_ptr<compute_image> motion_depth[2];
		shared_ptr<compute_image> output;
	} gather;
};
// contains all global state, can simply be cleared by setting to nullptr
extern unique_ptr<libwarp_state_struct> libwarp_state;
// none of the libwarp functions are able to run concurrently, must protect them via a global lock
extern safe_mutex libwarp_lock;

// internal libwarp/floor init
LIBWARP_ERROR_CODE libwarp_init();

// macro voodoo, make sure everything is initialized and only one has access to the warp state
#define LIBWARP_INIT_AND_LOCK GUARD(libwarp_lock); { \
	const auto err = libwarp_init(); if(err != LIBWARP_SUCCESS) { return err; } \
}

// actually builds the warp program for a specific camera setup
pair<LIBWARP_ERROR_CODE, shared_ptr<libwarp_state_struct::camera_setup_program>>
libwarp_build(const libwarp_camera_setup* const camera_setup);

// runs the specified warp kernel, all inlined and DCE'ed
template <WARP_KERNEL kernel_idx>
floor_inline_always LIBWARP_ERROR_CODE run_warp_kernel(const libwarp_camera_setup* const camera_setup,
													   const float& delta,
													   const uint32_t img_set = 0) {
	// build program for this camera setup if it hasn't been build already
	const auto prog = libwarp_build(camera_setup);
	if(prog.first != LIBWARP_SUCCESS) return prog.first;
	
	// global work-size == round screen dim to tile size
	const auto global_work_size = uint2(camera_setup->screen_width,
										camera_setup->screen_height).rounded_next_multiple(libwarp_state->tile_size);
	
	switch(kernel_idx) {
		case KERNEL_SCATTER_DEPTH_PASS: {
			const float clear_depth = numeric_limits<float>::max();
			libwarp_state->scatter.depth_buffer->fill(libwarp_state->dev_queue, &clear_depth, sizeof(clear_depth));
			
			libwarp_state->dev_queue->execute(prog.second->kernels[KERNEL_SCATTER_DEPTH_PASS],
											  global_work_size,
											  libwarp_state->tile_size,
											  libwarp_state->scatter.depth,
											  libwarp_state->scatter.motion,
											  libwarp_state->scatter.depth_buffer,
											  delta);
			return LIBWARP_SUCCESS;
		}
		case KERNEL_SCATTER_COLOR_DEPTH_TEST: {
			libwarp_state->dev_queue->execute(prog.second->kernels[KERNEL_SCATTER_COLOR_DEPTH_TEST],
											  global_work_size,
											  libwarp_state->tile_size,
											  libwarp_state->scatter.color,
											  libwarp_state->scatter.depth,
											  libwarp_state->scatter.motion,
											  libwarp_state->scatter.output,
											  libwarp_state->scatter.depth_buffer,
											  delta);
			return LIBWARP_SUCCESS;
		}
		case KERNEL_GATHER_FORWARD_ONLY: {
			libwarp_state->dev_queue->execute(prog.second->kernels[KERNEL_GATHER_FORWARD_ONLY],
											  global_work_size,
											  libwarp_state->tile_size,
											  libwarp_state->gather_forward.color,
											  libwarp_state->gather_forward.motion,
											  libwarp_state->gather_forward.output,
											  delta);
			return LIBWARP_SUCCESS;
		}
		case KERNEL_GATHER_BIDIRECTIONAL: {
			libwarp_state->dev_queue->execute(prog.second->kernels[KERNEL_GATHER_BIDIRECTIONAL],
											  global_work_size,
											  libwarp_state->tile_size,
											  libwarp_state->gather.color[img_set],
											  libwarp_state->gather.depth[img_set],
											  libwarp_state->gather.color[1u - img_set],
											  libwarp_state->gather.depth[1u - img_set],
											  libwarp_state->gather.motion[img_set * 2],
											  libwarp_state->gather.motion[img_set * 2 + 1],
											  libwarp_state->gather.motion_depth[img_set],
											  libwarp_state->gather.motion_depth[1u - img_set],
											  libwarp_state->gather.output,
											  delta);
			return LIBWARP_SUCCESS;
		}
		default: break;
	}
	floor_unreachable();
}

#endif
