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

#include "libwarp_internal.hpp"

#if defined(FLOOR_COMPUTE_HOST)
#include <libwarp/warp_kernels.hpp>
#endif

unique_ptr<libwarp_state_struct> libwarp_state;
safe_mutex libwarp_lock;

LIBWARP_ERROR_CODE libwarp_init() {
	if (!libwarp_state) {
		const bool init_libfloor = !floor::is_initialized();
		if (init_libfloor) {
			if (!floor::init(floor::init_state {
				.call_path = "",
				.data_path = "data/",
				.app_name = "libwarp",
				.console_only = true,
				.renderer = floor::RENDERER::NONE,
			})) {
				return LIBWARP_FLOOR_INIT_FAILURE;
			}
		}
		
		atexit([] {
			const auto destroy_libfloor = (libwarp_state && libwarp_state->did_init_libfloor);
			libwarp_state = nullptr;
			if (destroy_libfloor) {
				floor::destroy();
			}
		});
		libwarp_state = make_unique<libwarp_state_struct>();
		libwarp_state->did_init_libfloor = init_libfloor;
		
		// get compute context + device + create queue for it
		libwarp_state->ctx = floor::get_compute_context();
		if(libwarp_state->ctx == nullptr) return LIBWARP_NO_CONTEXT;
		libwarp_state->dev = libwarp_state->ctx->get_device(compute_device::TYPE::FASTEST);
		if(libwarp_state->dev == nullptr) return LIBWARP_NO_DEVICE;
		libwarp_state->dev_queue = libwarp_state->ctx->create_queue(*libwarp_state->dev);
		if(libwarp_state->dev_queue == nullptr) return LIBWARP_NO_QUEUE;
		
		// check if device supports 1024 work-items and tile-size of 32*32px (use it, if so)
		if(libwarp_state->dev->max_total_local_size == 1024 &&
		   libwarp_state->dev->max_local_size.x >= 32 &&
		   libwarp_state->dev->max_local_size.y >= 32 &&
		   // host-compute tile size is fixed
		   libwarp_state->ctx->get_compute_type() != COMPUTE_TYPE::HOST) {
			libwarp_state->tile_size = { 32, 32 };
		}
		
		// init done
		return LIBWARP_SUCCESS;
	}
	return LIBWARP_SUCCESS;
}

void libwarp_cleanup() REQUIRES(!libwarp_lock) {
	GUARD(libwarp_lock);
	if (libwarp_state == nullptr) return;
	
	libwarp_state->programs.clear();
	
	libwarp_state->scatter.color = nullptr;
	libwarp_state->scatter.depth = nullptr;
	libwarp_state->scatter.motion = nullptr;
	libwarp_state->scatter.output = nullptr;
	libwarp_state->scatter.depth_buffer = nullptr;
	
	libwarp_state->gather_forward.color = nullptr;
	libwarp_state->gather_forward.motion = nullptr;
	libwarp_state->gather_forward.output = nullptr;
	
	libwarp_state->gather.color[0] = nullptr;
	libwarp_state->gather.color[1] = nullptr;
	libwarp_state->gather.depth[0] = nullptr;
	libwarp_state->gather.depth[1] = nullptr;
	libwarp_state->gather.motion[0] = nullptr;
	libwarp_state->gather.motion[1] = nullptr;
	libwarp_state->gather.motion[2] = nullptr;
	libwarp_state->gather.motion[3] = nullptr;
	libwarp_state->gather.motion_depth[0] = nullptr;
	libwarp_state->gather.motion_depth[1] = nullptr;
	libwarp_state->gather.output = nullptr;
	
	libwarp_state->debug.debug_output = nullptr;
	libwarp_state->debug.depth = nullptr;
	libwarp_state->debug.motion = nullptr;
	libwarp_state->debug.motion_depth = nullptr;
}

void libwarp_destroy() REQUIRES(!libwarp_lock) {
	GUARD(libwarp_lock);
	const auto destroy_libfloor = (libwarp_state && libwarp_state->did_init_libfloor);
	libwarp_state = nullptr;
	if (destroy_libfloor) {
		floor::destroy();
	}
}

pair<LIBWARP_ERROR_CODE, shared_ptr<libwarp_state_struct::camera_setup_program>>
libwarp_build(const libwarp_camera_setup* const camera_setup) {
	// just in case ...
	if(camera_setup->screen_width == 0 || camera_setup->screen_height == 0) {
		return { LIBWARP_INVALID_SCREEN_DIM, {} };
	}
	
	// check if prog already exists for this setup
	for(const auto& prog : libwarp_state->programs) {
		if(memcmp(&prog.first, camera_setup, sizeof(libwarp_camera_setup)) == 0) {
			// does already exist, return it
			return { LIBWARP_SUCCESS, prog.second };
		}
	}
	
	// build it
	auto program = make_shared<libwarp_state_struct::camera_setup_program>();
#if !defined(__WINDOWS__)
	const string kernel_file_name = "/opt/libwarp/include/libwarp/warp_kernels.hpp";
#else
	string kernel_file_name = core::expand_path_with_env("%ProgramW6432%/libwarp/include/libwarp/warp_kernels.hpp");
	if(!file_io::is_file(kernel_file_name)) {
		kernel_file_name = core::expand_path_with_env("%ProgramFiles%/libwarp/include/libwarp/warp_kernels.hpp");
	}
#endif

	program->program = libwarp_state->ctx->add_program_file(kernel_file_name,
															// camera setup
															" -DLIBWARP_SCREEN_WIDTH=" + to_string(camera_setup->screen_width) +
															" -DLIBWARP_SCREEN_HEIGHT=" + to_string(camera_setup->screen_height) +
															" -DLIBWARP_SCREEN_FOV=" + to_string(camera_setup->field_of_view) + "f" +
															" -DLIBWARP_NEAR_PLANE=" + to_string(camera_setup->near_plane) + "f" +
															" -DLIBWARP_FAR_PLANE=" + to_string(camera_setup->far_plane) + "f" +
															" -DTILE_SIZE_X=" + to_string(libwarp_state->tile_size.x) +
															" -DTILE_SIZE_Y=" + to_string(libwarp_state->tile_size.y) +
															" -DDEFAULT_DEPTH_TYPE=" +
															(camera_setup->depth_type == LIBWARP_DEPTH_NORMALIZED ?
															 "depth_type::normalized" :
															 (camera_setup->depth_type == LIBWARP_DEPTH_Z_DIV_W ?
															  "depth_type::z_div_w" : "depth_type::linear")) +
															" -DNATIVE_DEPTH_IMAGE=" +
															(camera_setup->depth_type == LIBWARP_DEPTH_Z_DIV_W ? "0" : "1") +
															(camera_setup->is_screen_origin_top_left ?
															 " -DSCREEN_ORIGIN_LEFT_TOP=1" : " -DSCREEN_ORIGIN_LEFT_BOTTOM=1"));
	if(program == nullptr) return { LIBWARP_COMPILATION_FAILURE, {} };
	
	// retrieve kernels
	// NOTE: corresponds to WARP_KERNEL
	static const char* kernel_names[warp_kernel_count()] {
		"libwarp_warp_scatter_depth",
		"libwarp_warp_scatter_color",
		"libwarp_img_clear",
		"libwarp_single_px_fixup",
		"libwarp_warp_gather_forward",
		"libwarp_warp_gather",
		"libwarp_debug_depth_output",
		"libwarp_debug_motion_2d_output",
		"libwarp_debug_motion_3d_output",
		"libwarp_debug_motion_depth_output",
	};
	for(size_t i = 0; i < warp_kernel_count(); ++i) {
		program->kernels[i] = program->program->get_kernel(kernel_names[i]);
		if(program->kernels[i] == nullptr) {
			return { LIBWARP_NO_KERNEL, {} };
		}
	}
	libwarp_state->programs.emplace_back(*camera_setup, program);
	
	// success
	return { LIBWARP_SUCCESS, program };
}

LIBWARP_ERROR_CODE libwarp_prebuild(const libwarp_camera_setup* const camera_setup) REQUIRES(!libwarp_lock) {
	LIBWARP_INIT_AND_LOCK
	return libwarp_build(camera_setup).first;
}

LIBWARP_ERROR_CODE libwarp_scatter_floor(const libwarp_camera_setup* const camera_setup,
										 const float delta,
										 const bool clear_frame,
										 shared_ptr<compute_image> color_texture,
										 shared_ptr<compute_image> depth_texture,
										 shared_ptr<compute_image> motion_texture,
										 shared_ptr<compute_image> output_texture) REQUIRES(!libwarp_lock) {
	LIBWARP_INIT_AND_LOCK
	
	libwarp_state->scatter.color = color_texture;
	libwarp_state->scatter.depth = depth_texture;
	libwarp_state->scatter.motion = motion_texture;
	libwarp_state->scatter.output = output_texture;
	
	//
	const auto depth_buffer_size = sizeof(float) * camera_setup->screen_width * camera_setup->screen_height;
	if(libwarp_state->scatter.depth_buffer == nullptr ||
	   libwarp_state->scatter.depth_buffer->get_size() < depth_buffer_size) {
		libwarp_state->scatter.depth_buffer = libwarp_state->ctx->create_buffer(*libwarp_state->dev_queue, depth_buffer_size);
		if(libwarp_state->scatter.depth_buffer == nullptr) {
			return LIBWARP_DEPTH_BUFFER_FAILURE;
		}
	}
	
	// finally: exec kernels
	auto err = LIBWARP_SUCCESS;
	if(clear_frame) {
		err = run_warp_kernel<KERNEL_SCATTER_CLEAR>(camera_setup, delta);
	}
	if(err == LIBWARP_SUCCESS) {
		err = run_warp_kernel<KERNEL_SCATTER_DEPTH_PASS>(camera_setup, delta);
	}
	if(err == LIBWARP_SUCCESS) {
		err = run_warp_kernel<KERNEL_SCATTER_COLOR_DEPTH_TEST>(camera_setup, delta);
	}
	if(err == LIBWARP_SUCCESS) {
		err = run_warp_kernel<KERNEL_SCATTER_FIXUP>(camera_setup, delta);
	}
	return err;
}

LIBWARP_ERROR_CODE libwarp_gather_floor(const libwarp_camera_setup* const camera_setup,
										const float delta,
										shared_ptr<compute_image> color_current_texture,
										shared_ptr<compute_image> depth_current_texture,
										shared_ptr<compute_image> color_prev_texture,
										shared_ptr<compute_image> depth_prev_texture,
										shared_ptr<compute_image> motion_forward_texture,
										shared_ptr<compute_image> motion_backward_texture,
										shared_ptr<compute_image> motion_depth_forward_texture,
										shared_ptr<compute_image> motion_depth_backward_texture,
										shared_ptr<compute_image> output_texture) REQUIRES(!libwarp_lock) {
	LIBWARP_INIT_AND_LOCK
	
	// gather swaps images every other frame, so determine which set to use
	uint32_t img_set = 0;
	if(libwarp_state->gather.color[0] != nullptr &&
	   libwarp_state->gather.color[0] != color_current_texture) {
		img_set = 1; // use second set
	}
	
	libwarp_state->gather.color[img_set] = color_current_texture;
	libwarp_state->gather.depth[img_set] = depth_current_texture;
	libwarp_state->gather.color[1u - img_set] = color_prev_texture;
	libwarp_state->gather.depth[1u - img_set] = depth_prev_texture;
	libwarp_state->gather.motion_depth[img_set] = motion_depth_forward_texture;
	libwarp_state->gather.motion_depth[1u - img_set] = motion_depth_backward_texture;
	libwarp_state->gather.motion[img_set * 2] = motion_forward_texture;
	libwarp_state->gather.motion[img_set * 2 + 1] = motion_backward_texture;
	libwarp_state->gather.output = output_texture;
	
	// exec kernel
	return run_warp_kernel<KERNEL_GATHER_BIDIRECTIONAL>(camera_setup, delta, img_set);
}

LIBWARP_ERROR_CODE libwarp_gather_forward_only_floor(const libwarp_camera_setup* const camera_setup,
													 const float delta,
													 shared_ptr<compute_image> color_texture,
													 shared_ptr<compute_image> motion_texture,
													 shared_ptr<compute_image> output_texture) REQUIRES(!libwarp_lock) {
	LIBWARP_INIT_AND_LOCK
	
	libwarp_state->gather_forward.color = color_texture;
	libwarp_state->gather_forward.motion = motion_texture;
	libwarp_state->gather_forward.output = output_texture;

	// exec kernel
	return run_warp_kernel<KERNEL_GATHER_FORWARD_ONLY>(camera_setup, delta);
}
