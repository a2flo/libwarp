/*
 *  libwarp
 *  Copyright (C) 2015 - 2019 Florian Ziesche
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

#include <libwarp/libwarp.h>
#include <floor/floor/floor.hpp>
#include <floor/threading/thread_base.hpp>

#include "libwarp_internal.hpp"

#if defined(__APPLE__) && defined(__OBJC__)
#include <floor/compute/metal/metal_image.hpp>

floor_inline_always static bool libwarp_wrap_metal_texture(shared_ptr<compute_image>& img,
														   id <MTLTexture> metal_texture,
														   const bool read_write = false) {
	if(img == nullptr || ((metal_image*)img.get())->get_metal_image() != metal_texture) {
		img = make_shared<metal_image>(libwarp_state->dev, metal_texture, nullptr,
									   !read_write ? COMPUTE_MEMORY_FLAG::READ : COMPUTE_MEMORY_FLAG::READ_WRITE);
	}
	return (img != nullptr);
}

LIBWARP_ERROR_CODE libwarp_scatter_metal(const libwarp_camera_setup* const camera_setup,
										 const float delta,
										 const bool clear_frame,
										 id <MTLTexture> color_texture,
										 id <MTLTexture> depth_texture,
										 id <MTLTexture> motion_texture,
										 id <MTLTexture> output_texture) {
	LIBWARP_INIT_AND_LOCK;
	
	// wrap textures
	if(!libwarp_wrap_metal_texture(libwarp_state->scatter.color, color_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->scatter.depth, depth_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->scatter.motion, motion_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->scatter.output, output_texture, true)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	//
	const auto depth_buffer_size = sizeof(float) * camera_setup->screen_width * camera_setup->screen_height;
	if(libwarp_state->scatter.depth_buffer == nullptr ||
	   libwarp_state->scatter.depth_buffer->get_size() < depth_buffer_size) {
		libwarp_state->scatter.depth_buffer = libwarp_state->ctx->create_buffer(libwarp_state->dev, depth_buffer_size);
		if(libwarp_state->scatter.depth_buffer == nullptr) {
			return LIBWARP_DEPTH_BUFFER_FAILURE;
		}
	}
	
	// exec kernels
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

LIBWARP_ERROR_CODE libwarp_gather_metal(const libwarp_camera_setup* const camera_setup,
										const float delta,
										id <MTLTexture> color_current_texture,
										id <MTLTexture> depth_current_texture,
										id <MTLTexture> color_prev_texture,
										id <MTLTexture> depth_prev_texture,
										id <MTLTexture> motion_forward_texture,
										id <MTLTexture> motion_backward_texture,
										id <MTLTexture> motion_depth_forward_texture,
										id <MTLTexture> motion_depth_backward_texture,
										id <MTLTexture> output_texture) {
	LIBWARP_INIT_AND_LOCK;
	
	// wrap textures
	// gather swaps images every other frame, so determine which set to use
	uint32_t img_set = 0;
	if(libwarp_state->gather.color[0] != nullptr &&
	   ((metal_image*)libwarp_state->gather.color[0].get())->get_metal_image() != color_current_texture) {
		img_set = 1; // use second set
	}
	
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.color[img_set], color_current_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.depth[img_set], depth_current_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.color[1u - img_set], color_prev_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.depth[1u - img_set], depth_prev_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.motion_depth[img_set], motion_depth_forward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.motion_depth[1u - img_set], motion_depth_backward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.motion[img_set * 2], motion_forward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.motion[img_set * 2 + 1], motion_backward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	if(!libwarp_wrap_metal_texture(libwarp_state->gather.output, output_texture, true)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	// exec kernel
	return run_warp_kernel<KERNEL_GATHER_BIDIRECTIONAL>(camera_setup, delta, img_set);
}

LIBWARP_ERROR_CODE libwarp_gather_forward_only_metal(const libwarp_camera_setup* const camera_setup,
													 const float delta,
													 id <MTLTexture> color_texture,
													 id <MTLTexture> motion_texture,
													 id <MTLTexture> output_texture) {
	LIBWARP_INIT_AND_LOCK;
	
	// wrap textures
	if(!libwarp_wrap_metal_texture(libwarp_state->gather_forward.color, color_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather_forward.motion, motion_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_metal_texture(libwarp_state->gather_forward.output, output_texture, true)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	// exec kernel
	return run_warp_kernel<KERNEL_GATHER_FORWARD_ONLY>(camera_setup, delta);
}
#endif
