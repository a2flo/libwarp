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

#ifndef __LIBWARP_H__
#define __LIBWARP_H__

#include <stdint.h>

#if defined(__APPLE__) && defined(__OBJC__)
#include <Metal/MTLTexture.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif
	
	//! libwarp error/return codes
	typedef enum {
		//! all sucessful, no errors
		LIBWARP_SUCCESS					= 0,
		//! an (unknown) error occured
		LIBWARP_ERROR					= 1,
		//! failed to create a compute context
		LIBWARP_NO_CONTEXT				= 2,
		//! failed to acquire a compute device
		LIBWARP_NO_DEVICE				= 3,
		//! failed to create a compute queue
		LIBWARP_NO_QUEUE				= 4,
		//! failed to compile the warp program
		LIBWARP_COMPILATION_FAILURE		= 5,
		//! failed to retrieve or compile a warp kernel
		LIBWARP_NO_KERNEL				= 6,
		//! specified screen width/height are invalid
		LIBWARP_INVALID_SCREEN_DIM		= 7,
		//! failed to wrap an opengl or metal texture
		LIBWARP_IMAGE_WRAP_FAILURE		= 8,
		//! failed to acquire an image for compute use
		LIBWARP_IMAGE_ACQUIRE_FAILURE	= 9,
		//! failed to release an image from compute use
		LIBWARP_IMAGE_RELEASE_FAILURE	= 10,
		//! failed to create scatter depth buffer
		LIBWARP_DEPTH_BUFFER_FAILURE	= 11,
	} LIBWARP_ERROR_CODE;
	
	//! determines how depth values in the depth buffer should be interpreted
	typedef enum {
		//! normalized in [0, 1], default for OpenGL and Metal
		LIBWARP_DEPTH_NORMALIZED,
		//! z/w depth (manually written to a R32F texture in the shader)
		LIBWARP_DEPTH_Z_DIV_W,
		//! linear depth [0, far-plane]
		LIBWARP_DEPTH_LINEAR,
		//! log depth, computed in software (not supported yet)
		//LIBWARP_DEPTH_LOG,
	} LIBWARP_DEPTH_TYPE;
	
	//! all necessary camera state
	//! NOTE: program/kernels will be recompiled when this changes
	typedef struct libwarp_camera_setup {
		uint32_t screen_width;
		uint32_t screen_height;
		float field_of_view;
		float near_plane;
		float far_plane;
		LIBWARP_DEPTH_TYPE depth_type;
	} libwarp_camera_setup;
	
	//! scatter-based warping for use with OpenGL and CUDA/OpenCL/Host-Compute
	//! NOTE: whether this uses forward-predicted motion or backwards-correct motion,
	//!       solely depends on the data in the motion texture (not determined here)
	LIBWARP_ERROR_CODE libwarp_scatter(const libwarp_camera_setup* const camera_setup,
									   const float delta,
									   const uint32_t color_texture,
									   const uint32_t depth_texture,
									   const uint32_t motion_texture,
									   const uint32_t output_texture);
	
	//! gather-based warping for use with OpenGL and CUDA/OpenCL/Host-Compute
	//! NOTE: bidirectional warping
	LIBWARP_ERROR_CODE libwarp_gather(const libwarp_camera_setup* const camera_setup,
									  const float delta,
									  const uint32_t color_current_texture,
									  const uint32_t depth_current_texture,
									  const uint32_t color_prev_texture,
									  const uint32_t depth_prev_texture,
									  const uint32_t motion_forward_texture,
									  const uint32_t motion_backward_texture,
									  const uint32_t motion_depth_forward_texture,
									  const uint32_t motion_depth_backward_texture,
									  const uint32_t output_texture);
	
	//! gather-based warping for use with OpenGL and CUDA/OpenCL/Host-Compute
	//! NOTE: forward-only warping
	LIBWARP_ERROR_CODE libwarp_gather_forward_only(const libwarp_camera_setup* const camera_setup,
												   const float delta,
												   const uint32_t color_texture,
												   const uint32_t motion_texture,
												   const uint32_t output_texture);
	
#if defined(__APPLE__) && defined(__OBJC__)
	//! scatter-based warping for use with Metal
	//! NOTE: whether this uses forward-predicted motion or backwards-correct motion,
	//!       solely depends on the data in the motion texture (not determined here)
	LIBWARP_ERROR_CODE libwarp_scatter_metal(const libwarp_camera_setup* const camera_setup,
											 const float delta,
											 id <MTLTexture> color_texture,
											 id <MTLTexture> depth_texture,
											 id <MTLTexture> motion_texture,
											 id <MTLTexture> output_texture);
	
	//! gather-based warping for use with Metal
	//! NOTE: bidirectional warping
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
											id <MTLTexture> output_texture);
	
	//! gather-based warping for use with Metal
	//! NOTE: forward-only warping
	LIBWARP_ERROR_CODE libwarp_gather_forward_only_metal(const libwarp_camera_setup* const camera_setup,
														 const float delta,
														 id <MTLTexture> color_texture,
														 id <MTLTexture> motion_texture,
														 id <MTLTexture> output_texture);
#endif
	
	//! optional helper function that can be used to pre-build a program for the specified camera setup
	LIBWARP_ERROR_CODE libwarp_prebuild(const libwarp_camera_setup* const camera_setup);
	
	//! optional helper function that can be used to clear any run-time state
	void libwarp_cleanup();
	
	//! optional helper function that can be used to fully reset libwarp
	void libwarp_reset();
	
#if defined(__cplusplus)
}
#endif

#define LIBWARP_VERSION_STRINGIFY(ver) #ver
#define LIBWARP_VERSION_EVAL(ver) LIBWARP_VERSION_STRINGIFY(ver)

// <major>.<minor>.<revision><dev_stage>
#define LIBWARP_MAJOR_VERSION 0
#define LIBWARP_MINOR_VERSION 1
#define LIBWARP_REVISION_VERSION 0
#define LIBWARP_DEV_STAGE_VERSION 0xb3
#define LIBWARP_DEV_STAGE_VERSION_STR "b3"

#define LIBWARP_MAJOR_VERSION_STR LIBWARP_VERSION_EVAL(LIBWARP_MAJOR_VERSION)
#define LIBWARP_MINOR_VERSION_STR LIBWARP_VERSION_EVAL(LIBWARP_MINOR_VERSION)
#define LIBWARP_REVISION_VERSION_STR LIBWARP_VERSION_EVAL(LIBWARP_REVISION_VERSION)

// compatability version
#define LIBWARP_COMPAT_VERSION LIBWARP_MAJOR_VERSION_STR "." LIBWARP_MINOR_VERSION_STR "." LIBWARP_REVISION_VERSION_STR
// full version with current development stage info
#define LIBWARP_FULL_VERSION LIBWARP_COMPAT_VERSION LIBWARP_DEV_STAGE_VERSION_STR
// full version with dev stage, build time and build date
#define LIBWARP_VERSION_STRING "libwarp v" LIBWARP_FULL_VERSION " (" __TIME__ " " __DATE__ ")"

#endif
