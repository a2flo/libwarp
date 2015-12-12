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

#include <libwarp/libwarp.h>
#include <floor/floor/floor.hpp>
#include <floor/threading/thread_base.hpp>

#if defined(FLOOR_COMPUTE_HOST)
#include <libwarp/warp_kernels.hpp>
#endif

#if defined(__APPLE__) && defined(__OBJC__)
#include <floor/compute/metal/metal_image.hpp>
#endif

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
static unique_ptr<libwarp_state_struct> libwarp_state;
// none of the libwarp functions are able to run concurrently, must protect them via a global lock
static safe_mutex libwarp_lock;

static LIBWARP_ERROR_CODE libwarp_init() {
	static once_flag did_init;
	call_once(did_init, [] {
		// TODO: need proper gl init for opencl
		floor::init("", "data/", true, "config.json", true);
		atexit([] {
			libwarp_state = nullptr;
			floor::destroy();
		});
		libwarp_state = make_unique<libwarp_state_struct>();
		
		// get compute context + device + create queue for it
		libwarp_state->ctx = floor::get_compute_context();
		if(libwarp_state->ctx == nullptr) return LIBWARP_NO_CONTEXT;
		libwarp_state->dev = libwarp_state->ctx->get_device(compute_device::TYPE::FASTEST);
		if(libwarp_state->dev == nullptr) return LIBWARP_NO_DEVICE;
		libwarp_state->dev_queue = libwarp_state->ctx->create_queue(libwarp_state->dev);
		if(libwarp_state->dev_queue == nullptr) return LIBWARP_NO_QUEUE;
		
		// check if device supports 1024 work-items and tile-size of 32*32px (use it, if so)
		if(libwarp_state->dev->max_work_group_size == 1024 &&
		   libwarp_state->dev->max_work_group_item_sizes.x >= 32 &&
		   libwarp_state->dev->max_work_group_item_sizes.y >= 32 &&
		   // host-compute tile size is fixed
		   libwarp_state->ctx->get_compute_type() != COMPUTE_TYPE::HOST) {
			libwarp_state->tile_size = { 32, 32 };
		}
		
		// init done
		return LIBWARP_SUCCESS;
	});
	return LIBWARP_SUCCESS;
}
#define LIBWARP_INIT_AND_LOCK GUARD(libwarp_lock); { \
	const auto err = libwarp_init(); if(err != LIBWARP_SUCCESS) { return err; } \
}

void libwarp_cleanup() {
	GUARD(libwarp_lock);
	if(libwarp_state == nullptr) return;
	
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
}

void libwarp_reset() {
	GUARD(libwarp_lock);
	libwarp_state = nullptr;
}

static pair<LIBWARP_ERROR_CODE, shared_ptr<libwarp_state_struct::camera_setup_program>>
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
	program->program = libwarp_state->ctx->add_program_source("",
															  // magic
															  ""s +
#if !defined(__WINDOWS__)
															  " -isystem /usr/local/include/libwarp/" +
															  " -isystem /usr/include/libwarp/" +
															  " -isystem /opt/libwarp/include/libwarp/" +
#else
															  // TODO: where/what to include?
#endif
															  " -include warp_kernels.hpp" +
															  // camera setup
															  " -DLIBWARP_SCREEN_WIDTH=" + to_string(camera_setup->screen_width) +
															  " -DLIBWARP_SCREEN_HEIGHT=" + to_string(camera_setup->screen_height) +
															  " -DLIBWARP_SCREEN_FOV=" + to_string(camera_setup->field_of_view) +
															  " -DLIBWARP_NEAR_PLANE=" + to_string(camera_setup->near_plane) +
															  " -DLIBWARP_FAR_PLANE=" + to_string(camera_setup->far_plane) +
															  " -DTILE_SIZE_X=" + to_string(libwarp_state->tile_size.x) +
															  " -DTILE_SIZE_Y=" + to_string(libwarp_state->tile_size.y) +
															  " -DDEFAULT_DEPTH_TYPE=" +
															  (camera_setup->depth_type == LIBWARP_DEPTH_NORMALIZED ?
															   "depth_type::normalized" :
															   (camera_setup->depth_type == LIBWARP_DEPTH_Z_DIV_W ?
																"depth_type::z_div_w" : "depth_type::linear")));
	if(program == nullptr) return { LIBWARP_COMPILATION_FAILURE, {} };
	
	// retrieve kernels
	// NOTE: corresponds to WARP_KERNEL
	static const char* kernel_names[warp_kernel_count()] {
		"warp_scatter_depth",
		"warp_scatter_color",
		"img_clear",
		"single_px_fixup",
		"warp_gather_forward",
		"warp_gather",
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

LIBWARP_ERROR_CODE libwarp_prebuild(const libwarp_camera_setup* const camera_setup) {
	LIBWARP_INIT_AND_LOCK;
	return libwarp_build(camera_setup).first;
}

template <WARP_KERNEL kernel_idx>
static LIBWARP_ERROR_CODE run_warp_kernel(const libwarp_camera_setup* const camera_setup,
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

floor_inline_always static bool libwarp_wrap_gl_texture(shared_ptr<compute_image>& img, const uint32_t gl_texture,
														const bool read_write = false) {
	if(img == nullptr || img->get_opengl_object() != gl_texture) {
		img = libwarp_state->ctx->wrap_image(libwarp_state->dev, gl_texture, GL_TEXTURE_2D,
											 !read_write ? COMPUTE_MEMORY_FLAG::READ : COMPUTE_MEMORY_FLAG::READ_WRITE);
	}
	return (img != nullptr);
}

LIBWARP_ERROR_CODE libwarp_scatter(const libwarp_camera_setup* const camera_setup,
								   const float delta,
								   const uint32_t color_texture,
								   const uint32_t depth_texture,
								   const uint32_t motion_texture,
								   const uint32_t output_texture) {
	LIBWARP_INIT_AND_LOCK;
	
	// wrap textures
	if(!libwarp_wrap_gl_texture(libwarp_state->scatter.color, color_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->scatter.depth, depth_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->scatter.motion, motion_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->scatter.output, output_texture, true)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	if(!libwarp_state->scatter.color->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->scatter.depth->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->scatter.motion->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->scatter.output->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	
	//
	const auto depth_buffer_size = sizeof(float) * camera_setup->screen_width * camera_setup->screen_height;
	if(libwarp_state->scatter.depth_buffer == nullptr ||
	   libwarp_state->scatter.depth_buffer->get_size() < depth_buffer_size) {
		libwarp_state->scatter.depth_buffer = libwarp_state->ctx->create_buffer(depth_buffer_size);
		if(libwarp_state->scatter.depth_buffer == nullptr) {
			return LIBWARP_DEPTH_BUFFER_FAILURE;
		}
	}
	
	// finally: exec kernels
	auto err = run_warp_kernel<KERNEL_SCATTER_DEPTH_PASS>(camera_setup, delta);
	if(err == LIBWARP_SUCCESS) {
		err = run_warp_kernel<KERNEL_SCATTER_COLOR_DEPTH_TEST>(camera_setup, delta);
	}
	// delay error reporting, so images can be properly released again
	
	if(!libwarp_state->scatter.color->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->scatter.depth->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->scatter.motion->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->scatter.output->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	
	return err;
}

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
								  const uint32_t output_texture) {
	LIBWARP_INIT_AND_LOCK;
	
	// wrap textures
	// gather swaps images every other frame, so determine which set to use
	uint32_t img_set = 0;
	if(libwarp_state->gather.color[0] != nullptr &&
	   libwarp_state->gather.color[0]->get_opengl_object() != color_current_texture) {
		img_set = 1; // use second set
	}
	
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.color[img_set], color_current_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.depth[img_set], depth_current_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.color[1u - img_set], color_prev_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.depth[1u - img_set], depth_prev_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.motion_depth[img_set], motion_depth_forward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.motion_depth[1u - img_set], motion_depth_backward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.motion[img_set * 2], motion_forward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.motion[img_set * 2 + 1], motion_backward_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	if(!libwarp_wrap_gl_texture(libwarp_state->gather.output, output_texture, true)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	//
	if(!libwarp_state->gather.color[0]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather.color[1]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather.depth[0]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather.depth[1]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather.motion_depth[0]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather.motion_depth[1]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	
	if(!libwarp_state->gather.motion[img_set * 2]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather.motion[img_set * 2 + 1]->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	
	if(!libwarp_state->gather.output->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	
	// exec kernel
	const auto err = run_warp_kernel<KERNEL_GATHER_BIDIRECTIONAL>(camera_setup, delta, img_set);
	// delay error reporting, so images can be properly released again
	
	
	if(!libwarp_state->gather.color[0]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather.color[1]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather.depth[0]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather.depth[1]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather.motion_depth[0]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather.motion_depth[1]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	
	if(!libwarp_state->gather.motion[img_set * 2]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather.motion[img_set * 2 + 1]->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	
	if(!libwarp_state->gather.output->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	
	return err;
}

LIBWARP_ERROR_CODE libwarp_gather_forward_only(const libwarp_camera_setup* const camera_setup,
											   const float delta,
											   const uint32_t color_texture,
											   const uint32_t motion_texture,
											   const uint32_t output_texture) {
	LIBWARP_INIT_AND_LOCK;
	
	// wrap textures
	if(!libwarp_wrap_gl_texture(libwarp_state->gather_forward.color, color_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather_forward.motion, motion_texture)) return LIBWARP_IMAGE_WRAP_FAILURE;
	if(!libwarp_wrap_gl_texture(libwarp_state->gather_forward.output, output_texture, true)) return LIBWARP_IMAGE_WRAP_FAILURE;
	
	if(!libwarp_state->gather_forward.color->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather_forward.motion->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	if(!libwarp_state->gather_forward.output->acquire_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_ACQUIRE_FAILURE;
	
	// exec kernel
	const auto err = run_warp_kernel<KERNEL_GATHER_FORWARD_ONLY>(camera_setup, delta);
	// delay error reporting, so images can be properly released again
	
	if(!libwarp_state->gather_forward.color->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather_forward.motion->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	if(!libwarp_state->gather_forward.output->release_opengl_object(libwarp_state->dev_queue)) return LIBWARP_IMAGE_RELEASE_FAILURE;
	
	return err;
}

#if defined(__APPLE__) && defined(__OBJC__)
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
		libwarp_state->scatter.depth_buffer = libwarp_state->ctx->create_buffer(depth_buffer_size);
		if(libwarp_state->scatter.depth_buffer == nullptr) {
			return LIBWARP_DEPTH_BUFFER_FAILURE;
		}
	}
	
	// exec kernels
	const auto err = run_warp_kernel<KERNEL_SCATTER_DEPTH_PASS>(camera_setup, delta);
	if(err != LIBWARP_SUCCESS) return err;
	return run_warp_kernel<KERNEL_SCATTER_COLOR_DEPTH_TEST>(camera_setup, delta);
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
