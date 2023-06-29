#include <torch/all.h>
#include <torch/python.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <ATen/cuda/Atomic.cuh>

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_pixel_and_weights(
	const scalar_t * value_ptr,
	const scalar_t x_diff, const scalar_t y_diff, const int H, const int W, const int B, const int C){
	const scalar_t lt = * (value_ptr);
	const scalar_t rt = * (value_ptr+1);
	const scalar_t lb = * (value_ptr+W);
	const scalar_t rb = * (value_ptr+W+1);

	const scalar_t w1 = (1 - x_diff) * (1 - y_diff);             //(lower_x + 1 - x) * (lower_y + 1 - y);
	const scalar_t w2 = x_diff * (1 - y_diff);
	const scalar_t w3 = (1 - x_diff) * y_diff;
	const scalar_t w4 = x_diff * y_diff;

	const scalar_t result = lt*w1 + rt*w2 + lb*w3 + rb*w4;

	return result;
}

template <typename scalar_t>
__global__ void dcn_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> value,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid, 
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> mask,
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> result, 
	const int H, const int W, const int Hout, const int Wout, const int Pmax, const int Cmax){

	const int B = blockIdx.y;
	const int N = blockDim.x;
	const int n_thread = threadIdx.x;
	const int C = blockIdx.x;
	const int G = C/Cmax;

	int i,j,k,x,y;
	scalar_t v;
	
	for (i=n_thread; i<Hout*Wout; i=i+N){
		x = i%Wout;
		y = i/Wout;
		v = 0;
		const scalar_t *weight_ptr = &mask[B][y][x][G*Pmax];
		const scalar_t *grid_ptr = &grid[B][y][x][G*Pmax*2];

		for (j=0;j<Pmax;j++){
			const scalar_t xx = *(grid_ptr) * W + x;
			const scalar_t yy = *(grid_ptr+1) * H + y;
			const int lower_x = floor(xx);
			const int upper_x = lower_x + 1;
			const int lower_y = floor(yy);
			const int upper_y = lower_y + 1;
			const scalar_t x_diff = xx - lower_x;
			const scalar_t y_diff = yy - lower_y;
			if (lower_x>=0 && upper_x<=(W-1) && lower_y>=0 && upper_y<=(H-1)){
				v += *(weight_ptr+j) * get_pixel_and_weights(&value[B][C][lower_y][lower_x], x_diff, y_diff, H, W, B, C);
			}
		}
		*&result[B][C][y][x] = v;
	}
}


torch::Tensor dcn_cuda(const torch::Tensor value, const torch::Tensor offset, const torch::Tensor mask,\
						 const int P, const int G){
	// we use the grid within range 0-1 
	const int B = value.size(0);
	const int C = value.size(1);
	const int H = value.size(2);
	const int W = value.size(3);
	const int Hout = offset.size(1);
	const int Wout = offset.size(2);


	const int threads = 1024;
	const dim3 blocks(C,B);

	auto result = torch::zeros({B, C, Hout, Wout}, value.options());
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "generate grid sample", ([&] {
		dcn_cuda_kernel<scalar_t><<<blocks, threads>>>(
			value.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			offset.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			mask.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			result.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
			H, W, Hout, Wout, P, C/G);
	}));
	return result;
}
