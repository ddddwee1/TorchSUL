#include <torch/all.h>
#include <torch/python.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <ATen/cuda/Atomic.cuh>

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_pixel_and_weights(
	const scalar_t * value,
	const scalar_t x, const scalar_t y, const int H, const int W, const int B, const int C){
	// scalar_t ltx = floor(x+0.00001);
	// scalar_t lty = floor(y+0.00001);
	// scalar_t rtx = ceil(x+0.00001);
	// scalar_t rty = floor(y+0.00001);
	// scalar_t lbx = floor(x+0.00001);
	// scalar_t lby = ceil(y+0.00001);
	// scalar_t rbx = ceil(x+0.00001);
	// scalar_t rby = ceil(y+0.00001);
	const int lower_x = floor(x);
	const int upper_x = lower_x + 1;
	const int lower_y = floor(y);
	const int upper_y = lower_y + 1;

	// printf("Coord: %d %d %d %d\n",int(ltx), int(lty), int(rtx), int(rty));

	if (lower_x<0 || upper_x>(W-1) || lower_y<0 || upper_y>(H-1)){
		return 0;
	}

	// const scalar_t lt, rt, lb, rb;
	// lt = value[B][C][lower_y][lower_x];
	// rt = value[B][C][lower_y][upper_x];
	// lb = value[B][C][upper_y][lower_x];
	// rb = value[B][C][upper_y][upper_x];
	const scalar_t lt = * (value);
	const scalar_t rt = * (value+1);
	const scalar_t lb = * (value+W);
	const scalar_t rb = * (value+W+1);

	// const scalar_t w1,w2,w3,w4;
	const scalar_t w1 = (upper_x - x) * (upper_y - y);
	const scalar_t w2 = (x - lower_x) * (upper_y - y);
	const scalar_t w3 = (upper_x - x) * (y - lower_y);
	const scalar_t w4 = (x - lower_x) * (y - lower_y);

	const scalar_t result = lt*w1 + rt*w2 + lb*w3 + rb*w4;

	// printf("Value:\t%f %f %f %f\nWeight:\t%f %f %f %f\n", lt, rt, lb, rb, w1, w2, w3, w4);

	// return {lt, rt, lb, rb, w1, w2, w3, w4};
	// return lt*w1 + rt*w2 + lb*w3 + rb*w4;
	// return 0;
	// return lt * w1;
	return result;
}

template <typename scalar_t>
__global__ void dcn_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> value,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid, 
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> mask,
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> result, 
	const int H, const int W, const int Hout, const int Wout, const int Pmax, const int Cmax, const int Gmax){

	// we put channel and batch to block here. Combining to thread will boost speed?
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
			const scalar_t xx = *(grid_ptr) * W;
			const scalar_t yy = *(grid_ptr+1) * H;
			v += *(weight_ptr+j) * get_pixel_and_weights(&value[B][C][y][x], xx, yy, H, W, B, C);
		}
		*&result[B][C][y][x] = v;
	}
}


torch::Tensor dcn_cuda(torch::Tensor value, torch::Tensor grid, torch::Tensor mask){
	// we use the grid within range 0-1 
	// 
	// AT_ASSERTM(value.size(0)==grid.size(0), "Batch size of value and grid should be the same");
	// AT_ASSERTM(mask.size(0)==grid.size(0), "Batch size of mask and grid should be the same");
	// AT_ASSERTM(value.size(1)==grid.size(3), "Group number of value and grid should be the same");
	// AT_ASSERTM(value.size(1)==mask.size(3), "Group number of value and mask should be the same");
	// AT_ASSERTM(mask.size(4)==grid.size(4), "Point number of mask and grid should be the same");
	// AT_ASSERTM(mask.size(1)==grid.size(1), "Height of mask and grid should be the same");
	// AT_ASSERTM(mask.size(2)==grid.size(2), "Width of mask and grid should be the same");
	const int B = value.size(0);
	const int C = value.size(1);
	const int H = value.size(2);
	const int W = value.size(3);
	const int Hout = grid.size(1);
	const int Wout = grid.size(2);
	// const int P = grid.size(4);
	const int P=9;
	const int G=4;

	const int threads = 1024;
	const dim3 blocks(C,B);

	auto result = torch::zeros({B, C, Hout, Wout}, value.options());
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "generate grid sample", ([&] {
		dcn_cuda_kernel<scalar_t><<<blocks, threads>>>(
			value.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			mask.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			result.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
			H, W, Hout, Wout, P, C/G, G);
	}));
	return result;
}
