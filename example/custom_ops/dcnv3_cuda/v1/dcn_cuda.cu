#include <torch/all.h>
#include <torch/python.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_pixel_and_weights(
	const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> value,
	scalar_t x, scalar_t y, int H, int W, const int B, const int C, const int G){
	// scalar_t ltx = floor(x+0.00001);
	// scalar_t lty = floor(y+0.00001);
	// scalar_t rtx = ceil(x+0.00001);
	// scalar_t rty = floor(y+0.00001);
	// scalar_t lbx = floor(x+0.00001);
	// scalar_t lby = ceil(y+0.00001);
	// scalar_t rbx = ceil(x+0.00001);
	// scalar_t rby = ceil(y+0.00001);
	scalar_t lower_x = floor(x+0.00001);
	scalar_t upper_x = lower_x + 1;
	scalar_t lower_y = floor(y+0.00001);
	scalar_t upper_y = lower_y + 1;

	// printf("Coord: %d %d %d %d\n",int(ltx), int(lty), int(rtx), int(rty));

	if (lower_x<0 || upper_x>(W-1) || lower_y<0 || upper_y>(H-1)){
		return 0;
	}

	scalar_t lt, rt, lb, rb;
	lt = value[B][G][C][int(lower_y)][int(lower_x)];
	rt = value[B][G][C][int(lower_y)][int(upper_x)];
	lb = value[B][G][C][int(upper_y)][int(lower_x)];
	rb = value[B][G][C][int(upper_y)][int(upper_x)];

	scalar_t w1,w2,w3,w4;
	w1 = (upper_x - x) * (upper_y - y);
	w2 = (x - lower_x) * (upper_y - y);
	w3 = (upper_x - x) * (y - lower_y);
	w4 = (x - lower_x) * (y - lower_y);

	// printf("Value:\t%f %f %f %f\nWeight:\t%f %f %f %f\n", lt, rt, lb, rb, w1, w2, w3, w4);

	// return {lt, rt, lb, rb, w1, w2, w3, w4};
	return lt*w1 + rt*w2 + lb*w3 + rb*w4;
}

template <typename scalar_t>
__global__ void dcn_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> value,
	const torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> grid, 
	const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> mask,
	torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> result, 
	const int H, const int W, const int Hout, const int Wout, const int Pmax){

	// we put channel and batch to block here. Combining to thread will boost speed?
	const int B = blockIdx.y;
	const int C = blockIdx.x;
	const int G = blockIdx.z;
	const int N = blockDim.x;
	const int n_thread = threadIdx.x;

	int i,j;
	scalar_t x, y, v=0;
	// std::vector<scalar_t> buff;
	for (i=n_thread; i<Hout*Wout; i=i+N){
		x = i%Wout; 
		y = i/Wout;
		for (j=0;j<Pmax;j++){
			v += get_pixel_and_weights(value, grid[B][y][x][G][j][0]*(W-1), grid[B][y][x][G][j][1]*(H-1), H, W, B, C, G) * mask[B][y][x][G][j];
		}
		// printf("%f %d %d %d\n", v, B, C, G);
		// v = buff[0]*buff[4] + buff[1]*buff[5] + buff[2]*buff[6] + buff[3]*buff[7];
		result[B][G][C][y][x] = v;
	}
}

torch::Tensor dcn_cuda(torch::Tensor value, torch::Tensor grid, torch::Tensor mask){
	// we use the grid within range 0-1 
	// 
	AT_ASSERTM(value.size(0)==grid.size(0), "Batch size of value and grid should be the same");
	AT_ASSERTM(mask.size(0)==grid.size(0), "Batch size of mask and grid should be the same");
	AT_ASSERTM(value.size(1)==grid.size(3), "Group number of value and grid should be the same");
	AT_ASSERTM(value.size(1)==mask.size(3), "Group number of value and mask should be the same");
	AT_ASSERTM(mask.size(4)==grid.size(4), "Point number of mask and grid should be the same");
	AT_ASSERTM(mask.size(1)==grid.size(1), "Height of mask and grid should be the same");
	AT_ASSERTM(mask.size(2)==grid.size(2), "Width of mask and grid should be the same");
	const int B = value.size(0);
	const int G = value.size(1);
	const int C = value.size(2);
	const int H = value.size(3);
	const int W = value.size(4);
	const int Hout = grid.size(1);
	const int Wout = grid.size(2);
	const int P = grid.size(4);

	const int threads = 1024;
	const dim3 blocks(C,B,G);

	auto result = torch::zeros({B, G, C, Hout, Wout}, value.options());
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "generate grid sample", ([&] {
		dcn_cuda_kernel<scalar_t><<<blocks, threads>>>(
			value.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
			grid.packed_accessor32<scalar_t, 6, torch::RestrictPtrTraits>(),
			mask.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
			result.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
			H, W, Hout, Wout, P);
	}));
	return result;
}
