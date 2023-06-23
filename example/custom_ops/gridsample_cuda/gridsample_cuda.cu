#include <torch/all.h>
#include <torch/python.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_pixel_and_weights(
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> value,
	scalar_t x, scalar_t y, int H, int W, const int B, const int C){
	scalar_t ltx = floor(x+0.00001);
	scalar_t lty = floor(y+0.00001);
	scalar_t rtx = ceil(x+0.00001);
	scalar_t rty = floor(y+0.00001);
	scalar_t lbx = floor(x+0.00001);
	scalar_t lby = ceil(y+0.00001);
	scalar_t rbx = ceil(x+0.00001);
	scalar_t rby = ceil(y+0.00001);

	// printf("Coord: %d %d %d %d\n",int(ltx), int(lty), int(rtx), int(rty));

	if (ltx<0 || rbx>(W-1) || lty<0 || lty>(H-1)){
		return 0;
	}

	scalar_t lt, rt, lb, rb;
	lt = value[B][C][int(lty)][int(ltx)];
	rt = value[B][C][int(rty)][int(rtx)];
	lb = value[B][C][int(lby)][int(lbx)];
	rb = value[B][C][int(rby)][int(rbx)];

	scalar_t w1,w2,w3,w4;
	w1 = (rbx - x) * (rby - y);
	w2 = (x - ltx) * (rby - y);
	w3 = (rbx - x) * (y - lty);
	w4 = (x - ltx) * (y - lty);

	// printf("Value:\t%f %f %f %f\nWeight:\t%f %f %f %f\n", lt, rt, lb, rb, w1, w2, w3, w4);

	// return {lt, rt, lb, rb, w1, w2, w3, w4};
	return lt*w1 + rt*w2 + lb*w3 + rb*w4;
}

template <typename scalar_t>
__global__ void gridsample_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> value,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid, 
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> result, 
	const int H, const int W, const int Hout, const int Wout){

	// we put channel and batch to block here. Combining to thread will boost speed?
	const int B = blockIdx.y;
	const int C = blockIdx.x;
	const int N = blockDim.x;
	const int n_thread = threadIdx.x;

	int i;
	scalar_t x, y, v;
	// std::vector<scalar_t> buff;
	for (i=n_thread; i<Hout*Wout; i=i+N){
		x = n_thread%Wout; 
		y = n_thread/Wout;
		v = get_pixel_and_weights(value, grid[B][y][x][0]*(W-1), grid[B][y][x][1]*(H-1), H, W, B, C);
		// printf("%f\n", v);
		// v = buff[0]*buff[4] + buff[1]*buff[5] + buff[2]*buff[6] + buff[3]*buff[7];
		result[B][C][y][x] = v;
	}
}

torch::Tensor gridsample_cuda(torch::Tensor value, torch::Tensor grid){
	// we use the grid within range 0-1 
	// value: [B,C,H,W]
	// grid: [B,Hout,Wout,2]
	// 
	AT_ASSERTM(value.size(0)==grid.size(0), "Batch size of value and grid should be the same");
	const int B = value.size(0);
	const int C = value.size(1);
	const int H = value.size(2);
	const int W = value.size(3);
	const int Hout = grid.size(1);
	const int Wout = grid.size(2);

	const int threads = 1024;
	const dim3 blocks(C,B);

	auto result = torch::zeros({B, C, Hout, Wout}, value.options());
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "generate grid sample", ([&] {
		gridsample_cuda_kernel<scalar_t><<<blocks, threads>>>(
			value.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			result.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
			H, W, Hout, Wout);
	}));
	return result;
}
