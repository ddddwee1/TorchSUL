#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <vector>
#include <ATen/cuda/Atomic.cuh>

template <typename scalar_t>
__device__ __forceinline__ scalar_t bilinear_func(const scalar_t * value, const scalar_t xx, const scalar_t yy, const int x_lower, const int y_lower, const int W){
	const scalar_t diff_x = xx - x_lower;
	const scalar_t diff_y = yy - y_lower;

	const scalar_t v1 = *(value);
	const scalar_t v2 = *(value+1);
	const scalar_t v3 = *(value+W);
	const scalar_t v4 = *(value+W+1);

	const scalar_t w1 = (1 - diff_x) * (1 - diff_y);
	const scalar_t w2 = diff_x * (1 - diff_y);
	const scalar_t w3 = (1 - diff_x) * diff_y;
	const scalar_t w4 = diff_x * diff_y;

	return v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4;
}


template <typename scalar_t>
__global__ void gridsample_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 4> value, 
	const torch::PackedTensorAccessor32<scalar_t, 4> grid, 
	torch::PackedTensorAccessor32<scalar_t, 4> out,
	const int H, const int W, const int Ho, const int Wo){

	const int B = blockIdx.x;
	const int C = blockIdx.y;
	const int n_thread = threadIdx.x;
	const int thread_stride = blockDim.x;

	int x,y;
	for (int i=n_thread; i<Ho*Wo; i+=thread_stride){
		x = i / Wo;
		y = i % Wo;
		const scalar_t * grid_ptr = & grid[B][y][x][0];
		const scalar_t xx = *(grid_ptr) * (W-1);
		const scalar_t yy = *(grid_ptr+1) * (H-1);

		if (xx>=0 && xx<W-1 && yy>=0 && yy<H-1){
			const int y_lower = floor(yy);
			const int x_lower = floor(xx);
			const scalar_t * value_ptr = &value[B][C][y_lower][x_lower];
			*&out[B][C][y][x] = bilinear_func(value_ptr, xx, yy, x_lower, y_lower, W);
		}
	}
}


torch::Tensor gridsample_cuda(const torch::Tensor value, const torch::Tensor grid){
	const int B = value.size(0);
	const int C = value.size(1);
	const int Hin = value.size(2);
	const int Win = value.size(3);

	const int Hout = grid.size(1);
	const int Wout = grid.size(2);

	const int threads = 1024;
	const dim3 blocks(B,C);

	auto out = torch::zeros({B,C,Hout,Wout}, value.options());

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "generate grid sample", (
			[&]{gridsample_cuda_kernel<scalar_t><<<blocks, threads>>>(
				value.packed_accessor32<scalar_t, 4>(),
				grid.packed_accessor32<scalar_t, 4>(), 
				out.packed_accessor32<scalar_t, 4>(),
				Hin, Win, Hout, Wout);
			}
		)
	);
	return out;
}


template <typename scalar_t>
__device__ __forceinline__ void bilinear_back(const scalar_t * value, const scalar_t grad, scalar_t * value_grad, scalar_t * grid_grad, 
								const scalar_t xx, const scalar_t yy, const int x_lower, const int y_lower, const int H, const int W){
	const scalar_t diff_x = xx - x_lower;
	const scalar_t diff_y = yy - y_lower;

	const scalar_t v1 = *(value);
	const scalar_t v2 = *(value+1);
	const scalar_t v3 = *(value+W);
	const scalar_t v4 = *(value+W+1);

	const scalar_t w1 = (1 - diff_x) * (1 - diff_y);
	const scalar_t w2 = diff_x * (1 - diff_y);
	const scalar_t w3 = (1 - diff_x) * diff_y;
	const scalar_t w4 = diff_x * diff_y;

	gpuAtomicAdd(value_grad, grad*w1);
	gpuAtomicAdd(&*(value_grad+1), grad*w2);
	gpuAtomicAdd(&*(value_grad+W), grad*w3);
	gpuAtomicAdd(&*(value_grad+W+1), grad*w4);

	gpuAtomicAdd(grid_grad, grad*(W-1)*(-v1*(1-diff_y) + v2*(1-diff_y) - v3*diff_y + v4*diff_y));
	gpuAtomicAdd(grid_grad+1, grad*(H-1)*(-v1*(1-diff_x) - v2*diff_x + v3*(1-diff_x) + v4*diff_x));        //gpuAtomicAdd(at::Half *address, at::Half val)
}


template <typename scalar_t>
__global__ void gridsample_backward_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t,4> value, 
	const torch::PackedTensorAccessor32<scalar_t,4> grid,
	const torch::PackedTensorAccessor32<scalar_t,4> grad_out,
	torch::PackedTensorAccessor32<scalar_t, 4> value_grad,
	torch::PackedTensorAccessor32<scalar_t, 4> grid_grad,
	const int H, const int W, const int Ho, const int Wo){

	const int B = blockIdx.x;
	const int C = blockIdx.y;
	const int n_thread = threadIdx.x;
	const int thread_stride = blockDim.x;

	int x,y;
	for (int i=n_thread; i<Ho*Wo; i+=thread_stride){
		x = i / Wo;
		y = i % Wo;
		const scalar_t * grid_ptr = & grid[B][y][x][0];
		const scalar_t xx = *(grid_ptr) * (W-1);
		const scalar_t yy = *(grid_ptr+1) * (H-1);

		if (xx>=0 && xx<W-1 && yy>=0 && yy<H-1){
			const int y_lower = floor(yy);
			const int x_lower = floor(xx);
			const scalar_t * value_ptr = &value[B][C][y_lower][x_lower];
			scalar_t * value_grad_ = &value_grad[B][C][y_lower][x_lower];
			scalar_t * grid_grad_ = &grid_grad[B][y][x][0];
			const scalar_t grad_val = grad_out[B][C][y][x];
			bilinear_back(value_ptr, grad_val, value_grad_, grid_grad_, xx, yy, x_lower, y_lower, H, W);
		}
	}
}



std::vector<torch::Tensor> gridsample_backward_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor grad_out){
	const int B = value.size(0);
	const int C = value.size(1);
	const int Hin = value.size(2);
	const int Win = value.size(3);
	const int Hout = grid.size(1);
	const int Wout = grid.size(2);

	auto value_grad = torch::zeros_like(value);
	auto grid_grad = torch::zeros_like(grid);

	const int threads = 1024;
	const dim3 blocks(B,C);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "grid sample backward", (
			[&]{gridsample_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
				value.packed_accessor32<scalar_t, 4>(),
				grid.packed_accessor32<scalar_t, 4>(),
				grad_out.packed_accessor32<scalar_t, 4>(),
				value_grad.packed_accessor32<scalar_t, 4>(),
				grid_grad.packed_accessor32<scalar_t, 4>(),
				Hin, Win, Hout, Wout);
			}
		)
	);
	return {value_grad, grid_grad};
}
