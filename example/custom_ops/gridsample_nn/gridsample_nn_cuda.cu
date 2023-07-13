#include <torch/all.h>
#include <torch/python.h>
#include <vector>
#include <ATen/cuda/Atomic.cuh>

template <typename T>
__global__ void gridsample_nn_cuda_kernel(
			const torch::PackedTensorAccessor32<T, 4> value,
			const torch::PackedTensorAccessor32<T, 4> grid,
			torch::PackedTensorAccessor32<T, 4> out,
			const int H, const int W, const int Ho, const int Wo){
	const int thread_idx = threadIdx.x;
	const int B = blockIdx.y;
	const int C = blockIdx.x;
	const int N = blockDim.x;

	int x,y;
	for (int i=thread_idx; i<Ho*Wo; i+=N){
		x = i%Wo;
		y = i/Wo;
		const T xx = grid[B][y][x][0] * (W-1);
		const T yy = grid[B][y][x][1] * (H-1);
		const int x_rnd = round(xx);
		const int y_rnd = round(yy);
		if (xx>=0 && xx<(W-1) && yy>=0 && yy<(H-1)){
			*&out[B][C][y][x] = value[B][C][y_rnd][x_rnd];
		}
	}
}

torch::Tensor gridsample_nn_cuda(const torch::Tensor value, const torch::Tensor grid){
	// value: [B, C, H, W]
	// grid: [B, Ho, Wo, 2]
	const int B = value.size(0);
	const int C = value.size(1);
	const int H = value.size(2);
	const int W = value.size(3);

	const int Ho = grid.size(1);
	const int Wo = grid.size(2);

	const int threads = 1024;
	const dim3 blocks(C,B); 

	auto out = torch::zeros({B,C,Ho,Wo}, value.options());

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "gridsample nn", (
		[&] {gridsample_nn_cuda_kernel<<<blocks, threads>>>(
					value.packed_accessor32<scalar_t, 4>(),
					grid.packed_accessor32<scalar_t, 4>(),
					out.packed_accessor32<scalar_t, 4>(),
					H, W, Ho, Wo);
			}
		)
	);
	return out;
}

template <typename T> 
__device__ __forceinline__ void dv_dx(const T * value, T * grid_grad_ptr, 
				const T grad, const T xx, const T yy, const int H, const int W){
	const T v1 = *value;
	const T v2 = *(value+1);
	const T v3 = *(value+W);
	const T v4 = *(value+W+1);

	const T wx = round(xx) - floor(xx);    // wx=1  =>  x1=1
	const T wy = round(yy) - floor(yy);    // wy=1  =>  y1=1

	gpuAtomicAdd(grid_grad_ptr, grad*(W-1)*(-v1*(1-wy)+v2*(1-wy)-v3*wy+v4*wy));
	gpuAtomicAdd(grid_grad_ptr+1, grad*(H-1)*(-v1*(1-wx)-v2*wx+v3*(1-wx)+v4*wx));
}


template <typename T>
__global__ void gridsample_nn_back_cuda_kernel(
	const torch::PackedTensorAccessor32<T, 4> value,
	const torch::PackedTensorAccessor32<T, 4> grid,
	const torch::PackedTensorAccessor32<T, 4> grad_out,
	torch::PackedTensorAccessor32<T, 4> value_grad, 
	torch::PackedTensorAccessor32<T, 4> grid_grad,
	const int H, const int W, const int Ho, const int Wo){

	const int B = blockIdx.y;
	const int C = blockIdx.x;
	const int thread_idx = threadIdx.x;
	const int N = blockDim.x;

	int x,y;
	for (int i=thread_idx; i<Ho*Wo; i+=N){
		x = i%Wo;
		y = i/Wo;
		const T xx = grid[B][y][x][0] * (W-1);
		const T yy = grid[B][y][x][1] * (H-1);
		
		if (xx>=0 && xx<(W-1) && yy>=0 && yy<(H-1)){
			const T g = grad_out[B][C][y][x];
			const int x_rnd = round(xx);
			const int y_rnd = round(yy);
			const int x_lower = floor(xx);
			const int y_lower = floor(yy);
			gpuAtomicAdd(&value_grad[B][C][y_rnd][x_rnd], g); 

			dv_dx(&value[B][C][y_lower][x_lower], &grid_grad[B][y][x][0], g, xx, yy, H, W);
		}
	}
}


std::vector<torch::Tensor> gridsample_nn_back_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor grad_out){
	const int B = value.size(0);
	const int C = value.size(1);
	const int H = value.size(2);
	const int W = value.size(3);
	const int Ho = grid.size(1);
	const int Wo = grid.size(2);

	auto value_grad = torch::zeros_like(value);
	auto grid_grad = torch::zeros_like(grid);

	const int threads = 1024;
	const dim3 blocks(C,B);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "gridsample nn backward", (
			[&]{gridsample_nn_back_cuda_kernel<<<blocks, threads>>>(
				value.packed_accessor32<scalar_t, 4>(),
				grid.packed_accessor32<scalar_t, 4>(),
				grad_out.packed_accessor32<scalar_t, 4>(),
				value_grad.packed_accessor32<scalar_t, 4>(),
				grid_grad.packed_accessor32<scalar_t, 4>(),
				H, W, Ho, Wo);
			}
		)
	);
	return {value_grad, grid_grad};
}
