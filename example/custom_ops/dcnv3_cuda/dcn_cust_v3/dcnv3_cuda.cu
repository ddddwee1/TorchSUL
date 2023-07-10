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
__global__ void dcnv3_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> value, 
	const torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> grid, 
	const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> weight,
	torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> out,
	const int H, const int W, const int Ho, const int Wo, const int Pmax){

	const int C = blockIdx.x;
	const int G = blockIdx.y;
	const int B = blockIdx.z;
	const int n_thread = threadIdx.x;
	const int thread_stride = blockDim.x;

	int x,y,i,j;
	scalar_t v;
	for (i=n_thread; i<Ho*Wo; i=i+thread_stride){
		x = i % Wo;
		y = i / Wo;
		v = 0;
		const scalar_t * grid_ptr = & grid[B][G][y][x][0][0];
		const scalar_t * w_ptr = & weight[B][G][y][x][0];

		for (j=0; j<Pmax; j++){
			const scalar_t xx = *(grid_ptr+j*2) * (W-1);
			const scalar_t yy = *(grid_ptr+j*2+1) * (H-1);

			if (xx>=0 && xx<W-1 && yy>=0 && yy<H-1){
				const int y_lower = floor(yy);
				const int x_lower = floor(xx);
				v += *(w_ptr + j) * bilinear_func(&value[B][G][C][y_lower][x_lower], xx, yy, x_lower, y_lower, W);
				// printf("%d / %d %f\n",j, Pmax,v);
			}
		}
		*&out[B][G][C][y][x] = v;
	}
}


torch::Tensor dcnv3_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor weight){
	// value shape: [B,G,C,H,W]
	// grid shape: [B,Ho,Wo,G,P,2]
	// weight shape: [B,Ho,Wo,G,P]

	const int B = value.size(0);
	const int G = value.size(1);
	const int C = value.size(2); 
	const int Hin = value.size(3);
	const int Win = value.size(4);

	const int Hout = grid.size(2);
	const int Wout = grid.size(3);
	const int P = grid.size(4);

	const int threads = 1024;
	const dim3 blocks(C,G,B);

	auto out = torch::zeros({B,G,C,Hout,Wout}, value.options());

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "generate grid sample", (
			[&]{dcnv3_cuda_kernel<scalar_t><<<blocks, threads>>>(
				value.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
				grid.packed_accessor32<scalar_t, 6, torch::RestrictPtrTraits>(), 
				weight.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
				out.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
				Hin, Win, Hout, Wout, P);
			}
		)
	);
	return out;
}


template <typename scalar_t>
__device__ __forceinline__ void bilinear_back(const scalar_t * value, const scalar_t weight, const scalar_t grad, 
								scalar_t * value_grad, scalar_t * grid_grad, scalar_t * weight_grad,
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

	gpuAtomicAdd(value_grad, grad*w1*weight);
	gpuAtomicAdd(value_grad+1, grad*w2*weight);
	gpuAtomicAdd(value_grad+W, grad*w3*weight);
	gpuAtomicAdd(value_grad+W+1, grad*w4*weight);

	gpuAtomicAdd(grid_grad, weight*grad*(W-1)*(-v1*(1-diff_y) + v2*(1-diff_y) - v3*diff_y + v4*diff_y));
	gpuAtomicAdd(grid_grad+1, weight*grad*(H-1)*(-v1*(1-diff_x) - v2*diff_x + v3*(1-diff_x) + v4*diff_x));        

	gpuAtomicAdd(weight_grad, grad*(v1*w1 + v2*w2 + v3*w3 + v4*w4));
}


template <typename scalar_t>
__global__ void dcnv3_backward_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t,5> value, 
	const torch::PackedTensorAccessor32<scalar_t,6> grid,
	const torch::PackedTensorAccessor32<scalar_t,5> weight,
	const torch::PackedTensorAccessor32<scalar_t,5> grad_out,
	torch::PackedTensorAccessor32<scalar_t, 5> value_grad,
	torch::PackedTensorAccessor32<scalar_t, 6> grid_grad,
	torch::PackedTensorAccessor32<scalar_t, 5> weight_grad,
	const int H, const int W, const int Ho, const int Wo, const int Pmax){

	const int C = blockIdx.x;
	const int G = blockIdx.y;
	const int B = blockIdx.z;
	const int n_thread = threadIdx.x;
	const int thread_stride = blockDim.x;
	int x,y,i;

	// __shared__ scalar_t cache_grid_grad[Ho*Wo*2];
	// __shared__ scalar_t cache_weight_grad[Ho*Wo*2];
	// for (int i=n_thread; i<Ho*Wo; i+=thread_stride){
	// 	cache_grid_grad[i*2] = 0;
	// 	cache_grid_grad[i*2+1] = 0;
	// 	cache_weight_grad[i] = 0;
	// }
	// __syncthreads();

	
	for (i=n_thread; i<Ho*Wo; i+=thread_stride){
		x = i / Wo;
		y = i % Wo;    // swap x and y here, avoid concurrent write to the same memory address 
		const scalar_t * grid_ptr = & grid[B][G][y][x][0][0];
		const scalar_t grad_pix = grad_out[B][G][C][y][x];

		for (int j=0;j<Pmax;j++){
			const scalar_t xx = *(grid_ptr+j*2) * (W-1);
			const scalar_t yy = *(grid_ptr+j*2+1) * (H-1);

			if (xx>=0 && xx<W-1 && yy>=0 && yy<H-1){
				const int y_lower = floor(yy);
				const int x_lower = floor(xx);
				
				bilinear_back(&value[B][G][C][y_lower][x_lower], 
					weight[B][G][y][x][j], 
					grad_pix, 
					&value_grad[B][G][C][y_lower][x_lower], 
					&grid_grad[B][G][y][x][j][0],  //&cache_grid_grad[2*i],  // 
					&weight_grad[B][G][y][x][j],  //&cache_weight_grad[i],  // 
					xx, yy, x_lower, y_lower, H, W);
			}
		}
	}
	// __syncthreads();

	// for (i=0; i<Ho*Wo; i+=thread_stride){

	// }

}



std::vector<torch::Tensor> dcnv3_backward_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor weight, const torch::Tensor grad_out){
	const int B = value.size(0);
	const int G = value.size(1);
	const int C = value.size(2);
	const int Hin = value.size(3);
	const int Win = value.size(4);

	const int Hout = grid.size(2);
	const int Wout = grid.size(3);
	const int P = grid.size(4);

	auto value_grad = torch::zeros_like(value);
	auto grid_grad = torch::zeros_like(grid);
	auto weight_grad = torch::zeros_like(weight);

	const int threads = min(1024, Hout*Wout);
	const dim3 blocks(C,G,B);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.type(), "grid sample backward", (
			[&]{dcnv3_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
				value.packed_accessor32<scalar_t, 5>(),
				grid.packed_accessor32<scalar_t, 6>(),
				weight.packed_accessor32<scalar_t, 5>(),
				grad_out.packed_accessor32<scalar_t, 5>(),
				value_grad.packed_accessor32<scalar_t, 5>(),
				grid_grad.packed_accessor32<scalar_t, 6>(),
				weight_grad.packed_accessor32<scalar_t, 5>(),
				Hin, Win, Hout, Wout, P);
			}
		)
	);
	return {value_grad, grid_grad, weight_grad};
}
