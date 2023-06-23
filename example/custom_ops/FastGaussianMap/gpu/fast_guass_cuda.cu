#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t> 
__global__ void generate_hmap_cuda_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> np,
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> hmap, float sigma){
	const int n = blockIdx.x;
	const int c = threadIdx.x;
	const int nt = blockDim.x;
	const int b = blockIdx.y;
	const int mapsize = hmap.size(2) * hmap.size(3);
	int i,x,y;
	float e,dx,dy;
	if (np[b][n][2]>0){
		for (i=c;i<mapsize;i=i+nt){
			x = i%hmap.size(3);
			y = i/hmap.size(3);

			dx = float(x) - np[b][n][0];
			dy = float(y) - np[b][n][1];
			if (abs(dx)>3*sigma) {continue;}
			if (abs(dy)>3*sigma) {continue;}
			e = (dx * dx + dy * dy) / 2 / sigma / sigma;
			hmap[b][n][y][x] = exp(-e);
		}
	}
}

torch::Tensor generate_hmap_cuda(torch::Tensor pts, int H, int W, float sigma){
	// pts should be [np,2]
	const auto b = pts.size(0);
	const auto np = pts.size(1);
	auto hmap = torch::zeros({b, np, H, W}, pts.device());
	const int threads = 1024;
	const dim3 blocks(np, b);

	AT_DISPATCH_FLOATING_TYPES(pts.type(), "generate_hmap_cuda", ([&] {
		generate_hmap_cuda_kernel<scalar_t><<<blocks, threads>>>(
			pts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
			hmap.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			sigma);
	}));
	return hmap;
}
