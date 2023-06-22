#include "render_core.h"
#include <stdlib.h>

void _get_gaus(float* map, int sizeh, int sizew, float sigma, float* pts, int n_batch, int n_pts){
	int i,j,b,p;
	#pragma omp parallel for num_threads(16) private(p,b,i,j)
	for(b=0;b<n_batch;b++){
		for(p=0;p<n_pts;p++){
			float e,dx,dy,x,y;
			int idx;
			idx = b*n_pts*sizeh*sizew + p*sizeh*sizew;
			x = pts[(b*n_pts+p)*2];
			y = pts[(b*n_pts+p)*2 + 1];
			for(i=0;i<sizeh;i++){
				for (j=0;j<sizew;j++){
					dx = float(j) - x;
					dy = float(i) - y;
					if (abs(dx)>3*sigma) {idx++;continue;}
					if (abs(dy)>3*sigma) {idx++;continue;}
					e = (dx*dx + dy*dy) / 2 / sigma/ sigma;
					map[idx] += exp(-e);
					idx++;
				}
			}
		}
	}
	
}