#include "gmm_sample.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void gmm_sample_opt::init(){
    init_base();
}

void gmm_sample_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 

}

void gmm_sample_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, nout_, 2});
    shapes.push_back({num_, 2});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void gmm_sample_forward_kernel(const int nthreads, const scalar_t* const r1,  const scalar_t* const r2,
     const scalar_t* const wt, const scalar_t* const mean, const scalar_t* const std,
     scalar_t * const output, const int ng, const int nout) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % 2;
        int tn = index / 2 / nout;
        scalar_t tmp = 0;
        int i = 0;
        for(;i<ng;i++){
            tmp += wt[tn*ng+i];
            if(tmp>r1[index/2])
                break;
        }
        i = (i>=ng) ? ng-1 : i;
        int pid = (tn*ng + i)*2 + tw; 
        output[index] = r2[index] * std[pid] + mean[pid];
    }
}


std::vector<at::Tensor>  gmm_sample_opt::forward_cuda(at::Tensor wt, at::Tensor mean, at::Tensor std, at::Tensor r1, at::Tensor r2) 
{
    reshape(wt.size(0), wt.size(1), 1, 1);
    reshape_top(wt.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		wt.scalar_type(), "gmm_sample_forward_cuda", 
			([&] {
                    count = num_ * nout_ * 2;
                    gmm_sample_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, r1.data_ptr<scalar_t>(), r2.data_ptr<scalar_t>(), wt.data_ptr<scalar_t>(), mean.data_ptr<scalar_t>(),  
                        std.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), channel_, nout_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void gmm_sample_select_kernel(const int nthreads, const scalar_t* const input, 
     scalar_t * const output, const int nout, const scalar_t xb, const scalar_t yb) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        scalar_t tx,ty;
        int base;
        for(int i =0; i<nout; i++){
            base = (index * nout + i)*2;
            tx = input[base];
            ty = input[base+1];
            if(tx>-xb && tx<xb && ty>-yb && ty<yb)
                break;
        }
        output[index*2] = tx;
        output[index*2+1] = ty;
    }
}

std::vector<at::Tensor>  gmm_sample_opt::select(at::Tensor out, float x_bound, float y_bound) 
{
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		out.scalar_type(), "gmm_sample_select_cuda", 
			([&] {
                    count = num_;
                    scalar_t xb = x_bound, yb = y_bound;
                    gmm_sample_select_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, top_data_[0].data_ptr<scalar_t>(), top_data_[1].data_ptr<scalar_t>(), nout_, xb, yb);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
