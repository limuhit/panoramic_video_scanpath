#include "linear_mask.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void linear_mask_opt::init(){
    init_base();
}

template <typename scalar_t>
__global__ void linear_mask_forward_hidden_kernel(const int nthreads,  scalar_t * const mask, 
    const int channel, const int cin, const int cout, const int ncontext) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pc = index % channel;
        int pn = index / channel;
        int cg = pc / cin - ncontext;
        int ng = pn / cout - ncontext;
        if(cg<0 && ng<0){
            mask[index] = 1;
        }else if(cg<=ng){
            mask[index] = 1;
        }
    }
}


template <typename scalar_t>
__global__ void linear_mask_forward_hidden_out_kernel(const int nthreads,  scalar_t * const mask, 
    const int channel, const int cin, const int cout, const int ncontext) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pc = index % channel;
        int pn = index / channel;
        int cg = pc / cin - ncontext;
        int ng = pn / cout;
        if(cg<=ng){
            mask[index] = 1;
        }
    }
}

template <typename scalar_t>
__global__ void linear_mask_forward_kernel(const int nthreads,  scalar_t * const mask, 
    const int channel, const int cin, const int cout, const int ncontext) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pc = index % channel;
        int pn = index / channel;
        int cg = pc / cin - ncontext;
        int ng = pn / cout - ncontext;
        if(cg<0 && ng<0){
            mask[index] = 1;
        }else if(cg<ng){
            mask[index] = 1;
        }
    }
}


at::Tensor linear_mask_opt::forward_cuda() 
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_).requires_grad(false);
    mask_ =  at::zeros({nout_, channel_}, options);
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		mask_.scalar_type(), "linear_mask_forward_cuda", 
			([&] {
                count = nout_ * channel_;
                if(out_){
                    linear_mask_forward_hidden_out_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, mask_.data_ptr<scalar_t>(), channel_, cin_, cout_, ncontext_);
                }else{
                    if(hidden_){
                        linear_mask_forward_hidden_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, mask_.data_ptr<scalar_t>(), channel_, cin_, cout_, ncontext_);
    
                    }else{
                        linear_mask_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, mask_.data_ptr<scalar_t>(), channel_, cin_, cout_, ncontext_);
                    }

                }
                
                
                CUDA_POST_KERNEL_CHECK;
   			})
    );
    return mask_;
}
