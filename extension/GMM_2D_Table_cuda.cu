#include "GMM_2D_Table.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void GMM_2D_Table_opt::init(){
    init_base();
}

void GMM_2D_Table_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
}

void GMM_2D_Table_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, height_, width_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void GMM_2D_Table_forward_kernel(const int nthreads, const scalar_t* const wt, const scalar_t* const mean,
    const scalar_t* const std,  scalar_t * const output, const int height, const int width, const int ng, const scalar_t sq2,
    const scalar_t stride_h, const scalar_t stride_w, const scalar_t hb, const scalar_t wb, const scalar_t hf, const scalar_t wf){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pn = index / width / height;
        scalar_t qh = (ph - hb) * stride_h;
        scalar_t qw = (pw - wb) * stride_w;
        output[index] = 0;
        scalar_t ha,hb,wa,wb;
        for(int i=0;i<ng;i++){
            int ps = (pn*ng + i) * 2; 
            ha = (qh - hf - mean[ps+1]) * sq2 / std[ps+1];
            hb = (qh + hf - mean[ps+1]) * sq2 / std[ps+1];
            wa = (qw - wf - mean[ps]) * sq2 / std[ps];
            wb = (qw + wf - mean[ps]) * sq2 / std[ps];
            output[index] =  output[index] + (erf(hb)-erf(ha)) * (erf(wb)-erf(wa)) * 0.25 * wt[pn*ng + i];
        }
    }
}


std::vector<at::Tensor>  GMM_2D_Table_opt::forward_cuda(at::Tensor wt, at::Tensor mean, at::Tensor std) 
{
    reshape(wt.size(0), 1, h_out_, w_out_);
    reshape_top(wt.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		wt.scalar_type(), "GMM_2D_Table_forward_cuda", 
			([&] {
                    count = num_ * width_ * height_;
                    scalar_t stride_h = stride_h_, stride_w = stride_w_;
                    scalar_t hb = h_bias_, wb = w_bias_;
                    scalar_t hf = hf_, wf = wf_;
                    GMM_2D_Table_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, wt.data_ptr<scalar_t>(), mean.data_ptr<scalar_t>(), std.data_ptr<scalar_t>(),
                            top_data_[0].data_ptr<scalar_t>(), height_, width_, num_gaussian_, scalar_t(sq2_), 
                            stride_h, stride_w, hb, wb, hf, wf);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

