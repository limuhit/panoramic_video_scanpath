#include "vp2erp.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void vp2erp_opt::init(){
    init_base();
    wangle_ =  pi_ / 2 - fov_ / 2;
    num_ = -1;
    width_ = -1;
}

void vp2erp_opt::reshape(int num, int channel, int width){
    assert((width==2)&&("We only accept the input with the shape of N*C*2"));
    if (!reshape_base(num, channel, 1, width)) return; 

}

void vp2erp_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, width_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void vp2erp_forward_kernel(const int nthreads, const scalar_t* const input,  const scalar_t* const y,
     scalar_t * const output, const int channel, const scalar_t rad, 
     const scalar_t x_bias, const scalar_t y_bias, const scalar_t pi) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int base_y = index / channel * 3 * 3;
        scalar_t xa = rad;
        scalar_t xb = input[index*2] - x_bias + 0.5;
        scalar_t xc = -input[index*2+1] + y_bias - 0.5;
        scalar_t rn = sqrt(xa*xa + xb*xb + xc*xc);
        xa /= rn;
        xb /= rn;
        xc /= rn;
        scalar_t za = xa * y[base_y] + xb * y[base_y+1] + xc * y[base_y+2];
        scalar_t zb = xa * y[base_y+3] + xb * y[base_y+4] + xc * y[base_y+5];
        scalar_t zc = xa * y[base_y+6] + xb * y[base_y+7] + xc * y[base_y+8];
        scalar_t lat = asin(zc);
        scalar_t theta = atan(zb/za);
        if (za<=0){
            if(zb>0){
                theta = theta + pi;
            }else{
                theta = theta - pi;
            }
        }
        output[index*2] = theta;
        output[index*2+1] = lat;
    }
}

std::vector<at::Tensor>  vp2erp_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor rota) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "vp2erp_forward_cuda", 
			([&] {
                    count = num_ * channel_;
                    scalar_t rad = 0.5 * w_v_ * tan(wangle_);
                    scalar_t x_bias = 0.5 * w_v_;
                    scalar_t y_bias = 0.5 * h_v_;
                    vp2erp_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), rota.data_ptr<scalar_t>(), 
                            top_data_[0].data_ptr<scalar_t>(), channel_, rad, x_bias, y_bias, static_cast<scalar_t>(pi_));
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
