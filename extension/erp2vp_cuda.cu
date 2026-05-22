#include "erp2vp.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>

void erp2vp_opt::init(){
    init_base();
    wangle_ =  pi_ / 2 - fov_ / 2;
    num_ = -1;
    width_ = -1;
}

void erp2vp_opt::reshape(int num, int channel, int width){
    assert((width==2)&&("We only accept the input with the shape of N*C*2"));
    if (!reshape_base(num, channel, 1, width)) return; 
}

void erp2vp_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, width_});
    shapes.push_back({num_, channel_, 4});
    shapes.push_back({num_, channel_, 3});
    reshape_top_base(option,shapes);
}

void erp2vp_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void erp2vp_forward_kernel(int num, const scalar_t * const tf, const scalar_t * y, scalar_t * const out,
    scalar_t * const tfcs, scalar_t * const z,  const scalar_t rad, const scalar_t x_bias, const scalar_t y_bias, const int channel){
    CUDA_KERNEL_LOOP(i, num) {

        int pi = i / channel;
        int base_y = pi * 3 * 3;
        scalar_t ts = sin(tf[i*2]);
        scalar_t tc = cos(tf[i*2]);
        scalar_t fs = sin(tf[i*2+1]);
        scalar_t fc = cos(tf[i*2+1]);
        tfcs[i*4] = ts;
        tfcs[i*4+1] = tc;
        tfcs[i*4+2] = fs;
        tfcs[i*4+3] = fc;
        scalar_t xa = tc*fc;
        scalar_t xb = ts*fc;
        scalar_t xc = fs;
        scalar_t za = xa * y[base_y] + xb * y[base_y+3] + xc * y[base_y+6];
        scalar_t zb = xa * y[base_y+1] + xb * y[base_y+4] + xc * y[base_y+7];
        scalar_t zc = xa * y[base_y+2] + xb * y[base_y+5] + xc * y[base_y+8];
        scalar_t gamma = rad / za;
        out[i*2] = gamma*zb - 0.5 +  x_bias;
        out[i*2 + 1] = -gamma*zc - 0.5 + y_bias;
        z[i*3] = za;
        z[i*3+1] = zb;
        z[i*3+2] = zc;

    }
}

std::vector<at::Tensor>  erp2vp_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor rota) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "erp2vp_forward_cuda", 
			([&] {
                    count = num_ * channel_;
                    scalar_t rad = 0.5 * w_v_ * tan(wangle_);
                    scalar_t x_bias = 0.5 * w_v_;
                    scalar_t y_bias = 0.5 * h_v_;
                    erp2vp_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), rota.data_ptr<scalar_t>(), 
                            top_data_[0].data_ptr<scalar_t>(), top_data_[1].data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(),
                             rad, x_bias, y_bias, channel_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void erp2vp_backward_kernel(int num, scalar_t * const tf_diff, const scalar_t * y, scalar_t * const out_diff,
    scalar_t * const tfcs, scalar_t * const z,  const scalar_t rad, const scalar_t x_bias, const scalar_t y_bias, const int channel){
    CUDA_KERNEL_LOOP(i, num) {

        int base_y = i / channel * 3 * 3;
        scalar_t dp = 1 / z[i*3];
        scalar_t dpa = dp * rad;
        scalar_t dy = out_diff[i*2+1];
        scalar_t dx = out_diff[i*2];
        
        scalar_t dza = dpa * dp * (z[i*3+2]*dy -  z[i*3+1]*dx);
        scalar_t dzb = dpa * dx;
        scalar_t dzc = -dpa * dy;

        scalar_t dxa = dza * y[base_y] + dzb * y[base_y+1] + dzc * y[base_y+2];
        scalar_t dxb = dza * y[base_y+3] + dzb * y[base_y+4] + dzc * y[base_y+5];
        scalar_t dxc = dza * y[base_y+6] + dzb * y[base_y+7] + dzc * y[base_y+8]; 

        scalar_t ps = tfcs[i*4+2];
        scalar_t pc = tfcs[i*4+3];
        scalar_t ts = tfcs[i*4];
        scalar_t tc = tfcs[i*4+1];

        tf_diff[i*2] = pc*(tc*dxb-ts*dxa);
        tf_diff[i*2+1] = pc*dxc - ps*(tc*dxa+ts*dxb);
        
    }
}

std::vector<at::Tensor>  erp2vp_opt::backward_cuda(at::Tensor  top_diff, at::Tensor rota) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "erp2vp_backward_cuda", 
			([&] {
                count = num_ * channel_;
                    scalar_t rad = 0.5 * w_v_ * tan(wangle_);
                    scalar_t x_bias = 0.5 * w_v_;
                    scalar_t y_bias = 0.5 * h_v_;
                    erp2vp_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                         (count, bottom_diff_[0].data_ptr<scalar_t>(), rota.data_ptr<scalar_t>(), 
                         top_diff.data_ptr<scalar_t>(), top_data_[1].data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(),
                             rad, x_bias, y_bias, channel_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}