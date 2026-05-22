#include "pre_data.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void pre_data_opt::init(){
    init_base();
    CUDA_CHECK(cudaStreamCreate(&stream2_));
    CUDA_CHECK(cudaStreamCreate(&stream3_));
}

void pre_data_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 

}

void pre_data_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    nout_ = train_ ? nw_ : (num_ - 1 - windows_ - npred_) / stride_ + 1;
    shapes.push_back({bn_, nout_, windows_, channel_, height_, width_});
    shapes.push_back({bn_, nout_, windows_, 2*npred_+1, 2});
    shapes.push_back({bn_, nout_, npred_, 2});
    shapes.push_back({bn_, nout_, windows_, 2});
    top_data_.clear();
    for(int i=0; i<shapes.size(); i++)
        top_data_.push_back(at::empty(shapes[i], option));
}

template <typename scalar_t>
__global__ void pre_data_img_scale(const int nthreads, scalar_t* const input){
    CUDA_KERNEL_LOOP(index, nthreads) {
        input[index] = input[index] / 255;
    }
}

template <typename scalar_t>
__global__ void pre_data_path_scale(const int nthreads, scalar_t* const input, 
    const int inner_shape, const scalar_t x_b, const scalar_t y_b){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pc = (index / inner_shape) % 2;
        if(pc==0){
            input[index] = input[index] - x_b;
        }else{
            input[index] = input[index] - y_b;
        }
    }
}


template <typename scalar_t>
__global__ void pre_data_forward_img_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int * const randv, const int inner_shape, 
     const int * const start_idx, const int windows, const int stride, const int nout, const int num) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pw = (index / inner_shape) % windows;
        int pn = index / inner_shape / windows;
        int pnb = pn / nout;
        int pna = pn % nout;
        int qn = start_idx[pnb] + randv[pna]*stride;
        int qid = qn-windows+1+pw;
        if(qid>=0){
            output[index] = input[(pnb*num+qid)*inner_shape + ps];
        }else{
            output[index] = 0;
        }
    }
}

template <typename scalar_t>
__global__ void pre_data_forward_path_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int * const randv, const int pstride, const int lpred, 
     const int * const start_idx, const int npred,  const int pt_mid, const int windows, const int stride,
     const int nout, const int num) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pc = index % 2;
        int ps = (index / 2) % lpred;
        int pw = (index / 2 / lpred) % windows;
        int pn = index / 2 / lpred / windows;
        int pnb = pn / nout;
        int pna = pn % nout;
        int qn = start_idx[pnb] + randv[pna]*stride;
        int qid = qn-windows+1+pw;
        if(qid>=0){
            int tid = qid + ps - npred;
            if (tid<=qn){
                output[index] = input[((pnb*num+qid)*2+pc)*pstride+ps-npred+pt_mid];
            }else{
                output[index] = 0;
            }
        }else{
            output[index] = 0;
        }
        
    }
}

template <typename scalar_t>
__global__ void pre_data_forward_label_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int * const randv, const int pstride, 
     const int * const start_idx, const int npred,  const int pt_mid, const int stride,
     const int nout, const int num) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pc = index % 2;
        int ps = (index / 2) % npred;
        int pn = index / 2 / npred;
        int pnb = pn / nout;
        int pna = pn % nout;
        int qn = start_idx[pnb] + randv[pna]*stride;
        output[index] = input[((pnb*num+qn)*2+pc)*pstride+pt_mid+1+ps];
    }
}

std::vector<at::Tensor>  pre_data_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_path, at::Tensor rand_vec, at::Tensor start_idx) 
{
    bn_ = bottom_data.size(0);
    reshape(bottom_data.size(1), bottom_data.size(2), bottom_data.size(3), bottom_data.size(4));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "pre_data_forward_cuda", 
			([&] {
                    count = bn_ * num_ * channel_ * height_ * width_;
                    pre_data_img_scale<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count,bottom_data.data_ptr<scalar_t>());
                    count = bn_ * num_ * 2 * pathw_;
                    pre_data_path_scale<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream2_ >> >
                        (count,bottom_path.data_ptr<scalar_t>(),pathw_,scalar_t(x_mean_),scalar_t(y_mean_));
                    cudaDeviceSynchronize();
                    CUDA_POST_KERNEL_CHECK;

                    count = bn_ * nout_ * windows_ * channel_ * width_ * height_;
                    pre_data_forward_img_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),rand_vec.data_ptr<int>(),
                            channel_ * width_ * height_, start_idx.data_ptr<int>(), windows_, stride_, nout_, num_);
                    count = bn_ * nout_ * windows_ * (2*npred_+1) * 2;
                    pre_data_forward_path_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream2_ >> >
                        (count, bottom_path.data_ptr<scalar_t>(), top_data_[1].data_ptr<scalar_t>(),rand_vec.data_ptr<int>(),
                            pathw_, 2*npred_+1, start_idx.data_ptr<int>(), npred_, path_mid_, windows_, stride_, nout_, num_);
                    count = bn_ * nout_ * npred_ * 2;
                    pre_data_forward_label_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream3_ >> >
                        (count, bottom_path.data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(),rand_vec.data_ptr<int>(),
                            pathw_, start_idx.data_ptr<int>(), npred_,  path_mid_, stride_, nout_, num_);
                    cudaDeviceSynchronize();
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return {top_data_[0],top_data_[1],top_data_[2]};
}

std::vector<at::Tensor>  pre_data_opt::forward_cuda2(at::Tensor  bottom_data, at::Tensor bottom_path, at::Tensor raw_path, at::Tensor rand_vec, at::Tensor start_idx) 
{
    bn_ = bottom_data.size(0);
    reshape(bottom_data.size(1), bottom_data.size(2), bottom_data.size(3), bottom_data.size(4));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "pre_data_forward_cuda2", 
			([&] {
                    count = bn_ * num_ * channel_ * height_ * width_;
                    pre_data_img_scale<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count,bottom_data.data_ptr<scalar_t>());
                    count = bn_ * num_ * 2 * pathw_;
                    pre_data_path_scale<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream2_ >> >
                        (count,bottom_path.data_ptr<scalar_t>(),pathw_,scalar_t(x_mean_),scalar_t(y_mean_));
                    cudaDeviceSynchronize();
                    CUDA_POST_KERNEL_CHECK;
                    count = bn_ * nout_ * windows_ * channel_ * width_ * height_;
                    pre_data_forward_img_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),rand_vec.data_ptr<int>(),
                            channel_ * width_ * height_, start_idx.data_ptr<int>(), windows_, stride_, nout_, num_);
                    count = bn_ * nout_ * windows_ * 2;
                    pre_data_forward_img_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream2_ >> >
                        (count, raw_path.data_ptr<scalar_t>(), top_data_[3].data_ptr<scalar_t>(),rand_vec.data_ptr<int>(),
                            2, start_idx.data_ptr<int>(), windows_, stride_, nout_, num_);
                    count = bn_ * nout_ * npred_ * 2;
                    pre_data_forward_label_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream3_ >> >
                        (count, bottom_path.data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(),rand_vec.data_ptr<int>(),
                            pathw_, start_idx.data_ptr<int>(), npred_,  path_mid_, stride_, nout_, num_);
                    cudaDeviceSynchronize();
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
