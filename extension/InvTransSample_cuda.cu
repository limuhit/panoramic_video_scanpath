#include "InvTransSample.hpp"
#include <curand.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void InvTransSample_opt::init(){
    init_base();
}

void InvTransSample_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return;
    h_bias_ = (height_-1) / 2.;
	w_bias_ = (width_-1) / 2.;
}

void InvTransSample_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    shapes.push_back({num_,nsample_,2});
    shapes.push_back({num_,nsample_});
    shapes.push_back({num_,2});
    shapes.push_back({num_});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void InvTransSample_sum_kernel(const int nthreads, scalar_t* const input, const int inner_shape, const int cmod, const int omod){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pn = index / inner_shape;
        int ps = index % inner_shape;
        int pw = ps % cmod;
        int ph = pw / omod;
        if(ph>0){
            input[index] += input[index-pw+omod-1];
        }
    }
}

template <typename scalar_t>
__global__ void InvTransSample_forward_kernel(const int nthreads, const scalar_t* const tab,  const scalar_t* const randv,
     scalar_t * const output, const scalar_t * const old_p, scalar_t* const new_p,
     const int inner_shape, const int width, const int nsample,
     const scalar_t stride_h, const scalar_t stride_w, const scalar_t h_bias, const scalar_t w_bias) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pid = index / nsample;
        int ps = pid * inner_shape;
        int l = 0, r = inner_shape - 1, mid;
        scalar_t val = randv[index] * (tab[ps+r]+1e-6);
        while(l<r){
            mid = (l + r) / 2;
            if(tab[ps+mid]>=val){
                r = mid;
            }else{
                l = mid + 1;
            }
        }
        output[index*2] = (l % width - w_bias) * stride_w;
        output[index*2+1] = (l / width - h_bias) * stride_h;

        if(l>0){
            new_p[index] = old_p[pid] - log2(tab[ps+l]-tab[ps+l-1]);
        }else{
            new_p[index] = old_p[pid] - log2(tab[ps+l]);
        }
    }
}

std::vector<at::Tensor>  InvTransSample_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor  random_val, at::Tensor old_p)
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "InvTransSample_forward_cuda",
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    caffe_gpu_memcpy(count*sizeof(scalar_t), bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>());
                    int inner_shape = channel_ * width_ * height_;
                    int nloop = ceil(log2(inner_shape));
                    for(int i=0, omod=1; i<nloop; i++, omod*=2){
                        InvTransSample_sum_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_data_[0].data_ptr<scalar_t>(), inner_shape, omod*2, omod);
                    }
                    count = num_*nsample_;
                    InvTransSample_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count,  top_data_[0].data_ptr<scalar_t>(), random_val.data_ptr<scalar_t>(),
                             top_data_[1].data_ptr<scalar_t>(), old_p.data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(),
                             inner_shape, width_, nsample_, static_cast<scalar_t>(stride_h_), static_cast<scalar_t>(stride_w_),
                              static_cast<scalar_t>(h_bias_), static_cast<scalar_t>(w_bias_));
                    CUDA_POST_KERNEL_CHECK;
                }
            )
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void InvTransSample_select_kernel(const int nthreads, const scalar_t* const input, scalar_t* const output,
    const scalar_t* const vinput, scalar_t* const voutput, const int * idx){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pidx = idx[index];
        output[index*2] = input[pidx*2];
        output[index*2+1] = input[pidx*2+1];
        voutput[index] = vinput[pidx];
    }
}

void InvTransSample_opt::select_tran_idx(at::Tensor sort_idx){
    at::Tensor sort_cpu = sort_idx.to(torch::Device(torch::kCPU));
    didx_ = at::zeros({num_},at::kInt);
    eidx_ = at::zeros({num_},at::kInt);
    int * sidx = sort_cpu.data_ptr<int>();
    int * didx = didx_.data_ptr<int>();
    int * eidx = eidx_.data_ptr<int>();
    bool * flag = new bool[num_];
    memset(flag,0,num_*sizeof(bool));
    int pa;
    for(int i=0;i<num_;i++){
        pa = sidx[i] / nsample_;
        if(!flag[pa]){
            didx[pa] = sidx[i];
            eidx[pa] = pa;
            flag[pa] = true;
        }
    }
    for(int i=0, j=0;i<num_;i++){
        pa = sidx[i] / nsample_;
        if(didx[pa]!=sidx[i]){
            while(flag[j]) j++;
            didx[j] = sidx[i];
            eidx[j] = pa;
            j++;
        }
    }

    delete flag;
    didx_ = didx_.to(torch::Device(torch::kCUDA, device_));
    eidx_ = eidx_.to(torch::Device(torch::kCUDA, device_));
}

std::vector<at::Tensor> InvTransSample_opt::selcet_positions(at::Tensor sort_idx){
    select_tran_idx(sort_idx);
    int count = num_;
	AT_DISPATCH_FLOATING_TYPES(
		top_data_[1].scalar_type(), "InvTransSample_select_cuda",
			([&] {
                InvTransSample_select_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, top_data_[1].data_ptr<scalar_t>(),  top_data_[3].data_ptr<scalar_t>(),
                       top_data_[2].data_ptr<scalar_t>(), top_data_[4].data_ptr<scalar_t>(), didx_.data_ptr<int>());
                CUDA_POST_KERNEL_CHECK;
            })
    );
    return {eidx_,top_data_[3],top_data_[4]};
}

template <typename scalar_t>
__global__ void InvTransSample_select_thr_kernel(const int nthreads, const scalar_t* const input, scalar_t* const output,
    const scalar_t* const vinput, scalar_t* const voutput, const scalar_t *  vold,  int * const eidx,
    const int nsample, const scalar_t threshold){
    CUDA_KERNEL_LOOP(index, nthreads) {
        scalar_t lp;
        scalar_t lold = vold[index];
        int pidx = index*nsample;
        for(int pend = pidx + nsample-1; pidx<pend; pidx++){
            lp = vinput[pidx] - lold;
            if(lp<threshold) break;
        }
        eidx[index] = pidx;
        output[index*2] = input[pidx*2];
        output[index*2+1] = input[pidx*2+1];
        voutput[index] = vinput[pidx];
    }
}

std::vector<at::Tensor> InvTransSample_opt::selcet_threshold(at::Tensor old_p, float threshold){
    int count = num_;
    auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_);
    eidx_ = at::zeros({num_},options);
	AT_DISPATCH_FLOATING_TYPES(
		top_data_[1].scalar_type(), "InvTransSample_select_cuda",
			([&] {
                scalar_t thr = -log2(threshold);
                InvTransSample_select_thr_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, top_data_[1].data_ptr<scalar_t>(),  top_data_[3].data_ptr<scalar_t>(),  top_data_[2].data_ptr<scalar_t>(),
                        top_data_[4].data_ptr<scalar_t>(), old_p.data_ptr<scalar_t>(), eidx_.data_ptr<int>(), nsample_, thr);
                CUDA_POST_KERNEL_CHECK;
            })
    );
    return {eidx_,top_data_[3],top_data_[4]};
}
