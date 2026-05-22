#include "data_manager.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void data_manager_opt::init(){
    init_base();
    if(old_device_!=device_){
        streams_[0] = stream_;
        for(int i=1; i<ntensor_; i++){
            if(old_device_>=0) cudaStreamDestroy(streams_[i]);
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
    }
    old_device_ = device_;
}

void data_manager_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
}

void data_manager_opt::clear_tensors(){
    data_.clear();
    data_shape_.clear();
}

int __inline__ cal_inner_shape(at::IntArrayRef shape){
    int tshape = 1;
    for(int i = 1; i<shape.size(); i++){
        tshape *= shape[i];
    }
    return tshape;
}

void data_manager_opt::push_tensor(at::Tensor  bottom_data){
    int num = bottom_data.size(0);
    int inner_shape = cal_inner_shape(bottom_data.sizes());
    data_.push_back(bottom_data);
    data_shape_.push_back({num,inner_shape});
    assert(data_.size()<=ntensor_ && "the number of pushed tensor should be less than the parameter ntensor");
}



template<typename scalar_t>
void __global__ data_manager_copy_kernel(const int count, scalar_t * const data, const int * const fid, const int inner_shape){
    CUDA_KERNEL_LOOP(index, count) {
        int ps = index % inner_shape;
        int pn = index / inner_shape;
        int qn = fid[pn];
        if (qn!=pn){
            data[index] = data[qn*inner_shape+ps];
        }
    }
}

std::vector<at::Tensor>  data_manager_opt::forward_cuda(at::Tensor  copy_idx) 
{
    reshape(copy_idx.size(0), 1, 1, 1);
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		data_[0].scalar_type(), "data_manager_forward_cuda", 
        ([&] {
            for(int pi = 0; pi<data_.size(); pi++){
                count = data_shape_[pi][0] * data_shape_[pi][1];
                data_manager_copy_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, streams_[pi] >> >
                    (count, data_[pi].data_ptr<scalar_t>(), copy_idx.data_ptr<int>(), data_shape_[pi][1]);
            }
            sync();
            CUDA_POST_KERNEL_CHECK;
        })
    );
    return data_;
}

