#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
const static int max_level_ = 64;
class data_manager_opt: public base_opt{
	public:
		data_manager_opt(int ntensor, int device = 0, bool timeit=false){
			ntensor_ = ntensor;
			streams_ = new cudaStream_t[ntensor];
			base_opt_init(device,timeit);
		}
		~data_manager_opt(){
			for(int i=1;i<ntensor_;i++)
				cudaStreamDestroy(streams_[i]);
			delete streams_;
		}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void clear_tensors();
		void push_tensor(at::Tensor  bottom_data);
		void sync(){
			cudaDeviceSynchronize();
		}
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		int ntensor_;
		std::vector<at::Tensor> data_;
		std::vector<std::vector<int>> data_shape_;
		cudaStream_t * streams_;
		int old_device_ = -1;
};
