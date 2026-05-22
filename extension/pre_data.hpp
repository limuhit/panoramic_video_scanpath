#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class pre_data_opt: public base_opt{
	public:
		pre_data_opt(float x_mean, float y_mean, int pathw, int nw, int windows, int npred, int stride, bool train, int device = 0, bool timeit=false){
			pathw_ = pathw;
			path_mid_ = pathw / 2;
			nw_ = nw;
			windows_ = windows;
			npred_ = npred;
			stride_ = stride;
			train_ = train;
			x_mean_ = x_mean;
			y_mean_ = y_mean;
			base_opt_init(device,timeit);
		}
		~pre_data_opt()
		{
			cudaStreamDestroy(stream2_);
			cudaStreamDestroy(stream3_);
		}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_path, at::Tensor rand_vec, at::Tensor start_idx);
		std::vector<at::Tensor>  forward_cuda2(at::Tensor  bottom_data, at::Tensor bottom_path, at::Tensor raw_path, at::Tensor rand_vec, at::Tensor start_idx);
		int nw_, pathw_, path_mid_, nout_;
		int windows_;
		int npred_;
		int stride_;
		bool train_;
		int start_idx_;
		int bn_;
		cudaStream_t stream2_, stream3_;
		float x_mean_, y_mean_;
};
