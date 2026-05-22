#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class InvTransSample_opt: public base_opt{
	public:
		InvTransSample_opt( int nsample, float stride_h, float stride_w, int device = 0, bool timeit=false){
			nsample_ = nsample;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
			base_opt_init(device,timeit);
		}
		~InvTransSample_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options){}
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor random_v, at::Tensor old_p);
		std::vector<at::Tensor> selcet_positions(at::Tensor sort_idx);
		void select_tran_idx(at::Tensor sort_idx);
		std::vector<at::Tensor> selcet_threshold(at::Tensor old_p, float threshold);
		int nsample_;
		float stride_h_ , stride_w_, h_bias_, w_bias_;
		at::Tensor didx_, eidx_;
};
