#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "math.h"
class GMM_2D_Table_opt: public base_opt{
	public:
		GMM_2D_Table_opt(int num_gaussian, int hboard, int wboard, float stride_h, float stride_w, int device = 0, bool timeit=false){
			num_gaussian_ = num_gaussian;
			sq2_ = 1. / sqrt(2);
			h_out_ = hboard;
			w_out_ = wboard;
			stride_h_ = stride_h;
			stride_w_ = stride_w;
			hf_ = stride_h / 2;
			wf_ = stride_w / 2;
			h_bias_ = (h_out_-1) / 2;
			w_bias_ = (w_out_-1) / 2;
			base_opt_init(device,timeit);
		}
		~GMM_2D_Table_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor wt, at::Tensor mean, at::Tensor std);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff) {return {};}
		int num_gaussian_;
		float sq2_;
		float stride_h_ , stride_w_, h_bias_, w_bias_, hf_, wf_;
};
