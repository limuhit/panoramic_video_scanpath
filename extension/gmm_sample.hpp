#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class gmm_sample_opt: public base_opt{
	public:
		gmm_sample_opt(int nout, float thr, int device = 0, bool timeit=false){
			nout_ = nout;
			thr_ = thr;
			base_opt_init(device,timeit);
		}
		~gmm_sample_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor wt, at::Tensor mean, at::Tensor std, at::Tensor r1, at::Tensor r2);
		std::vector<at::Tensor>  select(at::Tensor out, float x_bound, float y_bound);
		int nout_;
		float thr_;
};
