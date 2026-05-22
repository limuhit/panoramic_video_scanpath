#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class erp2vp_opt: public base_opt{
	public:
		erp2vp_opt(float fov, int h_v, int w_v, int device = 0, bool timeit=false){
			pi_ = acos(-1.0);
			fov_ = fov / 180 * pi_;
			h_v_ = h_v;
			w_v_ = w_v;
			base_opt_init(device,timeit);
		}
		~erp2vp_opt(){}
		void init();
		void reshape(int num, int channel, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor rota);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff, at::Tensor rota);
		float fov_, pi_, wangle_;
		int h_v_;
		int w_v_; 
};
