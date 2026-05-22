#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class vp2erp_opt: public base_opt{
	public:
		vp2erp_opt(float fov, int h_v, int w_v, int device = 0, bool timeit=false){
			pi_ = acos(-1.0);
			fov_ = fov / 180 * pi_;
			h_v_ = h_v;
			w_v_ = w_v;
			base_opt_init(device,timeit);
		}
		~vp2erp_opt(){}
		void init();
		void reshape(int num, int channel, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor rota);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff){return {};}
		float fov_, pi_, wangle_;
		int h_v_;
		int w_v_;
};
