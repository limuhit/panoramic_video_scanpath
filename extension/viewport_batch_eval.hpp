#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class viewport_batch_eval_opt: public base_opt{
	public:
		viewport_batch_eval_opt(float fov, int h, int w, int windows, int npred, int topk, int device = 0, bool timeit=false){
			pi_ = acos(-1.0);
			fov_ = fov / 180 * pi_;
			h_out_ = h;
			w_out_ = w;
			windows_ = windows;
			npred_ = npred;
			batch_ = topk;
			base_opt_init(device,timeit);
		}
		~viewport_batch_eval_opt(){}
		void init();
		void set_video_sequence(at::Tensor  bottom_data, int start_idx,int len);
		bool reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_rota(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor video, at::Tensor theta_phi);
		float fov_, wangle_;
		float c_x_, c_y_, w_stride_, h_stride_, pi_, rad_;
		int batch_, windows_, npred_, idx_, len_;
		at::Tensor rota_;
		bool init_rota_;
};
