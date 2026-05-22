#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class linear_mask_opt: public base_opt{
	public:
		linear_mask_opt(int ncontext, int npred, int cin, int cout, bool hidden, bool out, int device = 0, bool timeit=false){
			ncontext_ = ncontext;
			npred_ = npred;
			cin_ = cin;
			cout_ = cout;
			hidden_ = hidden;
			ngroup_ = ncontext + npred;
			channel_ = cin_ * ngroup_;
			if(out){
				nout_ = cout_ * npred_;
			}else{
				nout_ = cout_ * ngroup_;
			}
			out_ = out;
			base_opt_init(device,timeit);
		}
		~linear_mask_opt(){}
		void init();
		at::Tensor  forward_cuda();
		int ncontext_, npred_, cin_, cout_, ngroup_;
		int nout_;
		bool hidden_, out_;
		at::Tensor mask_;
};
