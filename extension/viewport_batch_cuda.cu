#include "viewport_batch.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

template <typename scalar_t>
__global__ void vp_bt_init_xyz_kernel(int num, scalar_t * data, int height, int width, float w_stride, float h_stride, float c_x, float c_y){
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        scalar_t x = 1.;
        scalar_t y = (w - c_x + 0.5)*w_stride;
        scalar_t z = (h - c_y + 0.5)*h_stride;
        scalar_t r = sqrt(x*x + y*y + z*z);
        data[i*3] = x/r;
        data[i*3+1] = y/r;
        data[i*3+2] = -z/r;
    }
}

template <typename scalar_t>
__global__ void bt_rotation_kernels(const int nthreads, const scalar_t* const theta_phi, scalar_t * const r, 
    const int len, const int start_idx, const int wind, const scalar_t pi){
	CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % wind;
        int pn = index / wind;
        int pw = ps + start_idx;
        if(pw<0||pw>=len) continue;
        int pidx = pn*len + pw;
		scalar_t a11,a12,a13,a21,a22,a23,a31,a32,a33;
		scalar_t b11,b12,b13,b21,b22,b23,b31,b32,b33;
		scalar_t c,s;
        scalar_t t1 = theta_phi[pidx*2] / 180. * pi;
        scalar_t t2 = theta_phi[pidx*2+1] / 180. * pi;
		a11 = cos(t1);
		a12 = -sin(t1);
		a13 = 0;
		a21 = -a12;
		a22 = a11;
		a23 = 0;
		a31 = 0;
		a32 = 0;
		a33 = 1;
		c = cos(t2);
		s = sin(t2);
		b11 = c + (1-c)*a12*a12;
		b12 = (1-c)*a12*a22;
		b13 = -s * a22;
		b21 = (1-c)*a12*a22;
		b22 = c + (1-c)*a22*a22;
		b23 =  s * a12;
		b31 =  s * a22;
		b32 = -s * a12;
		b33 = c;
		r[index*9] = b11*a11 + b12*a21 + b13*a31;
		r[index*9+1] = b11*a12 + b12*a22 + b13*a32;
		r[index*9+2] = b11*a13 + b12*a23 + b13*a33;
		r[index*9+3] = b21*a11 + b22*a21 + b23*a31;
		r[index*9+4] = b21*a12 + b22*a22 + b23*a32;
		r[index*9+5] = b21*a13 + b22*a23 + b23*a33;
		r[index*9+6] = b31*a11 + b32*a21 + b33*a31;
		r[index*9+7] = b31*a12 + b32*a22 + b33*a32;
		r[index*9+8] = b31*a13 + b32*a23 + b33*a33;
	}
}

template <typename scalar_t>
__global__ void vp_bt_transpose_kernel(int num, const scalar_t * const x, const scalar_t * y, scalar_t * const z, 
    const int m, const int start_idx, const int windows, const int len){
    CUDA_KERNEL_LOOP(i, num) {
        int tb = i / m;
        int pw = tb % windows + start_idx;
        if(pw<0 ||  pw>=len) continue;
        int ibase = (i % m) *3;
        int obase = i * 3;
        int base_y = tb * 9;
        float xa = x[ibase];
        float xb = x[ibase + 1];
        float xc = x[ibase + 2];
        z[obase] = xa * y[base_y] + xb * y[base_y+1] + xc * y[base_y+2];
        z[obase + 1] = xa * y[base_y+3] + xb * y[base_y+4] + xc * y[base_y+5];
        z[obase + 2] = xa * y[base_y+6] + xb * y[base_y+7] + xc * y[base_y+8];
    }
}


void viewport_batch_opt::init(){
    init_base();
    height_ = -1;
    width_ = -1;
    float hfov = fov_ * h_out_ / w_out_ /2;
    float wfov = fov_ / 2;
    c_x_ = w_out_ / 2.0;
    c_y_ = h_out_ / 2.0;
    float pi_2 = pi_ / 2;
    wangle_ = pi_2 - wfov;
    float hangle = pi_2 - hfov;
    w_stride_ = 2 * sin(wfov) / sin(wangle_) / w_out_;
    h_stride_ = 2 * sin(hfov) / sin(hangle) / h_out_;
    rad_ = 0.5 * w_out_ * tan(wangle_);
    init_rota_ = false;
}

bool viewport_batch_opt::reshape(int num, int channel, int height, int width){
     return reshape_base(num, channel, height, width); 
}

void viewport_batch_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({batch_, windows_, channel_, h_out_, w_out_});
    shapes.push_back({h_out_, w_out_, 3});
    shapes.push_back({batch_, windows_, h_out_, w_out_, 3});
    shapes.push_back({batch_, windows_, h_out_, w_out_, 2});
    shapes.push_back({batch_, windows_, 2*npred_+1, 2});
    shapes.push_back({batch_, 2});
    reshape_top_base(option,shapes);
}

void viewport_batch_opt::reshape_rota(at::TensorOptions options){
    std::vector<int64_t> shapes = {batch_,windows_,9};
    if(!init_rota_){
        rota_ = at::empty(shapes, options);
    }else{
        if(!is_same_shape(rota_.sizes(),shapes)){
            rota_ = at::empty(shapes, options);
        }
    }
}

template <typename scalar_t>
__global__ void vp_bt_cal_xyz_kernel(int num, scalar_t * const xyz, scalar_t * tf,  scalar_t hx, scalar_t hy, scalar_t pi, 
    const int start_idx, const int windows, const int len, const int inner_shape){
    CUDA_KERNEL_LOOP(i, num) {
        int pw = (i / inner_shape) % windows + start_idx;
        if(pw<0 || pw>=len) continue;
        scalar_t lat = asin(xyz[i*3+2]);
        scalar_t tx = xyz[i*3];
        scalar_t ty = xyz[i*3+1];
        scalar_t theta = atan(ty/tx);
        if (tx<=0){
            if(ty>0){
                theta = theta + pi;
            }else{
                theta = theta - pi;
            }
        }
        tf[i*2] = (0.5 * theta / pi + 0.5) * hx - 0.5;
        tf[i*2+1] = (0.5 - lat / pi) * hy - 0.5;  
    }
}


template <typename scalar_t>
__global__ void viewport_bt_forward_kernel(const int nthreads, const scalar_t* const input,  
    const scalar_t * tf, scalar_t * const output, const int inner_shape,  const int hs, const int ws, 
    const int channel, const int start_idx, const int window, const int len) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pbase = index / inner_shape;
        int pc = pbase % channel;
        int qbase =  pbase / channel;
        int qw = qbase % window;
        int pw = qw + start_idx;
        if(pw<0 || pw>=len){
            output[index] = 0;
            continue;
        }
        int tbase = qw*channel+pc;
        int base = qbase*2*inner_shape;
        int tw = static_cast<int>(floor(tf[base + 2*ps]));
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]));
        int ah = th > 0 ? th : 0;
        int bh = th + 1 >= hs ? hs-1 : th + 1;
        int aw = (tw + ws) % ws;
        int bw = (tw + 1) % ws;
        scalar_t tx = tf[base + 2*ps] - tw;
        scalar_t ty = tf[base + 2*ps+1] - th;
        scalar_t ntx = 1. - tx;
        scalar_t nty = 1. - ty;
        output[index] = input[(tbase*hs+ah)*ws + aw]*ntx*nty + input[(tbase*hs+ah)*ws + bw]*tx*nty +  input[(tbase*hs+bh)*ws + aw]*ntx*ty + input[(tbase*hs+bh)*ws + bw]*tx*ty; 
    }
}



void viewport_batch_opt::set_video_sequence(at::Tensor  bottom_data, int start_idx, int len){
    idx_ = start_idx;
    len_ = len;
    assert("at least the startpoint should be given in prediction"&&(start_idx>=1));
    bool rp =  reshape(windows_, bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
    reshape_rota(bottom_data.options());
    if(rp){
        AT_DISPATCH_FLOATING_TYPES(
            bottom_data.scalar_type(), "viewport_forward_cuda", 
                ([&] {
                    int count = h_out_*w_out_;
                    vp_bt_init_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>> >
                        (count, top_data_[1].data_ptr<scalar_t>(),h_out_,w_out_,w_stride_,h_stride_, c_x_, c_y_);
                    CUDA_POST_KERNEL_CHECK;
                })
        );    
    }
}

template <typename scalar_t>
__global__ void erp2vp_bt_forward_kernel(int num, const scalar_t * const tf, const scalar_t * y, scalar_t * const out,
     const scalar_t rad, const int start_idx, const int end_idx,
    const int len, const int wind, const int dwind, const int npred, const scalar_t pi){
    CUDA_KERNEL_LOOP(i, num) {
        int ps = i % dwind;
        int t1 = i / dwind;
        int tw = t1 % wind;
        int pw = tw + start_idx;
        if(pw<0 || pw>len){
            out[i*2] = 0;
            out[i*2 + 1] = 0;
            continue;
        }
        int qw = pw + ps - npred;
        if(qw<0 || qw>=end_idx || pw==qw){
            out[i*2] = 0;
            out[i*2 + 1] = 0;
            continue;
        }
        int pn = t1 / wind;
        int base_y = (pn*wind + tw) * 9;
        int qbase = (pn*len + qw)*2; 
        scalar_t tf1 = tf[qbase] / 180. * pi;
        scalar_t tf2 = tf[qbase+1] / 180. * pi;
        scalar_t ts = sin(tf1);
        scalar_t tc = cos(tf1);
        scalar_t fs = sin(tf2);
        scalar_t fc = cos(tf2);
        scalar_t xa = tc*fc;
        scalar_t xb = ts*fc;
        scalar_t xc = fs;
        scalar_t za = xa * y[base_y]   + xb * y[base_y+3] + xc * y[base_y+6];
        scalar_t zb = xa * y[base_y+1] + xb * y[base_y+4] + xc * y[base_y+7];
        scalar_t zc = xa * y[base_y+2] + xb * y[base_y+5] + xc * y[base_y+8];
        scalar_t gamma = rad / za;
        out[i*2] = gamma*zb;
        out[i*2 + 1] = -gamma*zc;
    }
}

template <typename scalar_t>
__global__ void target_bt_forward_kernel(int num, const scalar_t * const tf, const scalar_t * y, scalar_t * const out,
     const scalar_t rad, const int start_idx,  const int len, const int wind,  const scalar_t pi){
    CUDA_KERNEL_LOOP(i, num) {
        int pw = start_idx;
        if(pw<0 || pw>len){
            out[i*2] = 0;
            out[i*2 + 1] = 0;
            continue;
        }
        int pn = i;
        int base_y = (pn*wind + wind-1) * 9;
        int qbase = (pn*len + pw)*2; 
        scalar_t tf1 = tf[qbase] / 180. * pi;
        scalar_t tf2 = tf[qbase+1] / 180. * pi;
        scalar_t ts = sin(tf1);
        scalar_t tc = cos(tf1);
        scalar_t fs = sin(tf2);
        scalar_t fc = cos(tf2);
        scalar_t xa = tc*fc;
        scalar_t xb = ts*fc;
        scalar_t xc = fs;
        scalar_t za = xa * y[base_y]   + xb * y[base_y+3] + xc * y[base_y+6];
        scalar_t zb = xa * y[base_y+1] + xb * y[base_y+4] + xc * y[base_y+7];
        scalar_t zc = xa * y[base_y+2] + xb * y[base_y+5] + xc * y[base_y+8];
        scalar_t gamma = rad / za;
        out[i*2] = gamma*zb;
        out[i*2 + 1] = -gamma*zc;
    }
}


std::vector<at::Tensor>  viewport_batch_opt::forward_cuda(at::Tensor video, at::Tensor theta_phi) 
{
	int count;
    int start_idx = idx_ - windows_;
	AT_DISPATCH_FLOATING_TYPES(
		theta_phi.scalar_type(), "viewport_forward_cuda", 
        ([&] {
            count = batch_* windows_;
            scalar_t spi = pi_;            
            bt_rotation_kernels<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                (count,theta_phi.data_ptr<scalar_t>(),rota_.data_ptr<scalar_t>(), len_, start_idx, windows_, spi);
            CUDA_POST_KERNEL_CHECK;
            count = batch_ * windows_ * h_out_ * w_out_;
            vp_bt_transpose_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                (count, top_data_[1].data_ptr<scalar_t>(), rota_.data_ptr<scalar_t>(), 
                    top_data_[2].data_ptr<scalar_t>(), h_out_*w_out_, start_idx, windows_, len_);
            scalar_t hx = width_;
            scalar_t hy = height_;
            vp_bt_cal_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                (count, top_data_[2].data_ptr<scalar_t>(), top_data_[3].data_ptr<scalar_t>(), hx, hy, spi,
                   start_idx, windows_, len_, h_out_*w_out_);
            count = count * channel_;
            viewport_bt_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                (count, video.data_ptr<scalar_t>(), top_data_[3].data_ptr<scalar_t>(),  top_data_[0].data_ptr<scalar_t>(), 
                    h_out_*w_out_, height_, width_, channel_, start_idx, windows_, len_);
            count = batch_ * windows_ * (2*npred_+1);
            scalar_t rad = rad_;
            erp2vp_bt_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                (count, theta_phi.data_ptr<scalar_t>(), rota_.data_ptr<scalar_t>(), top_data_[4].data_ptr<scalar_t>(),     
                    rad,  start_idx, idx_, len_, windows_, 2*npred_+1, npred_, spi);
            count = batch_;
            target_bt_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                (count, theta_phi.data_ptr<scalar_t>(), rota_.data_ptr<scalar_t>(), top_data_[5].data_ptr<scalar_t>(),
                    rad, idx_,  len_, windows_,  spi);
            CUDA_POST_KERNEL_CHECK;
        })
    );
    return {top_data_[0],top_data_[4],top_data_[5]};
}

template<typename scalar_t>
void __global__ fork_bt_copy(const int count, scalar_t * const data, const int * const fid, const int inner_shape){
    CUDA_KERNEL_LOOP(index, count) {
        int ps = index % inner_shape;
        int pn = index / inner_shape;
        int qn = fid[pn];
        if (qn!=pn){
            data[index] = data[qn*inner_shape+ps];
        }
    }
}

template <typename scalar_t>
__global__ void vp2erp_bt_forward_kernel(const int nthreads, const scalar_t* const input,  const scalar_t* const y,
     scalar_t * const output, const scalar_t rad, const scalar_t pi, const int start_idx, const int len, const int npred, const int window) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % npred + start_idx;
        if(pw>=len){
            continue;
        }
        int pn = index / npred;
        int base_y = (pn*window + window - 1) * 9;
        scalar_t xa = rad;
        scalar_t xb = input[index*2];
        scalar_t xc = -input[index*2+1];
        scalar_t rn = sqrt(xa*xa + xb*xb + xc*xc);
        xa /= rn;
        xb /= rn;
        xc /= rn;
        scalar_t za = xa * y[base_y] + xb * y[base_y+1] + xc * y[base_y+2];
        scalar_t zb = xa * y[base_y+3] + xb * y[base_y+4] + xc * y[base_y+5];
        scalar_t zc = xa * y[base_y+6] + xb * y[base_y+7] + xc * y[base_y+8];
        scalar_t lat = asin(zc);
        scalar_t theta = atan(zb/za);
        if (za<=0){
            if(zb>0){
                theta = theta + pi;
            }else{
                theta = theta - pi;
            }
        }
        int pidx = (pn*len + pw)*2; 
        output[pidx] = theta / pi * 180;
        output[pidx+1] =  lat / pi * 180;
    }
}

at::Tensor viewport_batch_opt::set_path_sequence(at::Tensor old_path, at::Tensor path_data, at::Tensor fork_idxs, bool p1){

    int count;
    AT_DISPATCH_FLOATING_TYPES(
		old_path.scalar_type(), "viewport_forward_cuda", 
        ([&] {
            count = batch_ * len_ * 2;
            fork_bt_copy<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                (count,old_path.data_ptr<scalar_t>(), fork_idxs.data_ptr<int>(),len_*2);
            count = batch_ * windows_ * 9;
            fork_bt_copy<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                (count,rota_.data_ptr<scalar_t>(), fork_idxs.data_ptr<int>(),windows_*9);
            
            int nl = npred_;
            if(p1){
                count = batch_ * (npred_+1);
                nl = npred_ + 1;
            }else{
                count = batch_ * npred_;
            } 
            vp2erp_bt_forward_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                (count, path_data.data_ptr<scalar_t>(),  rota_.data_ptr<scalar_t>(), old_path.data_ptr<scalar_t>(),
                     static_cast<scalar_t>(rad_) , static_cast<scalar_t>(pi_), idx_, len_, nl, windows_);
            CUDA_POST_KERNEL_CHECK;
        })
    );
    idx_ += npred_;
    return old_path;
}