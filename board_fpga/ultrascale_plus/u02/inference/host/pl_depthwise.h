#pragma once
#include "pl_conv_adapter.h"
#include "../hls/protocol.h"

class PlDepthwise : public ncnn::Layer {
    Backend* backend;
    int channels,kw,kh,sw,sh,left,right,top,bottom,count,bias_term,activation;
    std::vector<int8_t> weights;
    std::vector<float> scales,bias;
public:
    explicit PlDepthwise(Backend* b):backend(b) {one_blob_only=true;}
    int load_param(const ncnn::ParamDict& p) override {
        channels=p.get(0,0);kw=p.get(1,0);kh=p.get(11,kw);sw=p.get(3,1);sh=p.get(13,sw);
        left=p.get(4,0);right=p.get(15,left);top=p.get(14,left);bottom=p.get(16,top);
        count=p.get(6,0);bias_term=p.get(5,0);activation=p.get(9,0);
        if(channels<1 || channels>512 || p.get(7,0)!=channels ||
           (kw!=1 && kw!=3) || (kh!=1 && kh!=3) || count!=channels*kw*kh ||
           p.get(2,1)!=1 || p.get(12,1)!=1 || p.get(8,0) ||
           (activation!=0 && activation!=1) || left<0 || right<0 || top<0 || bottom<0)
            return -1;
        return 0;
    }
    int load_model(const ncnn::ModelBin& mb) override {
        ncnn::Mat w=mb.load(count,0);if(w.empty() || w.elemsize!=4)return -1;
        bias.assign(channels,0);weights.resize(count);scales.resize(channels);
        if(bias_term) {ncnn::Mat b=mb.load(channels,1);if(b.empty())return -1;
            std::copy((const float*)b,(const float*)b+channels,bias.begin());}
        const float* data=w;const int k=kw*kh;
        for(int ch=0;ch<channels;++ch) {
            float a=0;for(int q=0;q<k;++q) {if(!std::isfinite(data[ch*k+q]))return -1;a=std::max(a,std::abs(data[ch*k+q]));}
            scales[ch]=a>0?a/127.f:1.f;
            for(int q=0;q<k;++q) weights[ch*k+q]=pose_quantize(data[ch*k+q],scales[ch]);
        }
        return 0;
    }
    int forward(const ncnn::Mat& in,ncnn::Mat& out,const ncnn::Option& opt) const override {
        try {
            if(in.dims!=3 || in.c!=channels || in.elempack!=1 || in.elemsize!=4)return -1;
            const int ow=(in.w+left+right-kw)/sw+1,oh=(in.h+top+bottom-kh)/sh+1;
            PoseConv p={in.w,in.h,channels,kw,kh,sw,sh,left,top,ow,oh,0};
            if(!light_dw_valid(p,channels,ow*oh,kw*kh))throw std::runtime_error("Depthwise shape outside PL contract");
            out.create(ow,oh,channels,4u,1,opt.blob_allocator);if(out.empty())return -100;
            const int plane=in.w*in.h,pixels=ow*oh;
            std::unique_ptr<int8_t[]> image(new int8_t[size_t(channels)*plane]);std::vector<float> input_scale(channels);
            std::vector<unsigned char> valid(channels,1);
            std::vector<float> input_maxima(backend->collect_calibration?channels:0);
            const auto qstart=std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(opt.num_threads)
            for(int ch=0;ch<channels;++ch) {
                float a=0;const float* src=in.channel(ch);
                if(!pose_max_abs(src,plane,a)) {valid[ch]=0;continue;}
                input_scale[ch]=a>0?a/127.f:1.f;
                if(backend->collect_calibration) input_maxima[ch]=a;
                pose_quantize_array(src,image.get()+ch*plane,plane,input_scale[ch]);
            }
            for(int ch=0;ch<channels;++ch)if(!valid[ch])throw std::runtime_error("Nonfinite depthwise input");
            if(backend->collect_calibration)
                backend->observe_input(name,channels,in.h,in.w,input_maxima,input_scale);
            backend->quantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-qstart).count();
            std::unique_ptr<int32_t[]> scratch(new int32_t[size_t(channels)*pixels]);
            const int32_t* result=backend->run_depthwise(weights.data(),image.get(),scratch.get(),channels,pixels,kw*kh,p);
            const auto dstart=std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(opt.num_threads)
            for(int ch=0;ch<channels;++ch) {
                float* dst=out.channel(ch);
                int i=0;
#ifdef __ARM_NEON
                for(;i+4<=pixels;i+=4) {
                    float32x4_t v=vcvtq_f32_s32(vld1q_s32(result+ch*pixels+i));
                    v=vmulq_n_f32(v,input_scale[ch]);v=vmulq_n_f32(v,scales[ch]);
                    v=vaddq_f32(v,vdupq_n_f32(bias[ch]));
                    if(activation==1)v=vmaxq_f32(v,vdupq_n_f32(0.f));
                    vst1q_f32(dst+i,v);
                }
#endif
                for(;i<pixels;++i) {
                    const float v=result[ch*pixels+i]*input_scale[ch]*scales[ch]+bias[ch];
                    dst[i]=activation==1?std::max(0.f,v):v;
                }
            }
            backend->dequantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-dstart).count();
            return 0;
        } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return -1;}
    }
};
static ncnn::Layer* create_depthwise(void* userdata) {return new PlDepthwise((Backend*)userdata);}
