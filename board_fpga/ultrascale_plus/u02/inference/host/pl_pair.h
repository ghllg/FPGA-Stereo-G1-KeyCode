#pragma once
#include "pl_conv_adapter.h"
#include "../hls/pair_contract.h"

class PlPair : public ncnn::Layer {
    Backend* backend;
    int channels,kw,stride,pad,pw_count,dw_count,pw_bias,dw_bias,activation,input_channels;
    std::vector<int8_t> pw_weights,dw_weights;
    std::vector<float> pw_scales,dw_scales,pw_offsets,dw_offsets,mid_scales;
public:
    explicit PlPair(Backend* b):backend(b) {one_blob_only=true;}
    int load_param(const ncnn::ParamDict& p) override {
        channels=p.get(0,0);kw=p.get(1,0);stride=p.get(2,0);pad=p.get(3,0);
        pw_count=p.get(4,0);dw_count=p.get(5,0);pw_bias=p.get(6,0);dw_bias=p.get(7,0);activation=p.get(8,0);
        if(channels<1 || channels>512 || pw_count<=0 || pw_count%channels ||
           dw_count!=channels*kw*kw || (kw!=1 && kw!=3) || (stride!=1 && stride!=2) ||
           pad<0 || pad>1 || (activation!=0 && activation!=1))return -1;
        input_channels=pw_count/channels;
        if(input_channels>512)return -1;
        ncnn::Mat scales=p.get(30,ncnn::Mat());
        if(scales.empty() || scales.total()!=size_t(channels))return -1;
        mid_scales.assign((const float*)scales,(const float*)scales+channels);
        for(float s:mid_scales)if(!(s>0) || !std::isfinite(s))return -1;
        return 0;
    }
    int load_model(const ncnn::ModelBin& mb) override {
        for(int part=0;part<2;++part) {
            const int count=part?dw_count:pw_count,has_bias=part?dw_bias:pw_bias,k=count/channels;
            auto& weights=part?dw_weights:pw_weights;auto& scales=part?dw_scales:pw_scales;
            auto& offsets=part?dw_offsets:pw_offsets;
            ncnn::Mat w=mb.load(count,0);if(w.empty() || w.elemsize!=4)return -1;
            weights.resize(count);scales.resize(channels);offsets.assign(channels,0);
            if(has_bias) {ncnn::Mat b=mb.load(channels,1);if(b.empty())return -1;
                std::copy((const float*)b,(const float*)b+channels,offsets.begin());}
            const float* src=w;
            for(int ch=0;ch<channels;++ch) {
                float a=0;if(!std::isfinite(offsets[ch]))return -1;
                for(int j=0;j<k;++j) {if(!std::isfinite(src[ch*k+j]))return -1;a=std::max(a,std::abs(src[ch*k+j]));}
                scales[ch]=a>0?a/127.f:1.f;
                for(int j=0;j<k;++j)weights[ch*k+j]=pose_quantize(src[ch*k+j],scales[ch]);
            }
        }
        return 0;
    }
    int forward(const ncnn::Mat& in,ncnn::Mat& out,const ncnn::Option& opt) const override {
        try {
            if(in.dims!=3 || in.c!=input_channels || in.elempack!=1 || in.elemsize!=4)return -1;
            const int ow=(in.w+2*pad-kw)/stride+1,oh=(in.h+2*pad-kw)/stride+1;
            PoseConv dw={in.w,in.h,channels,kw,kw,stride,stride,pad,pad,ow,oh,0};
            if(!light_pair_valid(dw,input_channels,ow*oh))throw std::runtime_error("Pair outside PL contract");
            out.create(ow,oh,channels,4u,1,opt.blob_allocator);if(out.empty())return -100;
            const int plane=in.w*in.h,pixels=ow*oh;
            std::unique_ptr<int8_t[]> image(new int8_t[size_t(input_channels)*plane]);
            std::vector<float> maxima(input_channels);std::vector<unsigned char> valid(input_channels,1);
            std::vector<LightAffine> affine(channels);
            const auto qstart=std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(opt.num_threads)
            for(int ch=0;ch<input_channels;++ch)
                if(!pose_max_abs(in.channel(ch),plane,maxima[ch]))valid[ch]=0;
            float maximum=0;
            for(int ch=0;ch<input_channels;++ch) {if(!valid[ch])throw std::runtime_error("Nonfinite pair input");maximum=std::max(maximum,maxima[ch]);}
            const float scale=maximum>0?maximum/127.f:1.f;
#pragma omp parallel for num_threads(opt.num_threads)
            for(int ch=0;ch<input_channels;++ch)pose_quantize_array(in.channel(ch),image.get()+ch*plane,plane,scale);
            for(int ch=0;ch<channels;++ch) {
                const double ratio=double(scale)*pw_scales[ch]/mid_scales[ch];
                int shift=30;
                while(shift>1 && std::ldexp(ratio,shift)>2147483647.0)--shift;
                const double mul=std::ldexp(ratio,shift),bias=std::ldexp(double(pw_offsets[ch])/mid_scales[ch],shift);
                if(!std::isfinite(mul+bias) || mul<0 || mul>2147483647.0 || std::abs(bias)>double(int64_t(1)<<60))
                    throw std::runtime_error("Affine coefficient overflow");
                affine[ch]={int32_t(std::llround(mul)),shift,int64_t(std::llround(bias))};
            }
            backend->quantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-qstart).count();
            std::unique_ptr<int32_t[]> scratch(backend->hardware()?nullptr:new int32_t[size_t(channels)*pixels]);
            const int32_t* sums=backend->run_pw_dw(pw_weights.data(),dw_weights.data(),image.get(),affine.data(),scratch.get(),input_channels,dw);
            const auto dstart=std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(opt.num_threads)
            for(int ch=0;ch<channels;++ch) {
                float* dst=out.channel(ch);int i=0;
#ifdef __ARM_NEON
                for(;i+4<=pixels;i+=4) {
                    float32x4_t v=vcvtq_f32_s32(vld1q_s32(sums+ch*pixels+i));
                    v=vmulq_n_f32(v,mid_scales[ch]);v=vmulq_n_f32(v,dw_scales[ch]);
                    v=vaddq_f32(v,vdupq_n_f32(dw_offsets[ch]));if(activation)v=vmaxq_f32(v,vdupq_n_f32(0.f));
                    vst1q_f32(dst+i,v);
                }
#endif
                for(;i<pixels;++i) {const float v=sums[ch*pixels+i]*mid_scales[ch]*dw_scales[ch]+dw_offsets[ch];dst[i]=activation?std::max(0.f,v):v;}
            }
            backend->dequantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-dstart).count();
            return 0;
        } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return -1;}
    }
};
static ncnn::Layer* create_pair(void* userdata) {return new PlPair((Backend*)userdata);}
