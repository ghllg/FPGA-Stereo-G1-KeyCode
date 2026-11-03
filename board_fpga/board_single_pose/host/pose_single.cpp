#include <net.h>
#include <layer.h>
#include <modelbin.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <memory>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "backend.h"
#include "quantize.h"
#include "quant_thresholds.h"
#include "im2col.h"
#include "offload_plan.h"
#include "swish_lut.h"

static OffloadPlan layer_plan;

class LutSwish : public ncnn::Layer {
    PoseSwishLut table;
public:
    LutSwish() {one_blob_only=true;support_inplace=true;}
    int forward_inplace(ncnn::Mat& value,const ncnn::Option& opt) const override {
        if(value.elemsize!=4 || value.elempack!=1) return -1;
        const int count=value.w*value.h*value.d;
#pragma omp parallel for num_threads(opt.num_threads)
        for(int c=0;c<value.c;++c) {
            float* p=value.channel(c);
            table.apply(p,count);
        }
        return 0;
    }
};
static ncnn::Layer* create_lut_swish(void*) {return new LutSwish;}

// Bring-up quantization: per-output-channel weights, per-tensor dynamic activations.
// This requires accuracy validation; it is not QAT or a production INT8 model.
static int8_t quant(float v,float scale) {
    return pose_quantize(v,scale);
}
static uint64_t offload_threshold() {
    const char* value=std::getenv("POSE_PL_MIN_MACS");
    if(!value) return 0;
    const std::string text(value);
    if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("POSE_PL_MIN_MACS must be a nonnegative decimal integer");
    const uint64_t result=std::stoull(text);
    if(result>1000000000000ULL) throw std::runtime_error("POSE_PL_MIN_MACS exceeds supported range");
    return result;
}
class PlConvolution : public ncnn::Layer {
    Backend* backend;
    int outputs,kw,kh,sw,sh,dw,dh,pl,pr,pt,pb,count,bias_term,channels;
    std::vector<int8_t> weights;
    std::vector<float> scales,bias;
    std::unique_ptr<ncnn::Layer> cpu_conv;
    uint64_t minimum_macs=0;
    bool named_cpu=false;
public:
    explicit PlConvolution(Backend* b):backend(b) {
        one_blob_only=true; support_inplace=false;
        minimum_macs=offload_threshold();
    }
    int load_param(const ncnn::ParamDict& p) override {
        outputs=p.get(0,0); kw=p.get(1,0); kh=p.get(11,kw);
        sw=p.get(3,1); sh=p.get(13,sw); dw=p.get(2,1); dh=p.get(12,dw);
        pl=p.get(4,0); pr=p.get(15,pl); pt=p.get(14,pl); pb=p.get(16,pt);
        count=p.get(6,0); bias_term=p.get(5,0);
        if (outputs<=0 || kw<=0 || kh<=0 || sw<=0 || sh<=0 || dw<=0 || dh<=0 ||
            pl<0 || pr<0 || pt<0 || pb<0 || count<=0 || count%(outputs*kw*kh) ||
            p.get(8,0) || p.get(9,0) || p.get(19,0) || p.get(18,0.f)!=0.f) {
            std::cerr << "Unsupported convolution parameters; no CPU fallback\n"; return -1;
        }
        channels=count/(outputs*kw*kh);
        named_cpu=layer_plan.enabled && !layer_plan.select(name);
        if(minimum_macs || named_cpu) {
            cpu_conv.reset(ncnn::create_layer("Convolution"));
            if(!cpu_conv || cpu_conv->load_param(p)) return -1;
        }
        return 0;
    }
    int load_model(const ncnn::ModelBin& mb) override {
        ncnn::Mat w=mb.load(count,0);
        if (w.empty() || w.elemsize!=4) return -1;
        bias.resize(outputs,0.f);
        ncnn::Mat bias_data;
        if (bias_term) {
            bias_data=mb.load(outputs,1);
            if (bias_data.empty()) return -1;
            std::copy((const float*)bias_data,(const float*)bias_data+outputs,bias.begin());
        }
        if(cpu_conv) {
            ncnn::Mat tensors[2]={w,bias_data};
            if(cpu_conv->load_model(ncnn::ModelBinFromMatArray(tensors))) return -1;
        }
        weights.resize(count); scales.resize(outputs);
        const float* data=w;
        int k=count/outputs;
        for (int o=0;o<outputs;++o) {
            float maximum=0;
            for (int q=0;q<k;++q) {
                if (!std::isfinite(data[o*k+q])) return -1;
                maximum=std::max(maximum,std::abs(data[o*k+q]));
            }
            scales[o]=maximum>0 ? maximum/127.f : 1.f;
            for (int q=0;q<k;++q) weights[o*k+q]=quant(data[o*k+q],scales[o]);
        }
        return 0;
    }
    int create_pipeline(const ncnn::Option& opt) override {return cpu_conv?cpu_conv->create_pipeline(opt):0;}
    int destroy_pipeline(const ncnn::Option& opt) override {return cpu_conv?cpu_conv->destroy_pipeline(opt):0;}
    int forward(const ncnn::Mat& in,ncnn::Mat& out,const ncnn::Option& opt) const override {
        try {return forward_impl(in,out,opt);}
        catch (const std::exception& error) {
            // Do not propagate C++ exceptions through NCNN/OpenMP workers.
            std::cerr<<"Convolution M="<<outputs<<" input="<<in.w<<"x"<<in.h
                <<"x"<<in.c<<": "<<error.what()<<"\n";
            return -1;
        }
    }
    int forward_impl(const ncnn::Mat& in,ncnn::Mat& out,const ncnn::Option& opt) const {
        const auto conv_start=std::chrono::steady_clock::now();
        if (in.dims!=3 || in.c!=channels || in.elempack!=1 || in.elemsize!=4) return -1;
        int ow=(in.w+pl+pr-dw*(kw-1)-1)/sw+1;
        int oh=(in.h+pt+pb-dh*(kh-1)-1)/sh+1;
        if (ow<=0 || oh<=0) return -1;
        // Explicit opt-in partitioning, never a recovery path for PL errors.
        if(cpu_conv && (named_cpu || uint64_t(ow)*oh*count<minimum_macs)) {
            const int status=cpu_conv->forward(in,out,opt);
            if(status) return status;
            ++backend->cpu_selected_layers;
            backend->cpu_selected_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-conv_start).count();
            return 0;
        }
        out.create(ow,oh,outputs,4u,1,opt.blob_allocator);
        if (out.empty()) return -100;
        const auto quant_start=std::chrono::steady_clock::now();
        const char* conv_option=std::getenv("POSE_PL_CONV");
        const bool device_conv=conv_option && std::string(conv_option)=="1";
        const char* quant_option=std::getenv("POSE_PL_QUANT");
        const bool device_quant=quant_option && std::string(quant_option)=="1";
        if(device_quant && !device_conv) throw std::runtime_error("POSE_PL_QUANT requires POSE_PL_CONV=1");
        const bool scalar_quant=std::getenv("POSE_QUANT_SCALAR")!=nullptr;
        const bool vector_quant=!scalar_quant && std::getenv("POSE_QUANT_VECTOR")!=nullptr;
        float maximum=0;
        std::vector<float> channel_max(channels,0.f);
        std::vector<uint8_t> channel_valid(channels,1);
#pragma omp parallel for num_threads(opt.num_threads) if(opt.num_threads>1)
        for (int ch=0;ch<channels;++ch) {
            const float* p=in.channel(ch);
            if(scalar_quant) {
                for(int xy=0;xy<in.w*in.h;++xy) {
                    if(!std::isfinite(p[xy])) {channel_valid[ch]=0;break;}
                    channel_max[ch]=std::max(channel_max[ch],std::abs(p[xy]));
                }
            } else if(!pose_max_abs(p,in.w*in.h,channel_max[ch])) channel_valid[ch]=0;
        }
        for(int ch=0;ch<channels;++ch) {
            if(!channel_valid[ch]) return -1;
            maximum=std::max(maximum,channel_max[ch]);
        }
        const auto quant_scan_end=std::chrono::steady_clock::now();
        const float scale=maximum>0 ? maximum/127.f : 1.f;
        // Quantize each input pixel once; im2col reuses it for overlapping
        // convolution windows. Keep the original division/rounding arithmetic.
        const int input_pixels=in.w*in.h;
        // Every element below is assigned before use. Avoid zero-filling
        // temporary tensors that the next stage immediately overwrites.
        std::unique_ptr<int8_t[]> quantized(device_quant?nullptr:new int8_t[size_t(channels)*input_pixels]);
        uint32_t thresholds[128];
        if(device_quant) pose_make_thresholds(scale,thresholds);
        if(!device_quant) {
#pragma omp parallel for num_threads(opt.num_threads) if(opt.num_threads>1)
        for (int ch=0;ch<channels;++ch) {
            const float* src=in.channel(ch);
            if(!vector_quant) for(int xy=0;xy<input_pixels;++xy) quantized[size_t(ch)*input_pixels+xy]=quant(src[xy],scale);
            else pose_quantize_array(src,quantized.get()+size_t(ch)*input_pixels,input_pixels,scale);
        }
        }
        if(std::getenv("POSE_LAYER_PROFILE")) std::cerr<<"quant input="<<channels*input_pixels<<" scale="<<scale
            <<" scan_ms="<<std::chrono::duration<double,std::milli>(quant_scan_end-quant_start).count()
            <<" convert_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-quant_scan_end).count()<<"\n";
        backend->quantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-quant_start).count();
        const int k=channels*kw*kh, pixels=ow*oh;
        PoseConv shape={in.w,in.h,channels,kw,kh,sw,sh,pl,pt,ow,oh,0};
        if (device_conv && !pose_conv_valid(shape,1,k) && !pr && !pb && dw==1 && dh==1)
            pose_conv_fold_pointwise(shape);
        if (device_conv && (dw!=1 || dh!=1 || !pose_conv_valid(shape,1,k)))
            throw std::runtime_error("PL convolution does not support this layer shape");
        // Bound each invocation's DDR footprint. im2col is initially ARM-side.
        const char* chunk_env=std::getenv("POSE_GEMM_CHUNK");
        const int requested_chunk=chunk_env?std::stoi(chunk_env):4096;
        if (requested_chunk<1 || requested_chunk>25600) throw std::runtime_error("POSE_GEMM_CHUNK must be 1..25600");
        const int features=device_quant ? 576+((channels*input_pixels*4+63)&~63)+((channels*input_pixels+63)&~63)
            : 64+channels*input_pixels;
        const int dma_limit=device_conv ?
            (8*1024*1024-int(weights.size())-features-192)/(4*outputs) :
            (8*1024*1024-int(weights.size())-192)/(k+4*outputs);
        const int chunk=std::min(requested_chunk,dma_limit);
        if(chunk<1) throw std::runtime_error("Convolution cannot fit the 8 MiB DMA buffer");
        std::unique_ptr<int8_t[]> b(device_conv?nullptr:new int8_t[size_t(k)*std::min(chunk,pixels)]);
        std::unique_ptr<int32_t[]> sums(new int32_t[size_t(outputs)*std::min(chunk,pixels)]);
        for (int start=0;start<pixels;start+=chunk) {
            int n=std::min(chunk,pixels-start);
            const int32_t* result_sums;
            if (device_conv) {
                shape.start=start;
                if(device_quant) result_sums=backend->run_quantized_convolution(weights.data(),(const float*)in.data,
                    in.cstep,thresholds,sums.get(),outputs,n,k,shape,start==0);
                else result_sums=backend->run_convolution(weights.data(),quantized.get(),sums.get(),outputs,n,k,shape,start==0);
            } else {
            int8_t* direct=backend->prepare_input(outputs,n,k);
            int8_t* columns=direct?direct:b.get();
            const auto column_start=std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(opt.num_threads) if(opt.num_threads>1)
            for (int ch=0;ch<channels;++ch) {
                int q=ch*kw*kh;
                const int8_t* src=quantized.get()+size_t(ch)*input_pixels;
                if (kw==1 && kh==1 && sw==1 && sh==1 && pl==0 && pr==0 && pt==0 && pb==0) {
                    memcpy(columns+size_t(q)*n,src+start,n);
                    ++q;
                    continue;
                }
                for (int ky=0;ky<kh;++ky) for (int kx=0;kx<kw;++kx,++q)
                    pose_column(src,columns+size_t(q)*n,in.w,in.h,ow,start,n,
                                sw,sh,pl,pt,kx*dw,ky*dh);
            }
            backend->im2col_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-column_start).count();
            if (direct) result_sums=backend->run_prepared(weights.data());
            else {
                backend->run(weights.data(),columns,sums.get(),outputs,n,k);
                result_sums=sums.get();
            }
            }
            const auto dequant_start=std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(opt.num_threads) if(opt.num_threads>1)
            for (int o=0;o<outputs;++o) {
                float* dest=out.channel(o);
                int j=0;
#ifdef __ARM_NEON
                for (;j+4<=n;j+=4) {
                    float32x4_t v=vcvtq_f32_s32(vld1q_s32(result_sums+o*n+j));
                    v=vmulq_n_f32(v,scale);
                    v=vmulq_n_f32(v,scales[o]);
                    v=vaddq_f32(v,vdupq_n_f32(bias[o]));
                    vst1q_f32(dest+start+j,v);
                }
#endif
                for (;j<n;++j) dest[start+j]=result_sums[o*n+j]*scale*scales[o]+bias[o];
            }
            backend->dequantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-dequant_start).count();
        }
        backend->custom_conv_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-conv_start).count();
        ++backend->conv_layers;
        return 0;
    }
};
static ncnn::Layer* create_conv(void* userdata) {return new PlConvolution((Backend*)userdata);}

int main(int argc,char** argv) {
    if (argc!=8 && argc!=10) {
        std::cerr << "Usage: pose_single csim|pl|cpu PARAM BIN RGB320 SOURCE_W SOURCE_H OUTPUT.json [UIO DMA_NAME]\n";
        return 2;
    }
    try {
        std::string mode=argv[1];
        const uint64_t selected_min_macs=offload_threshold();
        layer_plan=OffloadPlan(std::getenv("POSE_PL_LAYERS"));
        if(layer_plan.enabled && (selected_min_macs || mode=="cpu"))
            throw std::runtime_error("POSE_PL_LAYERS requires pl/csim mode and no MAC threshold");
        int width=std::stoi(argv[5]),height=std::stoi(argv[6]);
        if (width<=0 || height<=0) throw std::runtime_error("Invalid original image dimensions");
        std::unique_ptr<Backend> backend;
        if (mode!="cpu") backend=make_backend(mode,argc==10?argv[8]:"/dev/uio0",argc==10?argv[9]:"udmabuf0");
        ncnn::Net net;
        const char* swish_option=std::getenv("POSE_SWISH_LUT");
        const bool swish_lut=swish_option && std::string(swish_option)=="1";
        const char* threads_env=std::getenv("POSE_THREADS");
        const int threads=threads_env?std::stoi(threads_env):1;
        if(threads<1 || threads>2) throw std::runtime_error("POSE_THREADS must be 1 or 2 on this board");
#ifndef _OPENMP
        if(threads!=1) throw std::runtime_error("Two threads require an OpenMP build");
#endif
        net.opt.num_threads=threads;
        net.opt.use_vulkan_compute=false;
        net.opt.use_packing_layout=std::getenv("POSE_NCNN_PACKING")!=nullptr;
        if(backend && (selected_min_macs || layer_plan.enabled) && net.opt.use_packing_layout)
            throw std::runtime_error("Selective offload currently requires packing disabled");
        net.opt.use_fp16_storage=false;
        net.opt.use_fp16_arithmetic=false;
        net.opt.use_bf16_storage=false;
        if (backend) net.register_custom_layer("Convolution",create_conv,nullptr,backend.get());
        if(swish_lut) net.register_custom_layer("Swish",create_lut_swish);
        if (net.load_param(argv[2]) || net.load_model(argv[3])) throw std::runtime_error("Model load failed");
        layer_plan.validate();
        std::vector<unsigned char> rgb(320*320*3);
        std::ifstream f(argv[4],std::ios::binary);
        if (!f.read((char*)rgb.data(),rgb.size()) || f.peek()!=EOF) throw std::runtime_error("Expected exactly RGB888 320x320");
        ncnn::Mat in=ncnn::Mat::from_pixels(rgb.data(),ncnn::Mat::PIXEL_RGB,320,320);
        const float norm[3]={1.f/255,1.f/255,1.f/255};
        in.substract_mean_normalize(nullptr,norm);
        auto start=std::chrono::steady_clock::now();
        auto ex=net.create_extractor();
        ncnn::Mat out;
        if (ex.input("in0",in) || ex.extract("out0",out)) throw std::runtime_error("Inference failed");
        double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if (out.dims!=2 || out.h!=56 || out.w!=2100) throw std::runtime_error("Model must be YOLO11n-pose 320, 56x2100 output");
        for (int i=0;i<56;++i) for (int j=0;j<2100;++j)
            if (!std::isfinite(out.row(i)[j])) throw std::runtime_error("Nonfinite network output");
        const float* scores=out.row(4);
        int best=std::max_element(scores,scores+2100)-scores;
        bool detected=scores[best]>=0.25f;
        double scale=std::min(320.0/width,320.0/height);
        int resized_w=int(std::round(width*scale)),resized_h=int(std::round(height*scale));
        int left=(320-resized_w)/2,top=(320-resized_h)/2;
        std::ofstream result(argv[7]);
        if (!result) throw std::runtime_error("Cannot open output JSON");
        result << std::setprecision(9) << "{\"schema\":\"single_pose_v1\",\"backend\":\"" << mode
               << "\",\"pl_acceleration\":" << (backend && backend->hardware() && backend->calls?"true":"false")
               << ",\"scope\":\"single_image_2d\",\"coordinate_space\":\"source_image_pixels\""
               << ",\"source_size\":[" << width << "," << height << "]"
               << ",\"infer_ms\":" << elapsed << ",\"gemm_calls\":" << (backend?backend->calls:0)
               << ",\"cpu_threads\":" << threads
               << ",\"swish_lut\":" << (swish_lut?"true":"false")
               << ",\"pl_min_macs\":" << (backend?selected_min_macs:0)
               << ",\"pl_layer_plan\":[";
        { bool first=true; for(const auto& name:layer_plan.requested) {
            if(!first) result << ",";
            result << "\"" << name << "\""; first=false;
        } }
        result << "]"
               << ",\"gemm_macs\":" << (backend?backend->macs:0)
               << ",\"direct_dma_calls\":" << (backend?backend->direct_calls:0)
               << ",\"pl_convolution_calls\":" << (backend?backend->convolution_calls:0)
               << ",\"pl_quantize_calls\":" << (backend?backend->pl_quantize_calls:0)
               << ",\"resident_input_reuses\":" << (backend?backend->resident_reuses:0)
               << ",\"profile\":{\"cached_dma\":" << (backend && backend->cached_dma?"true":"false")
               << ",\"directional_dma\":" << (backend && backend->directional_dma?"true":"false")
               << ",\"prepare_ms\":" << (backend?backend->prepare_ms:0)
               << ",\"pl_wait_ms\":" << (backend?backend->wait_ms:0)
               << ",\"readback_ms\":" << (backend?backend->readback_ms:0)
               << ",\"input_bytes\":" << (backend?backend->input_bytes:0)
               << ",\"output_bytes\":" << (backend?backend->output_bytes:0) << "}"
               << ",\"cpu_profile\":{\"custom_conv_ms\":" << (backend?backend->custom_conv_ms:0)
               << ",\"quantize_ms\":" << (backend?backend->quantize_ms:0)
               << ",\"im2col_ms\":" << (backend?backend->im2col_ms:0)
               << ",\"dequantize_ms\":" << (backend?backend->dequantize_ms:0)
               << ",\"other_ms\":" << std::max(0.0,elapsed-(backend?backend->custom_conv_ms:0))
               << ",\"conv_layers\":" << (backend?backend->conv_layers:0)
               << ",\"cpu_selected_conv_layers\":" << (backend?backend->cpu_selected_layers:0)
               << ",\"cpu_selected_conv_ms\":" << (backend?backend->cpu_selected_ms:0) << "}"
               << ",\"score\":" << scores[best] << ",\"detected\":" << (detected?"true":"false")
               << ",\"keypoint_order\":\"COCO17\",\"keypoints\":[";
        if (detected) for (int i=0;i<17;++i) {
            if (i) result << ",";
            result << "[" << (out.row(5+i*3)[best]-left)/scale << ","
                   << (out.row(6+i*3)[best]-top)/scale << "," << out.row(7+i*3)[best] << "]";
        }
        result << "]}\n"; result.close();
        if (!result) throw std::runtime_error("Output write failed");
        std::cerr << "SINGLE_POSE_OK backend=" << mode << " infer_ms=" << elapsed << "\n";
    } catch (const std::exception& e) {std::cerr << e.what() << "\n"; return 1;}
    return 0;
}
