// Forked from board_single_pose e48cb5e for the light network.
#include "backend.h"
#include "../hls/protocol.h"
#include "board_single_pose/hls/pose_gemm.h"
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <vector>
#include "im2col.h"
#include "quant_thresholds.h"
#include "stream_reference.h"
#include "stream_wire.h"
#ifdef U02_ARM_STEM_SMLAL
#include "arm_stem_smlal.h"
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#ifdef __linux__
#include <sys/mman.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include "backend_coherent.h"
#ifdef U02_STREAM_FP32_OUTPUT
#include "backend_stream_fp32.h"
#ifdef U02_STREAM_REGION_HOOK
#include "region_client.h"
#include "backend_region_dual.h"
#endif
#endif
#ifdef U02_STREAM_RESIDENT_PACKETS
#include "backend_stream_resident.h"
#endif
#include "backend_stream.h"
#endif

struct Csim : Backend {
    const int32_t* run_stream(const std::vector<StreamOp>& ops,const std::vector<int8_t>& weights,
                    const std::vector<StreamRequant>& coeff,const int8_t* image,int32_t* output) override {
        stream_reference(ops,weights,coeff,image,output
#ifdef U02_STREAM_TRACE
                         ,stream_trace_group
#endif
                         );
        if(const char* path=std::getenv("POSE_STREAM_FIXTURE")) {
            const auto packet=stream_pack(ops,weights,coeff);
            const uint32_t inbytes=ops.front().ci*ops.front().h*ops.front().w;
            const uint32_t outbytes=ops.back().co*ops.back().oh*ops.back().ow*4;
            const uint32_t record[6]={0x55304631u,uint32_t(packet.size()*8),inbytes,outbytes,uint32_t(ops.size()),uint32_t(stream_calls)};
            std::ofstream file(path,std::ios::binary|(stream_calls?std::ios::app:std::ios::trunc));
            file.write(reinterpret_cast<const char*>(record),sizeof(record));
            file.write(reinterpret_cast<const char*>(packet.data()),packet.size()*8);
            file.write(reinterpret_cast<const char*>(image),inbytes);
            file.write(reinterpret_cast<const char*>(output),outbytes);
            file.close();if(!file)throw std::runtime_error("Stream fixture write failed");
        }
        ++calls; ++stream_calls;
        for(const auto& op:ops) {
            if(op.kind) ++depthwise_calls; else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
        return output;
    }
    const int32_t* run_depthwise(const int8_t* a,const int8_t* image,int32_t* out,
        int channels,int count,int kernel,const PoseConv& p) override {
        if(!light_dw_valid(p,channels,count,kernel)) throw std::runtime_error("Invalid depthwise descriptor");
        for(int ch=0;ch<channels;++ch) for(int xy=0;xy<count;++xy) {
            const int pos=p.start+xy;int32_t sum=0;
            for(int ky=0;ky<p.kh;++ky) for(int kx=0;kx<p.kw;++kx) {
                const int y=pos/p.ow*p.sh-p.top+ky,x=pos%p.ow*p.sw-p.left+kx;
                if(y>=0 && y<p.h && x>=0 && x<p.w)
                    sum+=int32_t(a[ch*kernel+ky*p.kw+kx])*image[(ch*p.h+y)*p.w+x];
            }
            out[ch*count+xy]=sum;
        }
        ++calls;++depthwise_calls;macs+=uint64_t(channels)*count*kernel;
        return out;
    }
    const int32_t* run_pw_dw(const int8_t* pw,const int8_t* dw,const int8_t* image,
        const LightAffine* affine,int32_t* output,int input_channels,const PoseConv& p) override {
        if(!light_pair_valid(p,input_channels,p.ow*p.oh))throw std::runtime_error("Invalid fused descriptor");
        const int pixels=p.w*p.h;
        PoseConv first={p.w,p.h,input_channels,1,1,1,1,0,0,p.w,p.h,0};
        std::vector<int32_t> sums(size_t(p.channels)*pixels);
        std::vector<int8_t> mid(size_t(p.channels)*pixels);
        run_convolution(pw,image,sums.data(),p.channels,pixels,input_channels,first,true);
        for(int ch=0;ch<p.channels;++ch) {
            if(!light_affine_valid(affine[ch]))throw std::runtime_error("Invalid affine coefficients");
            for(int i=0;i<pixels;++i)mid[ch*pixels+i]=light_affine_relu(sums[ch*pixels+i],affine[ch]);
        }
        ++fused_pair_calls;
        return run_depthwise(dw,mid.data(),output,p.channels,p.ow*p.oh,p.kw*p.kh,p);
    }
    std::vector<int8_t> quant_image;
    const int8_t* quant_weights=nullptr;
    const float* quant_source=nullptr;
    size_t quant_stride=0;
    int quant_m=0;
    PoseConv quant_shape={};
    uint32_t quant_table[128]={};
    bool hardware() const override {return false;}
    const int32_t* run_quantized_convolution(const int8_t* a,const float* image,
        size_t stride,const uint32_t* thresholds,int32_t* scratch,
        int m,int n,int k,const PoseConv& p,bool first) override {
        if(!pose_conv_valid(p,n,k) || stride<size_t(p.w)*p.h)
            throw std::runtime_error("Invalid FP32 convolution input");
        const size_t plane=size_t(p.w)*p.h;
        uint32_t header[16],previous[16];pose_conv_header(header,p);pose_conv_header(previous,quant_shape);
        if(!first && (quant_weights!=a || quant_source!=image || quant_stride!=stride || quant_m!=m ||
            memcmp(header,previous,48) || memcmp(thresholds,quant_table,512)))
            throw std::runtime_error("Quantized convolution resident input contract changed");
        if(first) {
            quant_image.resize(plane*p.channels);
            for(int ch=0;ch<p.channels;++ch) for(size_t i=0;i<plane;++i) {
                uint32_t bits;memcpy(&bits,image+ch*stride+i,4);
                quant_image[ch*plane+i]=pose_threshold_quantize_bits(bits,thresholds);
            }
            ++pl_quantize_calls;
            quant_weights=a;quant_source=image;quant_stride=stride;quant_m=m;quant_shape=p;
            memcpy(quant_table,thresholds,512);
        }
        if(quant_image.size()!=plane*p.channels) throw std::runtime_error("No resident quantized image");
        return run_convolution(a,quant_image.data(),scratch,m,n,k,p,first);
    }
    void run(const int8_t* a,const int8_t* b,int32_t* c,int m,int n,int k) override {
        if (pose_gemm(a,b,c,m,n,k)) throw std::runtime_error("GEMM dimensions exceed IP limits");
        ++calls; macs+=uint64_t(m)*n*k;
    }
    const int32_t* run_convolution(const int8_t* a,const int8_t* image,int32_t* scratch,
        int m,int n,int k,const PoseConv& p,bool first) override {
        if (!pose_conv_valid(p,n,k)) throw std::runtime_error("Unsupported convolution descriptor");
        std::vector<int8_t> columns(size_t(k)*n);
        int q=0;
        for (int ch=0;ch<p.channels;++ch) for(int y=0;y<p.kh;++y) for(int x=0;x<p.kw;++x,++q)
            pose_column(image+size_t(ch)*p.w*p.h,columns.data()+size_t(q)*n,
                p.w,p.h,p.ow,p.start,n,p.sw,p.sh,p.left,p.top,x,y);
        run(a,columns.data(),scratch,m,n,k);
        ++convolution_calls;
        return scratch;
    }
};

// The streaming graph leaves exactly its 3x3 stem on ARM. Keep Csim intact as
// an independent arithmetic reference; accelerate only that explicit backend.
struct ArmIntegerStem : Csim {
    void run(const int8_t* a,const int8_t* b,int32_t* c,int m,int n,int k) override {
        if(m!=32 || n!=3072 || k!=27)
            throw std::runtime_error("Unexpected ARM integer stem shape");
#if defined(__aarch64__) && defined(U02_ARM_STEM_SMLAL)
        u02_arm_stem_smlal(a,b,c,cpu_threads);
        ++calls;macs+=uint64_t(m)*n*k;
#elif defined(__aarch64__)
        // Reuse each eight-pixel load across four output channels. Independent
        // accumulators also avoid a single widening-add dependency chain.
        #pragma omp parallel for num_threads(cpu_threads)
        for(int row=0;row<m;row+=4) for(int col=0;col<n;col+=8) {
            int32x4_t l0=vdupq_n_s32(0),h0=l0,l1=l0,h1=l0,l2=l0,h2=l0,l3=l0,h3=l0;
            for(int inner=0;inner<k;++inner) {
                const int8x8_t pixels=vld1_s8(b+inner*n+col);
                const int16x8_t p0=vmull_s8(vdup_n_s8(a[row*k+inner]),pixels);
                const int16x8_t p1=vmull_s8(vdup_n_s8(a[(row+1)*k+inner]),pixels);
                const int16x8_t p2=vmull_s8(vdup_n_s8(a[(row+2)*k+inner]),pixels);
                const int16x8_t p3=vmull_s8(vdup_n_s8(a[(row+3)*k+inner]),pixels);
                l0=vaddw_s16(l0,vget_low_s16(p0));h0=vaddw_s16(h0,vget_high_s16(p0));
                l1=vaddw_s16(l1,vget_low_s16(p1));h1=vaddw_s16(h1,vget_high_s16(p1));
                l2=vaddw_s16(l2,vget_low_s16(p2));h2=vaddw_s16(h2,vget_high_s16(p2));
                l3=vaddw_s16(l3,vget_low_s16(p3));h3=vaddw_s16(h3,vget_high_s16(p3));
            }
            vst1q_s32(c+row*n+col,l0);vst1q_s32(c+row*n+col+4,h0);
            vst1q_s32(c+(row+1)*n+col,l1);vst1q_s32(c+(row+1)*n+col+4,h1);
            vst1q_s32(c+(row+2)*n+col,l2);vst1q_s32(c+(row+2)*n+col+4,h2);
            vst1q_s32(c+(row+3)*n+col,l3);vst1q_s32(c+(row+3)*n+col+4,h3);
        }
        ++calls;macs+=uint64_t(m)*n*k;
#else
        Csim::run(a,b,c,m,n,k);
#endif
    }
};

std::unique_ptr<Backend> make_backend(const std::string& mode,const std::string& uio,const std::string& dma) {
    if (mode=="arm-stem") return std::unique_ptr<Backend>(new ArmIntegerStem);
    if (mode=="csim" || mode=="stream-csim") return std::unique_ptr<Backend>(new Csim);
#ifdef __linux__
#if defined(U02_STREAM_REGION_HOOK) && defined(U02_STREAM_FP32_OUTPUT)
    if(mode=="pl-region") {
        const char* path=std::getenv("U02_REGION_MODEL");
        if(!path || !*path)throw std::runtime_error("pl-region requires U02_REGION_MODEL pointing to its compiled model bundle");
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error("Cannot open fused model bundle");
        const auto bytes=file.tellg();
        if(bytes<128 || bytes>1048576)throw std::runtime_error("Invalid fused model bundle size");
        std::vector<unsigned char> constants(static_cast<size_t>(bytes));file.seekg(0);
        file.read(reinterpret_cast<char*>(constants.data()),constants.size());
        if(!file)throw std::runtime_error("Truncated fused model bundle");
        RegionModelBinding check(constants); // Validate before any device construction.
        return std::unique_ptr<Backend>(new RegionDualBackend<RegionDualClient>(constants));
    }
#endif
    if (mode=="pl") return std::unique_ptr<Backend>(new CoherentBackend);
    if (mode=="pl-stream") {
#ifdef U02_STREAM_FP32_OUTPUT
        return std::unique_ptr<Backend>(new StreamFp32Backend);
#elif defined(U02_STREAM_RESIDENT_PACKETS)
        return std::unique_ptr<Backend>(new StreamResidentBackend);
#else
        return std::unique_ptr<Backend>(new StreamCoherentBackend);
#endif
    }
#endif
    throw std::runtime_error("Backend must be csim or verified ZU5EV coherent-DMA pl");
}
