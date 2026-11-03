#include "backend.h"
#include "../hls/pose_gemm.h"
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <vector>
#include "im2col.h"
#include "quant_thresholds.h"
#ifdef __linux__
#include <sys/mman.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif

struct Csim : Backend {
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
#ifdef __linux__
static uint64_t number(const std::string& path) {
    std::ifstream f(path); std::string s;
    if (!(f>>s)) throw std::runtime_error("Cannot read "+path);
    return std::stoull(s,nullptr,0);
}
static void put(const std::string& path,uint64_t value) {
    std::ofstream f(path); f << value; f.close();
    if (!f) throw std::runtime_error("Cannot write "+path);
}
struct Uio : Backend {
    int fd=-1,dfd=-1;
    volatile uint32_t* regs=nullptr;
    uint8_t* buf=nullptr;
    size_t size=0; uint64_t phys=0;
    std::string sys;
    bool prepared=false;
    int pm=0,pn=0,pk=0;
    size_t as=0,bs=0,cs=0,bo=0,co=0,used=0;
    bool resident=false,partial_sync=false;
    const int8_t *resident_a=nullptr;
    const void* resident_image=nullptr;
    size_t resident_stride=0;
    bool resident_quant=false;
    uint32_t resident_thresholds[128]={};
    PoseConv resident_shape={};
    int resident_m=0,command_k=0;
    bool hardware() const override {return true;}
    void release() {
        if (buf) munmap(buf,size);
        if (regs) munmap((void*)regs,4096);
        if (dfd>=0) close(dfd);
        if (fd>=0) close(fd);
    }
    Uio(const std::string& device,const std::string& dma) {
        try {
            if (dma.find('/')!=std::string::npos) throw std::runtime_error("DMA argument must be a device name");
            const std::string name=device.substr(device.find_last_of('/')+1);
            std::ifstream f("/sys/class/uio/"+name+"/name"); std::string identity; std::getline(f,identity);
            if (identity!="pose-gemm") throw std::runtime_error("UIO device is not named pose-gemm; refusing MMIO");
            if (number("/sys/class/uio/"+name+"/maps/map0/addr")!=0x43c10000 ||
                number("/sys/class/uio/"+name+"/maps/map0/size")<4096 ||
                number("/sys/class/uio/"+name+"/maps/map0/offset")!=0)
                throw std::runtime_error("Unexpected UIO mapping");
            fd=open(device.c_str(),O_RDWR|O_SYNC);
            if (fd<0 || flock(fd,LOCK_EX|LOCK_NB)) throw std::runtime_error("Cannot exclusively open UIO");
            sys="/sys/class/u-dma-buf/"+dma;
            size=number(sys+"/size"); phys=number(sys+"/phys_addr");
            if (!size || phys+size>0x100000000ULL) throw std::runtime_error("DMA buffer exceeds 32-bit address space");
            // HP0 is non-coherent. Keep the explicit CPU/device cache syncs
            // below; cached CPU access avoids slow uncached bulk memcpy.
            const char* cache_option=std::getenv("POSE_DMA_CACHED");
            cached_dma=!(cache_option && std::string(cache_option)=="0");
            const char* direction_option=std::getenv("POSE_DMA_DIRECTIONAL");
            directional_dma=direction_option && std::string(direction_option)=="1";
            // This ARM32 5.4 kernel uses legacy arm_dma_ops: the udmabuf
            // dma_coherent informational attribute incorrectly reports the
            // generic fallback true. Its dma_sync_single_* calls still use
            // the ARM non-coherent cache operations. Never skip these syncs.
            dfd=open(("/dev/"+dma).c_str(),O_RDWR|(cached_dma?0:O_SYNC));
            if (dfd<0 || flock(dfd,LOCK_EX|LOCK_NB)) throw std::runtime_error("Cannot exclusively open DMA buffer");
            void* p=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            if (p==MAP_FAILED) throw std::runtime_error("UIO mmap failed");
            regs=(volatile uint32_t*)p;
            p=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
            if (p==MAP_FAILED) throw std::runtime_error("DMA mmap failed");
            buf=(uint8_t*)p;
            if (!(regs[0]&4)) throw std::runtime_error("Accelerator not idle; reset required");
            // A previous process may have used PL-written quantization
            // workspace. Establish CPU ownership once before this mode's
            // read-only input / write-only output invariant begins.
            if(directional_dma) sync("sync_for_cpu",size);
        } catch (...) {release(); throw;}
    }
    ~Uio() {release();}
    void sync(const char* attr,size_t used,size_t offset=0,unsigned direction=0) {
        // u-dma-buf 5.5 packed command: offset[63:32], size[31:4],
        // direction[3:2], trigger[0]. Buffers are aligned to 64 bytes.
        if (!used || used%64 || offset%64 || used>0xfffffff0ULL || offset>size || used>size-offset)
            throw std::runtime_error("Invalid packed DMA sync size");
        if(direction>2) throw std::runtime_error("Invalid DMA direction");
        put(sys+"/"+attr,(uint64_t(offset)<<32)|uint64_t(used)|(direction<<2)|1);
    }
    void reclaim_inputs(size_t bytes,size_t offset=0) {
        // In directional mode PL only reads A/B. Every previous C region
        // has already been returned to CPU ownership after completion, even
        // when the next layout moves the input/output boundary. PL quantized
        // workspace writes are prohibited in this mode.
        sync("sync_for_cpu",bytes,offset,directional_dma?1:0);
    }
    void layout(int m,int n,int k,size_t feature_bytes=0) {
        if (prepared) throw std::runtime_error("Previous prepared DMA input has not been submitted");
        if (m<1 || m>POSE_MAX_M || n<1 || n>POSE_MAX_N || k<1 || k>POSE_MAX_K)
            throw std::runtime_error("GEMM dimensions exceed IP limits");
        auto aligned=[](size_t v){return (v+63)&~size_t(63);};
        as=size_t(m)*k; bs=feature_bytes?feature_bytes:size_t(k)*n; cs=size_t(m)*n*4;
        bo=aligned(as); co=aligned(bo+bs); used=aligned(co+cs);
        if (used>size) throw std::runtime_error("DMA buffer too small");
        if (!(regs[0]&4)) throw std::runtime_error("Accelerator busy; refusing buffer overwrite");
        pm=m; pn=n; pk=k; command_k=k; partial_sync=false;
    }
    int8_t* prepare_input(int m,int n,int k) override {
        const char* option=std::getenv("POSE_DMA_DIRECT");
        if (!option || std::string(option)!="1") return nullptr;
        layout(m,n,k);
        resident=false;
        const auto start=std::chrono::steady_clock::now();
        // CPU ownership precedes host im2col writes, including when layouts
        // change and a former output region becomes an input region.
        reclaim_inputs(used);
        prepare_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        prepared=true;
        return reinterpret_cast<int8_t*>(buf+bo);
    }
    const int32_t* run_prepared(const int8_t* a) override {
        if (!prepared) throw std::runtime_error("No prepared DMA input");
        prepared=false;
        const auto start=std::chrono::steady_clock::now();
        memcpy(buf,a,as);
        execute(start);
        ++direct_calls;
        return reinterpret_cast<const int32_t*>(buf+co);
    }
    void run(const int8_t* a,const int8_t* b,int32_t* c,int m,int n,int k) override {
        layout(m,n,k);
        resident=false;
        const auto prepare_start=std::chrono::steady_clock::now();
        reclaim_inputs(used);
        memcpy(buf,a,as); memcpy(buf+bo,b,bs);
        execute(prepare_start);
        const auto start=std::chrono::steady_clock::now();
        memcpy(c,buf+co,cs);
        readback_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    const int32_t* run_convolution(const int8_t* a,const int8_t* image,int32_t* scratch,
        int m,int n,int k,const PoseConv& p,bool first) override {
        if (!pose_conv_valid(p,n,k)) throw std::runtime_error("Unsupported convolution descriptor");
        uint32_t header[16],previous[16];pose_conv_header(header,p);
        pose_conv_header(previous,resident_shape);
        if (!first && (!resident || resident_quant || resident_a!=a || resident_image!=image || resident_m!=m ||
            memcmp(header,previous,12*sizeof(uint32_t))))
            throw std::runtime_error("Convolution resident input contract changed");
        const size_t image_bytes=size_t(p.w)*p.h*p.channels;
        layout(m,n,k,64+image_bytes);
        command_k=k|POSE_CONV_FLAG;
        const auto begin=std::chrono::steady_clock::now();
        if (first) {
            reclaim_inputs(used);
            memcpy(buf,a,as);memcpy(buf+bo+64,image,image_bytes);
        } else {
            // Weights/image remain device-owned. Only the descriptor is
            // rewritten; the previous output was returned to CPU ownership.
            reclaim_inputs(64,bo);
            partial_sync=true;
        }
        memcpy(buf+bo,header,64);
        resident=false;
        execute(begin);
        resident=true;resident_quant=false;resident_a=a;resident_image=image;resident_m=m;resident_shape=p;
        ++convolution_calls;
        if (!first) ++resident_reuses;
        return reinterpret_cast<const int32_t*>(buf+co);
    }
    const int32_t* run_quantized_convolution(const int8_t* a,const float* image,
        size_t stride,const uint32_t* thresholds,int32_t* scratch,
        int m,int n,int k,const PoseConv& p,bool first) override {
        if(directional_dma) throw std::runtime_error("Directional DMA does not support PL-written quantization workspace");
        const size_t plane=size_t(p.w)*p.h,count=plane*p.channels;
        if(!pose_conv_valid(p,n,k) || stride<plane || count>1048576)
            throw std::runtime_error("Unsupported PL quantization input");
        uint32_t header[16],previous[16];pose_conv_header(header,p);pose_conv_header(previous,resident_shape);
        if(!first && (!resident || !resident_quant || resident_a!=a || resident_image!=image ||
            resident_m!=m || resident_stride!=stride || memcmp(header,previous,48) ||
            memcmp(thresholds,resident_thresholds,512)))
            throw std::runtime_error("Quantized convolution resident input contract changed");
        const size_t float_bytes=(count*4+63)&~size_t(63),int8_bytes=(count+63)&~size_t(63);
        layout(m,n,k,64+512+float_bytes+int8_bytes);
        command_k=k|POSE_CONV_FLAG;header[13]=first?1:2;
        const auto begin=std::chrono::steady_clock::now();
        if(first) {
            sync("sync_for_cpu",used);
            memcpy(buf,a,as);memcpy(buf+bo+64,thresholds,512);
            // NCNN channel planes can have alignment padding. Only logical
            // pixels are uploaded; the PL descriptor is densely packed CHW.
            for(int ch=0;ch<p.channels;++ch)
                memcpy(buf+bo+576+ch*plane*4,image+ch*stride,plane*4);
        } else {
            sync("sync_for_cpu",64,bo);partial_sync=true;
        }
        memcpy(buf+bo,header,64);resident=false;
        execute(begin);
        resident=true;resident_quant=true;resident_a=a;resident_image=image;
        resident_m=m;resident_shape=p;resident_stride=stride;
        memcpy(resident_thresholds,thresholds,512);
        ++convolution_calls;
        if(first) ++pl_quantize_calls;else ++resident_reuses;
        return reinterpret_cast<const int32_t*>(buf+co);
    }
    void execute(std::chrono::steady_clock::time_point prepare_start) {
        if(directional_dma) {
            if(partial_sync) sync("sync_for_device",64,bo,1);
            else sync("sync_for_device",co,0,1);
            sync("sync_for_device",used-co,co,2);
        } else if (partial_sync) {
            sync("sync_for_device",64,bo);
            sync("sync_for_device",used-co,co);
        } else sync("sync_for_device",used);
        regs[0x10/4]=phys;
        regs[0x1c/4]=phys+bo;
        regs[0x28/4]=phys+co;
        regs[0x34/4]=pm; regs[0x3c/4]=pn; regs[0x44/4]=command_k;
        __sync_synchronize(); regs[0]=1;
        auto start=std::chrono::steady_clock::now();
        prepare_ms+=std::chrono::duration<double,std::milli>(start-prepare_start).count();
        while (!(regs[0]&2)) {
            if (std::chrono::steady_clock::now()-start>std::chrono::seconds(30))
                throw std::runtime_error("PL timeout; stop inference and reset hardware before reuse");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        __sync_synchronize();
        const auto readback_start=std::chrono::steady_clock::now();
        wait_ms+=std::chrono::duration<double,std::milli>(readback_start-start).count();
        if (int32_t(regs[0x4c/4])!=0) throw std::runtime_error("PL rejected GEMM command");
        // Return C to the CPU. A quantized-convolution command also writes
        // its INT8 workspace in B, which remains device-owned for reuse.
        // The complete buffer is returned before the next CPU overwrite.
        sync("sync_for_cpu",used-co,co,directional_dma?2:0);
        readback_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-readback_start).count();
        input_bytes+=as+bs; output_bytes+=cs;
        ++calls; macs+=uint64_t(pm)*pn*pk;
    }
};
#endif
std::unique_ptr<Backend> make_backend(const std::string& mode,const std::string& uio,const std::string& dma) {
    if (mode=="csim") return std::unique_ptr<Backend>(new Csim);
#ifdef __linux__
    if (mode=="pl") return std::unique_ptr<Backend>(new Uio(uio,dma));
#endif
    throw std::runtime_error("Backend must be csim or pl (Linux UIO + u-dma-buf required for pl)");
}
