#include "pose_network_session.h"
#include <thread>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <cstdlib>

// Persistent workers are also used in serial mode, so A/B does not change
// which thread owns each NCNN session or its OpenMP context.
class PoseWorker {
    PoseNetworkSession& session;
    const ncnn::Mat& input;
    std::mutex mutex;
    std::condition_variable ready,finished;
    bool requested=false,busy=false,stopping=false;
    std::exception_ptr error;
    std::thread worker;
    void loop() {
        for(;;) {
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock,[&]{return requested || stopping;});
            if(stopping && !requested)return;
            requested=false;lock.unlock();
            std::exception_ptr failure;
            try {session.forward(input);}catch(...) {failure=std::current_exception();}
            lock.lock();if(failure)error=failure;busy=false;finished.notify_one();
        }
    }
public:
    PoseWorker(PoseNetworkSession& s,const ncnn::Mat& in):session(s),input(in),worker([this]{loop();}) {}
    ~PoseWorker() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;ready.notify_one();}
        worker.join();
    }
    void submit() {
        std::lock_guard<std::mutex> lock(mutex);
        if(error)std::rethrow_exception(error);
        if(busy || stopping)throw std::runtime_error("Worker already busy or stopped");
        busy=requested=true;ready.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);finished.wait(lock,[&]{return !busy;});
        if(error)std::rethrow_exception(error);
    }
};
static int env_int(const char* name,int fallback,int low,int high) {
    const char* value=std::getenv(name);const int result=value?std::stoi(value):fallback;
    if(result<low || result>high)throw std::runtime_error(std::string("Invalid ")+name);
    return result;
}
static ncnn::Mat load_input(const char* path) {
    ncnn::Mat result(96,128,3);std::ifstream file(path,std::ios::binary);
    for(int c=0;c<3;++c)if(!file.read(reinterpret_cast<char*>(static_cast<float*>(result.channel(c))),96*128*4))
        throw std::runtime_error("Truncated normalized pair input");
    if(file.peek()!=EOF)throw std::runtime_error("Extra pair input bytes");
    for(int c=0;c<3;++c)for(int i=0;i<96*128;++i)
        if(!std::isfinite(static_cast<const float*>(result.channel(c))[i]))
            throw std::runtime_error("Nonfinite pair input");
    return result;
}
static void timing_json(std::ostream& out,const PoseNetworkSession::Timing& t) {
    out<<"{\"start_ns\":"<<t.start_ns<<",\"end_ns\":"<<t.end_ns
       <<",\"forward_ms\":"<<(t.end_ns-t.start_ns)*1e-6<<",\"pl_calls\":"<<t.calls
       <<",\"pl_macs\":"<<t.macs<<",\"driver_wait_ms\":"<<t.wait_ms
       <<",\"submission_queue_ms\":"<<t.queue_ms<<",\"control_including_queue_ms\":"<<t.control_ms
       <<",\"quantize_ms\":"<<t.quantize_ms<<",\"dequantize_ms\":"<<t.dequantize_ms;
#ifdef U02_STREAM_BOUNDARY_PROFILE
    out<<",\"boundary_range_ms\":"<<t.boundary_range_ms<<",\"boundary_quant_ms\":"<<t.boundary_quant_ms
       <<",\"boundary_coeff_ms\":"<<t.boundary_coeff_ms<<",\"boundary_patch_ms\":"<<t.boundary_patch_ms;
#endif
#ifdef U02_STREAM_RANGE_CACHE
    out<<",\"range_hit_elements\":"<<t.range_hit_elements
       <<",\"range_verified_elements\":"<<t.range_verified_elements
       <<",\"range_published_elements\":"<<t.range_published_elements;
#endif
    out<<"}";
}
int main(int argc,char** argv) {
    try {
        if(argc!=13)throw std::runtime_error("Usage: pose_pair_network cpu|stream-csim|pl-stream PARAM BIN LEFT.f32 RIGHT.f32 OUTPUT_PREFIX SOURCE_W SOURCE_H ROI_X ROI_Y ROI_W ROI_H");
        const std::string mode=argv[1],execution=std::getenv("POSE_PAIR_EXECUTION")?std::getenv("POSE_PAIR_EXECUTION"):"parallel";
        if(mode!="cpu" && mode!="stream-csim" && mode!="pl-stream")throw std::runtime_error("Invalid complete network mode");
        if(execution!="serial" && execution!="parallel")throw std::runtime_error("Invalid pair execution");
        const int threads=env_int("POSE_THREADS",1,1,4),warmup=env_int("POSE_WARMUP",3,0,100),repeats=env_int("POSE_REPEAT",10,1,1000);
        const int sw=std::stoi(argv[7]),sh=std::stoi(argv[8]);
        const double rx=std::stod(argv[9]),ry=std::stod(argv[10]),rw=std::stod(argv[11]),rh=std::stod(argv[12]);
        if(sw<1 || sh<1 || !std::isfinite(rx+ry+rw+rh) || rw<=0 || rh<=0)throw std::runtime_error("Invalid source/ROI");
        const auto left_input=load_input(argv[4]),right_input=load_input(argv[5]);
        std::shared_ptr<StreamDualDevice> device;
        std::unique_ptr<Backend> left_backend,right_backend;
        if(mode=="pl-stream") {
#ifdef U02_STREAM_DUAL_FP32
            device=std::make_shared<StreamDualDevice>("/dev/u02_stream_dual_fp32",U02_STREAM_DUAL_FP32_ABI,U02_STREAM_DUAL_FP32_ARENA_BYTES);
            left_backend.reset(new StreamDualFp32Backend(device,0));right_backend.reset(new StreamDualFp32Backend(device,1));
#else
            device=std::make_shared<StreamDualDevice>();
            left_backend.reset(new StreamDualBackend(device,0));right_backend.reset(new StreamDualBackend(device,1));
#endif
        }
        PoseNetworkSession left(mode,argv[2],argv[3],threads,std::move(left_backend));
        PoseNetworkSession right(mode,argv[2],argv[3],threads,std::move(right_backend));
        PoseWorker left_worker(left,left_input),right_worker(right,right_input);
        struct Row {uint64_t start,end;PoseNetworkSession::Timing left,right;};
        std::vector<Row> rows;rows.reserve(repeats);
        uint64_t all_pl_calls=0;
        for(int i=-warmup;i<repeats;++i) {
            const auto start=PoseNetworkSession::now_ns();
            left_worker.submit();
            if(execution=="serial")left_worker.wait();
            right_worker.submit();left_worker.wait();right_worker.wait();
            const auto end=PoseNetworkSession::now_ns();
            if(mode=="pl-stream" && (left.timing.calls!=110 || right.timing.calls!=110))
                throw std::runtime_error("Both complete networks must issue 110 actual PL calls");
            all_pl_calls+=left.timing.calls+right.timing.calls;
            if(i>=0)rows.push_back({start,end,left.timing,right.timing});
        }
        left.save(std::string(argv[6])+".left.json",sw,sh,rx,ry,rw,rh);
        right.save(std::string(argv[6])+".right.json",sw,sh,rx,ry,rw,rh);
        std::vector<double> sorted;uint64_t measured_calls=0,measured_ns=0;
        for(const auto& row:rows) {sorted.push_back((row.end-row.start)*1e-6);measured_ns+=row.end-row.start;measured_calls+=row.left.calls+row.right.calls;}
        std::sort(sorted.begin(),sorted.end());
        std::ofstream out(std::string(argv[6])+".json");if(!out)throw std::runtime_error("Cannot save pair benchmark");
        out<<std::setprecision(12)<<"{\"scope\":\"two_complete_board_networks_on_fixed_normalized_inputs_no_capture_decode_network\","
           <<"\"backend\":\""<<mode<<"\",\"execution\":\""<<execution<<"\",\"dma_abi\":"<<(device?
#ifdef U02_STREAM_DUAL_FP32
               8
#else
               7
#endif
               :0)
           <<",\"stem_variant\":\"int16_smlal_int32_accumulate_when_pl_stream\",\"threads_per_network\":"<<threads
           <<",\"warmup_pairs\":"<<warmup<<",\"measured_pairs\":"<<repeats<<",\"all_pl_calls\":"<<all_pl_calls
           <<",\"measured_pl_calls\":"<<measured_calls<<",\"pair_wall_includes_worker_dispatch_and_output_retention\":true"
           <<",\"pair_wall_median_ms\":"<<(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*.5
           <<",\"pair_wall_p95_ms\":"<<sorted[size_t(std::ceil(sorted.size()*.95))-1]
           <<",\"processing_pairs_per_second\":"<<double(repeats)*1e9/measured_ns<<",\"rows\":[";
        for(size_t i=0;i<rows.size();++i) {
            if(i)out<<",";const auto& row=rows[i];
            out<<"{\"pair_start_ns\":"<<row.start<<",\"pair_end_ns\":"<<row.end<<",\"left\":";
            timing_json(out,row.left);out<<",\"right\":";timing_json(out,row.right);out<<"}";
        }
        out<<"]}\n";if(!out)throw std::runtime_error("Pair benchmark write failed");
        std::cout<<"PAIR_FULL_NETWORK_COMPLETED pairs="<<repeats<<" measured_pl_calls="<<measured_calls<<"\n";
        return 0;
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
