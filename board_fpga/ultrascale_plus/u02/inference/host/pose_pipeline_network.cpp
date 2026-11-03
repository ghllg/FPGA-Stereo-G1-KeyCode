// Isolated full-network integration for a bounded one/two-pair pipeline.
// Fixed normalized inputs only; no claim of camera/decode/network throughput.
#include "pose_network_worker.h"
#include "backend_stream_pipeline.h"
#include "pose_view_roi.h"
#include <array>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

static int env_number(const char* name,int fallback,int low,int high) {
    const char* value=std::getenv(name);if(!value)return fallback;
    size_t used=0;const int result=std::stoi(value,&used);
    if(value[used] || result<low || result>high)throw std::runtime_error(std::string("Invalid ")+name);
    return result;
}
static ncnn::Mat load_input(const char* path) {
    ncnn::Mat result(96,128,3);std::ifstream file(path,std::ios::binary);
    if(result.empty())throw std::runtime_error("Input allocation");
    for(int c=0;c<3;++c)if(!file.read(reinterpret_cast<char*>(static_cast<float*>(result.channel(c))),96*128*4))
        throw std::runtime_error("Truncated normalized pipeline input");
    if(file.peek()!=EOF)throw std::runtime_error("Extra pipeline input bytes");
    for(int c=0;c<3;++c)for(int i=0;i<96*128;++i)
        if(!std::isfinite(static_cast<const float*>(result.channel(c))[i]))throw std::runtime_error("Nonfinite pipeline input");
    return result;
}
static bool same_heatmap(const ncnn::Mat& a,const ncnn::Mat& b) {
    if(a.w!=24 || a.h!=32 || a.c!=17 || a.elempack!=1 || a.elemsize!=4 ||
       b.w!=24 || b.h!=32 || b.c!=17 || b.elempack!=1 || b.elemsize!=4)return false;
    for(int c=0;c<17;++c)if(memcmp(a.channel(c),b.channel(c),24*32*4))return false;
    return true;
}
static void timing_json(std::ostream& out,const PoseNetworkSession::Timing& t) {
    out<<"{\"start_ns\":"<<t.start_ns<<",\"end_ns\":"<<t.end_ns<<",\"pl_calls\":"<<t.calls
       <<",\"pl_macs\":"<<t.macs<<",\"driver_wait_ms\":"<<t.wait_ms<<",\"submission_queue_ms\":"<<t.queue_ms
       <<",\"control_including_queue_ms\":"<<t.control_ms<<",\"quantize_ms\":"<<t.quantize_ms
       <<",\"dequantize_ms\":"<<t.dequantize_ms<<"}";
}
int main(int argc,char** argv) {
    try {
        if(argc!=15)throw std::runtime_error("Usage: pose_pipeline_network cpu|stream-csim|pl-stream PARAM BIN INPUT0 INPUT1 INPUT2 INPUT3 OUTPUT_PREFIX SOURCE_W SOURCE_H ROI_X ROI_Y ROI_W ROI_H");
        const std::string mode=argv[1];
        if(mode!="cpu" && mode!="stream-csim" && mode!="pl-stream")throw std::runtime_error("Invalid complete network backend");
        const unsigned pairs=env_number("POSE_INFLIGHT_PAIRS",1,1,2),contexts=pairs*2;
        const int warmup=env_number("POSE_WARMUP",6,2,100),repeats=env_number("POSE_REPEAT",100,2,1000);
        if(warmup%2 || repeats%2)throw std::runtime_error("Balanced two-input-pattern run needs even pair counts");
        const int sw=std::stoi(argv[9]),sh=std::stoi(argv[10]);
        const double rx=std::stod(argv[11]),ry=std::stod(argv[12]),rw=std::stod(argv[13]),rh=std::stod(argv[14]);
        // Prepared tensors may come from a padded ROI extending outside the
        // source image. Match the existing Host's coordinate-mapping contract;
        // this entry never reads source pixels or crops an image buffer.
        if(sw<1 || sh<1 || !std::isfinite(rx+ry+rw+rh) || rw<=0 || rh<=0)
            throw std::runtime_error("Invalid source/ROI");
        const std::string prefix=argv[8];
        const auto rois=pose_view_rois(sw,sh,rx,ry,rw,rh);
        std::array<ncnn::Mat,4> inputs,expected,last_outputs,worker_inputs;
        std::array<PoseNetworkSession::Timing,4> last_timings;
        for(unsigned i=0;i<4;++i)inputs[i]=load_input(argv[4+i]);
        // Independent complete network reference on this A53, outside timing.
        // PL mode uses the original integer backend, not a PC-produced answer.
        {
            PoseNetworkSession reference(mode=="cpu"?"cpu":"stream-csim",argv[2],argv[3],1);
            for(unsigned i=0;i<4;++i) {
                reference.forward(inputs[i]);expected[i]=reference.output;
                reference.timing.calls=reference.timing.macs=0; // CPU oracle never counts as actual PL.
                const auto& r=rois[i%2];
                reference.save(prefix+".reference."+std::to_string(i)+".json",sw,sh,r[0],r[1],r[2],r[3]);
            }
        }
        std::shared_ptr<StreamPipelineDevice> device;
        if(mode=="pl-stream")device=std::make_shared<StreamPipelineDevice>(contexts);
        std::array<std::unique_ptr<PoseNetworkSession>,4> sessions;
        std::array<std::unique_ptr<PoseWorker>,4> workers;
        for(unsigned i=0;i<contexts;++i) {
            std::unique_ptr<Backend> backend;
            if(device)backend.reset(new StreamPipelineBackend(device,i));
            sessions[i].reset(new PoseNetworkSession(mode,argv[2],argv[3],1,std::move(backend)));
        }
        if(device)device->seal();
        for(unsigned i=0;i<contexts;++i)workers[i].reset(new PoseWorker(*sessions[i],worker_inputs[i]));
        struct Row {unsigned pair_id,slot,pattern;uint64_t start,end;PoseNetworkSession::Timing left,right;};
        std::vector<Row> rows;rows.reserve(repeats);
        uint64_t all_calls=0,measured_begin=0,measured_end=0;
        auto phase=[&](unsigned count,bool measured) {
            std::array<uint64_t,2> starts{};
            auto submit=[&](unsigned id) {
                const unsigned slot=id%pairs,pattern=id%2;
                worker_inputs[slot*2]=inputs[pattern*2];worker_inputs[slot*2+1]=inputs[pattern*2+1];
                starts[slot]=PoseNetworkSession::now_ns();
                workers[slot*2]->submit();workers[slot*2+1]->submit();
            };
            if(measured)measured_begin=PoseNetworkSession::now_ns();
            for(unsigned id=0;id<pairs && id<count;++id)submit(id);
            for(unsigned id=0;id<count;++id) {
                const unsigned slot=id%pairs,pattern=id%2;
                workers[slot*2]->wait();workers[slot*2+1]->wait();
                const uint64_t end=PoseNetworkSession::now_ns();
                auto left=sessions[slot*2]->timing,right=sessions[slot*2+1]->timing;
                if(!device)left.calls=right.calls=left.macs=right.macs=0;
                for(unsigned side=0;side<2;++side) {
                    const auto& session=*sessions[slot*2+side];
                    if(!same_heatmap(session.output,expected[pattern*2+side]))throw std::runtime_error("Pipeline full heatmap differs from independent board reference");
                    if((device && session.timing.calls!=110) || (mode=="cpu" && session.timing.calls))throw std::runtime_error("Incorrect actual PL call count");
                    if(device)all_calls+=session.timing.calls;
                    last_outputs[pattern*2+side]=session.output;last_timings[pattern*2+side]=session.timing;
                    if(!device)last_timings[pattern*2+side].calls=last_timings[pattern*2+side].macs=0;
                }
                if(measured)rows.push_back({id,slot,pattern,starts[slot],end,left,right});
                if(id+pairs<count)submit(id+pairs);
            }
            if(measured)measured_end=PoseNetworkSession::now_ns();
        };
        phase(warmup,false);phase(repeats,true);
        // Save actual verified outputs after timing. Every measured/warmup
        // image above was compared in full, not just the final four images.
        for(unsigned i=0;i<4;++i) {
            sessions[0]->output=last_outputs[i];sessions[0]->timing=last_timings[i];
            const auto& r=rois[i%2];
            sessions[0]->save(prefix+".actual."+std::to_string(i)+".json",sw,sh,r[0],r[1],r[2],r[3]);
        }
        std::vector<double> latency;uint64_t measured_calls=0;
        for(const auto& row:rows){latency.push_back((row.end-row.start)*1e-6);measured_calls+=row.left.calls+row.right.calls;}
        std::sort(latency.begin(),latency.end());
        std::ofstream out(prefix+".json");if(!out)throw std::runtime_error("Pipeline report open");
        out<<std::setprecision(12)<<"{\"scope\":\"complete_board_network_fixed_inputs_no_camera_decode_network\",\"backend\":\""<<mode
           <<"\",\"inflight_pairs\":"<<pairs<<",\"contexts\":"<<contexts<<",\"threads_per_network\":1,\"dma_abi\":"<<(device?9:0)
           <<",\"dma_arena_bytes\":"<<(device?U02_POSE_ARENA_BYTES:0)<<",\"dma_used_bytes\":"<<(device?U02_STREAM_PIPELINE_USED_BYTES:0)
           <<",\"shared_model_packets\":"<<(device?device->packet_count():0)<<",\"shared_model_payload_bytes\":"<<(device?device->model_bytes():0)
           <<",\"pl_cores\":"<<(device?1:0)<<",\"warmup_pairs\":"<<warmup<<",\"measured_pairs\":"<<repeats
           <<",\"all_pl_calls\":"<<all_calls<<",\"measured_pl_calls\":"<<measured_calls<<",\"all_full_heatmaps_exact\":true"
           <<",\"measured_start_ns\":"<<measured_begin<<",\"measured_end_ns\":"<<measured_end
           <<",\"processing_pairs_per_second\":"<<double(repeats)*1e9/(measured_end-measured_begin)
           <<",\"pair_latency_median_ms\":"<<(latency[(latency.size()-1)/2]+latency[latency.size()/2])*.5
           <<",\"pair_latency_p95_ms\":"<<latency[size_t(std::ceil(.95*latency.size()))-1]
           <<",\"throughput_uses_elapsed_window_not_inverse_pair_latency\":true,\"validation_in_measured_window\":true,\"rows\":[";
        for(size_t i=0;i<rows.size();++i) {
            const auto& row=rows[i];if(i)out<<",";
            out<<"{\"pair_id\":"<<row.pair_id<<",\"slot\":"<<row.slot<<",\"input_pattern\":"<<row.pattern
               <<",\"pair_start_ns\":"<<row.start<<",\"pair_end_ns\":"<<row.end<<",\"left\":";
            timing_json(out,row.left);out<<",\"right\":";timing_json(out,row.right);out<<"}";
        }
        out<<"]}\n";if(!out)throw std::runtime_error("Pipeline report write");
        std::cout<<"PIPELINE_WHOLE_NETWORK_COMPLETED pairs="<<repeats<<" contexts="<<contexts<<" measured_pl_calls="<<measured_calls<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
