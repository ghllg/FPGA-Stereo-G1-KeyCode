// Same recorded JPEG pairs, serial bounded contexts, one thread/network.
// Validation is outside measured time; throughput uses summed pair intervals.
#include "pose_network_session.h"
#include "backend_stream_pipeline.h"
#include "image_preprocess.h"
#include <algorithm>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
static volatile std::sig_atomic_t stopped=0;
static void stop_signal(int){stopped=1;}
struct Input{std::string left,right;uint64_t lt,rt;};
int main(int argc,char** argv){try{
    if(argc!=14)throw std::runtime_error("compare_sequence cpu|pl-stream PARAM BIN PAIRS_TSV OUTPUT_PREFIX LX LY LW LH RX RY RW RH");
    std::string mode=argv[1],prefix=argv[5];
    if(mode!="cpu"&&mode!="pl-stream")throw std::runtime_error("Invalid backend");
    std::signal(SIGTERM,stop_signal);std::signal(SIGINT,stop_signal);
    double roi[2][4];for(int s=0;s<2;++s)for(int j=0;j<4;++j)roi[s][j]=std::stod(argv[6+s*4+j]);
    for(auto& r:roi)if(!std::isfinite(r[0]+r[1]+r[2]+r[3])||r[0]<0||r[1]<0||r[2]<=0||r[3]<=0||r[0]+r[2]>1920||r[1]+r[3]>1080)throw std::runtime_error("Invalid ROI");
    std::ifstream manifest(argv[4]);std::vector<Input> inputs;Input in;
    while(manifest>>in.left>>in.right>>in.lt>>in.rt){if(inputs.size()>=9000)throw std::runtime_error("Pair limit");inputs.push_back(in);}
    if(!manifest.eof()||inputs.size()<2)throw std::runtime_error("Invalid pair list");
    std::shared_ptr<StreamPipelineDevice> device;
    if(mode=="pl-stream")device=std::make_shared<StreamPipelineDevice>(2);
    std::array<std::unique_ptr<PoseNetworkSession>,2> sessions;
    for(int s=0;s<2;++s){std::unique_ptr<Backend> b;if(device)b.reset(new StreamPipelineBackend(device,s));sessions[s].reset(new PoseNetworkSession(mode,argv[2],argv[3],1,std::move(b)));}
    if(device)device->seal();
    PoseNetworkSession reference(mode=="cpu"?"cpu":"stream-csim",argv[2],argv[3],1);
    std::ofstream frames(prefix+".ndjson");if(!frames)throw std::runtime_error("Cannot create output");
    std::vector<double> inference,latency;double total=0;uint64_t calls=0;
    for(int i=-2;i<int(inputs.size());++i){
        if(stopped||access("cancel",F_OK)==0)throw std::runtime_error("Cancelled between complete inferences");
        auto& row=inputs[std::max(0,i)];std::array<ncnn::Mat,2> tensors;
        double infer=0;uint64_t pair_calls=0;
        const auto start=PoseNetworkSession::now_ns();
        for(int side=0;side<2;++side){
            auto encoded=read_pose_image(side?row.right:row.left);tensors[side]=ncnn::Mat(96,128,3);
            auto& r=roi[side];prepare_pose_image(encoded,1920,1080,r[0],r[1],r[2],r[3],static_cast<float*>(tensors[side].data));
            sessions[side]->forward(tensors[side]);auto& t=sessions[side]->timing;
            infer+=(t.end_ns-t.start_ns)*1e-6;pair_calls+=t.calls;
            if((mode=="cpu"&&t.calls)||(mode=="pl-stream"&&t.calls!=110))throw std::runtime_error("Unexpected actual PL count");
        }
        double elapsed=(PoseNetworkSession::now_ns()-start)*1e-6;
        // Complete CPU oracle verification for every image, outside timing.
        for(int side=0;side<2;++side){reference.forward(tensors[side]);for(int c=0;c<17;++c)
            if(memcmp(reference.output.channel(c),sessions[side]->output.channel(c),768*4))throw std::runtime_error("Heatmap reference mismatch");}
        if(i<0)continue;
        total+=elapsed;latency.push_back(elapsed);inference.push_back(infer);calls+=pair_calls;
        frames<<std::setprecision(12)<<"{\"pair_id\":"<<i<<",\"inference_ms\":"<<infer<<",\"total_ms\":"<<elapsed<<",\"cameras\":[";
        for(int s=0;s<2;++s){if(s)frames<<',';auto& r=roi[s];auto pts=sessions[s]->points(r[0],r[1],r[2],r[3]);
            frames<<"{\"camera_id\":\""<<(s?"right":"left")<<"\",\"capture_timestamp_us\":"<<(s?row.rt:row.lt)<<",\"keypoints\":[";
            for(int k=0;k<17;++k){if(k)frames<<',';float x=pts[k*3],y=pts[k*3+1],score=pts[k*3+2];frames<<'['<<x<<','<<y<<','<<score<<','<<((score>=.3&&x>=0&&x<1920&&y>=0&&y<1080)?"true":"false")<<']';}frames<<"]}";
        }frames<<"]}\n";frames.flush();if(!frames)throw std::runtime_error("Result write failed");
        std::cout<<"PROGRESS "<<i+1<<' '<<inputs.size()<<'\n'<<std::flush;
    }
    auto percentile=[](std::vector<double> v,double p){std::sort(v.begin(),v.end());return v[size_t(std::ceil(p*v.size()))-1];};
    std::ofstream out(prefix+".json");out<<std::setprecision(12)<<"{\"backend\":\""<<mode<<"\",\"scope\":\"board_recorded_jpeg_serial_decode_preprocess_network_excludes_oracle_and_result_IO\",\"measured_pairs\":"<<inputs.size()<<",\"warmup_pairs\":2,\"inflight_pairs\":1,\"threads_per_network\":1,\"measured_pl_calls\":"<<calls<<",\"all_full_heatmaps_exact\":true,\"processing_pairs_per_second\":"<<inputs.size()*1000./total<<",\"pair_latency_median_ms\":"<<percentile(latency,.5)<<",\"pair_latency_p95_ms\":"<<percentile(latency,.95)<<",\"inference_median_ms\":"<<percentile(inference,.5)<<",\"timed_sum_ms\":"<<total<<",\"continuous_camera_throughput_measured\":false}\n";
    if(!out)throw std::runtime_error("Report write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
