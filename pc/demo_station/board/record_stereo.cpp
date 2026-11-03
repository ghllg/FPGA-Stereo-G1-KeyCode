#include "camera_capture.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <array>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/statvfs.h>
#include <unistd.h>

static volatile std::sig_atomic_t interrupted=0;
static void signal_stop(int){interrupted=1;}
static uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static std::string node_for(const std::string& bus){
    std::string found;
    for(int i=0;i<16;++i){
        const auto node="/dev/video"+std::to_string(i);
        int fd=open(node.c_str(),O_RDONLY|O_NONBLOCK|O_CLOEXEC);
        if(fd<0)continue;
        v4l2_capability cap{};int ok=ioctl(fd,VIDIOC_QUERYCAP,&cap);close(fd);
        uint32_t flags=(cap.capabilities&V4L2_CAP_DEVICE_CAPS)?cap.device_caps:cap.capabilities;
        if(ok==0 && (flags&V4L2_CAP_VIDEO_CAPTURE) && (flags&V4L2_CAP_STREAMING) && bus==reinterpret_cast<char*>(cap.bus_info)){
            if(!found.empty())throw std::runtime_error("Ambiguous camera identity");
            found=node;
        }
    }
    if(found.empty())throw std::runtime_error("Missing camera: "+bus);
    return found;
}
static void write_new(const std::string& path,const std::vector<uint8_t>& bytes){
    int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);
    if(fd<0)throw std::runtime_error("Cannot create image");
    size_t offset=0;
    while(offset<bytes.size()){
        auto n=write(fd,bytes.data()+offset,bytes.size()-offset);
        if(n<0 && errno==EINTR)continue;
        if(n<=0){close(fd);throw std::runtime_error("Image write failed");}
        offset+=size_t(n);
    }
    if(close(fd))throw std::runtime_error("Image close failed");
}
int main(int argc,char** argv){
    try{
        if(argc!=5)throw std::runtime_error("record_stereo SECONDS MAX_MIB LEFT_USB_BUS RIGHT_USB_BUS (new working directory)");
        int seconds=std::stoi(argv[1]),limit=std::stoi(argv[2]);
        if(seconds<1||seconds>300||limit<32||limit>8192)throw std::runtime_error("Capture bounds invalid");
        if(std::string(argv[3])==argv[4])throw std::runtime_error("Cameras must be distinct");
        if(access("left.csv",F_OK)==0||access("right.csv",F_OK)==0)throw std::runtime_error("Existing capture");
        struct statvfs disk{};
        if(statvfs(".",&disk)||uint64_t(disk.f_bavail)*disk.f_frsize<uint64_t(limit)*1048576+64*1048576)
            throw std::runtime_error("Insufficient board storage for configured bound");
        std::signal(SIGTERM,signal_stop);std::signal(SIGINT,signal_stop);
        std::array<std::unique_ptr<CameraCapture>,2> cameras;
        cameras[0].reset(new CameraCapture(node_for(argv[3]),1920,1080,false));
        cameras[1].reset(new CameraCapture(node_for(argv[4]),1920,1080,false));
        std::atomic<bool> start(false),stop(false);std::atomic<uint64_t> saved(0);
        std::array<std::exception_ptr,2> errors;
        std::array<unsigned,2> counts{{0,0}};
        const uint64_t deadline=now_ns()+uint64_t(seconds)*1000000000;
        auto run=[&](unsigned side){try{
            std::string name=side?"right":"left";
            std::ofstream csv(name+".csv");
            if(!csv)throw std::runtime_error("CSV open failed");
            csv<<"capture_index,sequence,source_timestamp_us,flags,receive_monotonic_ns,discarded_ready_frames,jpeg_bytes\n";
            while(!start.load())std::this_thread::yield();
            cameras[side]->start();
            uint64_t previous=0;
            while(now_ns()<deadline&&!stop.load()&&!interrupted&&access("cancel",F_OK)!=0){
                std::vector<uint8_t> jpeg;auto frame=cameras[side]->next(jpeg);auto received=now_ns();
                if(previous&&frame.timestamp_us<=previous)throw std::runtime_error("Non-monotonic timestamp");
                previous=frame.timestamp_us;
                uint64_t before=saved.fetch_add(jpeg.size());
                if(before+jpeg.size()>uint64_t(limit)*1048576)throw std::runtime_error("Recording storage bound reached");
                unsigned i=counts[side];write_new(name+"_"+std::to_string(i)+".jpg",jpeg);
                csv<<i<<','<<frame.sequence<<','<<frame.timestamp_us<<','<<frame.flags<<','<<received<<','<<frame.discarded_ready_frames<<','<<jpeg.size()<<'\n';
                csv.flush();if(!csv)throw std::runtime_error("CSV write failed");++counts[side];
            }
            cameras[side]->stop();
        }catch(...){errors[side]=std::current_exception();stop.store(true);}};
        std::thread left(run,0),right;
        try{right=std::thread(run,1);start.store(true);left.join();right.join();}
        catch(...){stop.store(true);start.store(true);if(left.joinable())left.join();if(right.joinable())right.join();throw;}
        for(auto e:errors)if(e)std::rethrow_exception(e);
        if(interrupted||access("cancel",F_OK)==0)throw std::runtime_error("Capture cancelled; partial data retained");
        if(!counts[0]||!counts[1])throw std::runtime_error("No camera frames");
        std::cout<<"{\"complete\":true,\"left_frames\":"<<counts[0]<<",\"right_frames\":"<<counts[1]<<",\"bytes\":"<<saved.load()<<",\"exposure_sync_verified\":false}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
