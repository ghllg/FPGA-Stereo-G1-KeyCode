// Capture-only UVC timestamp evidence. No accelerator, image decode or timestamp correction.
#include "camera_capture.h"
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

namespace {
constexpr uint32_t meta_cap=0x00800000;
constexpr v4l2_buf_type meta_type=static_cast<v4l2_buf_type>(13);
uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void require(bool ok,const char* text) { if(!ok)throw std::runtime_error(text); }
int call(int fd,unsigned long op,void* arg) { int rc;do {rc=ioctl(fd,op,arg);}while(rc<0 && errno==EINTR);return rc; }
struct Identity {std::string video,meta;};
Identity resolve(const std::string& bus) {
    Identity result;
    for(int i=0;i<8;++i) {
        const std::string name="/dev/video"+std::to_string(i);
        const int fd=open(name.c_str(),O_RDONLY|O_NONBLOCK|O_CLOEXEC);if(fd<0)continue;
        v4l2_capability cap{};const int rc=call(fd,VIDIOC_QUERYCAP,&cap);close(fd);
        if(rc<0 || bus!=reinterpret_cast<const char*>(cap.bus_info))continue;
        const uint32_t flags=(cap.capabilities&V4L2_CAP_DEVICE_CAPS)?cap.device_caps:cap.capabilities;
        if(flags&V4L2_CAP_VIDEO_CAPTURE) {require(result.video.empty(),"Duplicate video identity");result.video=name;}
        if(flags&meta_cap) {require(result.meta.empty(),"Duplicate metadata identity");result.meta=name;}
    }
    require(!result.video.empty() && !result.meta.empty(),"Camera video or metadata node absent");
    return result;
}
class Metadata {
    int fd=-1;bool streaming=false;
    struct Map {void* address;size_t size;};std::vector<Map> maps;
    void cleanup() noexcept {
        if(fd>=0 && streaming) {auto type=meta_type;call(fd,VIDIOC_STREAMOFF,&type);}
        for(const auto& map:maps)munmap(map.address,map.size);
        if(fd>=0)close(fd);fd=-1;
    }
public:
    explicit Metadata(const std::string& node) {
        try {
            fd=open(node.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);require(fd>=0,"Metadata open failed");
            v4l2_format format{};format.type=meta_type;
            require(call(fd,VIDIOC_G_FMT,&format)==0,"Metadata format query failed");
            uint32_t fourcc=0;std::memcpy(&fourcc,format.fmt.raw_data,4);
            require(fourcc==v4l2_fourcc('U','V','C','H'),"Unexpected metadata format");
            v4l2_requestbuffers req{};req.type=meta_type;req.memory=V4L2_MEMORY_MMAP;req.count=8;
            require(call(fd,VIDIOC_REQBUFS,&req)==0 && req.count>=2 && req.count<=32,"Metadata buffers unavailable");
            for(unsigned i=0;i<req.count;++i) {
                v4l2_buffer buf{};buf.type=meta_type;buf.memory=V4L2_MEMORY_MMAP;buf.index=i;
                require(call(fd,VIDIOC_QUERYBUF,&buf)==0,"Metadata QUERYBUF failed");
                void* address=mmap(nullptr,buf.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,buf.m.offset);
                require(address!=MAP_FAILED,"Metadata mmap failed");maps.push_back({address,buf.length});
                require(call(fd,VIDIOC_QBUF,&buf)==0,"Metadata QBUF failed");
            }
            auto type=meta_type;require(call(fd,VIDIOC_STREAMON,&type)==0,"Metadata STREAMON failed");streaming=true;
        }catch(...){cleanup();throw;}
    }
    ~Metadata(){cleanup();}
    void collect(std::ostream& out,const std::atomic<bool>& done) {
        const auto deadline=now_ns()+20000000000ull;unsigned count=0;
        while(!done && now_ns()<deadline && count<4096) {
            pollfd p{fd,POLLIN,0};const int ready=poll(&p,1,100);
            if(ready<0 && errno==EINTR)continue;
            require(ready>=0 && !(p.revents&(POLLERR|POLLHUP|POLLNVAL)),"Metadata polling failed");if(!ready)continue;
            v4l2_buffer buf{};buf.type=meta_type;buf.memory=V4L2_MEMORY_MMAP;
            if(call(fd,VIDIOC_DQBUF,&buf)<0){if(errno==EAGAIN)continue;throw std::runtime_error("Metadata DQBUF failed");}
            require(buf.index<maps.size() && buf.bytesused<=maps[buf.index].size,"Invalid metadata buffer");
            const auto* bytes=static_cast<const unsigned char*>(maps[buf.index].address);
            out<<"{\"sequence\":"<<buf.sequence<<",\"timestamp_us\":"<<uint64_t(buf.timestamp.tv_sec)*1000000+buf.timestamp.tv_usec
               <<",\"receive_ns\":"<<now_ns()<<",\"flags\":"<<buf.flags<<",\"bytes\":"<<buf.bytesused<<",\"hex\":\"";
            for(unsigned i=0;i<buf.bytesused;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(bytes[i]);
            out<<std::dec<<"\"}\n";
            require(call(fd,VIDIOC_QBUF,&buf)==0,"Metadata requeue failed");++count;
        }
        require(done || now_ns()<deadline,"Metadata probe deadline exceeded");
        require(bool(out),"Metadata output failed");
    }
};
}

int main() {
    try {
        std::array<Identity,2> nodes{{resolve("usb-xhci-hcd.0.auto-1.4"),resolve("usb-xhci-hcd.0.auto-1.2")}};
        std::array<std::unique_ptr<CameraCapture>,2> video;
        std::array<std::unique_ptr<Metadata>,2> meta;
        for(unsigned side=0;side<2;++side) {
            video[side].reset(new CameraCapture(nodes[side].video,1920,1080,false));
            meta[side].reset(new Metadata(nodes[side].meta));
        }
        std::atomic<bool> done{false},launch{false};std::array<std::exception_ptr,4> errors;
        std::array<std::thread,4> workers;
        try {
            for(unsigned side=0;side<2;++side) {
                workers[side]=std::thread([&,side] {
                    try {
                        std::ofstream out("video"+std::to_string(side)+".ndjson");require(bool(out),"Video evidence open failed");
                        while(!launch)std::this_thread::yield();video[side]->start();
                        std::vector<uint8_t> jpeg;
                        for(unsigned n=0;n<120 && !done;++n) {
                            const auto frame=video[side]->next(jpeg);
                            out<<"{\"sequence\":"<<frame.sequence<<",\"timestamp_us\":"<<frame.timestamp_us
                               <<",\"receive_ns\":"<<now_ns()<<",\"flags\":"<<frame.flags
                               <<",\"discarded_ready\":"<<frame.discarded_ready_frames<<"}\n";
                        }
                        require(bool(out),"Video evidence write failed");
                    }catch(...){errors[side]=std::current_exception();done=true;}
                });
                workers[side+2]=std::thread([&,side] {
                    try {std::ofstream out("meta"+std::to_string(side)+".ndjson");require(bool(out),"Metadata evidence open failed");meta[side]->collect(out,done);}
                    catch(...){errors[side+2]=std::current_exception();done=true;}
                });
            }
        }catch(...){done=true;launch=true;for(auto& t:workers)if(t.joinable())t.join();throw;}
        launch=true;workers[0].join();workers[1].join();done=true;
        workers[2].join();workers[3].join();
        for(const auto& error:errors)if(error)std::rethrow_exception(error);
        std::cout<<"U02_UVC_TIMESTAMP_PROBE_DONE video_frames_per_side=120 pl_calls=0 timestamp_adjustment_us=0\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
