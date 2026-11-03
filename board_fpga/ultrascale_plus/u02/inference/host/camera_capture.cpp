#include "camera_capture.h"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
int call(int fd,unsigned long request,void* value) {
    int result;do {result=ioctl(fd,request,value);} while(result<0 && errno==EINTR);
    return result;
}
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
constexpr size_t max_jpeg=16*1024*1024;
}

struct CameraCapture::State {
    int fd=-1;bool streaming=false,started_once=false;
    std::string bus;
    struct Mapping {void* data=MAP_FAILED;size_t size=0;};
    std::vector<Mapping> maps;
    void stop() {
        if(streaming) {
            int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            require(call(fd,VIDIOC_STREAMOFF,&type)==0,"Camera STREAMOFF failed");
            streaming=false;
        }
    }
    ~State() {
        if(streaming) {int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;call(fd,VIDIOC_STREAMOFF,&type);}
        for(auto& m:maps)if(m.data!=MAP_FAILED)munmap(m.data,m.size);
        if(fd>=0)close(fd);
    }
};

CameraCapture::CameraCapture(const std::string& node,int width,int height,bool start_immediately):state(new State) {
    require(node=="/dev/video0" || node=="/dev/video1" || node=="/dev/video2",
            "Use an explicitly identified camera node");
    require(width>0 && width<=4096 && height>0 && height<=4096,"Invalid camera dimensions");
    state->fd=open(node.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);
    require(state->fd>=0,"Cannot open camera node");
    v4l2_capability cap={};
    require(call(state->fd,VIDIOC_QUERYCAP,&cap)==0,"Camera QUERYCAP failed");
    state->bus=std::string(reinterpret_cast<const char*>(cap.bus_info),
        strnlen(reinterpret_cast<const char*>(cap.bus_info),sizeof(cap.bus_info)));
    require(!state->bus.empty(),"Camera physical bus identity missing");
    const uint32_t caps=(cap.capabilities&V4L2_CAP_DEVICE_CAPS)?cap.device_caps:cap.capabilities;
    // The SDK's older header predates META_CAPTURE. Require the actual image
    // capability instead of interpreting an unknown node from its name.
    require((caps&V4L2_CAP_VIDEO_CAPTURE) && (caps&V4L2_CAP_STREAMING),
            "Camera node must support image capture, not metadata");
    v4l2_format fmt={};fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width=width;fmt.fmt.pix.height=height;
    fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_MJPEG;fmt.fmt.pix.field=V4L2_FIELD_ANY;
    require(call(state->fd,VIDIOC_S_FMT,&fmt)==0,"Camera S_FMT failed");
    require(fmt.fmt.pix.width==unsigned(width) && fmt.fmt.pix.height==unsigned(height) &&
            fmt.fmt.pix.pixelformat==V4L2_PIX_FMT_MJPEG && fmt.fmt.pix.sizeimage<=max_jpeg,
            "Camera did not accept requested bounded MJPEG mode");
    v4l2_streamparm rate={};rate.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rate.parm.capture.timeperframe.numerator=1;rate.parm.capture.timeperframe.denominator=30;
    require(call(state->fd,VIDIOC_S_PARM,&rate)==0,"Camera frame-rate request failed");
#if defined(U02_STEREO_SOURCE_1080P) || defined(U02_CAPTURE_1080P)
    // The candidate must not silently negotiate a lower frame rate. Actual
    // sustained capture rate still needs source/receive timestamp measurement.
    require(rate.parm.capture.timeperframe.numerator && rate.parm.capture.timeperframe.denominator &&
            std::abs(double(rate.parm.capture.timeperframe.denominator)/rate.parm.capture.timeperframe.numerator-30.)<=.03,
            "1080p camera did not negotiate approximately 30 FPS");
#endif
    v4l2_requestbuffers req={};req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory=V4L2_MEMORY_MMAP;req.count=4;
    require(call(state->fd,VIDIOC_REQBUFS,&req)==0 && req.count>=2 && req.count<=4,
            "Camera requires two to four bounded buffers");
    state->maps.resize(req.count);
    for(unsigned i=0;i<req.count;++i) {
        v4l2_buffer buf={};buf.type=req.type;buf.memory=req.memory;buf.index=i;
        require(call(state->fd,VIDIOC_QUERYBUF,&buf)==0 && buf.length>0 && buf.length<=max_jpeg,
                "Invalid camera buffer length");
        auto& map=state->maps[i];map.size=buf.length;
        map.data=mmap(nullptr,map.size,PROT_READ|PROT_WRITE,MAP_SHARED,state->fd,buf.m.offset);
        require(map.data!=MAP_FAILED,"Camera mmap failed");
        require(call(state->fd,VIDIOC_QBUF,&buf)==0,"Camera initial QBUF failed");
    }
    if(start_immediately)start();
}
void CameraCapture::start() {
    require(!state->started_once,"Camera has already been started");
    int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    require(call(state->fd,VIDIOC_STREAMON,&type)==0,"Camera STREAMON failed");
    state->streaming=true;
    state->started_once=true;
}
CameraCapture::~CameraCapture()=default;
const std::string& CameraCapture::bus_id() const {return state->bus;}
void CameraCapture::stop() {state->stop();}

CameraFrame CameraCapture::next(std::vector<uint8_t>& jpeg) {
    require(state->streaming,"Camera is stopped");
    const auto start=std::chrono::steady_clock::now();
    CameraFrame frame;unsigned consumed=0;
    // Finite polling deadline and drain bound: never loop indefinitely when a
    // producer continues supplying new frames. Retain the newest ready frame.
    while(consumed<state->maps.size()) {
        v4l2_buffer buf={};buf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;buf.memory=V4L2_MEMORY_MMAP;
        if(call(state->fd,VIDIOC_DQBUF,&buf)<0) {
            require(errno==EAGAIN,"Camera DQBUF failed");
            if(consumed)break;
            const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-start).count();
            require(elapsed<2000,"Camera frame timeout");
            pollfd p={state->fd,POLLIN,0};
            const int ready=poll(&p,1,int(2000-elapsed));
            if(ready<0 && errno==EINTR)continue;
            require(ready>0 && !(p.revents&(POLLERR|POLLHUP|POLLNVAL)),"Camera poll timeout or error");
            continue;
        }
        require(buf.index<state->maps.size(),"Invalid camera buffer index");
        const auto& map=state->maps[buf.index];
        require(!(buf.flags&V4L2_BUF_FLAG_ERROR) && buf.bytesused>=4 &&
                buf.bytesused<=map.size && buf.bytesused<=max_jpeg,"Invalid camera frame length or flags");
        const uint8_t* bytes=static_cast<const uint8_t*>(map.data);
        require(bytes[0]==0xff && bytes[1]==0xd8 && bytes[buf.bytesused-2]==0xff && bytes[buf.bytesused-1]==0xd9,
                "Incomplete camera JPEG");
        jpeg.assign(bytes,bytes+buf.bytesused);
        frame.sequence=buf.sequence;frame.flags=buf.flags;
        require(buf.timestamp.tv_sec>=0 && buf.timestamp.tv_usec>=0 && buf.timestamp.tv_usec<1000000,
                "Invalid camera timestamp");
        frame.timestamp_us=uint64_t(buf.timestamp.tv_sec)*1000000+buf.timestamp.tv_usec;
        require(call(state->fd,VIDIOC_QBUF,&buf)==0,"Camera requeue failed");
        ++consumed;
    }
    frame.discarded_ready_frames=consumed-1;
    frame.capture_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return frame;
}
