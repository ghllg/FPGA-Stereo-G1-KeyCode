// Read-only standard V4L2 query: no S_CTRL, S_FMT, REQBUFS or STREAMON.
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>

static int query(int fd,unsigned long request,void* value) {
    int rc;do{rc=ioctl(fd,request,value);}while(rc<0 && errno==EINTR);return rc;
}
static std::string quoted(const unsigned char* text,size_t size) {
    std::string out="\"";
    for(size_t i=0;i<size && text[i];++i) {
        const unsigned c=text[i];
        if(c=='"' || c=='\\')out+='\\';
        if(c<32) {const char hex[]="0123456789abcdef";out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=char(c);
    }
    return out+'"';
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    try {
        std::ofstream out(argv[1]);if(!out)throw std::runtime_error("Cannot save read-only probe");
        out<<"{\"schema\":\"u02_v4l2_readonly_controls_v1\",\"stream_started\":false,\"controls_changed\":false,\"pl_calls\":0,\"devices\":[";
        unsigned count=0;
        for(unsigned index=0;index<4;++index) {
            const std::string node="/dev/video"+std::to_string(index);
            const int fd=open(node.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);if(fd<0)continue;
            v4l2_capability cap={};
            if(query(fd,VIDIOC_QUERYCAP,&cap)<0){close(fd);continue;}
            const unsigned bits=(cap.capabilities&V4L2_CAP_DEVICE_CAPS)?cap.device_caps:cap.capabilities;
            if(!(bits&V4L2_CAP_VIDEO_CAPTURE)){close(fd);continue;}
            const std::string bus(reinterpret_cast<const char*>(cap.bus_info),strnlen(reinterpret_cast<const char*>(cap.bus_info),sizeof(cap.bus_info)));
            if(bus!="usb-xhci-hcd.0.auto-1.4" && bus!="usb-xhci-hcd.0.auto-1.2") {close(fd);throw std::runtime_error("Unexpected physical camera bus");}
            if(count++)out<<",";
            out<<"{\"node\":\""<<node<<"\",\"driver\":"<<quoted(cap.driver,sizeof(cap.driver))<<",\"card\":"<<quoted(cap.card,sizeof(cap.card))
               <<",\"bus\":"<<quoted(cap.bus_info,sizeof(cap.bus_info))<<",\"capabilities\":"<<bits;
            v4l2_format format={};format.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            const int format_result=query(fd,VIDIOC_G_FMT,&format),format_errno=errno;
            out<<",\"get_format_errno\":"<<(format_result<0?format_errno:0);
            if(format_result==0)out<<",\"width\":"<<format.fmt.pix.width<<",\"height\":"<<format.fmt.pix.height<<",\"fourcc_integer\":"<<format.fmt.pix.pixelformat;
            v4l2_streamparm rate={};rate.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            const int rate_result=query(fd,VIDIOC_G_PARM,&rate),rate_errno=errno;
            out<<",\"get_rate_errno\":"<<(rate_result<0?rate_errno:0);
            if(rate_result==0)out<<",\"timeperframe\":["<<rate.parm.capture.timeperframe.numerator<<","<<rate.parm.capture.timeperframe.denominator<<"]";
            out<<",\"controls\":[";
            v4l2_queryctrl control={};control.id=V4L2_CTRL_FLAG_NEXT_CTRL;unsigned previous=0,n=0;
            while(query(fd,VIDIOC_QUERYCTRL,&control)==0) {
                if(control.id<=previous || n>=128){close(fd);throw std::runtime_error("Control enumeration not bounded");}
                if(n++)out<<",";
                out<<"{\"id\":"<<control.id<<",\"name\":"<<quoted(control.name,sizeof(control.name))<<",\"type\":"<<control.type<<",\"flags\":"<<control.flags
                   <<",\"min\":"<<control.minimum<<",\"max\":"<<control.maximum<<",\"step\":"<<control.step<<",\"default\":"<<control.default_value;
                if(control.type!=V4L2_CTRL_TYPE_CTRL_CLASS && control.type!=V4L2_CTRL_TYPE_BUTTON && !(control.flags&V4L2_CTRL_FLAG_DISABLED)) {
                    v4l2_control value={};value.id=control.id;const int rc=query(fd,VIDIOC_G_CTRL,&value),error=errno;
                    out<<",\"get_errno\":"<<(rc<0?error:0);if(rc==0)out<<",\"current\":"<<value.value;
                }
                if(control.type==V4L2_CTRL_TYPE_MENU && control.maximum-control.minimum<32) {
                    out<<",\"menu\":[";bool first=true;
                    for(int item=control.minimum;item<=control.maximum;++item) {
                        v4l2_querymenu menu={};menu.id=control.id;menu.index=item;
                        if(query(fd,VIDIOC_QUERYMENU,&menu)==0) {
                            if(!first)out<<",";first=false;
                            out<<"{\"value\":"<<item<<",\"name\":"<<quoted(menu.name,sizeof(menu.name))<<"}";
                        }
                    }
                    out<<"]";
                }
                out<<"}";previous=control.id;control={};control.id=previous|V4L2_CTRL_FLAG_NEXT_CTRL;
            }
            out<<"],\"enumeration_end_errno\":"<<errno<<"}";close(fd);
        }
        out<<"],\"capture_nodes\":"<<count<<",\"complete\":"<<(count==2?"true":"false")<<"}\n";
        if(count!=2)throw std::runtime_error("Expected two physical capture nodes");
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
