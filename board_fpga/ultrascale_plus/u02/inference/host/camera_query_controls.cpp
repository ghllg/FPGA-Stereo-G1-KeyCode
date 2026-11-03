// Read-only V4L2 capability/control inventory. No S_* ioctl, buffers, STREAMON,
// format negotiation, exposure adjustment, PL access or startup changes.
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static std::string quoted(const char* s,size_t size) {
    std::string out="\"";
    for(size_t i=0;i<size && s[i];++i) {
        const auto c=static_cast<unsigned char>(s[i]);
        if(c=='"' || c=='\\'){out+='\\';out+=char(c);}
        else if(c>=32 && c<127)out+=char(c);
        else out+='?';
    }
    return out+'"';
}
static int query(int fd,unsigned long request,void* value) {
    int rc;do {rc=ioctl(fd,request,value);}while(rc<0 && errno==EINTR);return rc;
}
struct Device {
    int fd;
    explicit Device(const char* path):fd(open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC)) {
        if(fd<0)throw std::runtime_error("Cannot open video node read-only");
    }
    ~Device(){close(fd);}
};
int main(int argc,char** argv) {
    try {
        if(argc<2 || argc>3)throw std::runtime_error("Usage: camera_query_controls /dev/videoN [/dev/videoM]");
        for(int i=1;i<argc;++i) {
            std::string path=argv[i];
            if(path.compare(0,10,"/dev/video") || path.size()==10 || path.find_first_not_of("0123456789",10)!=std::string::npos)
                throw std::runtime_error("Only explicit video nodes accepted");
        }
        std::ifstream boot_file("/proc/sys/kernel/random/boot_id");std::string boot;boot_file>>boot;
        std::cout<<"{\"scope\":\"read_only_stopped_camera_inventory_not_proof_of_prior_stream_controls\",\"boot_id\":"
                 <<quoted(boot.c_str(),boot.size())<<",\"configuration_written\":false,\"stream_started\":false,\"devices\":[";
        for(int i=1;i<argc;++i) {
            Device device(argv[i]);v4l2_capability cap{};
            if(query(device.fd,VIDIOC_QUERYCAP,&cap))throw std::runtime_error("VIDIOC_QUERYCAP failed");
            if(i>1)std::cout<<",";
            std::cout<<"{\"node\":"<<quoted(argv[i],std::strlen(argv[i]))
                <<",\"driver\":"<<quoted(reinterpret_cast<const char*>(cap.driver),sizeof cap.driver)
                <<",\"card\":"<<quoted(reinterpret_cast<const char*>(cap.card),sizeof cap.card)
                <<",\"bus_info\":"<<quoted(reinterpret_cast<const char*>(cap.bus_info),sizeof cap.bus_info)
                <<",\"capabilities\":"<<cap.capabilities;
            v4l2_format format{};format.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            const int format_rc=query(device.fd,VIDIOC_G_FMT,&format),format_error=format_rc?errno:0;
            std::cout<<",\"format_query_errno\":"<<format_error;
            if(!format_rc)std::cout<<",\"current_format\":{\"width\":"<<format.fmt.pix.width
                <<",\"height\":"<<format.fmt.pix.height<<",\"fourcc\":"<<format.fmt.pix.pixelformat<<"}";
            v4l2_streamparm parm{};parm.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            const int parm_rc=query(device.fd,VIDIOC_G_PARM,&parm),parm_error=parm_rc?errno:0;
            std::cout<<",\"interval_query_errno\":"<<parm_error;
            if(!parm_rc)std::cout<<",\"current_interval\":{\"numerator\":"<<parm.parm.capture.timeperframe.numerator
                <<",\"denominator\":"<<parm.parm.capture.timeperframe.denominator<<"}";
            std::cout<<",\"controls\":[";
            v4l2_queryctrl control{};control.id=V4L2_CTRL_FLAG_NEXT_CTRL;unsigned count=0;
            for(;;) {
                if(query(device.fd,VIDIOC_QUERYCTRL,&control)) {
                    if(errno!=EINVAL)throw std::runtime_error("Control enumeration failed");
                    break;
                }
                if(count>=256)throw std::runtime_error("Control inventory limit exceeded");
                if(count++)std::cout<<",";
                std::cout<<"{\"id\":"<<control.id<<",\"name\":"
                    <<quoted(reinterpret_cast<const char*>(control.name),sizeof control.name)
                    <<",\"type\":"<<control.type<<",\"flags\":"<<control.flags
                    <<",\"minimum\":"<<control.minimum<<",\"maximum\":"<<control.maximum
                    <<",\"step\":"<<control.step<<",\"default\":"<<control.default_value;
                if(control.type==V4L2_CTRL_TYPE_INTEGER || control.type==V4L2_CTRL_TYPE_BOOLEAN ||
                   control.type==V4L2_CTRL_TYPE_MENU || control.type==V4L2_CTRL_TYPE_INTEGER_MENU ||
                   control.type==V4L2_CTRL_TYPE_BITMASK) {
                    v4l2_control value{};value.id=control.id;
                    const int rc=query(device.fd,VIDIOC_G_CTRL,&value),error=rc?errno:0;
                    std::cout<<",\"value_query_errno\":"<<error;
                    if(!rc)std::cout<<",\"value\":"<<value.value;
                }
                if(control.type==V4L2_CTRL_TYPE_MENU && control.minimum>=0 && control.maximum-control.minimum<64) {
                    std::cout<<",\"menu\":[";bool first=true;
                    for(int item=control.minimum;item<=control.maximum;++item) {
                        v4l2_querymenu menu{};menu.id=control.id;menu.index=item;
                        if(query(device.fd,VIDIOC_QUERYMENU,&menu))continue;
                        if(!first)std::cout<<",";first=false;
                        std::cout<<"{\"index\":"<<item<<",\"name\":"
                            <<quoted(reinterpret_cast<const char*>(menu.name),sizeof menu.name)<<"}";
                    }
                    std::cout<<"]";
                }
                std::cout<<"}";control.id|=V4L2_CTRL_FLAG_NEXT_CTRL;
            }
            std::cout<<"]}";
        }
        std::cout<<"]}\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
