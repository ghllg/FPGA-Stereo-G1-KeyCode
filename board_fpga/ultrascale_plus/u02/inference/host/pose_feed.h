#pragma once
#include <algorithm>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <cstddef>
#include <unistd.h>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

// Nonblocking transport, at most one pending TCP frame. Under backpressure new
// display frames are dropped; inference and its DMA ownership are unaffected.
class PoseFeed {
    int fd_=-1;
    int client_=-1;
    bool tcp_=false;
    std::vector<uint8_t> pending_;
    size_t offset_=0;
    sockaddr_in peer_{};
    uint64_t session_=0;
    uint32_t frame_=0;
    uint8_t probe_bytes_[16]{};
    size_t probe_offset_=0;
    static void put(uint8_t* p,uint64_t value,int bytes) {
        for(int i=0;i<bytes;++i)p[i]=uint8_t(value>>(8*i));
    }
public:
    struct ClockProbe {uint32_t nonce=0;uint64_t pc_ns=0,board_receive_ns=0;};
    bool accept_client() {
        if(!tcp_)return false;
        if(client_<0){client_=accept4(fd_,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);probe_offset_=0;}
        return client_>=0;
    }
    ClockProbe receive_probe() {
        ClockProbe probe;
        if(!accept_client())return probe;
        while(probe_offset_<16) {
            ssize_t got=recv(client_,probe_bytes_+probe_offset_,16-probe_offset_,MSG_DONTWAIT);
            if(got>0){probe_offset_+=size_t(got);continue;}
            if(got<0 && (errno==EAGAIN || errno==EWOULDBLOCK))return probe;
            close(client_);client_=-1;probe_offset_=0;
            if(!pending_.empty()){pending_.clear();offset_=0;++dropped;}
            return probe;
        }
        probe_offset_=0;
        if(std::memcmp(probe_bytes_,"U02C",4)) {close(client_);client_=-1;return probe;}
        for(unsigned i=0;i<4;++i)probe.nonce|=uint32_t(probe_bytes_[4+i])<<(8*i);
        for(unsigned i=0;i<8;++i)probe.pc_ns|=uint64_t(probe_bytes_[8+i])<<(8*i);
        timespec now{};
        if(clock_gettime(CLOCK_MONOTONIC,&now))throw std::runtime_error("Clock probe failed");
        probe.board_receive_ns=uint64_t(now.tv_sec)*1000000000ull+now.tv_nsec;
        return probe;
    }
    uint64_t sent=0,dropped=0;
    uint64_t session_id() const {return session_;}
    uint32_t tcp_rtt_us() const {
        if(!tcp_ || client_<0)return 0;
        tcp_info info{};socklen_t bytes=sizeof(info);
        if(getsockopt(client_,IPPROTO_TCP,TCP_INFO,&info,&bytes) ||
           bytes<offsetof(tcp_info,tcpi_rtt)+sizeof(info.tcpi_rtt))return 0;
        return info.tcpi_rtt;
    }
    PoseFeed(const char* ip,int port,bool tcp=false):tcp_(tcp) {
        if(!ip || port<1024 || port>65535)throw std::runtime_error("Invalid pose display endpoint");
        peer_.sin_family=AF_INET;peer_.sin_port=htons(uint16_t(port));
        if(inet_pton(AF_INET,ip,&peer_.sin_addr)!=1)throw std::runtime_error("Pose display requires an IPv4 address");
        fd_=socket(AF_INET,(tcp?SOCK_STREAM:SOCK_DGRAM)|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if(fd_<0)throw std::runtime_error("Cannot create pose display socket");
        if(tcp){
            int reuse=1;setsockopt(fd_,SOL_SOCKET,SO_REUSEADDR,&reuse,sizeof(reuse));
            sockaddr_in local{};local.sin_family=AF_INET;local.sin_port=htons(uint16_t(port));
            const char* bind_ip=std::getenv("POSE_BIND_IP");
            if(inet_pton(AF_INET,bind_ip?bind_ip:"192.168.50.2",&local.sin_addr)!=1){
                close(fd_);fd_=-1;throw std::runtime_error("POSE_BIND_IP requires an IPv4 address");
            }
            if(bind(fd_,reinterpret_cast<sockaddr*>(&local),sizeof(local)) || listen(fd_,1)){
                close(fd_);fd_=-1;throw std::runtime_error("Cannot listen on board pose display endpoint");
            }
        }
        session_=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())^(uint64_t(getpid())<<32);
    }
    ~PoseFeed(){if(client_>=0)close(client_);if(fd_>=0)close(fd_);}
    PoseFeed(const PoseFeed&)=delete;
    PoseFeed& operator=(const PoseFeed&)=delete;
    bool flush() {
        while(offset_<pending_.size()){
            ssize_t written=::send(client_,pending_.data()+offset_,pending_.size()-offset_,MSG_DONTWAIT|MSG_NOSIGNAL);
            if(written>0){offset_+=size_t(written);continue;}
            if(written<0 && (errno==EAGAIN || errno==EWOULDBLOCK))return false;
            close(client_);client_=-1;pending_.clear();offset_=0;++dropped;return false;
        }
        if(!pending_.empty()){pending_.clear();offset_=0;++sent;}
        return true;
    }
    // Same proven TCP/chunk envelope as the display path, with a JSON-only
    // body. Schema 2 metadata identifies paired COCO17 messages to receivers.
    bool send_metadata(const std::string& metadata) {
        if(!tcp_ || metadata.empty() || metadata.size()>16384)
            throw std::runtime_error("Invalid paired TCP metadata");
        if(client_<0)client_=accept4(fd_,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);
        if(client_<0){++dropped;return false;}
        if(!flush()){++dropped;return false;}
        const size_t total=4+metadata.size();
        std::vector<uint8_t> body(total);put(body.data(),metadata.size(),4);
        std::memcpy(body.data()+4,metadata.data(),metadata.size());
        const size_t chunk=1176,count=(total+chunk-1)/chunk;const uint32_t frame=++frame_;
        for(size_t i=0;i<count;++i) {
            const size_t length=std::min(chunk,total-i*chunk);
            const size_t old=pending_.size();pending_.resize(old+4+24+length);
            uint8_t* packet=pending_.data()+old;put(packet,length+24,4);packet+=4;
            std::memcpy(packet,"U02P",4);put(packet+4,session_,8);put(packet+12,frame,4);
            put(packet+16,total,4);put(packet+20,i,2);put(packet+22,count,2);
            std::memcpy(packet+24,body.data()+i*chunk,length);
        }
        return flush();
    }
    bool send(const std::vector<uint8_t>& jpeg,const std::vector<float>& points,
              int width,int height,const std::string& backend,uint32_t sequence,
              uint64_t camera_timestamp,double forward_ms,double pipeline_ms,uint64_t calls) {
        if(tcp_){
            if(client_<0)client_=accept4(fd_,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);
            if(client_<0){++dropped;return false;}
            if(!flush()){++dropped;return false;}
        }
        if(points.size()!=51 || jpeg.empty() || width<1 || height<1)throw std::runtime_error("Invalid pose display frame");
        for(float p:points)if(!std::isfinite(p))throw std::runtime_error("Nonfinite pose display coordinate");
        if(backend!="cpu" && backend!="pl-stream" && backend!="stream-csim")throw std::runtime_error("Unsupported display backend");
        std::ostringstream json;json<<std::setprecision(9)
            <<"{\"schema\":1,\"whole_network_on_board\":true,\"backend\":\""<<backend
            <<"\",\"source_size\":["<<width<<","<<height<<"],\"sequence\":"<<sequence
            <<",\"camera_timestamp_us\":"<<camera_timestamp<<",\"exposure_timestamp_verified\":false"
            <<",\"forward_ms\":"<<forward_ms<<",\"pipeline_ms\":"<<pipeline_ms
            <<",\"pl_calls\":"<<calls<<",\"keypoints\":[";
        for(int c=0;c<17;++c){if(c)json<<",";json<<"["<<points[3*c]<<","<<points[3*c+1]<<","<<points[3*c+2]<<"]";}
        json<<"]}";const std::string metadata=json.str();
        const size_t total=4+metadata.size()+jpeg.size();
        if(metadata.size()>4096 || total>2*1024*1024)throw std::runtime_error("Pose display frame exceeds bound");
        std::vector<uint8_t> body(total);put(body.data(),metadata.size(),4);
        std::memcpy(body.data()+4,metadata.data(),metadata.size());
        std::memcpy(body.data()+4+metadata.size(),jpeg.data(),jpeg.size());
        const size_t chunk=1176,count=(total+chunk-1)/chunk;const uint32_t frame=++frame_;
        for(size_t i=0;i<count;++i){
            uint8_t packet[1200];std::memcpy(packet,"U02P",4);put(packet+4,session_,8);put(packet+12,frame,4);
            put(packet+16,total,4);put(packet+20,i,2);put(packet+22,count,2);
            const size_t length=std::min(chunk,total-i*chunk);std::memcpy(packet+24,body.data()+i*chunk,length);
            if(tcp_){
                const size_t old=pending_.size();pending_.resize(old+4+length+24);
                put(pending_.data()+old,length+24,4);std::memcpy(pending_.data()+old+4,packet,length+24);
            }else if(sendto(fd_,packet,length+24,MSG_DONTWAIT|MSG_NOSIGNAL,reinterpret_cast<sockaddr*>(&peer_),sizeof(peer_))!=ssize_t(length+24)){
                ++dropped;return false;
            }
        }
        if(tcp_)return flush();
        ++sent;return true;
    }
};
