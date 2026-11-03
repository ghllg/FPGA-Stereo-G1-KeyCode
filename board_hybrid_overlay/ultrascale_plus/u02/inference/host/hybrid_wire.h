#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace u02_hybrid {
static const size_t payload_bytes=65512, max_body_bytes=8*1024*1024;
inline void put(uint8_t* p,uint64_t value,unsigned bytes) {
    for(unsigned i=0;i<bytes;++i)p[i]=uint8_t(value>>(8*i));
}
inline uint32_t crc32(const std::vector<uint8_t>& bytes) {
    struct Table {
        uint32_t values[256];
        Table(){for(unsigned i=0;i<256;++i){uint32_t c=i;for(unsigned j=0;j<8;++j)c=(c>>1)^(0xedb88320u&uint32_t(-int(c&1)));values[i]=c;}}
    };
    static const Table table;
    uint32_t value=0xffffffffu;
    for(uint8_t b:bytes)value=table.values[(value^b)&255]^(value>>8);
    return value^0xffffffffu;
}
inline std::vector<uint8_t> encode(uint64_t session,uint32_t frame,const std::string& metadata,
                                   const std::vector<uint8_t>& left,const std::vector<uint8_t>& right) {
    const size_t total=4+metadata.size()+left.size()+right.size();
    if(metadata.empty() || metadata.size()>16384 || left.empty() || right.empty() ||
       left.size()>4*1024*1024 || right.size()>4*1024*1024 || total>max_body_bytes)
        throw std::runtime_error("Hybrid pair exceeds bounded wire limits");
    std::vector<uint8_t> body(total);put(body.data(),metadata.size(),4);
    std::memcpy(body.data()+4,metadata.data(),metadata.size());
    std::memcpy(body.data()+4+metadata.size(),left.data(),left.size());
    std::memcpy(body.data()+4+metadata.size()+left.size(),right.data(),right.size());
    const size_t count=(total+payload_bytes-1)/payload_bytes;
    std::vector<uint8_t> packets;packets.reserve(total+28*count);
    for(size_t i=0;i<count;++i){
        const size_t length=std::min(payload_bytes,total-i*payload_bytes),old=packets.size();
        packets.resize(old+28+length);uint8_t* p=packets.data()+old;
        put(p,length+24,4);p+=4;std::memcpy(p,"U02H",4);put(p+4,session,8);put(p+12,frame,4);
        put(p+16,total,4);put(p+20,i,2);put(p+22,count,2);std::memcpy(p+24,body.data()+i*payload_bytes,length);
    }
    return packets;
}
}
