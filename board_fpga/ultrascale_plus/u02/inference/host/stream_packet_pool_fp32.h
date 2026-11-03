#pragma once
#include "stream_fp32_wire.h"
#include "../../board/linux/u02_stream_resident_api.h"
#include <map>
#include <set>

// One sealed model per backend session. Packet identity and static descriptors
// stay fixed after registration; only stream_patch_requant's prefix may change.
// Storage is supplied by the owning backend (or ordinary memory in offline tests).
class StreamPacketPoolFp32 {
public:
    struct Slot {size_t offset,bytes,prefix;int group;};
private:
    struct Entry {Slot slot;const uint64_t* data;std::vector<uint8_t> fixed_header;};
    uint8_t* storage;
    size_t capacity,used=0,payload=0;
    std::map<const std::vector<uint64_t>*,Entry> entries;
    std::set<int> groups;
public:
    StreamPacketPoolFp32(uint8_t* memory,size_t bytes):storage(memory),capacity(bytes) {
        if(!memory || bytes<U02_POSE_ARENA_BYTES)throw std::runtime_error("Resident arena capacity");
    }
    void register_packet(int group,const std::vector<uint64_t>& packet) {
        if(group<0 || group>=110 || groups.count(group) || entries.count(&packet))
            throw std::runtime_error("Resident model packet registered twice; create a fresh backend for a new model");
        if(packet.size()<8 || packet.size()>STREAM_MAX_PACKET_BYTES/8)
            throw std::runtime_error("Resident packet size");
        uint32_t h[8];memcpy(h,packet.data(),sizeof(h));
        const size_t bytes=packet.size()*8;
        if(h[0]!=STREAM_PACKET_MAGIC || h[1]!=STREAM_FP32_PACKET_VERSION || !h[2] || h[2]>STREAM_MAX_OPS ||
           h[5]!=bytes || h[3]<32+h[2]*sizeof(StreamOp) || h[3]%64 || h[4]%64 || h[3]>h[4] || h[4]>bytes)
            throw std::runtime_error("Resident packet header");
        const size_t offset=(used+U02_STREAM_RESIDENT_ALIGNMENT-1)&~size_t(U02_STREAM_RESIDENT_ALIGNMENT-1);
        if(offset+bytes>U02_STREAM_RESIDENT_POOL_BYTES ||
           offset+U02_STREAM_PACKET_BYTES>U02_STREAM_RESIDENT_A_REGION_BYTES ||
           U02_STREAM_RESIDENT_C_OFFSET+U02_STREAM_OUTPUT_BYTES>capacity)
            throw std::runtime_error("Resident packet pool exhausted");
        Entry entry={{offset,bytes,h[4],group},packet.data(),{}};
        entry.fixed_header.assign(reinterpret_cast<const uint8_t*>(packet.data()),
                                  reinterpret_cast<const uint8_t*>(packet.data())+h[3]);
        memcpy(storage+offset,packet.data(),bytes);
        entries.emplace(&packet,std::move(entry));groups.insert(group);
        used=offset+bytes;payload+=bytes;
    }
    Slot update_prefix(const std::vector<uint64_t>& packet) {
        auto found=entries.find(&packet);
        if(found==entries.end())throw std::runtime_error("Resident packet is not registered");
        const auto& entry=found->second;
        if(packet.data()!=entry.data || packet.size()*8!=entry.slot.bytes ||
           memcmp(packet.data(),entry.fixed_header.data(),entry.fixed_header.size()))
            throw std::runtime_error("Resident packet identity or static descriptors changed");
        memcpy(storage+entry.slot.offset,packet.data(),entry.slot.prefix);
        return entry.slot;
    }
    size_t packet_count() const {return entries.size();}
    size_t used_bytes() const {return used;}
    size_t payload_bytes() const {return payload;}
};
