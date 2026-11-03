#pragma once
#include "stream_packet_pool.h"
#include "../../board/linux/u02_stream_pipeline_api.h"
#include <array>

// Registration is single-threaded, before workers start. Runtime mutation must
// hold the device submission lock through both prefix publication and DONE.
// Each layer still owns its source packet and dynamic coefficients privately.
class StreamSharedPacketPool {
    struct Model {
        std::vector<uint64_t> packet;
        size_t fixed_bytes=0, prefix_bytes=0;
    };
    struct Binding {unsigned context;int group;const uint64_t* data;size_t words;};
    StreamPacketPool pool;
    std::map<int,Model> model;
    std::map<const std::vector<uint64_t>*,Binding> bindings;
    std::array<std::set<int>,4> registered;
    unsigned contexts;
    bool sealed=false;
public:
    StreamSharedPacketPool(uint8_t* memory,size_t bytes,unsigned count):pool(memory,bytes),contexts(count) {
        if(count!=2 && count!=4)throw std::runtime_error("Shared pool needs two or four contexts");
    }
    void register_packet(unsigned context,int group,const std::vector<uint64_t>& packet) {
        if(sealed || context>=contexts || group<0 || group>=110 ||
           registered[context].count(group) || bindings.count(&packet))
            throw std::runtime_error("Shared packet registration/identity");
        auto found=model.find(group);
        if(found==model.end()) {
            if(context!=0)throw std::runtime_error("Canonical model must be registered first");
            Model item;item.packet=packet;
            auto inserted=model.emplace(group,std::move(item));
            try {pool.register_packet(group,inserted.first->second.packet);}
            catch(...) {model.erase(inserted.first);throw;}
            uint32_t h[8];memcpy(h,packet.data(),sizeof(h));
            inserted.first->second.fixed_bytes=h[3];inserted.first->second.prefix_bytes=h[4];
            found=inserted.first;
        } else {
            const Model& item=found->second;
            const size_t bytes=item.packet.size()*8;
            if(packet.size()!=item.packet.size() ||
               memcmp(packet.data(),item.packet.data(),item.fixed_bytes) ||
               memcmp(reinterpret_cast<const uint8_t*>(packet.data())+item.prefix_bytes,
                      reinterpret_cast<const uint8_t*>(item.packet.data())+item.prefix_bytes,
                      bytes-item.prefix_bytes))
                throw std::runtime_error("Shared contexts must have identical descriptors and weights");
        }
        bindings.emplace(&packet,Binding{context,group,packet.data(),packet.size()});
        registered[context].insert(group);
    }
    void seal() {
        if(sealed)throw std::runtime_error("Shared model already sealed");
        for(unsigned ctx=0;ctx<contexts;++ctx)
            if(registered[ctx].size()!=110)throw std::runtime_error("Incomplete shared whole-network model");
        sealed=true;
    }
    StreamPacketPool::Slot update_prefix(unsigned context,const std::vector<uint64_t>& packet) {
        if(!sealed)throw std::runtime_error("Unsealed shared model");
        auto found=bindings.find(&packet);
        if(found==bindings.end() || found->second.context!=context ||
           found->second.data!=packet.data() || found->second.words!=packet.size())
            throw std::runtime_error("Shared runtime packet ownership changed");
        Model& item=model.at(found->second.group);
        if(memcmp(packet.data(),item.packet.data(),item.fixed_bytes))
            throw std::runtime_error("Shared static descriptor changed");
        // The canonical descriptor/weight suffix was validated at registration;
        // only the private mutable coefficient prefix can replace it here.
        memcpy(item.packet.data(),packet.data(),item.prefix_bytes);
        return pool.update_prefix(item.packet);
    }
    size_t packet_count() const {return pool.packet_count();}
    size_t used_bytes() const {return pool.used_bytes();}
    size_t payload_bytes() const {return pool.payload_bytes();}
};
