#pragma once
#include "../hls/stream_region_model_packet.h"
#include <vector>
#include <cstring>
#include <stdexcept>

// Bind fused constants to the actual PlStream load_model results. Dynamic
// first-boundary coefficients are intentionally regenerated from FP32 ingress;
// every weight and every remaining coefficient must match the loaded model.
class RegionModelBinding {
    std::vector<unsigned char> model_;
    std::vector<RegionModelGroup> groups_;
    std::vector<unsigned char> checked_;
    bool failed_=false,sealed_=false;
    size_t find(unsigned id) const {
        for(size_t i=0;i<groups_.size();++i)if(groups_[i].id==id)return i;
        return groups_.size();
    }
    void require(bool value,const char* message) {
        if(!value){failed_=true;throw std::runtime_error(message);}
    }
public:
    explicit RegionModelBinding(const std::vector<unsigned char>& model={}):model_(model) {
        if(model_.empty())return;
        require(!region_model_validate(model_.data(),model_.size()),"Invalid fused model bundle");
        RegionModelHeader h;memcpy(&h,model_.data(),64);groups_.resize(h.groups);checked_.assign(h.groups,0);
        for(size_t i=0;i<groups_.size();++i)memcpy(&groups_[i],model_.data()+64+i*64,64);
    }
    void packet(unsigned id,const std::vector<uint64_t>& packet) {
        const size_t i=find(id);if(i==groups_.size())return;
        require(!failed_ && !sealed_,"Fused model binding is failed or sealed");
        const auto& g=groups_[i];const unsigned char* expected=model_.data()+g.packet_offset;
        const unsigned char* actual=reinterpret_cast<const unsigned char*>(packet.data());
        require(packet.size()*8==g.packet_bytes,"Fused packet size differs from loaded model");
        uint32_t h[8];memcpy(h,expected,32);
        require(!memcmp(actual,expected,h[3]),"Fused packet header/operators differ from loaded model");
        require(!memcmp(actual+h[4],expected+h[4],g.packet_bytes-h[4]),"Fused weights differ from loaded model");
        for(unsigned index=1;index<h[2];++index) {
            StreamOp op;memcpy(&op,expected+32+index*48,48);
            require(!memcmp(actual+h[3]+op.requant_offset*16,expected+h[3]+op.requant_offset*16,op.co*16),
                    "Fused internal/final coefficients differ from loaded model");
        }
        checked_[i]|=1;
    }
    void boundary(unsigned id,const std::vector<float>& weight,const std::vector<float>& bias,
                  const std::vector<float>& outgoing) {
        const size_t i=find(id);if(i==groups_.size())return;
        require(!failed_ && !sealed_,"Fused model binding is failed or sealed");
        const auto& g=groups_[i];
        require(weight.size()==g.channels && bias.size()==g.channels &&
                (outgoing.size()==1 || outgoing.size()==g.channels),"Fused boundary constant shape mismatch");
        require(!memcmp(weight.data(),model_.data()+g.weight_scale_offset,g.channels*4) &&
                !memcmp(bias.data(),model_.data()+g.bias_offset,g.channels*4),"Fused boundary weights/bias differ from loaded model");
        for(unsigned ch=0;ch<g.channels;++ch)
            require(!memcmp(&outgoing[outgoing.size()==1?0:ch],model_.data()+g.outgoing_offset+ch*4,4),
                    "Fused outgoing scales differ from loaded model");
        checked_[i]|=2;
    }
    const std::vector<unsigned char>& seal() {
        require(!failed_ && !groups_.empty(),"Fused model is missing or failed");
        for(auto state:checked_)require(state==3,"Fused constants were not checked against all loaded groups");
        sealed_=true;return model_;
    }
};
