#pragma once
#include "region_graph_binding_generated.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct RegionGraphView {float* data;unsigned c,h,w;size_t cstep;};
struct RegionScheduleExtension {
    virtual ~RegionScheduleExtension(){}
    virtual void bind(const char* param,unsigned layers,const std::vector<RegionGraphView>& views)=0;
    virtual void begin_forward()=0;
    virtual bool handle_node(unsigned node)=0;
    virtual void end_forward()=0;
};

// Transport can be the candidate RegionClient or a clearly identified test
// callback. This scheduler does not establish support for the remaining 101
// packet2 groups on the same hardware. Production activation requires that.
template<class Transport> class RegionGraphDispatch : public RegionScheduleExtension {
    Transport& transport;
    std::vector<unsigned char> model;
    std::vector<RegionGraphView> views;
    std::vector<std::vector<float>> staged;
    std::vector<unsigned char> published;
    bool bound=false,active=false,ran=false,failed=false;
    unsigned next=0;
    static size_t boundary_count(){return sizeof(REGION_GRAPH_BOUNDARIES)/sizeof(RegionGraphBoundary);}
    static void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
public:
    unsigned dispatch_callbacks=0,published_boundaries=0;
    RegionGraphDispatch(Transport& backend,const std::vector<unsigned char>& constants):transport(backend),model(constants){}
    void bind(const char* param,unsigned layers,const std::vector<RegionGraphView>& supplied) override {
        require(!active,"Cannot bind during a forward");
        require(param && std::string(param)==REGION_GRAPH_PARAM && layers==REGION_GRAPH_LAYERS &&
            supplied.size()==REGION_GRAPH_BLOBS,"Region requires its exact complete graph");
        for(const auto& b:REGION_GRAPH_BOUNDARIES) {
            const auto& view=supplied[b.blob];
            require(view.data && view.c==b.c && view.h==b.h && view.w==b.w &&
                view.cstep==size_t(b.h)*b.w,"Region boundary view is not dense FP32 CHW");
        }
        std::vector<std::vector<float>> replacement(boundary_count());
        for(size_t i=0;i<boundary_count();++i) {
            const auto& b=REGION_GRAPH_BOUNDARIES[i];
            if(b.flags&2)replacement[i].resize(size_t(b.c)*b.h*b.w);
        }
        const std::vector<unsigned char> metadata(REGION_GRAPH_METADATA,REGION_GRAPH_METADATA+sizeof(REGION_GRAPH_METADATA));
        bound=false;failed=true;
        transport.install(metadata,model);
        views=supplied;staged.swap(replacement);published.assign(boundary_count(),0);bound=true;failed=false;
    }
    void begin_forward() override {
        require(bound && !active && !failed,"Region is unbound, active or failed; do not silently retry");
        active=true;ran=false;next=0;std::fill(published.begin(),published.end(),0);
    }
    bool handle_node(unsigned node) override {
        require(active && !failed && node==next && node<REGION_GRAPH_LAYERS,"Region node order mismatch");
        ++next;
        try {
            if(node==REGION_GRAPH_ENTRY) {
                for(const auto& b:REGION_GRAPH_BOUNDARIES)if(b.flags&1)
                    transport.write_ingress(b.tensor,views[b.blob].data,size_t(b.c)*b.h*b.w);
                transport.run();++dispatch_callbacks;ran=true;
                for(size_t i=0;i<boundary_count();++i) {
                    const auto& b=REGION_GRAPH_BOUNDARIES[i];
                    if(b.flags&2)transport.read_egress(b.tensor,staged[i].data(),staged[i].size());
                }
            }
            for(size_t i=0;i<boundary_count();++i) {
                const auto& b=REGION_GRAPH_BOUNDARIES[i];
                if((b.flags&2) && b.birth==node) {
                    require(ran && !published[i],"Region output is unavailable or already published");
                    std::memcpy(views[b.blob].data,staged[i].data(),staged[i].size()*sizeof(float));
                    published[i]=1;++published_boundaries;
                }
            }
        }catch(...) {failed=true;throw;}
        return std::find(REGION_GRAPH_REPLACED,REGION_GRAPH_REPLACED+
            sizeof(REGION_GRAPH_REPLACED)/sizeof(unsigned),node)!=REGION_GRAPH_REPLACED+
            sizeof(REGION_GRAPH_REPLACED)/sizeof(unsigned);
    }
    void end_forward() override {
        require(active && !failed && ran && next==REGION_GRAPH_LAYERS,"Incomplete region graph forward");
        for(size_t i=0;i<boundary_count();++i)
            if(REGION_GRAPH_BOUNDARIES[i].flags&2)require(published[i],"Unpublished region boundary");
        active=false;
    }
};
