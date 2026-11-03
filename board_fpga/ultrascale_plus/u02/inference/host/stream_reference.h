#pragma once
#include "../hls/stream_contract.h"
#include <vector>
#include <stdexcept>
#ifdef U02_STREAM_TRACE
#include <iostream>
#endif

inline void stream_reference(const std::vector<StreamOp>& ops,const std::vector<int8_t>& weights,
                             const std::vector<StreamRequant>& coeff,const int8_t* image,int32_t* output
#ifdef U02_STREAM_TRACE
                             ,unsigned trace_group=0
#endif
                             ) {
    if(ops.empty() || ops.size()>6) throw std::runtime_error("Unsupported streaming chain length");
    std::vector<int8_t> input(image,image+ops[0].ci*ops[0].h*ops[0].w),next;
    for(size_t index=0;index<ops.size();++index) {
        const StreamOp& op=ops[index];
        const int kernel=op.kind?9:op.ci;
        if(!stream_op_valid(op) || size_t(op.weight_offset+op.co*kernel)>weights.size() ||
           input.size()!=size_t(op.ci*op.h*op.w)) throw std::runtime_error("Invalid stream descriptor");
        const bool last=index+1==ops.size();
#ifdef U02_STREAM_TRACE
        uint64_t clipped=0,rounded_zero=0,active=0,max_magnitude=0;
#endif
        if(!last) {
            if(size_t(op.requant_offset+op.co)>coeff.size()) throw std::runtime_error("Missing requant coefficients");
            next.resize(size_t(op.co)*op.oh*op.ow);
        }
        for(int ch=0;ch<op.co;++ch) for(int y=0;y<op.oh;++y) for(int x=0;x<op.ow;++x) {
            int32_t sum=0;
            if(op.kind==0) {
                for(int ci=0;ci<op.ci;++ci)
                    sum+=int32_t(weights[op.weight_offset+ch*op.ci+ci])*input[(ci*op.h+y)*op.w+x];
            } else {
                for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {
                    const int iy=y*op.stride-op.pad+ky,ix=x*op.stride-op.pad+kx;
                    if(iy>=0 && iy<op.h && ix>=0 && ix<op.w)
                        sum+=int32_t(weights[op.weight_offset+ch*9+ky*3+kx])*input[(ch*op.h+iy)*op.w+ix];
                }
            }
            const int pos=(ch*op.oh+y)*op.ow+x;
            if(last) output[pos]=sum;
            else {
                const StreamRequant& q=coeff[op.requant_offset+ch];
                if(q.multiplier<0 || q.multiplier>=8388608 || q.shift<0 || q.shift>40 ||
                   q.bias<-(int64_t(1)<<60) || q.bias>(int64_t(1)<<60))
                    throw std::runtime_error("Invalid requant coefficient");
                next[pos]=stream_requant(sum,q,op.relu!=0);
#ifdef U02_STREAM_TRACE
                const int64_t value=int64_t(sum)*q.multiplier+q.bias;
                const uint64_t magnitude=value<0?uint64_t(-value):uint64_t(value);
                const uint64_t rounded=(magnitude+(q.shift?(uint64_t(1)<<(q.shift-1)):0))>>q.shift;
                if(!op.relu || value>0) {
                    clipped+=rounded>127;
                    active+=value!=0;
                    rounded_zero+=value!=0 && rounded==0;
                    max_magnitude=std::max(max_magnitude,rounded);
                }
#endif
            }
        }
        if(!last) input.swap(next);
#ifdef U02_STREAM_TRACE
        if(!last)std::cerr<<"U02_REQUANT_TRACE group="<<trace_group<<" op="<<index
            <<" elements="<<op.co*op.oh*op.ow<<" clipped="<<clipped<<" active="<<active
            <<" rounded_zero="<<rounded_zero<<" max_magnitude="<<max_magnitude<<'\n';
#endif
    }
}
