#pragma once
#include "../../board/linux/u02_region_api.h"
#include "../../board/linux/u02_region_dual_api.h"
#include "../../board/linux/u02_region_binding_validation.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Candidate transport only; current formal driver does not implement ABI5.
// Construction is an explicit board operation after fresh programming proof.
// Caller serializes all methods; scratch may not be shared with another thread.
class RegionClient {
    int fd_=-1;
    float* scratch_=nullptr;
    uint64_t token_=0;
    bool pending_=false;
    bool result_valid_=false;
    unsigned abi_=U02_REGION_ABI;
    std::vector<U02rRegionTensor> tensors_;
    std::vector<unsigned char> ingress_written_;
    static void require(bool condition,const char* message) {
        if(!condition)throw std::runtime_error(message);
    }
    static std::runtime_error failure(const char* operation) {
        return std::runtime_error(std::string(operation)+": "+std::strerror(errno));
    }
    const U02rRegionTensor& tensor(unsigned id,unsigned flag) const {
        require(!pending_,"Region scratch is still device-owned; recover before use");
        require(token_ && id<tensors_.size(),"Region is not installed or tensor ID is invalid");
        const auto& result=tensors_[id];
        require((result.flags&flag)!=0,"Tensor is not a declared region boundary");
        return result;
    }
public:
    explicit RegionClient(const char* device="/dev/u02_region",unsigned expected_abi=U02_REGION_ABI) {
        require(expected_abi==U02_REGION_ABI || expected_abi==U02_REGION_DUAL_ABI,"Unsupported candidate ABI");
        abi_=expected_abi;
        try {
            fd_=::open(device,O_RDWR|O_CLOEXEC);if(fd_<0)throw failure("open region");
            u02_region_info info={};
            if(::ioctl(fd_,U02_REGION_INFO,&info))throw failure("region INFO");
            require(info.abi==abi_ && info.metadata_version==1 && info.model_version==1 &&
                info.metadata_capacity==U02_REGION_METADATA_CAPACITY && info.model_capacity==U02_REGION_MODEL_CAPACITY &&
                info.scratch_capacity==U02_REGION_SCRATCH_CAPACITY && !info.flags,"Unexpected region device ABI");
            for(uint64_t reserved:info.reserved)require(!reserved,"Nonzero INFO reserved field");
            void* mapping=::mmap(nullptr,info.scratch_capacity,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
            if(mapping==MAP_FAILED)throw failure("map owned region scratch");
            scratch_=static_cast<float*>(mapping);
        }catch(...) {if(fd_>=0)::close(fd_);fd_=-1;throw;}
    }
    RegionClient(const RegionClient&)=delete;
    RegionClient& operator=(const RegionClient&)=delete;
    ~RegionClient() {
        // A timed-out job must remain pinned by the driver after fd/VMA close.
        // Never unload, reset or free device DMA from this destructor.
        if(scratch_)::munmap(scratch_,U02_REGION_SCRATCH_CAPACITY);
        if(fd_>=0)::close(fd_);
    }
    void install(const std::vector<unsigned char>& metadata,const std::vector<unsigned char>& model) {
        require(!pending_,"Cannot replace device-owned region configuration");
        std::unique_ptr<u02_region_validation_workspace> workspace(new u02_region_validation_workspace);
        require(!u02_region_binding_validate(metadata.data(),metadata.size(),model.data(),model.size(),workspace.get()),"Invalid region program/model binding");
        U02rRegionHeader header;std::memcpy(&header,metadata.data(),sizeof(header));
        std::vector<U02rRegionTensor> tensors(header.tensors);
        std::memcpy(tensors.data(),metadata.data()+header.tensor_offset,tensors.size()*sizeof(tensors[0]));
        u02_region_install request={};
        request.metadata_ptr=reinterpret_cast<uintptr_t>(metadata.data());request.model_ptr=reinterpret_cast<uintptr_t>(model.data());
        request.metadata_bytes=metadata.size();request.model_bytes=model.size();
        // Discard the old client token before a possibly ambiguous INSTALL
        // failure (e.g. successful kernel swap followed by copy_to_user fault).
        token_=0;result_valid_=false;tensors_.clear();ingress_written_.clear();
        if(::ioctl(fd_,U02_REGION_INSTALL,&request))throw failure("region INSTALL");
        require(request.token!=0,"Region INSTALL did not return a generation token");
        token_=request.token;tensors_.swap(tensors);ingress_written_.assign(tensors_.size(),0);
    }
    void write_ingress(unsigned id,const float* input,size_t elements) {
        const auto& target=tensor(id,1);
        require(input && elements==target.bytes/4,"Region ingress shape mismatch");
        result_valid_=false;
        std::memcpy(scratch_+target.offset/4,input,target.bytes);
        ingress_written_[id]=1;
    }
    void read_egress(unsigned id,float* output,size_t elements) const {
        require(result_valid_,"No completed region result is available");
        const auto& source=tensor(id,2);
        require(output && elements==source.bytes/4,"Region egress shape mismatch");
        std::memcpy(output,scratch_+source.offset/4,source.bytes);
    }
    u02_region_job run(unsigned timeout_us=1000000) {
        require(!pending_ && token_,"Region cannot submit while uninstalled or device-owned");
        require(timeout_us && timeout_us<=U02_REGION_TIMEOUT_MAX_US,"Invalid region timeout");
        for(size_t i=0;i<tensors_.size();++i)
            if(tensors_[i].flags&1)require(ingress_written_[i]!=0,"Region ingress has not been initialized");
        u02_region_job job={};job.token=token_;job.timeout_us=timeout_us;
        pending_=true;result_valid_=false;
        if(::ioctl(fd_,U02_REGION_RUN,&job))throw failure("region RUN; ownership uncertain, recover before reuse");
        // Successful ioctl guarantees completion and CPU ownership, even if
        // arithmetic status reports a failed region. Do not consume its output.
        pending_=false;require(job.token==token_,"Region RUN returned the wrong token");
        require(job.status==0,"Region hardware returned an error; output is invalid");
        result_valid_=true;
        return job;
    }
    void recover() {
        if(::ioctl(fd_,U02_REGION_RECOVER))throw failure("region RECOVER; preserve pinned buffers");
        pending_=false;result_valid_=false;
    }
    u02_region_dual_job run_legacy(const void* packet,size_t packet_bytes,const void* input,size_t input_bytes,
            unsigned operators,float* output,size_t output_bytes,unsigned timeout_us=1000000) {
        require(abi_==U02_REGION_DUAL_ABI,"Ordinary groups require the dual ABI6 device");
        require(!pending_,"Shared compute engine remains device-owned; recover before reuse");
        require(packet && input && output && packet_bytes<=U02_REGION_DUAL_PACKET_CAPACITY &&
            input_bytes<=U02_REGION_DUAL_INPUT_CAPACITY && output_bytes<=U02_REGION_DUAL_OUTPUT_CAPACITY,
            "Invalid ordinary-group buffers");
        u02_region_dual_job job={};
        job.packet_ptr=reinterpret_cast<uintptr_t>(packet);job.input_ptr=reinterpret_cast<uintptr_t>(input);
        job.packet_bytes=packet_bytes;job.input_bytes=input_bytes;job.output_bytes=output_bytes;
        job.operators=operators;job.timeout_us=timeout_us;
        require(u02_region_dual_job_valid(&job),"Invalid ordinary-group request");
        result_valid_=false;
        std::fill(ingress_written_.begin(),ingress_written_.end(),0);
        pending_=true;
        if(::ioctl(fd_,U02_REGION_DUAL_LEGACY_RUN,&job))
            throw failure("dual ordinary RUN; ownership uncertain, recover before reuse");
        pending_=false;
        require(job.status==0,"Ordinary-group hardware error; output is invalid");
        std::memcpy(output,scratch_,output_bytes);
        return job;
    }
    bool pending() const {return pending_;}
};

class RegionDualClient : public RegionClient {
public:
    explicit RegionDualClient(const char* device="/dev/u02_region_dual")
        :RegionClient(device,U02_REGION_DUAL_ABI) {}
};
