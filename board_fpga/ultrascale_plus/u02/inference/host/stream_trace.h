#pragma once
#include <sstream>

// Diagnostic-only: capture the same fused-group boundaries in both graphs.
// lightmode must be disabled before inference so extraction cannot recompute.
inline void write_stream_trace(ncnn::Extractor& ex,const char* output) {
    std::ifstream param("model.stream.param");
    std::ofstream trace(std::string(output)+".fixture",std::ios::binary);
    if(!param || !trace)throw std::runtime_error("Cannot open boundary trace files");
    const uint32_t magic=0x55303254,count=110;
    trace.write((const char*)&magic,4);trace.write((const char*)&count,4);
    std::string line;unsigned captured=0;
    while(std::getline(param,line)) {
        std::istringstream row(line);std::string type,name,input,blob;int ni,no;
        if(!(row>>type) || type!="U02Stream")continue;
        if(!(row>>name>>ni>>no>>input>>blob) || ni!=1 || no!=1)
            throw std::runtime_error("Invalid boundary trace descriptor");
        ncnn::Mat value;
        if(ex.extract(blob.c_str(),value) || value.dims!=3 || value.elempack!=1 || value.elemsize!=4)
            throw std::runtime_error("Invalid boundary trace tensor");
        uint32_t header[]={captured,uint32_t(value.c),uint32_t(value.h),uint32_t(value.w)};
        trace.write((const char*)header,sizeof(header));
        for(int c=0;c<value.c;++c)
            trace.write((const char*)(const float*)value.channel(c),value.h*value.w*4);
        ++captured;
    }
    if(captured!=count || !trace)throw std::runtime_error("Incomplete boundary trace");
    std::cerr<<"U02_BOUNDARY_TRACE_COMPLETE groups="<<captured<<'\n';
}
