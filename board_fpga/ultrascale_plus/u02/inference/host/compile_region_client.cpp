#include "region_client.h"
static_assert(sizeof(u02_region_job)==64,"Region client job ABI");
// Compile-only artifact. No constructor/device access is executed by main.
void region_client_compile_probe(RegionClient& client,const std::vector<unsigned char>& metadata,
        const std::vector<unsigned char>& model,const float* input,float* output,size_t elements) {
    client.install(metadata,model);client.write_ingress(0,input,elements);
    client.run();client.read_egress(1,output,elements);
    if(client.pending())client.recover();
}
int main(){return 0;}
