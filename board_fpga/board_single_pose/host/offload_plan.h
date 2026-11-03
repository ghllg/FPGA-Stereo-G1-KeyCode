#pragma once
#include <set>
#include <string>
#include <stdexcept>

// Explicit, model-specific allowlist. An absent list preserves MAC-threshold mode.
struct OffloadPlan {
    bool enabled=false;
    std::set<std::string> requested, seen;
    explicit OffloadPlan(const char* value=nullptr) {
        if(!value) return;
        enabled=true;
        std::string text(value);
        if(text.empty()) throw std::runtime_error("POSE_PL_LAYERS must not be empty");
        size_t start=0;
        do {
            size_t end=text.find(',',start);
            std::string name=text.substr(start,end==std::string::npos?end:end-start);
            if(name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." )!=std::string::npos)
                throw std::runtime_error("Invalid layer name in POSE_PL_LAYERS");
            if(!requested.insert(name).second) throw std::runtime_error("Duplicate layer in POSE_PL_LAYERS");
            if(end==std::string::npos) break;
            start=end+1;
        } while(true);
    }
    bool select(const std::string& name) {
        if(!enabled) return true;
        if(!requested.count(name)) return false;
        seen.insert(name);
        return true;
    }
    void validate() const {
        if(seen!=requested) throw std::runtime_error("POSE_PL_LAYERS contains names absent from ordinary Convolution layers");
    }
};
