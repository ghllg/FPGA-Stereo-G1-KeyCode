#pragma once
#include <cstddef>
#include <cstdint>

namespace u02_stereo_nearest {
// A nearest neighbour is final only after both temporal neighbours are seen.
// Histories include consumed/dropped frames, so those cannot be rematched.
template<class Sequence,class Timestamp>
bool closed_nearest(const Sequence& history,uint64_t target,Timestamp timestamp,uint64_t& nearest) {
    if(history.size()<2 || target<=timestamp(history[0]) ||
       target>=timestamp(history[history.size()-1]))return false;
    size_t lo=0,hi=history.size();
    while(lo<hi) {
        const size_t mid=lo+(hi-lo)/2;
        if(timestamp(history[mid])<target)lo=mid+1;else hi=mid;
    }
    const uint64_t before=timestamp(history[lo-1]),after=timestamp(history[lo]);
    nearest=(target-before<=after-target)?before:after; // deterministic earlier tie
    return true;
}

template<class History,class Timestamp>
bool mutual(uint64_t left,uint64_t right,const History& left_history,
            const History& right_history,Timestamp timestamp,uint64_t limit_us) {
    const uint64_t delta=left>right?left-right:right-left;
    if(delta>limit_us)return false;
    uint64_t nearest_left=0,nearest_right=0;
    return closed_nearest(right_history,left,timestamp,nearest_right) && nearest_right==right &&
           closed_nearest(left_history,right,timestamp,nearest_left) && nearest_left==left;
}

template<class Candidates,class History,class CandidateTime,class HistoryTime>
bool latest(const Candidates& left,const Candidates& right,
            const History& left_history,const History& right_history,
            CandidateTime candidate_time,HistoryTime history_time,uint64_t limit_us,
            size_t& best_left,size_t& best_right) {
    bool found=false;uint64_t latest_sum=0;
    for(size_t l=0;l<left.size();++l)for(size_t r=0;r<right.size();++r) {
        const uint64_t lt=candidate_time(left[l]),rt=candidate_time(right[r]);
        if(mutual(lt,rt,left_history,right_history,history_time,limit_us) &&
           (!found || lt+rt>latest_sum)) {
            best_left=l;best_right=r;latest_sum=lt+rt;found=true;
        }
    }
    return found;
}
}
