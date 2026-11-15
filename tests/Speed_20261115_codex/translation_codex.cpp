#include <cmath>
#include <cstdint>
// Same exhaustive centres, earliest tie, mean refinement and <= radius rule as
// decision_signals.translation_consensus; O(N) auxiliary memory, no NxN matrix.
extern "C" int translation_consensus_codex(const double* d,int n,double radius,std::uint8_t* mask,double* centre) {
    if(n==0){centre[0]=centre[1]=0.;return 0;}
    int best=-1,winner=0;
    for(int i=0;i<n;++i){
        int count=0;
        for(int j=0;j<n;++j)
            if(std::hypot(d[2*j]-d[2*i],d[2*j+1]-d[2*i+1])<=radius)++count;
        if(count>best){best=count;winner=i;}
    }
    double sx=0.,sy=0.;int count=0;
    for(int j=0;j<n;++j)
        if(std::hypot(d[2*j]-d[2*winner],d[2*j+1]-d[2*winner+1])<=radius){sx+=d[2*j];sy+=d[2*j+1];++count;}
    centre[0]=sx/count;centre[1]=sy/count;
    int refined=0;
    for(int j=0;j<n;++j){
        mask[j]=std::hypot(d[2*j]-centre[0],d[2*j+1]-centre[1])<=radius;
        refined+=mask[j];
    }
    return refined;
}
