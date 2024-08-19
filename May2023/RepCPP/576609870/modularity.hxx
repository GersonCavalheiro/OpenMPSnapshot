#pragma once
#include <cmath>
#include <vector>
#include "_main.hxx"

using std::pow;
using std::vector;






inline double modularityCommunity(double cin, double ctot, double M, double R=1) {
ASSERT(cin>=0 && ctot>=0 && M>0 && R>0);
return cin/(2*M) - R*pow(ctot/(2*M), 2);
}



template <class T>
inline double modularityCommunities(const vector<T>& cin, const vector<T>& ctot, double M, double R=1) {
ASSERT(M>0 && R>0);
double a = 0;
for (size_t i=0, I=cin.size(); i<I; ++i)
a += modularityCommunity(cin[i], ctot[i], M, R);
return a;
}


template <class T>
inline double modularityCommunitiesOmp(const vector<T>& cin, const vector<T>& ctot, double M, double R=1) {
ASSERT(M>0 && R>0);
double a = 0;
size_t C = cin.size();
#pragma omp parallel for schedule(auto) reduction(+:a)
for (size_t i=0; i<C; ++i)
a += modularityCommunity(cin[i], ctot[i], M, R);
return a;
}





template <class G, class FC>
inline double modularityBy(const G& x, FC fc, double M, double R=1) {
ASSERT(M>0 && R>0);
size_t S = x.span();
vector<double> cin(S), ctot(S);
x.forEachVertexKey([&](auto u) {
size_t c = fc(u);
x.forEachEdge(u, [&](auto v, auto w) {
size_t d = fc(v);
if (c==d) cin[c] += w;
ctot[c] += w;
});
});
return modularityCommunities(cin, ctot, M, R);
}

template <class G, class FC>
inline double modularityByOmp(const G& x, FC fc, double M, double R=1) {
using K = typename G::key_type;
ASSERT(M>0 && R>0);
size_t S = x.span();
vector<double> cin(S), ctot(S);
#pragma omp parallel for schedule(auto)
for (K u=0; u<S; ++u) {
if (!x.hasVertex(u)) continue;
size_t c = fc(u);
x.forEachEdge(u, [&](auto v, auto w) {
size_t d = fc(v);
if (c==d) {
#pragma omp atomic
cin[c] += w;
}
#pragma omp atomic
ctot[c] += w;
});
}
return modularityCommunities(cin, ctot, M, R);
}



template <class G>
inline double modularity(const G& x, double M, double R=1) {
ASSERT(M>0 && R>0 && R<=1);
auto fc = [](auto u) { return u; };
return modularityBy(x, fc, M, R);
}

template <class G>
inline double modularityOmp(const G& x, double M, double R=1) {
ASSERT(M>0 && R>0 && R<=1);
auto fc = [](auto u) { return u; };
return modularityByOmp(x, fc, M, R);
}






inline double deltaModularity(double vcout, double vdout, double vtot, double ctot, double dtot, double M, double R=1) {
ASSERT(vcout>=0 && vdout>=0 && vtot>=0 && ctot>=0 && dtot>=0 && M>0 && R>0);
return (vcout-vdout)/M - R*vtot*(vtot+ctot-dtot)/(2*M*M);
}
