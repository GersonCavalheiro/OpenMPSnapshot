#pragma once
#include <cmath>
#include <vector>

using std::pow;
using std::vector;






template <class T>
inline T modularityCommunity(T cin, T ctot, T M, T R=T(1)) {
return cin/(2*M) - R*pow(ctot/(2*M), 2);
}



template <class T>
T modularityCommunities(const vector<T>& cin, const vector<T>& ctot, T M, T R=T(1)) {
T a = T();
for (size_t i=0, I=cin.size(); i<I; i++)
a += modularityCommunity(cin[i], ctot[i], M, R);
return a;
}



template <class G, class FC, class T>
auto modularity(const G& x, FC fc, T M, T R=T(1)) {
size_t S = x.span();
vector<T> cin(S), ctot(S);
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


template <class G, class T>
inline auto modularity(const G& x, T M, T R=T(1)) {
auto fc = [](auto u) { return u; };
return modularity(x, fc, M, R);
}






template <class T>
inline T deltaModularity(T vcout, T vdout, T vtot, T ctot, T dtot, T M, T R=T(1)) {
return (vcout-vdout)/M - R*vtot*(vtot+ctot-dtot)/(2*M*M);
}
