

#include <iostream>
#include <vector>
#include <numeric>
#include "interval.hh"
#include "omp_bf_count.hh"


size_t omp_bf_count( const std::vector<interval> &upd,
const std::vector<interval> &sub,
std::vector<int> &counts )
{
const int n = sub.size();
const int m = upd.size();
counts.resize(n);

std::cout << "omp_bf_count: " << std::flush;

for (int i=0; i<n; i++) {
int c = 0;
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(i, sub, upd) reduction(+:c)
#else
#pragma omp parallel for default(none) shared(i, m, sub, upd) reduction(+:c)
#endif
for (int j=0; j<m; j++) {
c += intersect(sub[i], upd[j]);
}
counts[i] = c;
}

const int n_intersections = std::accumulate(counts.begin(), counts.end(), 0);
return n_intersections;
}
