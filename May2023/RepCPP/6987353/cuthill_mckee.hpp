#ifndef AMGCL_REORDER_CUTHILL_MCKEE_HPP
#define AMGCL_REORDER_CUTHILL_MCKEE_HPP





#include <vector>
#include <algorithm>

#include <amgcl/backend/interface.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace reorder {

template <bool reverse = false>
struct cuthill_mckee {
template <class Matrix, class Vector>
static void get(const Matrix &A, Vector &perm) {
const ptrdiff_t n = backend::rows(A);


ptrdiff_t initialNode = 0; 
ptrdiff_t maxDegree   = 0;

std::vector<ptrdiff_t> degree(n);
std::vector<ptrdiff_t> levelSet(n, 0);
std::vector<ptrdiff_t> nextSameDegree(n, -1);

#pragma omp parallel
{
ptrdiff_t maxd = 0;
#pragma omp for
for(ptrdiff_t i = 0; i < n; ++i) {
ptrdiff_t row_width = 0;
for(auto a = backend::row_begin(A, i); a; ++a, ++row_width);
degree[i] = row_width;
maxd = std::max(maxd, degree[i]);
}
#pragma omp critical
{
maxDegree = std::max(maxDegree, maxd);
}
}

std::vector<ptrdiff_t> firstWithDegree(maxDegree + 1, -1);
std::vector<ptrdiff_t> nFirstWithDegree(maxDegree + 1);

perm[0] = initialNode;
ptrdiff_t currentLevelSet = 1;
levelSet[initialNode] = currentLevelSet;
ptrdiff_t maxDegreeInCurrentLevelSet = degree[initialNode];
firstWithDegree[maxDegreeInCurrentLevelSet] = initialNode;

for (ptrdiff_t next = 1; next < n; ) {
ptrdiff_t nMDICLS = 0;
std::fill(nFirstWithDegree.begin(), nFirstWithDegree.end(), -1);
bool empty = true; 

ptrdiff_t firstVal  = reverse ? maxDegreeInCurrentLevelSet : 0;
ptrdiff_t finalVal  = reverse ? -1 : maxDegreeInCurrentLevelSet + 1;
ptrdiff_t increment = reverse ? -1 : 1;

for(ptrdiff_t soughtDegree = firstVal; soughtDegree != finalVal; soughtDegree += increment)
{
ptrdiff_t node = firstWithDegree[soughtDegree];
while (node > 0) {
for(auto a = backend::row_begin(A, node); a; ++a) {
ptrdiff_t c = a.col();
if (levelSet[c] == 0) {
levelSet[c] = currentLevelSet + 1;
perm[next] = c;
++next;
empty = false; 
nextSameDegree[c] = nFirstWithDegree[degree[c]];
nFirstWithDegree[degree[c]] = c;
nMDICLS = std::max(nMDICLS, degree[c]);
}
}
node = nextSameDegree[node];
}
}

++currentLevelSet;
maxDegreeInCurrentLevelSet = nMDICLS;
for(ptrdiff_t i = 0; i <= nMDICLS; ++i)
firstWithDegree[i] = nFirstWithDegree[i];

if (empty) {
bool found = false;
for(ptrdiff_t i = 0; i < n; ++i) {
if (levelSet[i] == 0) {
perm[next] = i;
++next;
levelSet[i] = currentLevelSet;
maxDegreeInCurrentLevelSet = degree[i];
firstWithDegree[maxDegreeInCurrentLevelSet] = i;
found = true;
break;
}
}
precondition(found, "Internal consistency error at skyline_lu");
}
}
}
};

} 
} 

#endif
