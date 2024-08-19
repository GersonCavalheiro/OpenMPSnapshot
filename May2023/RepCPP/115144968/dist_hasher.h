#pragma once

#include <functional>

namespace hpmr {
template <class K, class H>
class DistHasher {
public:
DistHasher() {
int n_procs;
MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
n_procs_u = static_cast<size_t>(n_procs);
}

size_t operator()(const K& key) const { return hasher(key) / n_procs_u; }

private:
H hasher;

size_t n_procs_u;
};
}  
