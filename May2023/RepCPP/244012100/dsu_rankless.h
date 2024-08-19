#ifndef __DSU_RANKLESS_H
#define __DSU_RANKLESS_H

#include <atomic>
#include <omp.h>
#include <stdexcept>

#include "defs.h"
#include "parallel_array.h"


struct DSU {
const u32 NUM_THREADS;

ParallelArray<atomic_u32> parent;

DSU(u32 size, u32 NUM_THREADS = omp_get_max_threads()) : NUM_THREADS(NUM_THREADS), parent(size) {
if (size == 0) {
throw std::invalid_argument("DSU size cannot be zero");
}

#pragma omp parallel for shared(parent) num_threads(NUM_THREADS)
for (u32 i = 0; i < size; ++i) parent[i] = i;
}

u32 size() const {
return parent.size();
}

void check_out_of_range(u32 id) const {
if (id >= size()) {
throw std::out_of_range("Node id out of range");
}
}

u32 get_parent(u32 id) const {
return parent[id];
}


u32 find_root(u32 id) {
check_out_of_range(id);

while (id != parent[id]) {
u32 value = parent[id];
u32 grandparent = parent[value];


if (value != grandparent) {
parent[id].compare_exchange_strong(value, grandparent);
}

id = grandparent;
}

return id;
}


bool same_set(u32 id1, u32 id2) {
check_out_of_range(id1);
check_out_of_range(id2);

while (true) {
id1 = find_root(id1);
id2 = find_root(id2);

if (id1 == id2) {
return true;
} else if (parent[id1] == id1) {
return false;
}
}
}


void unite(u32 id1, u32 id2) {
check_out_of_range(id1);
check_out_of_range(id2);

while (true) {
id1 = find_root(id1);
id2 = find_root(id2);


if (id1 == id2) return;

if (id1 > id2) {
std::swap(id1, id2);
}


if (!parent[id2].compare_exchange_strong(id2, id1)) {
continue;
}

break;
}
}
};

#endif
