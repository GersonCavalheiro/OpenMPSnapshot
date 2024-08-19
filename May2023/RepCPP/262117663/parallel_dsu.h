#ifndef __DSU_H
#define __DSU_H

#include <atomic>
#include <omp.h>
#include <stdexcept>

#include "defs.h"
#include "parallel_array.h"


struct ParallelDSU {
const u32 NUM_THREADS;

u32 dsu_size;
atomic_u64* data;

const u32 BINARY_BUCKET_SIZE = 32;
const u64 RANK_MASK = 0xFFFFFFFF00000000ULL;

ParallelDSU(u32 size, u32 NUM_THREADS = omp_get_max_threads()) : NUM_THREADS(NUM_THREADS), dsu_size(size) {
if (size == 0) {
throw std::invalid_argument("DSU size cannot be zero");
}

data = static_cast<atomic_u64*>(operator new[] (size * sizeof(atomic_u64)));;

#pragma omp parallel for shared(data) num_threads(NUM_THREADS)
for (u32 i = 0; i < size; ++i) data[i] = i;
}

u32 size() const {
return dsu_size;
}

void check_out_of_range(u32 id) const {
if (id >= size()) {
throw std::out_of_range("Node id out of range");
}
}

u64 encode_node(u32 parent, u32 rank) {
return (static_cast<u64>(rank) << BINARY_BUCKET_SIZE) | parent;
}

u32 get_parent(u32 id) const {
return static_cast<u32>(data[id]);
}

u32 get_rank(u32 id) const {
return static_cast<u32>(data[id] >> BINARY_BUCKET_SIZE);
}


u32 find_root(u32 id) {
check_out_of_range(id);

while (id != get_parent(id)) {
u64 value = data[id];
u32 grandparent = get_parent(static_cast<u32>(value));
u64 new_value = (value & RANK_MASK) | grandparent;


if (value != new_value) {
data[id].compare_exchange_strong(value, new_value);
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
} else if (get_parent(id1) == id1) {
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

u32 rank1 = get_rank(id1);
u32 rank2 = get_rank(id2);


if (rank1 < rank2 || (rank1 == rank2 && id1 > id2)) {
std::swap(rank1, rank2);
std::swap(id1, id2);
}

u64 old_value = encode_node(id2, rank2);
u64 new_value = encode_node(id1, rank2);


if (!data[id2].compare_exchange_strong(old_value, new_value)) {
continue;
}


if (rank1 == rank2) {
old_value = encode_node(id1, rank1);
new_value = encode_node(id1, rank1 + 1);

data[id1].compare_exchange_strong(old_value, new_value);
}

break;
}
}
};

#endif
