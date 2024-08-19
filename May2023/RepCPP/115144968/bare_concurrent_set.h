#pragma once

#include <functional>
#include "bare_concurrent_container.h"
#include "bare_set.h"

namespace hpmr {
template <class K, class H = std::hash<K>>
class BareConcurrentSet : public BareConcurrentContainer<K, void, BareSet<K, H>, H> {
public:
void set(const K& key, const size_t hash_value);

void async_set(const K& key, const size_t hash_value);

void sync();

protected:
using BareConcurrentContainer<K, void, BareSet<K, H>, H>::n_segments;

using BareConcurrentContainer<K, void, BareSet<K, H>, H>::segments;

using BareConcurrentContainer<K, void, BareSet<K, H>, H>::segment_locks;

using BareConcurrentContainer<K, void, BareSet<K, H>, H>::thread_caches;
};

template <class K, class H>
void BareConcurrentSet<K, H>::set(const K& key, const size_t hash_value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
segments.at(segment_id).set(key, hash_value);
omp_unset_lock(&lock);
}

template <class K, class H>
void BareConcurrentSet<K, H>::async_set(const K& key, const size_t hash_value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
if (omp_test_lock(&lock)) {
segments.at(segment_id).set(key, hash_value);
omp_unset_lock(&lock);
} else {
const int thread_id = omp_get_thread_num();
thread_caches.at(thread_id).set(key, hash_value);
}
}

template <class K, class H>
void BareConcurrentSet<K, H>::sync() {
#pragma omp parallel
{
const int thread_id = omp_get_thread_num();
const auto& handler = [&](const K& key, const size_t hash_value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
segments.at(segment_id).set(key, hash_value);
omp_unset_lock(&lock);
};
thread_caches.at(thread_id).for_each(handler);
thread_caches.at(thread_id).clear();
}
}
}  