#pragma once

#include <omp.h>
#include <functional>
#include <numeric>
#include "bare_map.h"

namespace hpmr {
template <class K, class V, class H = std::hash<K>>
class BareConcurrentMap {
public:
BareConcurrentMap();

BareConcurrentMap(const BareConcurrentMap& m);

~BareConcurrentMap();

void reserve(const size_t n_keys_min);

void set_max_load_factor(const float max_load_factor);

float get_max_load_factor() const { return max_load_factor; };

size_t get_n_keys() const;

size_t get_n_buckets() const;

float get_load_factor();

void set(
const K& key,
const size_t hash_value,
const V& value,
const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite);

void async_set(
const K& key,
const size_t hash_value,
const V& value,
const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite);

void sync(const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite);

void unset(const K& key, const size_t hash_value);

V get(const K& key, const size_t hash_value, const V& default_value = V());

bool has(const K& key, const size_t hash_value);

void clear();

void clear_and_shrink();

std::string to_string();

void from_string(const std::string& str);

void for_each(
const std::function<void(const K& key, const size_t hash_value, const V& value)>& handler,
const bool verbose = false);

private:
float max_load_factor;

size_t n_segments;

std::vector<BareMap<K, V, H>> segments;

size_t n_threads;

std::vector<BareMap<K, V, H>> thread_caches;

std::vector<omp_lock_t> segment_locks;

constexpr static size_t N_SEGMENTS_PER_THREAD = 8;

bool has_big_prime_factors(const int num);
};

template <class K, class V, class H>
BareConcurrentMap<K, V, H>::BareConcurrentMap() {
max_load_factor = BareMap<K, V, H>::DEFAULT_MAX_LOAD_FACTOR;
n_threads = omp_get_max_threads();
thread_caches.resize(n_threads);
if (!has_big_prime_factors(n_threads)) {
throw std::invalid_argument("N threads has big prime factors, which affects performance.");
}
n_segments = n_threads * N_SEGMENTS_PER_THREAD;
segments.resize(n_segments);
segment_locks.resize(n_segments);
for (auto& lock : segment_locks) omp_init_lock(&lock);
}

template <class K, class V, class H>
BareConcurrentMap<K, V, H>::BareConcurrentMap(const BareConcurrentMap& m) {
max_load_factor = m.max_load_factor;
n_threads = omp_get_max_threads();
thread_caches.resize(n_threads);
n_segments = m.n_segments;
segments = m.segments;
segment_locks.resize(n_segments);
for (auto& lock : segment_locks) omp_init_lock(&lock);
}

template <class K, class V, class H>
BareConcurrentMap<K, V, H>::~BareConcurrentMap() {
for (auto& lock : segment_locks) omp_destroy_lock(&lock);
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::reserve(const size_t n_keys_min) {
const size_t n_segment_keys_min = n_keys_min / n_segments;
for (size_t i = 0; i < n_segments; i++) segments.at(i).reserve(n_segment_keys_min);
const size_t n_thread_keys_est = n_keys_min / 1000;
for (size_t i = 0; i < n_threads; i++) thread_caches.at(i).reserve(n_thread_keys_est);
};

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::set_max_load_factor(const float max_load_factor) {
this->max_load_factor = max_load_factor;
for (size_t i = 0; i < n_segments; i++) segments.at(i).max_load_factor = max_load_factor;
for (size_t i = 0; i < n_threads; i++) thread_caches.at(i).max_load_factor = max_load_factor;
}

template <class K, class V, class H>
size_t BareConcurrentMap<K, V, H>::get_n_keys() const {
size_t n_keys = 0;
for (size_t i = 0; i < n_segments; i++) n_keys += segments.at(i).get_n_keys();
return n_keys;
}

template <class K, class V, class H>
size_t BareConcurrentMap<K, V, H>::get_n_buckets() const {
size_t n_buckets = 0;
for (size_t i = 0; i < n_segments; i++) n_buckets += segments.at(i).get_n_buckets();
return n_buckets;
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::async_set(
const K& key,
const size_t hash_value,
const V& value,
const std::function<void(V&, const V&)>& reducer) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
if (omp_test_lock(&lock)) {
segments.at(segment_id).set(key, hash_value, value, reducer);
omp_unset_lock(&lock);
} else {
const int thread_id = omp_get_thread_num();
thread_caches.at(thread_id).set(key, hash_value, value, reducer);
}
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::sync(const std::function<void(V&, const V&)>& reducer) {
#pragma omp parallel
{
const int thread_id = omp_get_thread_num();
const auto& handler = [&](const K& key, const size_t hash_value, const V& value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
segments.at(segment_id).set(key, hash_value, value, reducer);
omp_unset_lock(&lock);
};
thread_caches.at(thread_id).for_each(handler);
thread_caches.at(thread_id).clear();
}
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::set(
const K& key,
const size_t hash_value,
const V& value,
const std::function<void(V&, const V&)>& reducer) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
segments.at(segment_id).set(key, hash_value, value, reducer);
omp_unset_lock(&lock);
}

template <class K, class V, class H>
V BareConcurrentMap<K, V, H>::get(const K& key, const size_t hash_value, const V& default_value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
V res = segments.at(segment_id).get(key, hash_value, default_value);
omp_unset_lock(&lock);
return res;
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::unset(const K& key, const size_t hash_value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
segments.at(segment_id).unset(key, hash_value);
omp_unset_lock(&lock);
}

template <class K, class V, class H>
bool BareConcurrentMap<K, V, H>::has(const K& key, const size_t hash_value) {
const size_t segment_id = hash_value % n_segments;
auto& lock = segment_locks[segment_id];
omp_set_lock(&lock);
bool res = segments.at(segment_id).has(key, hash_value);
omp_unset_lock(&lock);
return res;
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::clear() {
for (size_t i = 0; i < n_segments; i++) segments.at(i).clear();
for (size_t i = 0; i < n_threads; i++) thread_caches.at(i).clear();
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::clear_and_shrink() {
for (size_t i = 0; i < n_segments; i++) segments.at(i).clear_and_shrink();
for (size_t i = 0; i < n_threads; i++) thread_caches.at(i).clear_and_shrink();
}

template <class K, class V, class H>
std::string BareConcurrentMap<K, V, H>::to_string() {
std::vector<std::string> ostrs(n_segments);
size_t total_size = 0;
#pragma omp parallel for
for (size_t i = 0; i < n_segments; i++) {
auto& lock = segment_locks[i];
omp_set_lock(&lock);
hps::serialize_to_string(segments.at(i), ostrs.at(i));
omp_unset_lock(&lock);
#pragma omp atomic
total_size += ostrs[i].size();
}
std::string str;
str.reserve(total_size + n_segments * 8);
hps::OutputBuffer<std::string> ob_str(str);
hps::Serializer<float, std::string>::serialize(max_load_factor, ob_str);
for (size_t i = 0; i < n_segments; i++) {
hps::Serializer<std::string, std::string>::serialize(ostrs[i], ob_str);
}
ob_str.flush();
return str;
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::from_string(const std::string& str) {
std::vector<std::string> istrs(n_segments);
hps::InputBuffer<std::string> ib_str(str);
hps::Serializer<float, std::string>::parse(max_load_factor, ib_str);
for (size_t i = 0; i < n_segments; i++) {
hps::Serializer<std::string, std::string>::parse(istrs[i], ib_str);
}
#pragma omp parallel for
for (size_t i = 0; i < n_segments; i++) {
auto& lock = segment_locks[i];
omp_set_lock(&lock);
hps::parse_from_string(segments.at(i), istrs[i]);
omp_unset_lock(&lock);
}
}

template <class K, class V, class H>
void BareConcurrentMap<K, V, H>::for_each(
const std::function<void(const K& key, const size_t hash_value, const V& value)>& handler,
const bool verbose) {
#pragma omp parallel for schedule(static, 1)
for (size_t i = 0; i < n_segments; i++) {
segments.at(i).for_each(handler);
if (verbose && omp_get_thread_num() == 0) {
printf("%zu/%zu ", i / n_threads, N_SEGMENTS_PER_THREAD);
}
}
if (verbose) printf("#\n");
}

template <class K, class V, class H>
bool BareConcurrentMap<K, V, H>::has_big_prime_factors(const int num) {
constexpr int SMALL_PRIMES[] = {2, 3, 5, 7};
constexpr int N_SMALL_PRIMES = sizeof(SMALL_PRIMES) / sizeof(int);
int remain = num;
for (int i = 0; i < N_SMALL_PRIMES; i++) {
const int prime = SMALL_PRIMES[i];
while (remain % prime == 0) remain /= prime;
}
return remain == 1;
}

}  
