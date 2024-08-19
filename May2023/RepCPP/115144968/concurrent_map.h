

#pragma once

#include "bare_concurrent_map.h"
#include "dist_map.h"
#include "reducer.h"

namespace hpmr {

template <class K, class V, class H = std::hash<K>>
class ConcurrentMap {
public:
void reserve(const size_t n_keys_min) { bare_map.reserve(n_keys_min); }

size_t get_n_keys() const { return bare_map.get_n_keys(); }

size_t get_n_buckets() const { return bare_map.get_n_buckets(); };

float get_load_factor() const { return bare_map.get_load_factor(); }

float get_max_load_factor() const { return bare_map.get_max_load_factor(); }

void set_max_load_factor(const float max_load_factor) {
bare_map.set_max_load_factor(max_load_factor);
}

void set(
const K& key,
const V& value,
const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite) {
bare_map.set(key, hasher(key), value, reducer);
}

void async_set(
const K& key,
const V& value,
const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite) {
bare_map.async_set(key, hasher(key), value, reducer);
}

void sync(const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite) {
bare_map.sync(reducer);
}

V get(const K& key, const V& default_value = V()) {
return bare_map.get(key, hasher(key), default_value);
}

void unset(const K& key) { bare_map.unset(key, hasher(key)); }

bool has(const K& key) { return bare_map.has(key, hasher(key)); }

void clear() { bare_map.clear(); }

void clear_and_shrink() { bare_map.clear_and_shrink(); }

void for_each(
const std::function<void(const K& key, const size_t hash_value, const V& value)>& handler,
const bool verbose) {
bare_map.for_each(handler, verbose);
}



private:
H hasher;

BareConcurrentMap<K, V, H> bare_map;
};

}  