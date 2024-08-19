#pragma once

#include <cassert>
#include <functional>
#include <vector>
#include "bare_hash_container.h"
#include "hash_entry.h"
#include "reducer.h"

namespace hpmr {

template <class K, class H = std::hash<K>>
class BareSet : public BareHashContainer<K, void, H> {
public:
void set(const K& key, const size_t hash_value);

void for_each(const std::function<void(const K& key, const size_t hash_value)>& handler) const;

using BareHashContainer<K, void, H>::max_load_factor;

using BareHashContainer<K, void, H>::reserve_n_buckets;

protected:
using BareHashContainer<K, void, H>::n_keys;

using BareHashContainer<K, void, H>::n_buckets;

using BareHashContainer<K, void, H>::buckets;

using BareHashContainer<K, void, H>::check_balance;
};

template <class K, class H>
void BareSet<K, H>::set(const K& key, const size_t hash_value) {
size_t bucket_id = hash_value % n_buckets;
size_t n_probes = 0;
while (n_probes < n_buckets) {
if (!buckets.at(bucket_id).filled) {
buckets.at(bucket_id).fill(key, hash_value);
n_keys++;
if (n_buckets * max_load_factor <= n_keys) reserve_n_buckets(n_buckets * 2);
break;
} else if (buckets.at(bucket_id).hash_value == hash_value && buckets.at(bucket_id).key == key) {
break;
} else {
n_probes++;
bucket_id = (bucket_id + 1) % n_buckets;
}
}
check_balance(n_probes);
}

template <class K, class H>
void BareSet<K, H>::for_each(
const std::function<void(const K& key, const size_t hash_value)>& handler) const {
if (n_keys == 0) return;
for (size_t i = 0; i < n_buckets; i++) {
if (buckets.at(i).filled) {
handler(buckets.at(i).key, buckets.at(i).hash_value);
}
}
}
}  

namespace hps {
template <class K, class H, class B>
class Serializer<hpmr::BareSet<K, H>, B> {
public:
static void serialize(const hpmr::BareSet<K, H>& set, OutputBuffer<B>& buf) {
set.serialize(buf);
}
static void parse(hpmr::BareSet<K, H>& set, InputBuffer<B>& buf) { set.parse(buf); }
};
}  
