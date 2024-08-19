#pragma once

#include "bare_concurrent_set.h"

namespace hpmr {

template <class K, class H = std::hash<K>>
class ConcurrentSet : public BareConcurrentSet<K, H> {
public:
void set(const K& key) { BareConcurrentSet<K, H>::set(key, hasher(key)); }

void async_set(const K& key) { BareConcurrentSet<K, H>::async_set(key, hasher(key)); }

void unset(const K& key) { BareConcurrentSet<K, H>::unset(key, hasher(key)); }

bool has(const K& key) { return BareConcurrentSet<K, H>::has(key, hasher(key)); }

private:
H hasher;
};

}  