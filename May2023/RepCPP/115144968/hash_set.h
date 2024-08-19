#pragma once

#include "bare_set.h"

namespace hpmr {
template <class K, class H = std::hash<K>>
class HashSet : public BareSet<K, H> {
public:
void set(const K& key) { BareSet<K, H>::set(key, hasher(key)); }

void unset(const K& key) { BareSet<K, H>::unset(key, hasher(key)); }

bool has(const K& key) { return BareSet<K, H>::has(key, hasher(key)); }

private:
H hasher;
};
}  

namespace hps {
template <class K, class H, class B>
class Serializer<hpmr::HashSet<K, H>, B> {
public:
static void serialize(const hpmr::HashSet<K, H>& set, OutputBuffer<B>& buf) {
set.serialize(buf);
}
static void parse(hpmr::HashSet<K, H>& set, InputBuffer<B>& buf) { set.parse(buf); }
};
}  
