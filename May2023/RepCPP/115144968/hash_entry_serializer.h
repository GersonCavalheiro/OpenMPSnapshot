#pragma once

#include "../hps/src/hps.h"
#include "hash_entry.h"

namespace hps {
template <class K, class V, class B>
class Serializer<hpmr::HashEntry<K, V>, B> {
public:
static void serialize(const hpmr::HashEntry<K, V>& entry, OutputBuffer<B>& ob) {
Serializer<bool, B>::serialize(entry.filled, ob);
if (entry.filled) {
Serializer<K, B>::serialize(entry.key, ob);
Serializer<size_t, B>::serialize(entry.hash_value, ob);
Serializer<V, B>::serialize(entry.value, ob);
}
}
static void parse(hpmr::HashEntry<K, V>& entry, InputBuffer<B>& ib) {
Serializer<bool, B>::parse(entry.filled, ib);
if (entry.filled) {
Serializer<K, B>::parse(entry.key, ib);
Serializer<size_t, B>::parse(entry.hash_value, ib);
Serializer<V, B>::parse(entry.value, ib);
}
}
};

template <class K, class B>
class Serializer<hpmr::HashEntry<K, void>, B> {
public:
static void serialize(const hpmr::HashEntry<K, void>& entry, OutputBuffer<B>& ob) {
Serializer<bool, B>::serialize(entry.filled, ob);
if (entry.filled) {
Serializer<K, B>::serialize(entry.key, ob);
Serializer<size_t, B>::serialize(entry.hash_value, ob);
}
}
static void parse(hpmr::HashEntry<K, void>& entry, InputBuffer<B>& ib) {
Serializer<bool, B>::parse(entry.filled, ib);
if (entry.filled) {
Serializer<K, B>::parse(entry.key, ib);
Serializer<size_t, B>::parse(entry.hash_value, ib);
}
}
};
}  
