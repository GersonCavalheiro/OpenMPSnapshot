#pragma once

namespace hpmr {
template <class K, class V>
class HashEntry {
public:
K key;

size_t hash_value;

V value;

bool filled;

HashEntry() : filled(false){};

void fill(const K& key, const size_t hash_value, const V& value) {
this->key = key;
this->hash_value = hash_value;
this->value = value;
filled = true;
}
};

template <class K>
class HashEntry<K, void> {
public:
K key;

size_t hash_value;

bool filled;

HashEntry() : filled(false){};

void fill(const K& key, const size_t hash_value) {
this->key = key;
this->hash_value = hash_value;
filled = true;
}
};

}  
