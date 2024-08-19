#pragma once

#include <unordered_map>
#include <stdexcept>
#include <vector>

#include "dictionary.h"

using std::runtime_error;
using std::unordered_map;
using std::vector;

template <typename K, typename V>
class STLHashTable : public Dictionary<K,V> {
public:
int getSize();
bool isEmpty();
void insert(K key, V value);
void update(K key, V value);
V get(K key);
bool contains(K key);
void remove(K key);
std::vector<K> getKeys();
std::vector<pair<K,V>> getItems();
private:
unordered_map<K,V> actualDictionary;
};

template <typename K, typename V>
int STLHashTable<K,V>::getSize() {
return this->actualDictionary.size();
}

template <typename K, typename V>
bool STLHashTable<K,V>::isEmpty() {
return this->actualDictionary.empty();
}

template <typename K, typename V>
void STLHashTable<K,V>::insert(K key, V value) {
if (this->contains(key)) {
throw runtime_error("STLHashTable::insert: key already present");
}
this->actualDictionary[key] = value;
}

template <typename K, typename V>
void STLHashTable<K,V>::update(K key, V value) {
if (!this->contains(key)) {
throw runtime_error("STLHashTable::update: key not present");
}
this->actualDictionary[key] = value;
}

template <typename K, typename V>
V STLHashTable<K,V>::get(K key) {
if (!this->contains(key)) {
throw runtime_error("STLHashTable::get: key not present");
}
return this->actualDictionary[key];
}

template <typename K, typename V>
bool STLHashTable<K,V>::contains(K key) {
try {
this->actualDictionary.at(key); 
return true;
} catch (std::out_of_range& e) {
return false;
}
}

template <typename K, typename V>
void STLHashTable<K,V>::remove(K key) {
if (!this->contains(key)) {
throw runtime_error("STLHashTable::remove: key not present");
}
this->actualDictionary.erase(key);
}

template <typename K, typename V>
std::vector<K> STLHashTable<K,V>::getKeys() {
vector<K> vec;
for (auto it = actualDictionary.begin();
it != actualDictionary.end();
it++) {
vec.push_back(it->first);
}
return vec;
}

template <typename K, typename V>
std::vector<pair<K,V>> STLHashTable<K,V>::getItems() {
vector<pair<K,V>> vec;
for (auto it = actualDictionary.begin();
it != actualDictionary.end();
it++) {
vec.push_back(pair<K,V>(it->first, it->second));
}
return vec;
}

