#pragma once

#include <vector>
#include <utility>

using std::pair;
using std::vector;


template <typename K, typename V>
class Dictionary {
public:
virtual ~Dictionary() {}


virtual int getSize() = 0;


virtual bool isEmpty() = 0;


virtual void insert(K key, V value) = 0;


virtual void update(K key, V value) = 0;


virtual V get(K key) = 0;


virtual bool contains(K key) = 0;


virtual void remove(K key) = 0;


virtual std::vector<K> getKeys() = 0;


virtual std::vector<pair<K,V>> getItems() = 0;

public:
Dictionary() { }
private:
Dictionary(const Dictionary& other) = delete;
Dictionary& operator=(const Dictionary& other) = delete;
};

