
#ifndef ABSL_CONTAINER_FLAT_HASH_MAP_H_
#define ABSL_CONTAINER_FLAT_HASH_MAP_H_

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/internal/container_memory.h"
#include "absl/container/internal/hash_function_defaults.h"  
#include "absl/container/internal/raw_hash_map.h"  
#include "absl/memory/memory.h"

namespace absl {
namespace container_internal {
template <class K, class V>
struct FlatHashMapPolicy;
}  

template <class K, class V,
class Hash = absl::container_internal::hash_default_hash<K>,
class Eq = absl::container_internal::hash_default_eq<K>,
class Allocator = std::allocator<std::pair<const K, V>>>
class flat_hash_map : public absl::container_internal::raw_hash_map<
absl::container_internal::FlatHashMapPolicy<K, V>,
Hash, Eq, Allocator> {
using Base = typename flat_hash_map::raw_hash_map;

public:
flat_hash_map() {}
using Base::Base;

using Base::begin;

using Base::cbegin;

using Base::cend;

using Base::end;

using Base::capacity;

using Base::empty;

using Base::max_size;

using Base::size;

using Base::clear;

using Base::erase;

using Base::insert;

using Base::insert_or_assign;

using Base::emplace;

using Base::emplace_hint;

using Base::try_emplace;

using Base::extract;

using Base::merge;

using Base::swap;

using Base::rehash;

using Base::reserve;

using Base::at;

using Base::contains;

using Base::count;

using Base::equal_range;

using Base::find;

using Base::operator[];

using Base::bucket_count;

using Base::load_factor;

using Base::max_load_factor;

using Base::get_allocator;

using Base::hash_function;

using Base::key_eq;
};

namespace container_internal {

template <class K, class V>
struct FlatHashMapPolicy {
using slot_type = container_internal::slot_type<K, V>;
using key_type = K;
using mapped_type = V;
using init_type = std::pair< key_type, mapped_type>;

template <class Allocator, class... Args>
static void construct(Allocator* alloc, slot_type* slot, Args&&... args) {
slot_type::construct(alloc, slot, std::forward<Args>(args)...);
}

template <class Allocator>
static void destroy(Allocator* alloc, slot_type* slot) {
slot_type::destroy(alloc, slot);
}

template <class Allocator>
static void transfer(Allocator* alloc, slot_type* new_slot,
slot_type* old_slot) {
slot_type::transfer(alloc, new_slot, old_slot);
}

template <class F, class... Args>
static decltype(absl::container_internal::DecomposePair(
std::declval<F>(), std::declval<Args>()...))
apply(F&& f, Args&&... args) {
return absl::container_internal::DecomposePair(std::forward<F>(f),
std::forward<Args>(args)...);
}

static size_t space_used(const slot_type*) { return 0; }

static std::pair<const K, V>& element(slot_type* slot) { return slot->value; }

static V& value(std::pair<const K, V>* kv) { return kv->second; }
static const V& value(const std::pair<const K, V>* kv) { return kv->second; }
};

}  

namespace container_algorithm_internal {

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
struct IsUnorderedContainer<
absl::flat_hash_map<Key, T, Hash, KeyEqual, Allocator>> : std::true_type {};

}  

}  

#endif  
