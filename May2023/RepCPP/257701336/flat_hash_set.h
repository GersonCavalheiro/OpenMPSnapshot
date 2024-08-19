#ifndef ABSL_CONTAINER_FLAT_HASH_SET_H_
#define ABSL_CONTAINER_FLAT_HASH_SET_H_

#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/macros.h"
#include "absl/container/internal/container_memory.h"
#include "absl/container/internal/hash_function_defaults.h"  
#include "absl/container/internal/raw_hash_set.h"  
#include "absl/memory/memory.h"

namespace absl {
namespace container_internal {
template <typename T>
struct FlatHashSetPolicy;
}  

template <class T, class Hash = absl::container_internal::hash_default_hash<T>,
class Eq = absl::container_internal::hash_default_eq<T>,
class Allocator = std::allocator<T>>
class flat_hash_set
: public absl::container_internal::raw_hash_set<
absl::container_internal::FlatHashSetPolicy<T>, Hash, Eq, Allocator> {
using Base = typename flat_hash_set::raw_hash_set;

public:
flat_hash_set() {}
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

using Base::emplace;

using Base::emplace_hint;

using Base::extract;

using Base::merge;

using Base::swap;

using Base::rehash;

using Base::reserve;

using Base::contains;

using Base::count;

using Base::equal_range;

using Base::find;

using Base::bucket_count;

using Base::load_factor;

using Base::max_load_factor;

using Base::get_allocator;

using Base::hash_function;

using Base::key_eq;
};

namespace container_internal {

template <class T>
struct FlatHashSetPolicy {
using slot_type = T;
using key_type = T;
using init_type = T;
using constant_iterators = std::true_type;

template <class Allocator, class... Args>
static void construct(Allocator* alloc, slot_type* slot, Args&&... args) {
absl::allocator_traits<Allocator>::construct(*alloc, slot,
std::forward<Args>(args)...);
}

template <class Allocator>
static void destroy(Allocator* alloc, slot_type* slot) {
absl::allocator_traits<Allocator>::destroy(*alloc, slot);
}

template <class Allocator>
static void transfer(Allocator* alloc, slot_type* new_slot,
slot_type* old_slot) {
construct(alloc, new_slot, std::move(*old_slot));
destroy(alloc, old_slot);
}

static T& element(slot_type* slot) { return *slot; }

template <class F, class... Args>
static decltype(absl::container_internal::DecomposeValue(
std::declval<F>(), std::declval<Args>()...))
apply(F&& f, Args&&... args) {
return absl::container_internal::DecomposeValue(
std::forward<F>(f), std::forward<Args>(args)...);
}

static size_t space_used(const T*) { return 0; }
};
}  

namespace container_algorithm_internal {

template <class Key, class Hash, class KeyEqual, class Allocator>
struct IsUnorderedContainer<absl::flat_hash_set<Key, Hash, KeyEqual, Allocator>>
: std::true_type {};

}  

}  

#endif  
