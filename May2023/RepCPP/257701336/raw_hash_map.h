
#ifndef ABSL_CONTAINER_INTERNAL_RAW_HASH_MAP_H_
#define ABSL_CONTAINER_INTERNAL_RAW_HASH_MAP_H_

#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/container/internal/container_memory.h"
#include "absl/container/internal/raw_hash_set.h"  

namespace absl {
namespace container_internal {

template <class Policy, class Hash, class Eq, class Alloc>
class raw_hash_map : public raw_hash_set<Policy, Hash, Eq, Alloc> {
template <class P>
using MappedReference = decltype(P::value(
std::addressof(std::declval<typename raw_hash_map::reference>())));

template <class P>
using MappedConstReference = decltype(P::value(
std::addressof(std::declval<typename raw_hash_map::const_reference>())));

using KeyArgImpl = container_internal::KeyArg<IsTransparent<Eq>::value &&
IsTransparent<Hash>::value>;

public:
using key_type = typename Policy::key_type;
using mapped_type = typename Policy::mapped_type;
template <class K>
using key_arg = typename KeyArgImpl::template type<K, key_type>;

static_assert(!std::is_reference<key_type>::value, "");
static_assert(!std::is_reference<mapped_type>::value, "");

using iterator = typename raw_hash_map::raw_hash_set::iterator;
using const_iterator = typename raw_hash_map::raw_hash_set::const_iterator;

raw_hash_map() {}
using raw_hash_map::raw_hash_set::raw_hash_set;

template <class K = key_type, class V = mapped_type, K* = nullptr,
V* = nullptr>
std::pair<iterator, bool> insert_or_assign(key_arg<K>&& k, V&& v) {
return insert_or_assign_impl(std::forward<K>(k), std::forward<V>(v));
}

template <class K = key_type, class V = mapped_type, K* = nullptr>
std::pair<iterator, bool> insert_or_assign(key_arg<K>&& k, const V& v) {
return insert_or_assign_impl(std::forward<K>(k), v);
}

template <class K = key_type, class V = mapped_type, V* = nullptr>
std::pair<iterator, bool> insert_or_assign(const key_arg<K>& k, V&& v) {
return insert_or_assign_impl(k, std::forward<V>(v));
}

template <class K = key_type, class V = mapped_type>
std::pair<iterator, bool> insert_or_assign(const key_arg<K>& k, const V& v) {
return insert_or_assign_impl(k, v);
}

template <class K = key_type, class V = mapped_type, K* = nullptr,
V* = nullptr>
iterator insert_or_assign(const_iterator, key_arg<K>&& k, V&& v) {
return insert_or_assign(std::forward<K>(k), std::forward<V>(v)).first;
}

template <class K = key_type, class V = mapped_type, K* = nullptr>
iterator insert_or_assign(const_iterator, key_arg<K>&& k, const V& v) {
return insert_or_assign(std::forward<K>(k), v).first;
}

template <class K = key_type, class V = mapped_type, V* = nullptr>
iterator insert_or_assign(const_iterator, const key_arg<K>& k, V&& v) {
return insert_or_assign(k, std::forward<V>(v)).first;
}

template <class K = key_type, class V = mapped_type>
iterator insert_or_assign(const_iterator, const key_arg<K>& k, const V& v) {
return insert_or_assign(k, v).first;
}

template <class K = key_type, class... Args,
typename std::enable_if<
!std::is_convertible<K, const_iterator>::value, int>::type = 0,
K* = nullptr>
std::pair<iterator, bool> try_emplace(key_arg<K>&& k, Args&&... args) {
return try_emplace_impl(std::forward<K>(k), std::forward<Args>(args)...);
}

template <class K = key_type, class... Args,
typename std::enable_if<
!std::is_convertible<K, const_iterator>::value, int>::type = 0>
std::pair<iterator, bool> try_emplace(const key_arg<K>& k, Args&&... args) {
return try_emplace_impl(k, std::forward<Args>(args)...);
}

template <class K = key_type, class... Args, K* = nullptr>
iterator try_emplace(const_iterator, key_arg<K>&& k, Args&&... args) {
return try_emplace(std::forward<K>(k), std::forward<Args>(args)...).first;
}

template <class K = key_type, class... Args>
iterator try_emplace(const_iterator, const key_arg<K>& k, Args&&... args) {
return try_emplace(k, std::forward<Args>(args)...).first;
}

template <class K = key_type, class P = Policy>
MappedReference<P> at(const key_arg<K>& key) {
auto it = this->find(key);
if (it == this->end()) std::abort();
return Policy::value(&*it);
}

template <class K = key_type, class P = Policy>
MappedConstReference<P> at(const key_arg<K>& key) const {
auto it = this->find(key);
if (it == this->end()) std::abort();
return Policy::value(&*it);
}

template <class K = key_type, class P = Policy, K* = nullptr>
MappedReference<P> operator[](key_arg<K>&& key) {
return Policy::value(&*try_emplace(std::forward<K>(key)).first);
}

template <class K = key_type, class P = Policy>
MappedReference<P> operator[](const key_arg<K>& key) {
return Policy::value(&*try_emplace(key).first);
}

private:
template <class K, class V>
std::pair<iterator, bool> insert_or_assign_impl(K&& k, V&& v) {
auto res = this->find_or_prepare_insert(k);
if (res.second)
this->emplace_at(res.first, std::forward<K>(k), std::forward<V>(v));
else
Policy::value(&*this->iterator_at(res.first)) = std::forward<V>(v);
return {this->iterator_at(res.first), res.second};
}

template <class K = key_type, class... Args>
std::pair<iterator, bool> try_emplace_impl(K&& k, Args&&... args) {
auto res = this->find_or_prepare_insert(k);
if (res.second)
this->emplace_at(res.first, std::piecewise_construct,
std::forward_as_tuple(std::forward<K>(k)),
std::forward_as_tuple(std::forward<Args>(args)...));
return {this->iterator_at(res.first), res.second};
}
};

}  
}  

#endif  
