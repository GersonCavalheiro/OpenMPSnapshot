

#pragma once

#include "pybind11.h"
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <iostream>
#include <list>
#include <deque>
#include <valarray>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) 
#endif

#ifdef __has_include
#  if defined(PYBIND11_CPP17) && __has_include(<optional>)
#    include <optional>
#    define PYBIND11_HAS_OPTIONAL 1
#  endif
#  if defined(PYBIND11_CPP14) && (__has_include(<experimental/optional>) && \
!__has_include(<optional>))
#    include <experimental/optional>
#    define PYBIND11_HAS_EXP_OPTIONAL 1
#  endif
#  if defined(PYBIND11_CPP17) && __has_include(<variant>)
#    include <variant>
#    define PYBIND11_HAS_VARIANT 1
#  endif
#elif defined(_MSC_VER) && defined(PYBIND11_CPP17)
#  include <optional>
#  include <variant>
#  define PYBIND11_HAS_OPTIONAL 1
#  define PYBIND11_HAS_VARIANT 1
#endif

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T, typename U>
using forwarded_type = conditional_t<
std::is_lvalue_reference<T>::value, remove_reference_t<U> &, remove_reference_t<U> &&>;

template <typename T, typename U>
forwarded_type<T, U> forward_like(U &&u) {
return std::forward<detail::forwarded_type<T, U>>(std::forward<U>(u));
}

template <typename Type, typename Key> struct set_caster {
using type = Type;
using key_conv = make_caster<Key>;

bool load(handle src, bool convert) {
if (!isinstance<pybind11::set>(src))
return false;
auto s = reinterpret_borrow<pybind11::set>(src);
value.clear();
for (auto entry : s) {
key_conv conv;
if (!conv.load(entry, convert))
return false;
value.insert(cast_op<Key &&>(std::move(conv)));
}
return true;
}

template <typename T>
static handle cast(T &&src, return_value_policy policy, handle parent) {
if (!std::is_lvalue_reference<T>::value)
policy = return_value_policy_override<Key>::policy(policy);
pybind11::set s;
for (auto &&value : src) {
auto value_ = reinterpret_steal<object>(key_conv::cast(forward_like<T>(value), policy, parent));
if (!value_ || !s.add(value_))
return handle();
}
return s.release();
}

PYBIND11_TYPE_CASTER(type, _("Set[") + key_conv::name + _("]"));
};

template <typename Type, typename Key, typename Value> struct map_caster {
using key_conv   = make_caster<Key>;
using value_conv = make_caster<Value>;

bool load(handle src, bool convert) {
if (!isinstance<dict>(src))
return false;
auto d = reinterpret_borrow<dict>(src);
value.clear();
for (auto it : d) {
key_conv kconv;
value_conv vconv;
if (!kconv.load(it.first.ptr(), convert) ||
!vconv.load(it.second.ptr(), convert))
return false;
value.emplace(cast_op<Key &&>(std::move(kconv)), cast_op<Value &&>(std::move(vconv)));
}
return true;
}

template <typename T>
static handle cast(T &&src, return_value_policy policy, handle parent) {
dict d;
return_value_policy policy_key = policy;
return_value_policy policy_value = policy;
if (!std::is_lvalue_reference<T>::value) {
policy_key = return_value_policy_override<Key>::policy(policy_key);
policy_value = return_value_policy_override<Value>::policy(policy_value);
}
for (auto &&kv : src) {
auto key = reinterpret_steal<object>(key_conv::cast(forward_like<T>(kv.first), policy_key, parent));
auto value = reinterpret_steal<object>(value_conv::cast(forward_like<T>(kv.second), policy_value, parent));
if (!key || !value)
return handle();
d[key] = value;
}
return d.release();
}

PYBIND11_TYPE_CASTER(Type, _("Dict[") + key_conv::name + _(", ") + value_conv::name + _("]"));
};

template <typename Type, typename Value> struct list_caster {
using value_conv = make_caster<Value>;

bool load(handle src, bool convert) {
if (!isinstance<sequence>(src) || isinstance<str>(src))
return false;
auto s = reinterpret_borrow<sequence>(src);
value.clear();
reserve_maybe(s, &value);
for (auto it : s) {
value_conv conv;
if (!conv.load(it, convert))
return false;
value.push_back(cast_op<Value &&>(std::move(conv)));
}
return true;
}

private:
template <typename T = Type,
enable_if_t<std::is_same<decltype(std::declval<T>().reserve(0)), void>::value, int> = 0>
void reserve_maybe(sequence s, Type *) { value.reserve(s.size()); }
void reserve_maybe(sequence, void *) { }

public:
template <typename T>
static handle cast(T &&src, return_value_policy policy, handle parent) {
if (!std::is_lvalue_reference<T>::value)
policy = return_value_policy_override<Value>::policy(policy);
list l(src.size());
size_t index = 0;
for (auto &&value : src) {
auto value_ = reinterpret_steal<object>(value_conv::cast(forward_like<T>(value), policy, parent));
if (!value_)
return handle();
PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); 
}
return l.release();
}

PYBIND11_TYPE_CASTER(Type, _("List[") + value_conv::name + _("]"));
};

template <typename Type, typename Alloc> struct type_caster<std::vector<Type, Alloc>>
: list_caster<std::vector<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::deque<Type, Alloc>>
: list_caster<std::deque<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::list<Type, Alloc>>
: list_caster<std::list<Type, Alloc>, Type> { };

template <typename ArrayType, typename Value, bool Resizable, size_t Size = 0> struct array_caster {
using value_conv = make_caster<Value>;

private:
template <bool R = Resizable>
bool require_size(enable_if_t<R, size_t> size) {
if (value.size() != size)
value.resize(size);
return true;
}
template <bool R = Resizable>
bool require_size(enable_if_t<!R, size_t> size) {
return size == Size;
}

public:
bool load(handle src, bool convert) {
if (!isinstance<sequence>(src))
return false;
auto l = reinterpret_borrow<sequence>(src);
if (!require_size(l.size()))
return false;
size_t ctr = 0;
for (auto it : l) {
value_conv conv;
if (!conv.load(it, convert))
return false;
value[ctr++] = cast_op<Value &&>(std::move(conv));
}
return true;
}

template <typename T>
static handle cast(T &&src, return_value_policy policy, handle parent) {
list l(src.size());
size_t index = 0;
for (auto &&value : src) {
auto value_ = reinterpret_steal<object>(value_conv::cast(forward_like<T>(value), policy, parent));
if (!value_)
return handle();
PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); 
}
return l.release();
}

PYBIND11_TYPE_CASTER(ArrayType, _("List[") + value_conv::name + _<Resizable>(_(""), _("[") + _<Size>() + _("]")) + _("]"));
};

template <typename Type, size_t Size> struct type_caster<std::array<Type, Size>>
: array_caster<std::array<Type, Size>, Type, false, Size> { };

template <typename Type> struct type_caster<std::valarray<Type>>
: array_caster<std::valarray<Type>, Type, true> { };

template <typename Key, typename Compare, typename Alloc> struct type_caster<std::set<Key, Compare, Alloc>>
: set_caster<std::set<Key, Compare, Alloc>, Key> { };

template <typename Key, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_set<Key, Hash, Equal, Alloc>>
: set_caster<std::unordered_set<Key, Hash, Equal, Alloc>, Key> { };

template <typename Key, typename Value, typename Compare, typename Alloc> struct type_caster<std::map<Key, Value, Compare, Alloc>>
: map_caster<std::map<Key, Value, Compare, Alloc>, Key, Value> { };

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
: map_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> { };

template<typename T> struct optional_caster {
using value_conv = make_caster<typename T::value_type>;

template <typename T_>
static handle cast(T_ &&src, return_value_policy policy, handle parent) {
if (!src)
return none().inc_ref();
policy = return_value_policy_override<typename T::value_type>::policy(policy);
return value_conv::cast(*std::forward<T_>(src), policy, parent);
}

bool load(handle src, bool convert) {
if (!src) {
return false;
} else if (src.is_none()) {
return true;  
}
value_conv inner_caster;
if (!inner_caster.load(src, convert))
return false;

value.emplace(cast_op<typename T::value_type &&>(std::move(inner_caster)));
return true;
}

PYBIND11_TYPE_CASTER(T, _("Optional[") + value_conv::name + _("]"));
};

#if PYBIND11_HAS_OPTIONAL
template<typename T> struct type_caster<std::optional<T>>
: public optional_caster<std::optional<T>> {};

template<> struct type_caster<std::nullopt_t>
: public void_caster<std::nullopt_t> {};
#endif

#if PYBIND11_HAS_EXP_OPTIONAL
template<typename T> struct type_caster<std::experimental::optional<T>>
: public optional_caster<std::experimental::optional<T>> {};

template<> struct type_caster<std::experimental::nullopt_t>
: public void_caster<std::experimental::nullopt_t> {};
#endif

struct variant_caster_visitor {
return_value_policy policy;
handle parent;

using result_type = handle; 

template <typename T>
result_type operator()(T &&src) const {
return make_caster<T>::cast(std::forward<T>(src), policy, parent);
}
};

template <template<typename...> class Variant>
struct visit_helper {
template <typename... Args>
static auto call(Args &&...args) -> decltype(visit(std::forward<Args>(args)...)) {
return visit(std::forward<Args>(args)...);
}
};

template <typename Variant> struct variant_caster;

template <template<typename...> class V, typename... Ts>
struct variant_caster<V<Ts...>> {
static_assert(sizeof...(Ts) > 0, "Variant must consist of at least one alternative.");

template <typename U, typename... Us>
bool load_alternative(handle src, bool convert, type_list<U, Us...>) {
auto caster = make_caster<U>();
if (caster.load(src, convert)) {
value = cast_op<U>(caster);
return true;
}
return load_alternative(src, convert, type_list<Us...>{});
}

bool load_alternative(handle, bool, type_list<>) { return false; }

bool load(handle src, bool convert) {
if (convert && load_alternative(src, false, type_list<Ts...>{}))
return true;
return load_alternative(src, convert, type_list<Ts...>{});
}

template <typename Variant>
static handle cast(Variant &&src, return_value_policy policy, handle parent) {
return visit_helper<V>::call(variant_caster_visitor{policy, parent},
std::forward<Variant>(src));
}

using Type = V<Ts...>;
PYBIND11_TYPE_CASTER(Type, _("Union[") + detail::concat(make_caster<Ts>::name...) + _("]"));
};

#if PYBIND11_HAS_VARIANT
template <typename... Ts>
struct type_caster<std::variant<Ts...>> : variant_caster<std::variant<Ts...>> { };
#endif

NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const handle &obj) {
os << (std::string) str(obj);
return os;
}

NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
