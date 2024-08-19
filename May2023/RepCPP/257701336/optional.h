#ifndef ABSL_TYPES_OPTIONAL_H_
#define ABSL_TYPES_OPTIONAL_H_

#include "absl/base/config.h"
#include "absl/utility/utility.h"

#ifdef ABSL_HAVE_STD_OPTIONAL

#include <optional>

namespace absl {
using std::bad_optional_access;
using std::optional;
using std::make_optional;
using std::nullopt_t;
using std::nullopt;
}  

#else  

#include <cassert>
#include <functional>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/types/bad_optional_access.h"

#if defined(__clang__)
#if __has_feature(cxx_inheriting_constructors)
#define ABSL_OPTIONAL_USE_INHERITING_CONSTRUCTORS 1
#endif
#elif (defined(__GNUC__) &&                                       \
(__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 8)) || \
(__cpp_inheriting_constructors >= 200802) ||                  \
(defined(_MSC_VER) && _MSC_VER >= 1910)
#define ABSL_OPTIONAL_USE_INHERITING_CONSTRUCTORS 1
#endif

namespace absl {

template <typename T>
class optional;

struct nullopt_t {
struct init_t {};
static init_t init;

explicit constexpr nullopt_t(init_t& ) {}
};

extern const nullopt_t nullopt;

namespace optional_internal {

struct empty_struct {};
template <typename T, bool unused = std::is_trivially_destructible<T>::value>
class optional_data_dtor_base {
struct dummy_type {
static_assert(sizeof(T) % sizeof(empty_struct) == 0, "");
empty_struct data[sizeof(T) / sizeof(empty_struct)];
};

protected:
bool engaged_;
union {
dummy_type dummy_;
T data_;
};

void destruct() noexcept {
if (engaged_) {
data_.~T();
engaged_ = false;
}
}

constexpr optional_data_dtor_base() noexcept : engaged_(false), dummy_{{}} {}

template <typename... Args>
constexpr explicit optional_data_dtor_base(in_place_t, Args&&... args)
: engaged_(true), data_(absl::forward<Args>(args)...) {}

~optional_data_dtor_base() { destruct(); }
};

template <typename T>
class optional_data_dtor_base<T, true> {
struct dummy_type {
static_assert(sizeof(T) % sizeof(empty_struct) == 0, "");
empty_struct data[sizeof(T) / sizeof(empty_struct)];
};

protected:
bool engaged_;
union {
dummy_type dummy_;
T data_;
};
void destruct() noexcept { engaged_ = false; }

constexpr optional_data_dtor_base() noexcept : engaged_(false), dummy_{{}} {}

template <typename... Args>
constexpr explicit optional_data_dtor_base(in_place_t, Args&&... args)
: engaged_(true), data_(absl::forward<Args>(args)...) {}
};

template <typename T>
class optional_data_base : public optional_data_dtor_base<T> {
protected:
using base = optional_data_dtor_base<T>;
#if ABSL_OPTIONAL_USE_INHERITING_CONSTRUCTORS
using base::base;
#else
optional_data_base() = default;

template <typename... Args>
constexpr explicit optional_data_base(in_place_t t, Args&&... args)
: base(t, absl::forward<Args>(args)...) {}
#endif

template <typename... Args>
void construct(Args&&... args) {
::new (static_cast<void*>(&this->dummy_)) T(std::forward<Args>(args)...);
this->engaged_ = true;
}

template <typename U>
void assign(U&& u) {
if (this->engaged_) {
this->data_ = std::forward<U>(u);
} else {
construct(std::forward<U>(u));
}
}
};

template <typename T,
bool unused = absl::is_trivially_copy_constructible<T>::value&&
absl::is_trivially_copy_assignable<typename std::remove_cv<
T>::type>::value&& std::is_trivially_destructible<T>::value>
class optional_data;

template <typename T>
class optional_data<T, true> : public optional_data_base<T> {
protected:
#if ABSL_OPTIONAL_USE_INHERITING_CONSTRUCTORS
using optional_data_base<T>::optional_data_base;
#else
optional_data() = default;

template <typename... Args>
constexpr explicit optional_data(in_place_t t, Args&&... args)
: optional_data_base<T>(t, absl::forward<Args>(args)...) {}
#endif
};

template <typename T>
class optional_data<T, false> : public optional_data_base<T> {
protected:
#if ABSL_OPTIONAL_USE_INHERITING_CONSTRUCTORS
using optional_data_base<T>::optional_data_base;
#else
template <typename... Args>
constexpr explicit optional_data(in_place_t t, Args&&... args)
: optional_data_base<T>(t, absl::forward<Args>(args)...) {}
#endif

optional_data() = default;

optional_data(const optional_data& rhs) {
if (rhs.engaged_) {
this->construct(rhs.data_);
}
}

optional_data(optional_data&& rhs) noexcept(
absl::default_allocator_is_nothrow::value ||
std::is_nothrow_move_constructible<T>::value) {
if (rhs.engaged_) {
this->construct(std::move(rhs.data_));
}
}

optional_data& operator=(const optional_data& rhs) {
if (rhs.engaged_) {
this->assign(rhs.data_);
} else {
this->destruct();
}
return *this;
}

optional_data& operator=(optional_data&& rhs) noexcept(
std::is_nothrow_move_assignable<T>::value&&
std::is_nothrow_move_constructible<T>::value) {
if (rhs.engaged_) {
this->assign(std::move(rhs.data_));
} else {
this->destruct();
}
return *this;
}
};

enum class copy_traits { copyable = 0, movable = 1, non_movable = 2 };

template <copy_traits>
class optional_ctor_base;

template <>
class optional_ctor_base<copy_traits::copyable> {
public:
constexpr optional_ctor_base() = default;
optional_ctor_base(const optional_ctor_base&) = default;
optional_ctor_base(optional_ctor_base&&) = default;
optional_ctor_base& operator=(const optional_ctor_base&) = default;
optional_ctor_base& operator=(optional_ctor_base&&) = default;
};

template <>
class optional_ctor_base<copy_traits::movable> {
public:
constexpr optional_ctor_base() = default;
optional_ctor_base(const optional_ctor_base&) = delete;
optional_ctor_base(optional_ctor_base&&) = default;
optional_ctor_base& operator=(const optional_ctor_base&) = default;
optional_ctor_base& operator=(optional_ctor_base&&) = default;
};

template <>
class optional_ctor_base<copy_traits::non_movable> {
public:
constexpr optional_ctor_base() = default;
optional_ctor_base(const optional_ctor_base&) = delete;
optional_ctor_base(optional_ctor_base&&) = delete;
optional_ctor_base& operator=(const optional_ctor_base&) = default;
optional_ctor_base& operator=(optional_ctor_base&&) = default;
};

template <copy_traits>
class optional_assign_base;

template <>
class optional_assign_base<copy_traits::copyable> {
public:
constexpr optional_assign_base() = default;
optional_assign_base(const optional_assign_base&) = default;
optional_assign_base(optional_assign_base&&) = default;
optional_assign_base& operator=(const optional_assign_base&) = default;
optional_assign_base& operator=(optional_assign_base&&) = default;
};

template <>
class optional_assign_base<copy_traits::movable> {
public:
constexpr optional_assign_base() = default;
optional_assign_base(const optional_assign_base&) = default;
optional_assign_base(optional_assign_base&&) = default;
optional_assign_base& operator=(const optional_assign_base&) = delete;
optional_assign_base& operator=(optional_assign_base&&) = default;
};

template <>
class optional_assign_base<copy_traits::non_movable> {
public:
constexpr optional_assign_base() = default;
optional_assign_base(const optional_assign_base&) = default;
optional_assign_base(optional_assign_base&&) = default;
optional_assign_base& operator=(const optional_assign_base&) = delete;
optional_assign_base& operator=(optional_assign_base&&) = delete;
};

template <typename T>
constexpr copy_traits get_ctor_copy_traits() {
return std::is_copy_constructible<T>::value
? copy_traits::copyable
: std::is_move_constructible<T>::value ? copy_traits::movable
: copy_traits::non_movable;
}

template <typename T>
constexpr copy_traits get_assign_copy_traits() {
return absl::is_copy_assignable<T>::value &&
std::is_copy_constructible<T>::value
? copy_traits::copyable
: absl::is_move_assignable<T>::value &&
std::is_move_constructible<T>::value
? copy_traits::movable
: copy_traits::non_movable;
}

template <typename T, typename U>
struct is_constructible_convertible_from_optional
: std::integral_constant<
bool, std::is_constructible<T, optional<U>&>::value ||
std::is_constructible<T, optional<U>&&>::value ||
std::is_constructible<T, const optional<U>&>::value ||
std::is_constructible<T, const optional<U>&&>::value ||
std::is_convertible<optional<U>&, T>::value ||
std::is_convertible<optional<U>&&, T>::value ||
std::is_convertible<const optional<U>&, T>::value ||
std::is_convertible<const optional<U>&&, T>::value> {};

template <typename T, typename U>
struct is_constructible_convertible_assignable_from_optional
: std::integral_constant<
bool, is_constructible_convertible_from_optional<T, U>::value ||
std::is_assignable<T&, optional<U>&>::value ||
std::is_assignable<T&, optional<U>&&>::value ||
std::is_assignable<T&, const optional<U>&>::value ||
std::is_assignable<T&, const optional<U>&&>::value> {};

bool convertible_to_bool(bool);

template <typename T, typename = size_t>
struct optional_hash_base {
optional_hash_base() = delete;
optional_hash_base(const optional_hash_base&) = delete;
optional_hash_base(optional_hash_base&&) = delete;
optional_hash_base& operator=(const optional_hash_base&) = delete;
optional_hash_base& operator=(optional_hash_base&&) = delete;
};

template <typename T>
struct optional_hash_base<T, decltype(std::hash<absl::remove_const_t<T> >()(
std::declval<absl::remove_const_t<T> >()))> {
using argument_type = absl::optional<T>;
using result_type = size_t;
size_t operator()(const absl::optional<T>& opt) const {
if (opt) {
return std::hash<absl::remove_const_t<T> >()(*opt);
} else {
return static_cast<size_t>(0x297814aaad196e6dULL);
}
}
};

}  


template <typename T>
class optional : private optional_internal::optional_data<T>,
private optional_internal::optional_ctor_base<
optional_internal::get_ctor_copy_traits<T>()>,
private optional_internal::optional_assign_base<
optional_internal::get_assign_copy_traits<T>()> {
using data_base = optional_internal::optional_data<T>;

public:
typedef T value_type;


constexpr optional() noexcept {}

constexpr optional(nullopt_t) noexcept {}  

optional(const optional& src) = default;

optional(optional&& src) = default;

template <typename... Args>
constexpr explicit optional(in_place_t, Args&&... args)
: data_base(in_place_t(), absl::forward<Args>(args)...) {}

template <typename U, typename... Args,
typename = typename std::enable_if<std::is_constructible<
T, std::initializer_list<U>&, Args&&...>::value>::type>
constexpr explicit optional(in_place_t, std::initializer_list<U> il,
Args&&... args)
: data_base(in_place_t(), il, absl::forward<Args>(args)...) {
}

template <
typename U = T,
typename std::enable_if<
absl::conjunction<absl::negation<std::is_same<
in_place_t, typename std::decay<U>::type> >,
absl::negation<std::is_same<
optional<T>, typename std::decay<U>::type> >,
std::is_convertible<U&&, T>,
std::is_constructible<T, U&&> >::value,
bool>::type = false>
constexpr optional(U&& v) : data_base(in_place_t(), absl::forward<U>(v)) {}

template <
typename U = T,
typename std::enable_if<
absl::conjunction<absl::negation<std::is_same<
in_place_t, typename std::decay<U>::type>>,
absl::negation<std::is_same<
optional<T>, typename std::decay<U>::type>>,
absl::negation<std::is_convertible<U&&, T>>,
std::is_constructible<T, U&&>>::value,
bool>::type = false>
explicit constexpr optional(U&& v)
: data_base(in_place_t(), absl::forward<U>(v)) {}

template <typename U,
typename std::enable_if<
absl::conjunction<
absl::negation<std::is_same<T, U> >,
std::is_constructible<T, const U&>,
absl::negation<
optional_internal::
is_constructible_convertible_from_optional<T, U> >,
std::is_convertible<const U&, T> >::value,
bool>::type = false>
optional(const optional<U>& rhs) {
if (rhs) {
this->construct(*rhs);
}
}

template <typename U,
typename std::enable_if<
absl::conjunction<
absl::negation<std::is_same<T, U>>,
std::is_constructible<T, const U&>,
absl::negation<
optional_internal::
is_constructible_convertible_from_optional<T, U>>,
absl::negation<std::is_convertible<const U&, T>>>::value,
bool>::type = false>
explicit optional(const optional<U>& rhs) {
if (rhs) {
this->construct(*rhs);
}
}

template <typename U,
typename std::enable_if<
absl::conjunction<
absl::negation<std::is_same<T, U> >,
std::is_constructible<T, U&&>,
absl::negation<
optional_internal::
is_constructible_convertible_from_optional<T, U> >,
std::is_convertible<U&&, T> >::value,
bool>::type = false>
optional(optional<U>&& rhs) {
if (rhs) {
this->construct(std::move(*rhs));
}
}

template <
typename U,
typename std::enable_if<
absl::conjunction<
absl::negation<std::is_same<T, U>>, std::is_constructible<T, U&&>,
absl::negation<
optional_internal::is_constructible_convertible_from_optional<
T, U>>,
absl::negation<std::is_convertible<U&&, T>>>::value,
bool>::type = false>
explicit optional(optional<U>&& rhs) {
if (rhs) {
this->construct(std::move(*rhs));
}
}

~optional() = default;


optional& operator=(nullopt_t) noexcept {
this->destruct();
return *this;
}

optional& operator=(const optional& src) = default;

optional& operator=(optional&& src) = default;

template <
typename U = T,
typename = typename std::enable_if<absl::conjunction<
absl::negation<
std::is_same<optional<T>, typename std::decay<U>::type>>,
absl::negation<
absl::conjunction<std::is_scalar<T>,
std::is_same<T, typename std::decay<U>::type>>>,
std::is_constructible<T, U>, std::is_assignable<T&, U>>::value>::type>
optional& operator=(U&& v) {
this->assign(std::forward<U>(v));
return *this;
}

template <
typename U,
typename = typename std::enable_if<absl::conjunction<
absl::negation<std::is_same<T, U>>,
std::is_constructible<T, const U&>, std::is_assignable<T&, const U&>,
absl::negation<
optional_internal::
is_constructible_convertible_assignable_from_optional<
T, U>>>::value>::type>
optional& operator=(const optional<U>& rhs) {
if (rhs) {
this->assign(*rhs);
} else {
this->destruct();
}
return *this;
}

template <typename U,
typename = typename std::enable_if<absl::conjunction<
absl::negation<std::is_same<T, U>>, std::is_constructible<T, U>,
std::is_assignable<T&, U>,
absl::negation<
optional_internal::
is_constructible_convertible_assignable_from_optional<
T, U>>>::value>::type>
optional& operator=(optional<U>&& rhs) {
if (rhs) {
this->assign(std::move(*rhs));
} else {
this->destruct();
}
return *this;
}


ABSL_ATTRIBUTE_REINITIALIZES void reset() noexcept { this->destruct(); }

template <typename... Args,
typename = typename std::enable_if<
std::is_constructible<T, Args&&...>::value>::type>
T& emplace(Args&&... args) {
this->destruct();
this->construct(std::forward<Args>(args)...);
return reference();
}

template <typename U, typename... Args,
typename = typename std::enable_if<std::is_constructible<
T, std::initializer_list<U>&, Args&&...>::value>::type>
T& emplace(std::initializer_list<U> il, Args&&... args) {
this->destruct();
this->construct(il, std::forward<Args>(args)...);
return reference();
}


void swap(optional& rhs) noexcept(
std::is_nothrow_move_constructible<T>::value&&
std::is_trivial<T>::value) {
if (*this) {
if (rhs) {
using std::swap;
swap(**this, *rhs);
} else {
rhs.construct(std::move(**this));
this->destruct();
}
} else {
if (rhs) {
this->construct(std::move(*rhs));
rhs.destruct();
} else {
}
}
}


const T* operator->() const {
assert(this->engaged_);
return std::addressof(this->data_);
}
T* operator->() {
assert(this->engaged_);
return std::addressof(this->data_);
}

constexpr const T& operator*() const & { return reference(); }
T& operator*() & {
assert(this->engaged_);
return reference();
}
constexpr const T&& operator*() const && {
return absl::move(reference());
}
T&& operator*() && {
assert(this->engaged_);
return std::move(reference());
}

constexpr explicit operator bool() const noexcept { return this->engaged_; }

constexpr bool has_value() const noexcept { return this->engaged_; }

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif  
constexpr const T& value() const & {
return static_cast<bool>(*this)
? reference()
: (optional_internal::throw_bad_optional_access(), reference());
}
T& value() & {
return static_cast<bool>(*this)
? reference()
: (optional_internal::throw_bad_optional_access(), reference());
}
T&& value() && {  
return std::move(
static_cast<bool>(*this)
? reference()
: (optional_internal::throw_bad_optional_access(), reference()));
}
constexpr const T&& value() const && {  
return absl::move(
static_cast<bool>(*this)
? reference()
: (optional_internal::throw_bad_optional_access(), reference()));
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif  

template <typename U>
constexpr T value_or(U&& v) const& {
static_assert(std::is_copy_constructible<value_type>::value,
"optional<T>::value_or: T must by copy constructible");
static_assert(std::is_convertible<U&&, value_type>::value,
"optional<T>::value_or: U must be convertible to T");
return static_cast<bool>(*this)
? **this
: static_cast<T>(absl::forward<U>(v));
}
template <typename U>
T value_or(U&& v) && {  
static_assert(std::is_move_constructible<value_type>::value,
"optional<T>::value_or: T must by copy constructible");
static_assert(std::is_convertible<U&&, value_type>::value,
"optional<T>::value_or: U must be convertible to T");
return static_cast<bool>(*this) ? std::move(**this)
: static_cast<T>(std::forward<U>(v));
}

private:
constexpr const T& reference() const { return this->data_; }
T& reference() { return this->data_; }

static_assert(
!std::is_same<nullopt_t, typename std::remove_cv<T>::type>::value,
"optional<nullopt_t> is not allowed.");
static_assert(
!std::is_same<in_place_t, typename std::remove_cv<T>::type>::value,
"optional<in_place_t> is not allowed.");
static_assert(!std::is_reference<T>::value,
"optional<reference> is not allowed.");
};


template <typename T,
typename std::enable_if<std::is_move_constructible<T>::value,
bool>::type = false>
void swap(optional<T>& a, optional<T>& b) noexcept(noexcept(a.swap(b))) {
a.swap(b);
}

template <typename T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v) {
return optional<typename std::decay<T>::type>(absl::forward<T>(v));
}

template <typename T, typename... Args>
constexpr optional<T> make_optional(Args&&... args) {
return optional<T>(in_place_t(), absl::forward<Args>(args)...);
}

template <typename T, typename U, typename... Args>
constexpr optional<T> make_optional(std::initializer_list<U> il,
Args&&... args) {
return optional<T>(in_place_t(), il,
absl::forward<Args>(args)...);
}



template <typename T, typename U>
constexpr auto operator==(const optional<T>& x, const optional<U>& y)
-> decltype(optional_internal::convertible_to_bool(*x == *y)) {
return static_cast<bool>(x) != static_cast<bool>(y)
? false
: static_cast<bool>(x) == false ? true
: static_cast<bool>(*x == *y);
}

template <typename T, typename U>
constexpr auto operator!=(const optional<T>& x, const optional<U>& y)
-> decltype(optional_internal::convertible_to_bool(*x != *y)) {
return static_cast<bool>(x) != static_cast<bool>(y)
? true
: static_cast<bool>(x) == false ? false
: static_cast<bool>(*x != *y);
}
template <typename T, typename U>
constexpr auto operator<(const optional<T>& x, const optional<U>& y)
-> decltype(optional_internal::convertible_to_bool(*x < *y)) {
return !y ? false : !x ? true : static_cast<bool>(*x < *y);
}
template <typename T, typename U>
constexpr auto operator>(const optional<T>& x, const optional<U>& y)
-> decltype(optional_internal::convertible_to_bool(*x > *y)) {
return !x ? false : !y ? true : static_cast<bool>(*x > *y);
}
template <typename T, typename U>
constexpr auto operator<=(const optional<T>& x, const optional<U>& y)
-> decltype(optional_internal::convertible_to_bool(*x <= *y)) {
return !x ? true : !y ? false : static_cast<bool>(*x <= *y);
}
template <typename T, typename U>
constexpr auto operator>=(const optional<T>& x, const optional<U>& y)
-> decltype(optional_internal::convertible_to_bool(*x >= *y)) {
return !y ? true : !x ? false : static_cast<bool>(*x >= *y);
}

template <typename T>
constexpr bool operator==(const optional<T>& x, nullopt_t) noexcept {
return !x;
}
template <typename T>
constexpr bool operator==(nullopt_t, const optional<T>& x) noexcept {
return !x;
}
template <typename T>
constexpr bool operator!=(const optional<T>& x, nullopt_t) noexcept {
return static_cast<bool>(x);
}
template <typename T>
constexpr bool operator!=(nullopt_t, const optional<T>& x) noexcept {
return static_cast<bool>(x);
}
template <typename T>
constexpr bool operator<(const optional<T>&, nullopt_t) noexcept {
return false;
}
template <typename T>
constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept {
return static_cast<bool>(x);
}
template <typename T>
constexpr bool operator<=(const optional<T>& x, nullopt_t) noexcept {
return !x;
}
template <typename T>
constexpr bool operator<=(nullopt_t, const optional<T>&) noexcept {
return true;
}
template <typename T>
constexpr bool operator>(const optional<T>& x, nullopt_t) noexcept {
return static_cast<bool>(x);
}
template <typename T>
constexpr bool operator>(nullopt_t, const optional<T>&) noexcept {
return false;
}
template <typename T>
constexpr bool operator>=(const optional<T>&, nullopt_t) noexcept {
return true;
}
template <typename T>
constexpr bool operator>=(nullopt_t, const optional<T>& x) noexcept {
return !x;
}


template <typename T, typename U>
constexpr auto operator==(const optional<T>& x, const U& v)
-> decltype(optional_internal::convertible_to_bool(*x == v)) {
return static_cast<bool>(x) ? static_cast<bool>(*x == v) : false;
}
template <typename T, typename U>
constexpr auto operator==(const U& v, const optional<T>& x)
-> decltype(optional_internal::convertible_to_bool(v == *x)) {
return static_cast<bool>(x) ? static_cast<bool>(v == *x) : false;
}
template <typename T, typename U>
constexpr auto operator!=(const optional<T>& x, const U& v)
-> decltype(optional_internal::convertible_to_bool(*x != v)) {
return static_cast<bool>(x) ? static_cast<bool>(*x != v) : true;
}
template <typename T, typename U>
constexpr auto operator!=(const U& v, const optional<T>& x)
-> decltype(optional_internal::convertible_to_bool(v != *x)) {
return static_cast<bool>(x) ? static_cast<bool>(v != *x) : true;
}
template <typename T, typename U>
constexpr auto operator<(const optional<T>& x, const U& v)
-> decltype(optional_internal::convertible_to_bool(*x < v)) {
return static_cast<bool>(x) ? static_cast<bool>(*x < v) : true;
}
template <typename T, typename U>
constexpr auto operator<(const U& v, const optional<T>& x)
-> decltype(optional_internal::convertible_to_bool(v < *x)) {
return static_cast<bool>(x) ? static_cast<bool>(v < *x) : false;
}
template <typename T, typename U>
constexpr auto operator<=(const optional<T>& x, const U& v)
-> decltype(optional_internal::convertible_to_bool(*x <= v)) {
return static_cast<bool>(x) ? static_cast<bool>(*x <= v) : true;
}
template <typename T, typename U>
constexpr auto operator<=(const U& v, const optional<T>& x)
-> decltype(optional_internal::convertible_to_bool(v <= *x)) {
return static_cast<bool>(x) ? static_cast<bool>(v <= *x) : false;
}
template <typename T, typename U>
constexpr auto operator>(const optional<T>& x, const U& v)
-> decltype(optional_internal::convertible_to_bool(*x > v)) {
return static_cast<bool>(x) ? static_cast<bool>(*x > v) : false;
}
template <typename T, typename U>
constexpr auto operator>(const U& v, const optional<T>& x)
-> decltype(optional_internal::convertible_to_bool(v > *x)) {
return static_cast<bool>(x) ? static_cast<bool>(v > *x) : true;
}
template <typename T, typename U>
constexpr auto operator>=(const optional<T>& x, const U& v)
-> decltype(optional_internal::convertible_to_bool(*x >= v)) {
return static_cast<bool>(x) ? static_cast<bool>(*x >= v) : false;
}
template <typename T, typename U>
constexpr auto operator>=(const U& v, const optional<T>& x)
-> decltype(optional_internal::convertible_to_bool(v >= *x)) {
return static_cast<bool>(x) ? static_cast<bool>(v >= *x) : true;
}

}  

namespace std {

template <typename T>
struct hash<absl::optional<T> >
: absl::optional_internal::optional_hash_base<T> {};

}  

#undef ABSL_OPTIONAL_USE_INHERITING_CONSTRUCTORS
#undef ABSL_MSVC_CONSTEXPR_BUG_IN_UNION_LIKE_CLASS

#endif  

#endif  
