
#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/addressof.h>
#include <hydra/detail/external/hydra_thrust/swap.h>

#define HYDRA_THRUST_OPTIONAL_VERSION_MAJOR 0
#define HYDRA_THRUST_OPTIONAL_VERSION_MINOR 2

#include <exception>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

#if (defined(_MSC_VER) && _MSC_VER == 1900)
#define HYDRA_THRUST_OPTIONAL_MSVC2015
#endif

#if (defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ <= 9 &&              \
!defined(__clang__))
#define HYDRA_THRUST_OPTIONAL_GCC49
#endif

#if (defined(__GNUC__) && __GNUC__ == 5 && __GNUC_MINOR__ <= 4 &&              \
!defined(__clang__))
#define HYDRA_THRUST_OPTIONAL_GCC54
#endif

#if (defined(__GNUC__) && __GNUC__ == 5 && __GNUC_MINOR__ <= 5 &&              \
!defined(__clang__))
#define HYDRA_THRUST_OPTIONAL_GCC55
#endif

#if (defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ <= 9 &&              \
!defined(__clang__))
#define HYDRA_THRUST_OPTIONAL_NO_CONSTRR

#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_CONSTRUCTIBLE(T)                                     \
std::has_trivial_copy_constructor<T>::value
#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_ASSIGNABLE(T) std::has_trivial_copy_assign<T>::value

#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_DESTRUCTIBLE(T) std::is_trivially_destructible<T>::value

#elif (defined(__GNUC__) && __GNUC__ < 8 &&                                                \
!defined(__clang__))
#ifndef HYDRA_THRUST_GCC_LESS_8_TRIVIALLY_COPY_CONSTRUCTIBLE_MUTEX
#define HYDRA_THRUST_GCC_LESS_8_TRIVIALLY_COPY_CONSTRUCTIBLE_MUTEX
HYDRA_THRUST_BEGIN_NS
namespace detail {
template<class T>
struct is_trivially_copy_constructible : std::is_trivially_copy_constructible<T>{};
#ifdef _GLIBCXX_VECTOR
template<class T, class A>
struct is_trivially_copy_constructible<std::vector<T,A>>
: std::is_trivially_copy_constructible<T>{};
#endif      
}
HYDRA_THRUST_END_NS
#endif

#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_CONSTRUCTIBLE(T)                                     \
hydra_thrust::detail::is_trivially_copy_constructible<T>::value
#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_ASSIGNABLE(T)                                        \
std::is_trivially_copy_assignable<T>::value
#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_DESTRUCTIBLE(T) std::is_trivially_destructible<T>::value
#else
#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_CONSTRUCTIBLE(T)                                     \
std::is_trivially_copy_constructible<T>::value
#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_ASSIGNABLE(T)                                        \
std::is_trivially_copy_assignable<T>::value
#define HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_DESTRUCTIBLE(T) std::is_trivially_destructible<T>::value
#endif

#if __cplusplus > 201103L
#define HYDRA_THRUST_OPTIONAL_CPP14
#endif

#if (__cplusplus == 201103L || defined(HYDRA_THRUST_OPTIONAL_MSVC2015) ||                \
defined(HYDRA_THRUST_OPTIONAL_GCC49))
#define HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR
#else
#define HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR constexpr
#endif

HYDRA_THRUST_BEGIN_NS
#ifndef HYDRA_THRUST_MONOSTATE_INPLACE_MUTEX
#define HYDRA_THRUST_MONOSTATE_INPLACE_MUTEX
class monostate {};

struct in_place_t {
explicit in_place_t() = default;
};
static constexpr in_place_t in_place{};
#endif

template <class T> class optional;

namespace detail {
#ifndef HYDRA_THRUST_TRAITS_MUTEX
#define HYDRA_THRUST_TRAITS_MUTEX
template <class T> using remove_const_t = typename std::remove_const<T>::type;
template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <class T> using decay_t = typename std::decay<T>::type;
template <bool E, class T = void>
using enable_if_t = typename std::enable_if<E, T>::type;
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <class...> struct conjunction : std::true_type {};
template <class B> struct conjunction<B> : B {};
template <class B, class... Bs>
struct conjunction<B, Bs...>
: std::conditional<bool(B::value), conjunction<Bs...>, B>::type {};

#if defined(_LIBCPP_VERSION) && __cplusplus == 201103L
#define HYDRA_THRUST_OPTIONAL_LIBCXX_MEM_FN_WORKAROUND
#endif

#ifdef HYDRA_THRUST_OPTIONAL_LIBCXX_MEM_FN_WORKAROUND
template <class T> struct is_pointer_to_non_const_member_func : std::false_type{};
template <class T, class Ret, class... Args>
struct is_pointer_to_non_const_member_func<Ret (T::*) (Args...)> : std::true_type{};
template <class T, class Ret, class... Args>
struct is_pointer_to_non_const_member_func<Ret (T::*) (Args...)&> : std::true_type{};
template <class T, class Ret, class... Args>
struct is_pointer_to_non_const_member_func<Ret (T::*) (Args...)&&> : std::true_type{};        
template <class T, class Ret, class... Args>
struct is_pointer_to_non_const_member_func<Ret (T::*) (Args...) volatile> : std::true_type{};
template <class T, class Ret, class... Args>
struct is_pointer_to_non_const_member_func<Ret (T::*) (Args...) volatile&> : std::true_type{};
template <class T, class Ret, class... Args>
struct is_pointer_to_non_const_member_func<Ret (T::*) (Args...) volatile&&> : std::true_type{};        

template <class T> struct is_const_or_const_ref : std::false_type{};
template <class T> struct is_const_or_const_ref<T const&> : std::true_type{};
template <class T> struct is_const_or_const_ref<T const> : std::true_type{};    
#endif

__hydra_thrust_exec_check_disable__
template <typename Fn, typename... Args,
#ifdef HYDRA_THRUST_OPTIONAL_LIBCXX_MEM_FN_WORKAROUND
typename = enable_if_t<!(is_pointer_to_non_const_member_func<Fn>::value 
&& is_const_or_const_ref<Args...>::value)>, 
#endif
typename = enable_if_t<std::is_member_pointer<decay_t<Fn>>::value>,
int = 0>
__host__ __device__
constexpr auto invoke(Fn &&f, Args &&... args) noexcept(
noexcept(std::mem_fn(f)(std::forward<Args>(args)...)))
-> decltype(std::mem_fn(f)(std::forward<Args>(args)...)) {
return std::mem_fn(f)(std::forward<Args>(args)...);
}

__hydra_thrust_exec_check_disable__
template <typename Fn, typename... Args,
typename = enable_if_t<!std::is_member_pointer<decay_t<Fn>>::value>>
__host__ __device__
constexpr auto invoke(Fn &&f, Args &&... args) noexcept(
noexcept(std::forward<Fn>(f)(std::forward<Args>(args)...)))
-> decltype(std::forward<Fn>(f)(std::forward<Args>(args)...)) {
return std::forward<Fn>(f)(std::forward<Args>(args)...);
}

template <class F, class, class... Us> struct invoke_result_impl;

template <class F, class... Us>
struct invoke_result_impl<
F, decltype(detail::invoke(std::declval<F>(), std::declval<Us>()...), void()),
Us...> {
using type = decltype(detail::invoke(std::declval<F>(), std::declval<Us>()...));
};

template <class F, class... Us>
using invoke_result = invoke_result_impl<F, void, Us...>;

template <class F, class... Us>
using invoke_result_t = typename invoke_result<F, Us...>::type;
#endif

template <class...> struct voider { using type = void; };
template <class... Ts> using void_t = typename voider<Ts...>::type;

template <class T> struct is_optional_impl : std::false_type {};
template <class T> struct is_optional_impl<optional<T>> : std::true_type {};
template <class T> using is_optional = is_optional_impl<decay_t<T>>;

template <class U>
using fixup_void = conditional_t<std::is_void<U>::value, monostate, U>;

template <class F, class U, class = invoke_result_t<F, U>>
using get_map_return = optional<fixup_void<invoke_result_t<F, U>>>;

template <class F, class = void, class... U> struct returns_void_impl;
template <class F, class... U>
struct returns_void_impl<F, void_t<invoke_result_t<F, U...>>, U...>
: std::is_void<invoke_result_t<F, U...>> {};
template <class F, class... U>
using returns_void = returns_void_impl<F, void, U...>;

template <class T, class... U>
using enable_if_ret_void = enable_if_t<returns_void<T &&, U...>::value>;

template <class T, class... U>
using disable_if_ret_void = enable_if_t<!returns_void<T &&, U...>::value>;

template <class T, class U>
using enable_forward_value =
detail::enable_if_t<std::is_constructible<T, U &&>::value &&
!std::is_same<detail::decay_t<U>, in_place_t>::value &&
!std::is_same<optional<T>, detail::decay_t<U>>::value>;

template <class T, class U, class Other>
using enable_from_other = detail::enable_if_t<
std::is_constructible<T, Other>::value &&
!std::is_constructible<T, optional<U> &>::value &&
!std::is_constructible<T, optional<U> &&>::value &&
!std::is_constructible<T, const optional<U> &>::value &&
!std::is_constructible<T, const optional<U> &&>::value &&
!std::is_convertible<optional<U> &, T>::value &&
!std::is_convertible<optional<U> &&, T>::value &&
!std::is_convertible<const optional<U> &, T>::value &&
!std::is_convertible<const optional<U> &&, T>::value>;

template <class T, class U>
using enable_assign_forward = detail::enable_if_t<
!std::is_same<optional<T>, detail::decay_t<U>>::value &&
!detail::conjunction<std::is_scalar<T>,
std::is_same<T, detail::decay_t<U>>>::value &&
std::is_constructible<T, U>::value && std::is_assignable<T &, U>::value>;

template <class T, class U, class Other>
using enable_assign_from_other = detail::enable_if_t<
std::is_constructible<T, Other>::value &&
std::is_assignable<T &, Other>::value &&
!std::is_constructible<T, optional<U> &>::value &&
!std::is_constructible<T, optional<U> &&>::value &&
!std::is_constructible<T, const optional<U> &>::value &&
!std::is_constructible<T, const optional<U> &&>::value &&
!std::is_convertible<optional<U> &, T>::value &&
!std::is_convertible<optional<U> &&, T>::value &&
!std::is_convertible<const optional<U> &, T>::value &&
!std::is_convertible<const optional<U> &&, T>::value &&
!std::is_assignable<T &, optional<U> &>::value &&
!std::is_assignable<T &, optional<U> &&>::value &&
!std::is_assignable<T &, const optional<U> &>::value &&
!std::is_assignable<T &, const optional<U> &&>::value>;

#ifdef _MSC_VER
template <class T, class U = T> struct is_swappable : std::true_type {};

template <class T, class U = T> struct is_nothrow_swappable : std::true_type {};
#else
namespace swap_adl_tests {
struct tag {};

template <class T> tag swap(T &, T &);
template <class T, std::size_t N> tag swap(T (&a)[N], T (&b)[N]);

template <class, class> std::false_type can_swap(...) noexcept(false);
template <class T, class U,
class = decltype(swap(std::declval<T &>(), std::declval<U &>()))>
std::true_type can_swap(int) noexcept(noexcept(swap(std::declval<T &>(),
std::declval<U &>())));

template <class, class> std::false_type uses_std(...);
template <class T, class U>
std::is_same<decltype(swap(std::declval<T &>(), std::declval<U &>())), tag>
uses_std(int);

template <class T>
struct is_std_swap_noexcept
: std::integral_constant<bool,
std::is_nothrow_move_constructible<T>::value &&
std::is_nothrow_move_assignable<T>::value> {};

template <class T, std::size_t N>
struct is_std_swap_noexcept<T[N]> : is_std_swap_noexcept<T> {};

template <class T, class U>
struct is_adl_swap_noexcept
: std::integral_constant<bool, noexcept(can_swap<T, U>(0))> {};
} 

template <class T, class U = T>
struct is_swappable
: std::integral_constant<
bool,
decltype(detail::swap_adl_tests::can_swap<T, U>(0))::value &&
(!decltype(detail::swap_adl_tests::uses_std<T, U>(0))::value ||
(std::is_move_assignable<T>::value &&
std::is_move_constructible<T>::value))> {};

template <class T, std::size_t N>
struct is_swappable<T[N], T[N]>
: std::integral_constant<
bool,
decltype(detail::swap_adl_tests::can_swap<T[N], T[N]>(0))::value &&
(!decltype(
detail::swap_adl_tests::uses_std<T[N], T[N]>(0))::value ||
is_swappable<T, T>::value)> {};

template <class T, class U = T>
struct is_nothrow_swappable
: std::integral_constant<
bool,
is_swappable<T, U>::value &&
((decltype(detail::swap_adl_tests::uses_std<T, U>(0))::value
&&detail::swap_adl_tests::is_std_swap_noexcept<T>::value) ||
(!decltype(detail::swap_adl_tests::uses_std<T, U>(0))::value &&
detail::swap_adl_tests::is_adl_swap_noexcept<T,
U>::value))> {
};
#endif

template <class T, bool = ::std::is_trivially_destructible<T>::value>
struct optional_storage_base {
__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional_storage_base() noexcept
: m_dummy(), m_has_value(false) {}

__hydra_thrust_exec_check_disable__
template <class... U>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional_storage_base(in_place_t, U &&... u)
: m_value(std::forward<U>(u)...), m_has_value(true) {}

__hydra_thrust_exec_check_disable__
__host__ __device__
~optional_storage_base() {
if (m_has_value) {
m_value.~T();
m_has_value = false;
}
}

struct dummy {};
union {
dummy m_dummy;
T m_value;
};

bool m_has_value;
};

template <class T> struct optional_storage_base<T, true> {
__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional_storage_base() noexcept
: m_dummy(), m_has_value(false) {}

__hydra_thrust_exec_check_disable__
template <class... U>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional_storage_base(in_place_t, U &&... u)
: m_value(std::forward<U>(u)...), m_has_value(true) {}


struct dummy {};
union {
dummy m_dummy;
T m_value;
};

bool m_has_value = false;
};

template <class T> struct optional_operations_base : optional_storage_base<T> {
using optional_storage_base<T>::optional_storage_base;

__hydra_thrust_exec_check_disable__
__host__ __device__
void hard_reset() noexcept {
get().~T();
this->m_has_value = false;
}

__hydra_thrust_exec_check_disable__
template <class... Args>
__host__ __device__
void construct(Args &&... args) noexcept {
new (addressof(this->m_value)) T(std::forward<Args>(args)...);
this->m_has_value = true;
}

__hydra_thrust_exec_check_disable__
template <class Opt>
__host__ __device__
void assign(Opt &&rhs) {
if (this->has_value()) {
if (rhs.has_value()) {
this->m_value = std::forward<Opt>(rhs).get();
} else {
this->m_value.~T();
this->m_has_value = false;
}
}

if (rhs.has_value()) {
construct(std::forward<Opt>(rhs).get());
}
}

__hydra_thrust_exec_check_disable__
__host__ __device__
bool has_value() const { return this->m_has_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &get() & { return this->m_value; }
__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR const T &get() const & { return this->m_value; }
__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &&get() && { return std::move(this->m_value); }
#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr const T &&get() const && { return std::move(this->m_value); }
#endif
};

template <class T, bool = HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_CONSTRUCTIBLE(T)>
struct optional_copy_base : optional_operations_base<T> {
using optional_operations_base<T>::optional_operations_base;
};

template <class T>
struct optional_copy_base<T, false> : optional_operations_base<T> {
using optional_operations_base<T>::optional_operations_base;

__hydra_thrust_exec_check_disable__
optional_copy_base() = default;
__hydra_thrust_exec_check_disable__
__host__ __device__
optional_copy_base(const optional_copy_base &rhs) {
if (rhs.has_value()) {
this->construct(rhs.get());
} else {
this->m_has_value = false;
}
}

__hydra_thrust_exec_check_disable__
optional_copy_base(optional_copy_base &&rhs) = default;
__hydra_thrust_exec_check_disable__
optional_copy_base &operator=(const optional_copy_base &rhs) = default;
__hydra_thrust_exec_check_disable__
optional_copy_base &operator=(optional_copy_base &&rhs) = default;
};

#ifndef HYDRA_THRUST_OPTIONAL_GCC49
template <class T, bool = std::is_trivially_move_constructible<T>::value>
struct optional_move_base : optional_copy_base<T> {
using optional_copy_base<T>::optional_copy_base;
};
#else
template <class T, bool = false> struct optional_move_base;
#endif
template <class T> struct optional_move_base<T, false> : optional_copy_base<T> {
using optional_copy_base<T>::optional_copy_base;

__hydra_thrust_exec_check_disable__
optional_move_base() = default;
__hydra_thrust_exec_check_disable__
optional_move_base(const optional_move_base &rhs) = default;

__hydra_thrust_exec_check_disable__
__host__ __device__
optional_move_base(optional_move_base &&rhs) noexcept(
std::is_nothrow_move_constructible<T>::value) {
if (rhs.has_value()) {
this->construct(std::move(rhs.get()));
} else {
this->m_has_value = false;
}
}
__hydra_thrust_exec_check_disable__
optional_move_base &operator=(const optional_move_base &rhs) = default;
__hydra_thrust_exec_check_disable__
optional_move_base &operator=(optional_move_base &&rhs) = default;
};

template <class T, bool = HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_ASSIGNABLE(T) &&
HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_COPY_CONSTRUCTIBLE(T) &&
HYDRA_THRUST_OPTIONAL_IS_TRIVIALLY_DESTRUCTIBLE(T)>
struct optional_copy_assign_base : optional_move_base<T> {
using optional_move_base<T>::optional_move_base;
};

template <class T>
struct optional_copy_assign_base<T, false> : optional_move_base<T> {
using optional_move_base<T>::optional_move_base;

__hydra_thrust_exec_check_disable__
optional_copy_assign_base() = default;
__hydra_thrust_exec_check_disable__
optional_copy_assign_base(const optional_copy_assign_base &rhs) = default;

__hydra_thrust_exec_check_disable__
optional_copy_assign_base(optional_copy_assign_base &&rhs) = default;
__hydra_thrust_exec_check_disable__
__host__ __device__
optional_copy_assign_base &operator=(const optional_copy_assign_base &rhs) {
this->assign(rhs);
return *this;
}
__hydra_thrust_exec_check_disable__
optional_copy_assign_base &
operator=(optional_copy_assign_base &&rhs) = default;
};

#ifndef HYDRA_THRUST_OPTIONAL_GCC49
template <class T, bool = std::is_trivially_destructible<T>::value
&&std::is_trivially_move_constructible<T>::value
&&std::is_trivially_move_assignable<T>::value>
struct optional_move_assign_base : optional_copy_assign_base<T> {
using optional_copy_assign_base<T>::optional_copy_assign_base;
};
#else
template <class T, bool = false> struct optional_move_assign_base;
#endif

template <class T>
struct optional_move_assign_base<T, false> : optional_copy_assign_base<T> {
using optional_copy_assign_base<T>::optional_copy_assign_base;

__hydra_thrust_exec_check_disable__
optional_move_assign_base() = default;
__hydra_thrust_exec_check_disable__
optional_move_assign_base(const optional_move_assign_base &rhs) = default;

__hydra_thrust_exec_check_disable__
optional_move_assign_base(optional_move_assign_base &&rhs) = default;

__hydra_thrust_exec_check_disable__
optional_move_assign_base &
operator=(const optional_move_assign_base &rhs) = default;

__hydra_thrust_exec_check_disable__
__host__ __device__
optional_move_assign_base &
operator=(optional_move_assign_base &&rhs) noexcept(
std::is_nothrow_move_constructible<T>::value
&&std::is_nothrow_move_assignable<T>::value) {
this->assign(std::move(rhs));
return *this;
}
};

template <class T, bool EnableCopy = std::is_copy_constructible<T>::value,
bool EnableMove = std::is_move_constructible<T>::value>
struct optional_delete_ctor_base {
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(const optional_delete_ctor_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(optional_delete_ctor_base &&) noexcept = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(const optional_delete_ctor_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(optional_delete_ctor_base &&) noexcept = default;
};

template <class T> struct optional_delete_ctor_base<T, true, false> {
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(const optional_delete_ctor_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(optional_delete_ctor_base &&) noexcept = delete;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(const optional_delete_ctor_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(optional_delete_ctor_base &&) noexcept = default;
};

template <class T> struct optional_delete_ctor_base<T, false, true> {
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(const optional_delete_ctor_base &) = delete;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(optional_delete_ctor_base &&) noexcept = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(const optional_delete_ctor_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(optional_delete_ctor_base &&) noexcept = default;
};

template <class T> struct optional_delete_ctor_base<T, false, false> {
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(const optional_delete_ctor_base &) = delete;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base(optional_delete_ctor_base &&) noexcept = delete;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(const optional_delete_ctor_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_ctor_base &
operator=(optional_delete_ctor_base &&) noexcept = default;
};

template <class T,
bool EnableCopy = (std::is_copy_constructible<T>::value &&
std::is_copy_assignable<T>::value),
bool EnableMove = (std::is_move_constructible<T>::value &&
std::is_move_assignable<T>::value)>
struct optional_delete_assign_base {
__hydra_thrust_exec_check_disable__
optional_delete_assign_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(const optional_delete_assign_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(optional_delete_assign_base &&) noexcept =
default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(const optional_delete_assign_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(optional_delete_assign_base &&) noexcept = default;
};

template <class T> struct optional_delete_assign_base<T, true, false> {
__hydra_thrust_exec_check_disable__
optional_delete_assign_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(const optional_delete_assign_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(optional_delete_assign_base &&) noexcept =
default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(const optional_delete_assign_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(optional_delete_assign_base &&) noexcept = delete;
};

template <class T> struct optional_delete_assign_base<T, false, true> {
__hydra_thrust_exec_check_disable__
optional_delete_assign_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(const optional_delete_assign_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(optional_delete_assign_base &&) noexcept =
default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(const optional_delete_assign_base &) = delete;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(optional_delete_assign_base &&) noexcept = default;
};

template <class T> struct optional_delete_assign_base<T, false, false> {
__hydra_thrust_exec_check_disable__
optional_delete_assign_base() = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(const optional_delete_assign_base &) = default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base(optional_delete_assign_base &&) noexcept =
default;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(const optional_delete_assign_base &) = delete;
__hydra_thrust_exec_check_disable__
optional_delete_assign_base &
operator=(optional_delete_assign_base &&) noexcept = delete;
};

} 

struct nullopt_t {
struct do_not_use {};
__host__ __device__
constexpr explicit nullopt_t(do_not_use, do_not_use) noexcept {}
};
static constexpr nullopt_t nullopt{nullopt_t::do_not_use{},
nullopt_t::do_not_use{}};

class bad_optional_access : public std::exception {
public:
bad_optional_access() = default;
__host__
const char *what() const noexcept { return "Optional has no value"; }
};

template <class T>
class optional : private detail::optional_move_assign_base<T>,
private detail::optional_delete_ctor_base<T>,
private detail::optional_delete_assign_base<T> {
using base = detail::optional_move_assign_base<T>;

static_assert(!std::is_same<T, in_place_t>::value,
"instantiation of optional with in_place_t is ill-formed");
static_assert(!std::is_same<detail::decay_t<T>, nullopt_t>::value,
"instantiation of optional with nullopt_t is ill-formed");

public:
#if defined(HYDRA_THRUST_OPTIONAL_CPP14) && !defined(HYDRA_THRUST_OPTIONAL_GCC49) &&               \
!defined(HYDRA_THRUST_OPTIONAL_GCC54) && !defined(HYDRA_THRUST_OPTIONAL_GCC55)
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto and_then(F &&f) & {
using result = detail::invoke_result_t<F, T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto and_then(F &&f) && {
using result = detail::invoke_result_t<F, T &&>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto and_then(F &&f) const & {
using result = detail::invoke_result_t<F, const T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto and_then(F &&f) const && {
using result = detail::invoke_result_t<F, const T &&>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: result(nullopt);
}
#endif
#else
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR detail::invoke_result_t<F, T &> and_then(F &&f) & {
using result = detail::invoke_result_t<F, T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR detail::invoke_result_t<F, T &&> and_then(F &&f) && {
using result = detail::invoke_result_t<F, T &&>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr detail::invoke_result_t<F, const T &> and_then(F &&f) const & {
using result = detail::invoke_result_t<F, const T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr detail::invoke_result_t<F, const T &&> and_then(F &&f) const && {
using result = detail::invoke_result_t<F, const T &&>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: result(nullopt);
}
#endif
#endif

#if defined(HYDRA_THRUST_OPTIONAL_CPP14) && !defined(HYDRA_THRUST_OPTIONAL_GCC49) &&               \
!defined(HYDRA_THRUST_OPTIONAL_GCC54) && !defined(HYDRA_THRUST_OPTIONAL_GCC55)
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto map(F &&f) & {
return optional_map_impl(*this, std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto map(F &&f) && {
return optional_map_impl(std::move(*this), std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto map(F &&f) const & {
return optional_map_impl(*this, std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto map(F &&f) const && {
return optional_map_impl(std::move(*this), std::forward<F>(f));
}
#else
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR decltype(optional_map_impl(std::declval<optional &>(),
std::declval<F &&>()))
map(F &&f) & {
return optional_map_impl(*this, std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR decltype(optional_map_impl(std::declval<optional &&>(),
std::declval<F &&>()))
map(F &&f) && {
return optional_map_impl(std::move(*this), std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr decltype(optional_map_impl(std::declval<const optional &>(),
std::declval<F &&>()))
map(F &&f) const & {
return optional_map_impl(*this, std::forward<F>(f));
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr decltype(optional_map_impl(std::declval<const optional &&>(),
std::declval<F &&>()))
map(F &&f) const && {
return optional_map_impl(std::move(*this), std::forward<F>(f));
}
#endif
#endif

__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) & {
if (has_value())
return *this;

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) & {
return has_value() ? *this : std::forward<F>(f)();
}

__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) && {
if (has_value())
return std::move(*this);

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) && {
return has_value() ? std::move(*this) : std::forward<F>(f)();
}

__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) const & {
if (has_value())
return *this;

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) const & {
return has_value() ? *this : std::forward<F>(f)();
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) const && {
if (has_value())
return std::move(*this);

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) const && {
return has_value() ? std::move(*this) : std::forward<F>(f)();
}
#endif

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u);
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u);
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) const & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) const && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u);
}
#endif

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u)();
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u)();
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) const & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u)();
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) const && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u)();
}
#endif

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
constexpr optional<typename std::decay<U>::type> conjunction(U &&u) const {
using result = optional<detail::decay_t<U>>;
return has_value() ? result{u} : result{nullopt};
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(const optional &rhs) & {
return has_value() ? *this : rhs;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(const optional &rhs) const & {
return has_value() ? *this : rhs;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(const optional &rhs) && {
return has_value() ? std::move(*this) : rhs;
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(const optional &rhs) const && {
return has_value() ? std::move(*this) : rhs;
}
#endif

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(optional &&rhs) & {
return has_value() ? *this : std::move(rhs);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(optional &&rhs) const & {
return has_value() ? *this : std::move(rhs);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(optional &&rhs) && {
return has_value() ? std::move(*this) : std::move(rhs);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(optional &&rhs) const && {
return has_value() ? std::move(*this) : std::move(rhs);
}
#endif

__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() & {
optional ret = *this;
reset();
return ret;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() const & {
optional ret = *this;
reset();
return ret;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() && {
optional ret = std::move(*this);
reset();
return ret;
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() const && {
optional ret = std::move(*this);
reset();
return ret;
}
#endif

using value_type = T;

__hydra_thrust_exec_check_disable__
constexpr optional() noexcept = default;

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional(nullopt_t) noexcept {}

__hydra_thrust_exec_check_disable__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional(const optional &rhs) = default;

__hydra_thrust_exec_check_disable__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional(optional &&rhs) = default;

__hydra_thrust_exec_check_disable__
template <class... Args>
__host__ __device__
constexpr explicit optional(
detail::enable_if_t<std::is_constructible<T, Args...>::value, in_place_t>,
Args &&... args)
: base(in_place, std::forward<Args>(args)...) {}

__hydra_thrust_exec_check_disable__
template <class U, class... Args>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR explicit optional(
detail::enable_if_t<std::is_constructible<T, std::initializer_list<U> &,
Args &&...>::value,
in_place_t>,
std::initializer_list<U> il, Args &&... args) {
this->construct(il, std::forward<Args>(args)...);
}

__hydra_thrust_exec_check_disable__
template <
class U = T,
detail::enable_if_t<std::is_convertible<U &&, T>::value> * = nullptr,
detail::enable_forward_value<T, U> * = nullptr>
__host__ __device__
constexpr optional(U &&u) : base(in_place, std::forward<U>(u)) {}

__hydra_thrust_exec_check_disable__
template <
class U = T,
detail::enable_if_t<!std::is_convertible<U &&, T>::value> * = nullptr,
detail::enable_forward_value<T, U> * = nullptr>
__host__ __device__
constexpr explicit optional(U &&u) : base(in_place, std::forward<U>(u)) {}

__hydra_thrust_exec_check_disable__
template <
class U, detail::enable_from_other<T, U, const U &> * = nullptr,
detail::enable_if_t<std::is_convertible<const U &, T>::value> * = nullptr>
__host__ __device__
optional(const optional<U> &rhs) {
this->construct(*rhs);
}

__hydra_thrust_exec_check_disable__
template <class U, detail::enable_from_other<T, U, const U &> * = nullptr,
detail::enable_if_t<!std::is_convertible<const U &, T>::value> * =
nullptr>
__host__ __device__
explicit optional(const optional<U> &rhs) {
this->construct(*rhs);
}

__hydra_thrust_exec_check_disable__
template <
class U, detail::enable_from_other<T, U, U &&> * = nullptr,
detail::enable_if_t<std::is_convertible<U &&, T>::value> * = nullptr>
__host__ __device__
optional(optional<U> &&rhs) {
this->construct(std::move(*rhs));
}

__hydra_thrust_exec_check_disable__
template <
class U, detail::enable_from_other<T, U, U &&> * = nullptr,
detail::enable_if_t<!std::is_convertible<U &&, T>::value> * = nullptr>
__host__ __device__
explicit optional(optional<U> &&rhs) {
this->construct(std::move(*rhs));
}

__hydra_thrust_exec_check_disable__
~optional() = default;

__hydra_thrust_exec_check_disable__
__host__ __device__
optional &operator=(nullopt_t) noexcept {
if (has_value()) {
this->m_value.~T();
this->m_has_value = false;
}

return *this;
}

__hydra_thrust_exec_check_disable__
optional &operator=(const optional &rhs) = default;

__hydra_thrust_exec_check_disable__
optional &operator=(optional &&rhs) = default;

__hydra_thrust_exec_check_disable__
template <class U = T, detail::enable_assign_forward<T, U> * = nullptr>
__host__ __device__
optional &operator=(U &&u) {
if (has_value()) {
this->m_value = std::forward<U>(u);
} else {
this->construct(std::forward<U>(u));
}

return *this;
}

__hydra_thrust_exec_check_disable__
template <class U,
detail::enable_assign_from_other<T, U, const U &> * = nullptr>
__host__ __device__
optional &operator=(const optional<U> &rhs) {
if (has_value()) {
if (rhs.has_value()) {
this->m_value = *rhs;
} else {
this->hard_reset();
}
}

if (rhs.has_value()) {
this->construct(*rhs);
}

return *this;
}

__hydra_thrust_exec_check_disable__
template <class U, detail::enable_assign_from_other<T, U, U> * = nullptr>
__host__ __device__
optional &operator=(optional<U> &&rhs) {
if (has_value()) {
if (rhs.has_value()) {
this->m_value = std::move(*rhs);
} else {
this->hard_reset();
}
}

if (rhs.has_value()) {
this->construct(std::move(*rhs));
}

return *this;
}

__hydra_thrust_exec_check_disable__
template <class... Args>
__host__ __device__
T &emplace(Args &&... args) {
static_assert(std::is_constructible<T, Args &&...>::value,
"T must be constructible with Args");

*this = nullopt;
this->construct(std::forward<Args>(args)...);
return value();
}

__hydra_thrust_exec_check_disable__
template <class U, class... Args>
__host__ __device__
detail::enable_if_t<
std::is_constructible<T, std::initializer_list<U> &, Args &&...>::value,
T &>
emplace(std::initializer_list<U> il, Args &&... args) {
*this = nullopt;
this->construct(il, std::forward<Args>(args)...);
return value();    
}

__hydra_thrust_exec_check_disable__
__host__ __device__
void
swap(optional &rhs) noexcept(std::is_nothrow_move_constructible<T>::value
&&detail::is_nothrow_swappable<T>::value) {
if (has_value()) {
if (rhs.has_value()) {
using hydra_thrust::swap;
swap(**this, *rhs);
} else {
new (addressof(rhs.m_value)) T(std::move(this->m_value));
this->m_value.T::~T();
}
} else if (rhs.has_value()) {
new (addressof(this->m_value)) T(std::move(rhs.m_value));
rhs.m_value.T::~T();
}
}

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr const T *operator->() const {
return addressof(this->m_value);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T *operator->() {
return addressof(this->m_value);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &operator*() & { return this->m_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr const T &operator*() const & { return this->m_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &&operator*() && {
return std::move(this->m_value);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr const T &&operator*() const && { return std::move(this->m_value); }
#endif

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr bool has_value() const noexcept { return this->m_has_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr explicit operator bool() const noexcept {
return this->m_has_value;
}

__host__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &value() & {
if (has_value())
return this->m_value;
throw bad_optional_access();
}
__host__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR const T &value() const & {
if (has_value())
return this->m_value;
throw bad_optional_access();
}
__host__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &&value() && {
if (has_value())
return std::move(this->m_value);
throw bad_optional_access();
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__host__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR const T &&value() const && {
if (has_value())
return std::move(this->m_value);
throw bad_optional_access();
}
#endif

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
constexpr T value_or(U &&u) const & {
static_assert(std::is_copy_constructible<T>::value &&
std::is_convertible<U &&, T>::value,
"T must be copy constructible and convertible from U");
return has_value() ? **this : static_cast<T>(std::forward<U>(u));
}

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T value_or(U &&u) && {
static_assert(std::is_move_constructible<T>::value &&
std::is_convertible<U &&, T>::value,
"T must be move constructible and convertible from U");
return has_value() ? **this : static_cast<T>(std::forward<U>(u));
}

__hydra_thrust_exec_check_disable__
__host__ __device__
void reset() noexcept {
if (has_value()) {
this->m_value.~T();
this->m_has_value = false;
}
}
};

__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator==(const optional<T> &lhs,
const optional<U> &rhs) {
return lhs.has_value() == rhs.has_value() &&
(!lhs.has_value() || *lhs == *rhs);
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator!=(const optional<T> &lhs,
const optional<U> &rhs) {
return lhs.has_value() != rhs.has_value() ||
(lhs.has_value() && *lhs != *rhs);
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator<(const optional<T> &lhs,
const optional<U> &rhs) {
return rhs.has_value() && (!lhs.has_value() || *lhs < *rhs);
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator>(const optional<T> &lhs,
const optional<U> &rhs) {
return lhs.has_value() && (!rhs.has_value() || *lhs > *rhs);
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator<=(const optional<T> &lhs,
const optional<U> &rhs) {
return !lhs.has_value() || (rhs.has_value() && *lhs <= *rhs);
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator>=(const optional<T> &lhs,
const optional<U> &rhs) {
return !rhs.has_value() || (lhs.has_value() && *lhs >= *rhs);
}

__hydra_thrust_exec_check_disable__
template <class T>
__host__ __device__
inline constexpr bool operator==(const optional<T> &lhs, nullopt_t) noexcept {
return !lhs.has_value();
}
__hydra_thrust_exec_check_disable__
template <class T>
__host__ __device__
inline constexpr bool operator==(nullopt_t, const optional<T> &rhs) noexcept {
return !rhs.has_value();
}
__hydra_thrust_exec_check_disable__
template <class T>
__host__ __device__
inline constexpr bool operator!=(const optional<T> &lhs, nullopt_t) noexcept {
return lhs.has_value();
}
__hydra_thrust_exec_check_disable__
template <class T>
__host__ __device__
inline constexpr bool operator!=(nullopt_t, const optional<T> &rhs) noexcept {
return rhs.has_value();
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator<(const optional<T> &, nullopt_t) noexcept {
return false;
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator<(nullopt_t, const optional<T> &rhs) noexcept {
return rhs.has_value();
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator<=(const optional<T> &lhs, nullopt_t) noexcept {
return !lhs.has_value();
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator<=(nullopt_t, const optional<T> &) noexcept {
return true;
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator>(const optional<T> &lhs, nullopt_t) noexcept {
return lhs.has_value();
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator>(nullopt_t, const optional<T> &) noexcept {
return false;
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator>=(const optional<T> &, nullopt_t) noexcept {
return true;
}
__hydra_thrust_exec_check_disable__                                                    
template <class T>                                                               
__host__ __device__       
inline constexpr bool operator>=(nullopt_t, const optional<T> &rhs) noexcept {
return !rhs.has_value();
}

__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator==(const optional<T> &lhs, const U &rhs) {
return lhs.has_value() ? *lhs == rhs : false;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator==(const U &lhs, const optional<T> &rhs) {
return rhs.has_value() ? lhs == *rhs : false;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator!=(const optional<T> &lhs, const U &rhs) {
return lhs.has_value() ? *lhs != rhs : true;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator!=(const U &lhs, const optional<T> &rhs) {
return rhs.has_value() ? lhs != *rhs : true;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator<(const optional<T> &lhs, const U &rhs) {
return lhs.has_value() ? *lhs < rhs : true;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator<(const U &lhs, const optional<T> &rhs) {
return rhs.has_value() ? lhs < *rhs : false;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator<=(const optional<T> &lhs, const U &rhs) {
return lhs.has_value() ? *lhs <= rhs : true;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator<=(const U &lhs, const optional<T> &rhs) {
return rhs.has_value() ? lhs <= *rhs : false;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator>(const optional<T> &lhs, const U &rhs) {
return lhs.has_value() ? *lhs > rhs : false;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator>(const U &lhs, const optional<T> &rhs) {
return rhs.has_value() ? lhs > *rhs : true;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator>=(const optional<T> &lhs, const U &rhs) {
return lhs.has_value() ? *lhs >= rhs : false;
}
__hydra_thrust_exec_check_disable__
template <class T, class U>
__host__ __device__
inline constexpr bool operator>=(const U &lhs, const optional<T> &rhs) {
return rhs.has_value() ? lhs >= *rhs : true;
}

__hydra_thrust_exec_check_disable__
template <class T,
detail::enable_if_t<std::is_move_constructible<T>::value> * = nullptr,
detail::enable_if_t<detail::is_swappable<T>::value> * = nullptr>
__host__ __device__
void swap(optional<T> &lhs,
optional<T> &rhs) noexcept(noexcept(lhs.swap(rhs))) {
return lhs.swap(rhs);
}

namespace detail {
struct i_am_secret {};
} 

__hydra_thrust_exec_check_disable__
template <class T = detail::i_am_secret, class U,
class Ret =
detail::conditional_t<std::is_same<T, detail::i_am_secret>::value,
detail::decay_t<U>, T>>
__host__ __device__
inline constexpr optional<Ret> make_optional(U &&v) {
return optional<Ret>(std::forward<U>(v));
}

__hydra_thrust_exec_check_disable__
template <class T, class... Args>
__host__ __device__
inline constexpr optional<T> make_optional(Args &&... args) {
return optional<T>(in_place, std::forward<Args>(args)...);
}
__hydra_thrust_exec_check_disable__
template <class T, class U, class... Args>
__host__ __device__
inline constexpr optional<T> make_optional(std::initializer_list<U> il,
Args &&... args) {
return optional<T>(in_place, il, std::forward<Args>(args)...);
}

#if __cplusplus >= 201703L
template <class T> optional(T)->optional<T>;
#endif

namespace detail {
#ifdef HYDRA_THRUST_OPTIONAL_CPP14
__hydra_thrust_exec_check_disable__
template <class Opt, class F,
class Ret = decltype(detail::invoke(std::declval<F>(),
*std::declval<Opt>())),
detail::enable_if_t<!std::is_void<Ret>::value> * = nullptr>
__host__ __device__
constexpr auto optional_map_impl(Opt &&opt, F &&f) {
return opt.has_value()
? detail::invoke(std::forward<F>(f), *std::forward<Opt>(opt))
: optional<Ret>(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class Opt, class F,
class Ret = decltype(detail::invoke(std::declval<F>(),
*std::declval<Opt>())),
detail::enable_if_t<std::is_void<Ret>::value> * = nullptr>
__host__ __device__
auto optional_map_impl(Opt &&opt, F &&f) {
if (opt.has_value()) {
detail::invoke(std::forward<F>(f), *std::forward<Opt>(opt));
return make_optional(monostate{});
}

return optional<monostate>(nullopt);
}
#else
__hydra_thrust_exec_check_disable__
template <class Opt, class F,
class Ret = decltype(detail::invoke(std::declval<F>(),
*std::declval<Opt>())),
detail::enable_if_t<!std::is_void<Ret>::value> * = nullptr>
__host__ __device__
constexpr auto optional_map_impl(Opt &&opt, F &&f) -> optional<Ret> {
return opt.has_value()
? detail::invoke(std::forward<F>(f), *std::forward<Opt>(opt))
: optional<Ret>(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class Opt, class F,
class Ret = decltype(detail::invoke(std::declval<F>(),
*std::declval<Opt>())),
detail::enable_if_t<std::is_void<Ret>::value> * = nullptr>
__host__ __device__
auto optional_map_impl(Opt &&opt, F &&f) -> optional<monostate> {
if (opt.has_value()) {
detail::invoke(std::forward<F>(f), *std::forward<Opt>(opt));
return monostate{};
}

return nullopt;
}
#endif
} 

template <class T> class optional<T &> {
public:
#if defined(HYDRA_THRUST_OPTIONAL_CPP14) && !defined(HYDRA_THRUST_OPTIONAL_GCC49) &&               \
!defined(HYDRA_THRUST_OPTIONAL_GCC54) && !defined(HYDRA_THRUST_OPTIONAL_GCC55)
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto and_then(F &&f) & {
using result = detail::invoke_result_t<F, T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto and_then(F &&f) && {
using result = detail::invoke_result_t<F, T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto and_then(F &&f) const & {
using result = detail::invoke_result_t<F, const T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto and_then(F &&f) const && {
using result = detail::invoke_result_t<F, const T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}
#endif
#else
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR detail::invoke_result_t<F, T &> and_then(F &&f) & {
using result = detail::invoke_result_t<F, T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR detail::invoke_result_t<F, T &> and_then(F &&f) && {
using result = detail::invoke_result_t<F, T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr detail::invoke_result_t<F, const T &> and_then(F &&f) const & {
using result = detail::invoke_result_t<F, const T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr detail::invoke_result_t<F, const T &> and_then(F &&f) const && {
using result = detail::invoke_result_t<F, const T &>;
static_assert(detail::is_optional<result>::value,
"F must return an optional");

return has_value() ? detail::invoke(std::forward<F>(f), **this)
: result(nullopt);
}
#endif
#endif

#if defined(HYDRA_THRUST_OPTIONAL_CPP14) && !defined(HYDRA_THRUST_OPTIONAL_GCC49) &&               \
!defined(HYDRA_THRUST_OPTIONAL_GCC54) && !defined(HYDRA_THRUST_OPTIONAL_GCC55)
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto map(F &&f) & {
return detail::optional_map_impl(*this, std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR auto map(F &&f) && {
return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto map(F &&f) const & {
return detail::optional_map_impl(*this, std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr auto map(F &&f) const && {
return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
}
#else
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR decltype(detail::optional_map_impl(std::declval<optional &>(),
std::declval<F &&>()))
map(F &&f) & {
return detail::optional_map_impl(*this, std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR decltype(detail::optional_map_impl(std::declval<optional &&>(),
std::declval<F &&>()))
map(F &&f) && {
return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
}

__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr decltype(detail::optional_map_impl(std::declval<const optional &>(),
std::declval<F &&>()))
map(F &&f) const & {
return detail::optional_map_impl(*this, std::forward<F>(f));
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F>
__host__ __device__
constexpr decltype(detail::optional_map_impl(std::declval<const optional &&>(),
std::declval<F &&>()))
map(F &&f) const && {
return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
}
#endif
#endif

__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T>
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) & {
if (has_value())
return *this;

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T>
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) & {
return has_value() ? *this : std::forward<F>(f)();
}

__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) && {
if (has_value())
return std::move(*this);

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) && {
return has_value() ? std::move(*this) : std::forward<F>(f)();
}

__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) const & {
if (has_value())
return *this;

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR or_else(F &&f) const & {
return has_value() ? *this : std::forward<F>(f)();
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F, detail::enable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) const && {
if (has_value())
return std::move(*this);

std::forward<F>(f)();
return nullopt;
}

__hydra_thrust_exec_check_disable__
template <class F, detail::disable_if_ret_void<F> * = nullptr>
__host__ __device__
optional<T> or_else(F &&f) const && {
return has_value() ? std::move(*this) : std::forward<F>(f)();
}
#endif

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u);
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u);
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) const & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
U map_or(F &&f, U &&u) const && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u);
}
#endif

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u)();
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u)();
}

__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) const & {
return has_value() ? detail::invoke(std::forward<F>(f), **this)
: std::forward<U>(u)();
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
template <class F, class U>
__host__ __device__
detail::invoke_result_t<U> map_or_else(F &&f, U &&u) const && {
return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this))
: std::forward<U>(u)();
}
#endif

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
constexpr optional<typename std::decay<U>::type> conjunction(U &&u) const {
using result = optional<detail::decay_t<U>>;
return has_value() ? result{u} : result{nullopt};
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(const optional &rhs) & {
return has_value() ? *this : rhs;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(const optional &rhs) const & {
return has_value() ? *this : rhs;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(const optional &rhs) && {
return has_value() ? std::move(*this) : rhs;
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(const optional &rhs) const && {
return has_value() ? std::move(*this) : rhs;
}
#endif

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(optional &&rhs) & {
return has_value() ? *this : std::move(rhs);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(optional &&rhs) const & {
return has_value() ? *this : std::move(rhs);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional disjunction(optional &&rhs) && {
return has_value() ? std::move(*this) : std::move(rhs);
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional disjunction(optional &&rhs) const && {
return has_value() ? std::move(*this) : std::move(rhs);
}
#endif

__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() & {
optional ret = *this;
reset();
return ret;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() const & {
optional ret = *this;
reset();
return ret;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() && {
optional ret = std::move(*this);
reset();
return ret;
}

#ifndef HYDRA_THRUST_OPTIONAL_NO_CONSTRR
__hydra_thrust_exec_check_disable__
__host__ __device__
optional take() const && {
optional ret = std::move(*this);
reset();
return ret;
}
#endif

using value_type = T &;

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional() noexcept : m_value(nullptr) {}

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr optional(nullopt_t) noexcept : m_value(nullptr) {}

__hydra_thrust_exec_check_disable__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional(const optional &rhs) noexcept = default;

__hydra_thrust_exec_check_disable__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR optional(optional &&rhs) = default;

__hydra_thrust_exec_check_disable__
template <class U = T,
detail::enable_if_t<!detail::is_optional<detail::decay_t<U>>::value>
* = nullptr>
__host__ __device__
constexpr optional(U &&u) : m_value(addressof(u)) {
static_assert(std::is_lvalue_reference<U>::value, "U must be an lvalue");
}

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
constexpr explicit optional(const optional<U> &rhs) : optional(*rhs) {}

__hydra_thrust_exec_check_disable__
~optional() = default;

__hydra_thrust_exec_check_disable__
__host__ __device__
optional &operator=(nullopt_t) noexcept {
m_value = nullptr;
return *this;
}

__hydra_thrust_exec_check_disable__
optional &operator=(const optional &rhs) = default;

__hydra_thrust_exec_check_disable__
template <class U = T,
detail::enable_if_t<!detail::is_optional<detail::decay_t<U>>::value>
* = nullptr>
__host__ __device__
optional &operator=(U &&u) {
static_assert(std::is_lvalue_reference<U>::value, "U must be an lvalue");
m_value = addressof(u);
return *this;
}

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
optional &operator=(const optional<U> &rhs) {
m_value = addressof(rhs.value());
return *this;
}

__hydra_thrust_exec_check_disable__
template <class... Args>
__host__ __device__
T &emplace(Args &&... args) noexcept {
static_assert(std::is_constructible<T, Args &&...>::value,
"T must be constructible with Args");

*this = nullopt;
this->construct(std::forward<Args>(args)...);
}

__hydra_thrust_exec_check_disable__
__host__ __device__
void swap(optional &rhs) noexcept { std::swap(m_value, rhs.m_value); }

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr const T *operator->() const { return m_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T *operator->() { return m_value; }

__hydra_thrust_exec_check_disable__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &operator*() { return *m_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr const T &operator*() const { return *m_value; }

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr bool has_value() const noexcept { return m_value != nullptr; }

__hydra_thrust_exec_check_disable__
__host__ __device__
constexpr explicit operator bool() const noexcept {
return m_value != nullptr;
}

__host__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T &value() {
if (has_value())
return *m_value;
throw bad_optional_access();
}
__host__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR const T &value() const {
if (has_value())
return *m_value;
throw bad_optional_access();
}

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
constexpr T value_or(U &&u) const & {
static_assert(std::is_copy_constructible<T>::value &&
std::is_convertible<U &&, T>::value,
"T must be copy constructible and convertible from U");
return has_value() ? **this : static_cast<T>(std::forward<U>(u));
}

__hydra_thrust_exec_check_disable__
template <class U>
__host__ __device__
HYDRA_THRUST_OPTIONAL_CPP11_CONSTEXPR T value_or(U &&u) && {
static_assert(std::is_move_constructible<T>::value &&
std::is_convertible<U &&, T>::value,
"T must be move constructible and convertible from U");
return has_value() ? **this : static_cast<T>(std::forward<U>(u));
}

__hydra_thrust_exec_check_disable__
void reset() noexcept { m_value = nullptr; }

private:
T *m_value;
};

HYDRA_THRUST_END_NS

namespace std {
template <class T> struct hash<hydra_thrust::optional<T>> {
__hydra_thrust_exec_check_disable__
__host__ __device__
::std::size_t operator()(const hydra_thrust::optional<T> &o) const {
if (!o.has_value())
return 0;

return std::hash<hydra_thrust::detail::remove_const_t<T>>()(*o);
}
};
} 

#endif 

