

#ifndef BOOST_OUTCOME_BASIC_RESULT_HPP
#define BOOST_OUTCOME_BASIC_RESULT_HPP

#include "config.hpp"
#include "convert.hpp"
#include "detail/basic_result_final.hpp"

#include "policy/all_narrow.hpp"
#include "policy/terminate.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"  
#endif

BOOST_OUTCOME_V2_NAMESPACE_EXPORT_BEGIN

template <class R, class S, class NoValuePolicy>  
class basic_result;

namespace detail
{
template <class value_type, class error_type> struct result_predicates
{
static constexpr bool implicit_constructors_enabled =  
!(trait::is_error_type<std::decay_t<value_type>>::value &&
trait::is_error_type<std::decay_t<error_type>>::value)  
&& ((!detail::is_implicitly_constructible<value_type, error_type> &&
!detail::is_implicitly_constructible<error_type, value_type>)       
|| (trait::is_error_type<std::decay_t<error_type>>::value            
&& !detail::is_implicitly_constructible<error_type, value_type>  
&& std::is_integral<value_type>::value));                        

template <class T>
static constexpr bool enable_value_converting_constructor =                                                      
implicit_constructors_enabled                                                                                    
&& !is_in_place_type_t<std::decay_t<T>>::value                                                                   
&& !trait::is_error_type_enum<error_type, std::decay_t<T>>::value                                                
&& ((detail::is_implicitly_constructible<value_type, T> && !detail::is_implicitly_constructible<error_type, T>)  
|| (std::is_same<value_type, std::decay_t<T>>::value                                                         
&& detail::is_implicitly_constructible<value_type, T>) );  


template <class T>
static constexpr bool enable_error_converting_constructor =                                                      
implicit_constructors_enabled                                                                                    
&& !is_in_place_type_t<std::decay_t<T>>::value                                                                   
&& !trait::is_error_type_enum<error_type, std::decay_t<T>>::value                                                
&& ((!detail::is_implicitly_constructible<value_type, T> && detail::is_implicitly_constructible<error_type, T>)  
|| (std::is_same<error_type, std::decay_t<T>>::value                                                         
&& detail::is_implicitly_constructible<error_type, T>) );  

template <class ErrorCondEnum>
static constexpr bool enable_error_condition_converting_constructor =         
!is_in_place_type_t<std::decay_t<ErrorCondEnum>>::value                       
&& trait::is_error_type_enum<error_type, std::decay_t<ErrorCondEnum>>::value  
;  

template <class T, class U, class V>
static constexpr bool enable_compatible_conversion =  
(std::is_void<T>::value ||
detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>)  
&&(std::is_void<U>::value ||
detail::is_explicitly_constructible<error_type, typename basic_result<T, U, V>::error_type>)  
;

template <class T, class U, class V>
static constexpr bool enable_make_error_code_compatible_conversion =  
trait::is_error_code_available<std::decay_t<error_type>>::value       
&& !enable_compatible_conversion<T, U, V>                             
&& (std::is_void<T>::value ||
detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>)  
&&detail::is_explicitly_constructible<error_type,
typename trait::is_error_code_available<U>::type>;  

template <class T, class U, class V>
static constexpr bool enable_make_exception_ptr_compatible_conversion =  
trait::is_exception_ptr_available<std::decay_t<error_type>>::value       
&& !enable_compatible_conversion<T, U, V>                                
&& (std::is_void<T>::value ||
detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>)         
&&detail::is_explicitly_constructible<error_type, typename trait::is_exception_ptr_available<U>::type>;  

struct disable_inplace_value_error_constructor;
template <class... Args>
using choose_inplace_value_error_constructor = std::conditional_t<                               
detail::is_constructible<value_type, Args...> && detail::is_constructible<error_type, Args...>,  
disable_inplace_value_error_constructor,                                                         
std::conditional_t<                                                                              
detail::is_constructible<value_type, Args...>,                                                   
value_type,                                                                                      
std::conditional_t<                                                                              
detail::is_constructible<error_type, Args...>,                                                   
error_type,                                                                                      
disable_inplace_value_error_constructor>>>;
template <class... Args>
static constexpr bool enable_inplace_value_error_constructor =
implicit_constructors_enabled  
&& !std::is_same<choose_inplace_value_error_constructor<Args...>, disable_inplace_value_error_constructor>::value;
};

template <class T, class U> constexpr inline const U &extract_value_from_success(const success_type<U> &v) { return v.value(); }
template <class T, class U> constexpr inline U &&extract_value_from_success(success_type<U> &&v) { return static_cast<success_type<U> &&>(v).value(); }
template <class T> constexpr inline T extract_value_from_success(const success_type<void> & ) { return T{}; }

template <class T, class U, class V> constexpr inline const U &extract_error_from_failure(const failure_type<U, V> &v) { return v.error(); }
template <class T, class U, class V> constexpr inline U &&extract_error_from_failure(failure_type<U, V> &&v)
{
return static_cast<failure_type<U, V> &&>(v).error();
}
template <class T, class V> constexpr inline T extract_error_from_failure(const failure_type<void, V> & ) { return T{}; }

template <class T> struct is_basic_result
{
static constexpr bool value = false;
};
template <class R, class S, class T> struct is_basic_result<basic_result<R, S, T>>
{
static constexpr bool value = true;
};
}  


template <class T> using is_basic_result = detail::is_basic_result<std::decay_t<T>>;

template <class T> static constexpr bool is_basic_result_v = detail::is_basic_result<std::decay_t<T>>::value;

namespace concepts
{
#if defined(__cpp_concepts)

template <class U>
concept BOOST_OUTCOME_GCC6_CONCEPT_BOOL basic_result =
BOOST_OUTCOME_V2_NAMESPACE::is_basic_result<U>::value ||
(requires(U v) { BOOST_OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>(v); } &&    
detail::convertible<U, BOOST_OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>> &&  
detail::base_of<BOOST_OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>, U>);
#else
namespace detail
{
inline no_match match_basic_result(...);
template <class R, class S, class NVP, class T,                                                                      
typename = typename T::value_type,                                                                         
typename = typename T::error_type,                                                                         
typename = typename T::no_value_policy_type,                                                               
typename std::enable_if_t<std::is_convertible<T, BOOST_OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP>>::value &&  
std::is_base_of<BOOST_OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP>, T>::value,
bool> = true>
inline BOOST_OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP> match_basic_result(BOOST_OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP> &&, T &&);

template <class U>
static constexpr bool basic_result = BOOST_OUTCOME_V2_NAMESPACE::is_basic_result<U>::value ||
!std::is_same<no_match, decltype(match_basic_result(std::declval<BOOST_OUTCOME_V2_NAMESPACE::detail::devoid<U>>(),
std::declval<BOOST_OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
}  

template <class U> static constexpr bool basic_result = detail::basic_result<U>;
#endif
}  


namespace hooks
{

template <class T, class U> constexpr inline void hook_result_construction(T * , U && ) noexcept {}

template <class T, class U> constexpr inline void hook_result_copy_construction(T * , U && ) noexcept {}

template <class T, class U> constexpr inline void hook_result_move_construction(T * , U && ) noexcept {}

template <class T, class U, class... Args>
constexpr inline void hook_result_in_place_construction(T * , in_place_type_t<U> , Args &&... ) noexcept
{
}


template <class R, class S, class NoValuePolicy> constexpr inline uint16_t spare_storage(const detail::basic_result_storage<R, S, NoValuePolicy> *r) noexcept
{
return r->_state._status.spare_storage_value;
}

template <class R, class S, class NoValuePolicy>
constexpr inline void set_spare_storage(detail::basic_result_storage<R, S, NoValuePolicy> *r, uint16_t v) noexcept
{
r->_state._status.spare_storage_value = v;
}
}  


template <class R, class S, class NoValuePolicy>  
class BOOST_OUTCOME_NODISCARD basic_result : public detail::basic_result_final<R, S, NoValuePolicy>
{
static_assert(trait::type_can_be_used_in_basic_result<R>, "The type R cannot be used in a basic_result");
static_assert(trait::type_can_be_used_in_basic_result<S>, "The type S cannot be used in a basic_result");
static_assert(std::is_void<S>::value || std::is_default_constructible<S>::value, "The type S must be void or default constructible");

using base = detail::basic_result_final<R, S, NoValuePolicy>;

struct implicit_constructors_disabled_tag
{
};
struct value_converting_constructor_tag
{
};
struct error_converting_constructor_tag
{
};
struct error_condition_converting_constructor_tag
{
};
struct explicit_valueornone_converting_constructor_tag
{
};
struct explicit_valueorerror_converting_constructor_tag
{
};
struct explicit_compatible_copy_conversion_tag
{
};
struct explicit_compatible_move_conversion_tag
{
};
struct explicit_make_error_code_compatible_copy_conversion_tag
{
};
struct explicit_make_error_code_compatible_move_conversion_tag
{
};
struct explicit_make_exception_ptr_compatible_copy_conversion_tag
{
};
struct explicit_make_exception_ptr_compatible_move_conversion_tag
{
};

public:
using value_type = R;
using error_type = S;
using no_value_policy_type = NoValuePolicy;

using value_type_if_enabled = typename base::_value_type;
using error_type_if_enabled = typename base::_error_type;

template <class T, class U = S, class V = NoValuePolicy> using rebind = basic_result<T, U, V>;

protected:
struct predicate
{
using base = detail::result_predicates<value_type, error_type>;

static constexpr bool constructors_enabled = !std::is_same<std::decay_t<value_type>, std::decay_t<error_type>>::value;

static constexpr bool implicit_constructors_enabled = constructors_enabled && base::implicit_constructors_enabled;

template <class T>
static constexpr bool enable_value_converting_constructor =  
constructors_enabled                                         
&& !std::is_same<std::decay_t<T>, basic_result>::value       
&& base::template enable_value_converting_constructor<T>;

template <class T>
static constexpr bool enable_error_converting_constructor =  
constructors_enabled                                         
&& !std::is_same<std::decay_t<T>, basic_result>::value       
&& base::template enable_error_converting_constructor<T>;

template <class ErrorCondEnum>
static constexpr bool enable_error_condition_converting_constructor =  
constructors_enabled                                                   
&& !std::is_same<std::decay_t<ErrorCondEnum>, basic_result>::value     
&& base::template enable_error_condition_converting_constructor<ErrorCondEnum>;

template <class T, class U, class V>
static constexpr bool enable_compatible_conversion =          
constructors_enabled                                          
&& !std::is_same<basic_result<T, U, V>, basic_result>::value  
&& base::template enable_compatible_conversion<T, U, V>;

template <class T, class U, class V>
static constexpr bool enable_make_error_code_compatible_conversion =  
constructors_enabled                                                  
&& !std::is_same<basic_result<T, U, V>, basic_result>::value          
&& base::template enable_make_error_code_compatible_conversion<T, U, V>;

template <class T, class U, class V>
static constexpr bool enable_make_exception_ptr_compatible_conversion =  
constructors_enabled                                                     
&& !std::is_same<basic_result<T, U, V>, basic_result>::value             
&& base::template enable_make_exception_ptr_compatible_conversion<T, U, V>;

template <class... Args>
static constexpr bool enable_inplace_value_constructor =  
constructors_enabled                                      
&& (std::is_void<value_type>::value                       
|| detail::is_constructible<value_type, Args...>);

template <class... Args>
static constexpr bool enable_inplace_error_constructor =  
constructors_enabled                                      
&& (std::is_void<error_type>::value                       
|| detail::is_constructible<error_type, Args...>);

template <class... Args>
static constexpr bool enable_inplace_value_error_constructor =  
constructors_enabled                                            
&&base::template enable_inplace_value_error_constructor<Args...>;
template <class... Args> using choose_inplace_value_error_constructor = typename base::template choose_inplace_value_error_constructor<Args...>;
};

public:

basic_result() = delete;

basic_result(basic_result && ) = default;  

basic_result(const basic_result & ) = default;

basic_result &operator=(basic_result && ) = default;  

basic_result &operator=(const basic_result & ) = default;
~basic_result() = default;


BOOST_OUTCOME_TEMPLATE(class Arg, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!predicate::constructors_enabled && (sizeof...(Args) >= 0)))
basic_result(Arg && , Args &&... ) = delete;  


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED((predicate::constructors_enabled && !predicate::implicit_constructors_enabled  
&& (detail::is_implicitly_constructible<value_type, T> || detail::is_implicitly_constructible<error_type, T>) )))
basic_result(T && , implicit_constructors_disabled_tag  = implicit_constructors_disabled_tag()) =
delete;  


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_value_converting_constructor<T>))
constexpr basic_result(T &&t, value_converting_constructor_tag  = value_converting_constructor_tag()) noexcept(
detail::is_nothrow_constructible<value_type, T>)  
: base{in_place_type<typename base::value_type>, static_cast<T &&>(t)}
{
using namespace hooks;
hook_result_construction(this, static_cast<T &&>(t));
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_error_converting_constructor<T>))
constexpr basic_result(T &&t, error_converting_constructor_tag  = error_converting_constructor_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T>)  
: base{in_place_type<typename base::error_type>, static_cast<T &&>(t)}
{
using namespace hooks;
hook_result_construction(this, static_cast<T &&>(t));
}

BOOST_OUTCOME_TEMPLATE(class ErrorCondEnum)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(error_type(make_error_code(ErrorCondEnum()))),  
BOOST_OUTCOME_TPRED(predicate::template enable_error_condition_converting_constructor<ErrorCondEnum>))
constexpr basic_result(ErrorCondEnum &&t, error_condition_converting_constructor_tag  = error_condition_converting_constructor_tag()) noexcept(
noexcept(error_type(make_error_code(static_cast<ErrorCondEnum &&>(t)))))  
: base{in_place_type<typename base::error_type>, make_error_code(t)}
{
using namespace hooks;
hook_result_construction(this, static_cast<ErrorCondEnum &&>(t));
}


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(convert::value_or_error<basic_result, std::decay_t<T>>::enable_result_inputs || !concepts::basic_result<T>),  
BOOST_OUTCOME_TEXPR(convert::value_or_error<basic_result, std::decay_t<T>>{}(std::declval<T>())))
constexpr explicit basic_result(T &&o,
explicit_valueorerror_converting_constructor_tag  = explicit_valueorerror_converting_constructor_tag())  
: basic_result{convert::value_or_error<basic_result, std::decay_t<T>>{}(static_cast<T &&>(o))}
{
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V>))
constexpr explicit basic_result(
const basic_result<T, U, V> &o,
explicit_compatible_copy_conversion_tag  =
explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>)
: base{typename base::compatible_conversion_tag(), o}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V>))
constexpr explicit basic_result(
basic_result<T, U, V> &&o,
explicit_compatible_move_conversion_tag  =
explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>)
: base{typename base::compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<T, U, V>))
constexpr explicit basic_result(const basic_result<T, U, V> &o,
explicit_make_error_code_compatible_copy_conversion_tag  =
explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
&&noexcept(make_error_code(std::declval<U>())))
: base{typename base::make_error_code_compatible_conversion_tag(), o}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<T, U, V>))
constexpr explicit basic_result(basic_result<T, U, V> &&o,
explicit_make_error_code_compatible_move_conversion_tag  =
explicit_make_error_code_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
&&noexcept(make_error_code(std::declval<U>())))
: base{typename base::make_error_code_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<T, U, V>))
constexpr explicit basic_result(const basic_result<T, U, V> &o,
explicit_make_exception_ptr_compatible_copy_conversion_tag  =
explicit_make_exception_ptr_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
&&noexcept(make_exception_ptr(std::declval<U>())))
: base{typename base::make_exception_ptr_compatible_conversion_tag(), o}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<T, U, V>))
constexpr explicit basic_result(basic_result<T, U, V> &&o,
explicit_make_exception_ptr_compatible_move_conversion_tag  =
explicit_make_exception_ptr_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
&&noexcept(make_exception_ptr(std::declval<U>())))
: base{typename base::make_exception_ptr_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
}


BOOST_OUTCOME_TEMPLATE(class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<Args...>))
constexpr explicit basic_result(in_place_type_t<value_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, Args...>)
: base{_, static_cast<Args &&>(args)...}
{
using namespace hooks;
hook_result_in_place_construction(this, in_place_type<value_type>, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class U, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<std::initializer_list<U>, Args...>))
constexpr explicit basic_result(in_place_type_t<value_type_if_enabled> _, std::initializer_list<U> il,
Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, std::initializer_list<U>, Args...>)
: base{_, il, static_cast<Args &&>(args)...}
{
using namespace hooks;
hook_result_in_place_construction(this, in_place_type<value_type>, il, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<Args...>))
constexpr explicit basic_result(in_place_type_t<error_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, Args...>)
: base{_, static_cast<Args &&>(args)...}
{
using namespace hooks;
hook_result_in_place_construction(this, in_place_type<error_type>, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class U, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<std::initializer_list<U>, Args...>))
constexpr explicit basic_result(in_place_type_t<error_type_if_enabled> _, std::initializer_list<U> il,
Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, std::initializer_list<U>, Args...>)
: base{_, il, static_cast<Args &&>(args)...}
{
using namespace hooks;
hook_result_in_place_construction(this, in_place_type<error_type>, il, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class A1, class A2, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_value_error_constructor<A1, A2, Args...>))
constexpr basic_result(A1 &&a1, A2 &&a2, Args &&... args) noexcept(noexcept(
typename predicate::template choose_inplace_value_error_constructor<A1, A2, Args...>(std::declval<A1>(), std::declval<A2>(), std::declval<Args>()...)))
: basic_result(in_place_type<typename predicate::template choose_inplace_value_error_constructor<A1, A2, Args...>>, static_cast<A1 &&>(a1),
static_cast<A2 &&>(a2), static_cast<Args &&>(args)...)
{

using namespace hooks;
}


constexpr basic_result(const success_type<void> &o) noexcept(std::is_nothrow_default_constructible<value_type>::value)  
: base{in_place_type<value_type_if_enabled>}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, void, void>))
constexpr basic_result(const success_type<T> &o) noexcept(detail::is_nothrow_constructible<value_type, T>)  
: base{in_place_type<value_type_if_enabled>, detail::extract_value_from_success<value_type>(o)}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void>))
constexpr basic_result(success_type<T> &&o) noexcept(detail::is_nothrow_constructible<value_type, T>)  
: base{in_place_type<value_type_if_enabled>, detail::extract_value_from_success<value_type>(static_cast<success_type<T> &&>(o))}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<success_type<T> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<void, T, void>))
constexpr basic_result(const failure_type<T> &o, explicit_compatible_copy_conversion_tag  = explicit_compatible_copy_conversion_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T>)  
: base{in_place_type<error_type_if_enabled>, detail::extract_error_from_failure<error_type>(o)}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<void, T, void>))
constexpr basic_result(failure_type<T> &&o, explicit_compatible_move_conversion_tag  = explicit_compatible_move_conversion_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T>)  
: base{in_place_type<error_type_if_enabled>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o))}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<failure_type<T> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<void, T, void>))
constexpr basic_result(const failure_type<T> &o,
explicit_make_error_code_compatible_copy_conversion_tag  =
explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>())))  
: base{in_place_type<error_type_if_enabled>, make_error_code(detail::extract_error_from_failure<error_type>(o))}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<void, T, void>))
constexpr basic_result(failure_type<T> &&o,
explicit_make_error_code_compatible_move_conversion_tag  =
explicit_make_error_code_compatible_move_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>())))  
: base{in_place_type<error_type_if_enabled>, make_error_code(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<failure_type<T> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<void, T, void>))
constexpr basic_result(const failure_type<T> &o,
explicit_make_exception_ptr_compatible_copy_conversion_tag  =
explicit_make_exception_ptr_compatible_copy_conversion_tag()) noexcept(noexcept(make_exception_ptr(std::declval<T>())))  
: base{in_place_type<error_type_if_enabled>, make_exception_ptr(detail::extract_error_from_failure<error_type>(o))}
{
using namespace hooks;
hook_result_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<void, T, void>))
constexpr basic_result(failure_type<T> &&o,
explicit_make_exception_ptr_compatible_move_conversion_tag  =
explicit_make_exception_ptr_compatible_move_conversion_tag()) noexcept(noexcept(make_exception_ptr(std::declval<T>())))  
: base{in_place_type<error_type_if_enabled>, make_exception_ptr(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
{
using namespace hooks;
hook_result_move_construction(this, static_cast<failure_type<T> &&>(o));
}


constexpr void swap(basic_result &o) noexcept((std::is_void<value_type>::value || detail::is_nothrow_swappable<value_type>::value)  
&& (std::is_void<error_type>::value || detail::is_nothrow_swappable<error_type>::value))
{
constexpr bool value_throws = !std::is_void<value_type>::value && !detail::is_nothrow_swappable<value_type>::value;
constexpr bool error_throws = !std::is_void<error_type>::value && !detail::is_nothrow_swappable<error_type>::value;
detail::basic_result_storage_swap<value_throws, error_throws>(*this, o);
}


auto as_failure() const & { return failure(this->assume_error()); }

auto as_failure() && { return failure(static_cast<basic_result &&>(*this).assume_error()); }

#ifdef __APPLE__
failure_type<error_type> _xcode_workaround_as_failure() &&;
#endif
};


template <class R, class S, class P> inline void swap(basic_result<R, S, P> &a, basic_result<R, S, P> &b) noexcept(noexcept(a.swap(b)))
{
a.swap(b);
}

#if !defined(NDEBUG)
static_assert(std::is_trivially_copyable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copyable!");
static_assert(std::is_trivially_assignable<basic_result<int, long, policy::all_narrow>, basic_result<int, long, policy::all_narrow>>::value,
"result<int> is not trivially assignable!");
static_assert(std::is_trivially_destructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially destructible!");
static_assert(std::is_trivially_copy_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copy constructible!");
static_assert(std::is_trivially_move_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially move constructible!");
static_assert(std::is_trivially_copy_assignable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copy assignable!");
static_assert(std::is_trivially_move_assignable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially move assignable!");
static_assert(std::is_standard_layout<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not a standard layout type!");
#endif

BOOST_OUTCOME_V2_NAMESPACE_END

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
