

#ifndef BOOST_OUTCOME_BASIC_OUTCOME_HPP
#define BOOST_OUTCOME_BASIC_OUTCOME_HPP

#include "config.hpp"

#include "basic_result.hpp"
#include "detail/basic_outcome_exception_observers.hpp"
#include "detail/basic_outcome_failure_observers.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"  
#endif

BOOST_OUTCOME_V2_NAMESPACE_EXPORT_BEGIN

template <class R, class S, class P, class NoValuePolicy>  
class basic_outcome;

namespace detail
{
template <class value_type, class error_type, class exception_type> struct outcome_predicates
{
using result = result_predicates<value_type, error_type>;

static constexpr bool implicit_constructors_enabled =                
result::implicit_constructors_enabled                                
&& !detail::is_implicitly_constructible<value_type, exception_type>  
&& !detail::is_implicitly_constructible<error_type, exception_type>  
&& !detail::is_implicitly_constructible<exception_type, value_type>  
&& !detail::is_implicitly_constructible<exception_type, error_type>;

template <class T>
static constexpr bool enable_value_converting_constructor =  
implicit_constructors_enabled                                
&&result::template enable_value_converting_constructor<T>    
&& !detail::is_implicitly_constructible<exception_type, T>;  

template <class T>
static constexpr bool enable_error_converting_constructor =  
implicit_constructors_enabled                                
&&result::template enable_error_converting_constructor<T>    
&& !detail::is_implicitly_constructible<exception_type, T>;  

template <class ErrorCondEnum>
static constexpr bool enable_error_condition_converting_constructor = result::template enable_error_condition_converting_constructor<ErrorCondEnum>  
&& !detail::is_implicitly_constructible<exception_type, ErrorCondEnum>;

template <class T>
static constexpr bool enable_exception_converting_constructor =  
implicit_constructors_enabled                                    
&& !is_in_place_type_t<std::decay_t<T>>::value                   
&& !detail::is_implicitly_constructible<value_type, T> && !detail::is_implicitly_constructible<error_type, T> &&
detail::is_implicitly_constructible<exception_type, T>;

template <class T, class U>
static constexpr bool enable_error_exception_converting_constructor =                                         
implicit_constructors_enabled                                                                                 
&& !is_in_place_type_t<std::decay_t<T>>::value                                                                
&& !detail::is_implicitly_constructible<value_type, T> && detail::is_implicitly_constructible<error_type, T>  
&& !detail::is_implicitly_constructible<value_type, U> && detail::is_implicitly_constructible<exception_type, U>;

template <class T, class U, class V, class W>
static constexpr bool enable_compatible_conversion =  
(std::is_void<T>::value ||
detail::is_explicitly_constructible<value_type, typename basic_outcome<T, U, V, W>::value_type>)  
&&(std::is_void<U>::value ||
detail::is_explicitly_constructible<error_type, typename basic_outcome<T, U, V, W>::error_type>)  
&&(std::is_void<V>::value ||
detail::is_explicitly_constructible<exception_type, typename basic_outcome<T, U, V, W>::exception_type>)  
;

template <class T, class U, class V, class W>
static constexpr bool enable_make_error_code_compatible_conversion =  
trait::is_error_code_available<std::decay_t<error_type>>::value       
&& !enable_compatible_conversion<T, U, V, W>                          
&& (std::is_void<T>::value ||
detail::is_explicitly_constructible<value_type, typename basic_outcome<T, U, V, W>::value_type>)  
&&detail::is_explicitly_constructible<error_type,
typename trait::is_error_code_available<U>::type>  
&& (std::is_void<V>::value ||
detail::is_explicitly_constructible<exception_type, typename basic_outcome<T, U, V, W>::exception_type>);  

struct disable_inplace_value_error_exception_constructor;
template <class... Args>
using choose_inplace_value_error_exception_constructor = std::conditional_t<  
((static_cast<int>(detail::is_constructible<value_type, Args...>) + static_cast<int>(detail::is_constructible<error_type, Args...>) +
static_cast<int>(detail::is_constructible<exception_type, Args...>)) > 1),  
disable_inplace_value_error_exception_constructor,                            
std::conditional_t<                                                           
detail::is_constructible<value_type, Args...>,                                
value_type,                                                                   
std::conditional_t<                                                           
detail::is_constructible<error_type, Args...>,                                
error_type,                                                                   
std::conditional_t<                                                           
detail::is_constructible<exception_type, Args...>,                            
exception_type,                                                               
disable_inplace_value_error_exception_constructor>>>>;
template <class... Args>
static constexpr bool enable_inplace_value_error_exception_constructor =  
implicit_constructors_enabled &&
!std::is_same<choose_inplace_value_error_exception_constructor<Args...>, disable_inplace_value_error_exception_constructor>::value;
};

template <class Base, class R, class S, class P, class NoValuePolicy>
using select_basic_outcome_failure_observers =  
std::conditional_t<trait::is_error_code_available<S>::value && trait::is_exception_ptr_available<P>::value,
basic_outcome_failure_observers<Base, R, S, P, NoValuePolicy>, Base>;

template <class T, class U, class V> constexpr inline const V &extract_exception_from_failure(const failure_type<U, V> &v) { return v.exception(); }
template <class T, class U, class V> constexpr inline V &&extract_exception_from_failure(failure_type<U, V> &&v)
{
return static_cast<failure_type<U, V> &&>(v).exception();
}
template <class T, class U> constexpr inline const U &extract_exception_from_failure(const failure_type<U, void> &v) { return v.error(); }
template <class T, class U> constexpr inline U &&extract_exception_from_failure(failure_type<U, void> &&v)
{
return static_cast<failure_type<U, void> &&>(v).error();
}

template <class T> struct is_basic_outcome
{
static constexpr bool value = false;
};
template <class R, class S, class T, class N> struct is_basic_outcome<basic_outcome<R, S, T, N>>
{
static constexpr bool value = true;
};
}  


template <class T> using is_basic_outcome = detail::is_basic_outcome<std::decay_t<T>>;

template <class T> static constexpr bool is_basic_outcome_v = detail::is_basic_outcome<std::decay_t<T>>::value;

namespace concepts
{
#if defined(__cpp_concepts)

template <class U>
concept BOOST_OUTCOME_GCC6_CONCEPT_BOOL basic_outcome =
BOOST_OUTCOME_V2_NAMESPACE::is_basic_outcome<U>::value ||
(requires(U v) {
BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>(v);
} &&  
detail::convertible<
U, BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>> &&  
detail::base_of<
BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>, U>);
#else
namespace detail
{
inline no_match match_basic_outcome(...);
template <class R, class S, class P, class NVP, class T,                                                                 
typename = typename T::value_type,                                                                             
typename = typename T::error_type,                                                                             
typename = typename T::exception_type,                                                                         
typename = typename T::no_value_policy_type,                                                                   
typename std::enable_if_t<std::is_convertible<T, BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP>>::value &&  
std::is_base_of<BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP>, T>::value,
bool> = true>
inline BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP> match_basic_outcome(BOOST_OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP> &&, T &&);

template <class U>
static constexpr bool basic_outcome =
BOOST_OUTCOME_V2_NAMESPACE::is_basic_outcome<U>::value ||
!std::is_same<no_match, decltype(match_basic_outcome(std::declval<BOOST_OUTCOME_V2_NAMESPACE::detail::devoid<U>>(),
std::declval<BOOST_OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
}  

template <class U> static constexpr bool basic_outcome = detail::basic_outcome<U>;
#endif
}  

namespace hooks
{

template <class T, class... U> constexpr inline void hook_outcome_construction(T * , U &&... ) noexcept {}

template <class T, class U> constexpr inline void hook_outcome_copy_construction(T * , U && ) noexcept {}

template <class T, class U> constexpr inline void hook_outcome_move_construction(T * , U && ) noexcept {}

template <class T, class U, class... Args>
constexpr inline void hook_outcome_in_place_construction(T * , in_place_type_t<U> , Args &&... ) noexcept
{
}


template <class R, class S, class P, class NoValuePolicy, class U>
constexpr inline void override_outcome_exception(basic_outcome<R, S, P, NoValuePolicy> *o, U &&v) noexcept;
}  


template <class R, class S, class P, class NoValuePolicy>  
class BOOST_OUTCOME_NODISCARD basic_outcome
#if defined(BOOST_OUTCOME_DOXYGEN_IS_IN_THE_HOUSE) || defined(BOOST_OUTCOME_STANDARDESE_IS_IN_THE_HOUSE)
: public detail::basic_outcome_failure_observers<detail::basic_result_final<R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>,
public detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>,
public detail::basic_result_final<R, S, NoValuePolicy>
#else
: public detail::select_basic_outcome_failure_observers<
detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>
#endif
{
static_assert(trait::type_can_be_used_in_basic_result<P>, "The exception_type cannot be used");
static_assert(std::is_void<P>::value || std::is_default_constructible<P>::value, "exception_type must be void or default constructible");
using base = detail::select_basic_outcome_failure_observers<
detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>;
friend struct policy::base;
template <class T, class U, class V, class W>  
friend class basic_outcome;
template <class T, class U, class V, class W, class X>
friend constexpr inline void hooks::override_outcome_exception(basic_outcome<T, U, V, W> *o, X &&v) noexcept;  

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
struct exception_converting_constructor_tag
{
};
struct error_exception_converting_constructor_tag
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
struct error_failure_tag
{
};
struct exception_failure_tag
{
};

struct disable_in_place_value_type
{
};
struct disable_in_place_error_type
{
};
struct disable_in_place_exception_type
{
};

public:
using value_type = R;
using error_type = S;
using exception_type = P;
using no_value_policy_type = NoValuePolicy;

template <class T, class U = S, class V = P, class W = NoValuePolicy> using rebind = basic_outcome<T, U, V, W>;

protected:
struct predicate
{
using base = detail::outcome_predicates<value_type, error_type, exception_type>;

static constexpr bool constructors_enabled =
(!std::is_same<std::decay_t<value_type>, std::decay_t<error_type>>::value || (std::is_void<value_type>::value && std::is_void<error_type>::value))  
&& (!std::is_same<std::decay_t<value_type>, std::decay_t<exception_type>>::value ||
(std::is_void<value_type>::value && std::is_void<exception_type>::value))  
&& (!std::is_same<std::decay_t<error_type>, std::decay_t<exception_type>>::value ||
(std::is_void<error_type>::value && std::is_void<exception_type>::value))  
;

static constexpr bool implicit_constructors_enabled = constructors_enabled && base::implicit_constructors_enabled;

template <class T>
static constexpr bool enable_value_converting_constructor =  
constructors_enabled                                         
&& !std::is_same<std::decay_t<T>, basic_outcome>::value      
&& base::template enable_value_converting_constructor<T>;

template <class T>
static constexpr bool enable_error_converting_constructor =  
constructors_enabled                                         
&& !std::is_same<std::decay_t<T>, basic_outcome>::value      
&& base::template enable_error_converting_constructor<T>;

template <class ErrorCondEnum>
static constexpr bool enable_error_condition_converting_constructor =  
constructors_enabled                                                   
&& !std::is_same<std::decay_t<ErrorCondEnum>, basic_outcome>::value    
&& base::template enable_error_condition_converting_constructor<ErrorCondEnum>;

template <class T>
static constexpr bool enable_exception_converting_constructor =  
constructors_enabled                                             
&& !std::is_same<std::decay_t<T>, basic_outcome>::value          
&& base::template enable_exception_converting_constructor<T>;

template <class T, class U>
static constexpr bool enable_error_exception_converting_constructor =  
constructors_enabled                                                   
&& !std::is_same<std::decay_t<T>, basic_outcome>::value                
&& base::template enable_error_exception_converting_constructor<T, U>;

template <class T, class U, class V, class W>
static constexpr bool enable_compatible_conversion =               
constructors_enabled                                               
&& !std::is_same<basic_outcome<T, U, V, W>, basic_outcome>::value  
&& base::template enable_compatible_conversion<T, U, V, W>;

template <class T, class U, class V, class W>
static constexpr bool enable_make_error_code_compatible_conversion =  
constructors_enabled                                                  
&& !std::is_same<basic_outcome<T, U, V, W>, basic_outcome>::value     
&& base::template enable_make_error_code_compatible_conversion<T, U, V, W>;

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
static constexpr bool enable_inplace_exception_constructor =  
constructors_enabled                                          
&& (std::is_void<exception_type>::value                       
|| detail::is_constructible<exception_type, Args...>);

template <class... Args>
static constexpr bool enable_inplace_value_error_exception_constructor =  
constructors_enabled                                                      
&&base::template enable_inplace_value_error_exception_constructor<Args...>;
template <class... Args>
using choose_inplace_value_error_exception_constructor = typename base::template choose_inplace_value_error_exception_constructor<Args...>;
};

public:
using value_type_if_enabled =
std::conditional_t<std::is_same<value_type, error_type>::value || std::is_same<value_type, exception_type>::value, disable_in_place_value_type, value_type>;
using error_type_if_enabled =
std::conditional_t<std::is_same<error_type, value_type>::value || std::is_same<error_type, exception_type>::value, disable_in_place_error_type, error_type>;
using exception_type_if_enabled = std::conditional_t<std::is_same<exception_type, value_type>::value || std::is_same<exception_type, error_type>::value,
disable_in_place_exception_type, exception_type>;

protected:
detail::devoid<exception_type> _ptr;

public:

BOOST_OUTCOME_TEMPLATE(class Arg, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED((!predicate::constructors_enabled && sizeof...(Args) >= 0)))
basic_outcome(Arg && , Args &&... ) = delete;  


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED((predicate::constructors_enabled && !predicate::implicit_constructors_enabled  
&& (detail::is_implicitly_constructible<value_type, T> || detail::is_implicitly_constructible<error_type, T> ||
detail::is_implicitly_constructible<exception_type, T>) )))
basic_outcome(T && , implicit_constructors_disabled_tag  = implicit_constructors_disabled_tag()) =
delete;  


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_value_converting_constructor<T>))
constexpr basic_outcome(T &&t, value_converting_constructor_tag  = value_converting_constructor_tag()) noexcept(
detail::is_nothrow_constructible<value_type, T>)  
: base{in_place_type<typename base::_value_type>, static_cast<T &&>(t)}
, _ptr()
{
using namespace hooks;
hook_outcome_construction(this, static_cast<T &&>(t));
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_error_converting_constructor<T>))
constexpr basic_outcome(T &&t, error_converting_constructor_tag  = error_converting_constructor_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T>)  
: base{in_place_type<typename base::_error_type>, static_cast<T &&>(t)}
, _ptr()
{
using namespace hooks;
hook_outcome_construction(this, static_cast<T &&>(t));
}

BOOST_OUTCOME_TEMPLATE(class ErrorCondEnum)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(error_type(make_error_code(ErrorCondEnum()))),  
BOOST_OUTCOME_TPRED(predicate::template enable_error_condition_converting_constructor<ErrorCondEnum>))
constexpr basic_outcome(ErrorCondEnum &&t, error_condition_converting_constructor_tag  = error_condition_converting_constructor_tag()) noexcept(
noexcept(error_type(make_error_code(static_cast<ErrorCondEnum &&>(t)))))  
: base{in_place_type<typename base::_error_type>, make_error_code(t)}
{
using namespace hooks;
hook_outcome_construction(this, static_cast<ErrorCondEnum &&>(t));
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_exception_converting_constructor<T>))
constexpr basic_outcome(T &&t, exception_converting_constructor_tag  = exception_converting_constructor_tag()) noexcept(
detail::is_nothrow_constructible<exception_type, T>)  
: base()
, _ptr(static_cast<T &&>(t))
{
using namespace hooks;
this->_state._status.set_have_exception(true);
hook_outcome_construction(this, static_cast<T &&>(t));
}

BOOST_OUTCOME_TEMPLATE(class T, class U)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_error_exception_converting_constructor<T, U>))
constexpr basic_outcome(T &&a, U &&b, error_exception_converting_constructor_tag  = error_exception_converting_constructor_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>)  
: base{in_place_type<typename base::_error_type>, static_cast<T &&>(a)}
, _ptr(static_cast<U &&>(b))
{
using namespace hooks;
this->_state._status.set_have_exception(true);
hook_outcome_construction(this, static_cast<T &&>(a), static_cast<U &&>(b));
}


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(convert::value_or_error<basic_outcome, std::decay_t<T>>::enable_result_inputs || !concepts::basic_result<T>),    
BOOST_OUTCOME_TPRED(convert::value_or_error<basic_outcome, std::decay_t<T>>::enable_outcome_inputs || !concepts::basic_outcome<T>),  
BOOST_OUTCOME_TEXPR(convert::value_or_error<basic_outcome, std::decay_t<T>>{}(std::declval<T>())))
constexpr explicit basic_outcome(T &&o,
explicit_valueorerror_converting_constructor_tag  = explicit_valueorerror_converting_constructor_tag())  
: basic_outcome{convert::value_or_error<basic_outcome, std::decay_t<T>>{}(static_cast<T &&>(o))}
{
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V, class W)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V, W>))
constexpr explicit basic_outcome(
const basic_outcome<T, U, V, W> &o,
explicit_compatible_copy_conversion_tag  =
explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
&&detail::is_nothrow_constructible<exception_type, V>)
: base{typename base::compatible_conversion_tag(), o}
, _ptr(o._ptr)
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V, class W)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V, W>))
constexpr explicit basic_outcome(
basic_outcome<T, U, V, W> &&o,
explicit_compatible_move_conversion_tag  =
explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
&&detail::is_nothrow_constructible<exception_type, V>)
: base{typename base::compatible_conversion_tag(), static_cast<basic_outcome<T, U, V, W> &&>(o)}
, _ptr(static_cast<typename basic_outcome<T, U, V, W>::exception_type &&>(o._ptr))
{
using namespace hooks;
hook_outcome_move_construction(this, static_cast<basic_outcome<T, U, V, W> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_compatible_conversion<T, U, V>))
constexpr explicit basic_outcome(
const basic_result<T, U, V> &o,
explicit_compatible_copy_conversion_tag  =
explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
&&detail::is_nothrow_constructible<exception_type>)
: base{typename base::compatible_conversion_tag(), o}
, _ptr()
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_compatible_conversion<T, U, V>))
constexpr explicit basic_outcome(
basic_result<T, U, V> &&o,
explicit_compatible_move_conversion_tag  =
explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
&&detail::is_nothrow_constructible<exception_type>)
: base{typename base::compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
, _ptr()
{
using namespace hooks;
hook_outcome_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_make_error_code_compatible_conversion<T, U, V>))
constexpr explicit basic_outcome(const basic_result<T, U, V> &o,
explicit_make_error_code_compatible_copy_conversion_tag  =
explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
&&noexcept(make_error_code(std::declval<U>())) &&
detail::is_nothrow_constructible<exception_type>)
: base{typename base::make_error_code_compatible_conversion_tag(), o}
, _ptr()
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_make_error_code_compatible_conversion<T, U, V>))
constexpr explicit basic_outcome(basic_result<T, U, V> &&o,
explicit_make_error_code_compatible_move_conversion_tag  =
explicit_make_error_code_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
&&noexcept(make_error_code(std::declval<U>())) &&
detail::is_nothrow_constructible<exception_type>)
: base{typename base::make_error_code_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
, _ptr()
{
using namespace hooks;
hook_outcome_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
}



BOOST_OUTCOME_TEMPLATE(class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<Args...>))
constexpr explicit basic_outcome(in_place_type_t<value_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, Args...>)
: base{_, static_cast<Args &&>(args)...}
, _ptr()
{
using namespace hooks;
hook_outcome_in_place_construction(this, in_place_type<value_type>, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class U, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<std::initializer_list<U>, Args...>))
constexpr explicit basic_outcome(in_place_type_t<value_type_if_enabled> _, std::initializer_list<U> il,
Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, std::initializer_list<U>, Args...>)
: base{_, il, static_cast<Args &&>(args)...}
, _ptr()
{
using namespace hooks;
hook_outcome_in_place_construction(this, in_place_type<value_type>, il, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<Args...>))
constexpr explicit basic_outcome(in_place_type_t<error_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, Args...>)
: base{_, static_cast<Args &&>(args)...}
, _ptr()
{
using namespace hooks;
hook_outcome_in_place_construction(this, in_place_type<error_type>, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class U, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<std::initializer_list<U>, Args...>))
constexpr explicit basic_outcome(in_place_type_t<error_type_if_enabled> _, std::initializer_list<U> il,
Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, std::initializer_list<U>, Args...>)
: base{_, il, static_cast<Args &&>(args)...}
, _ptr()
{
using namespace hooks;
hook_outcome_in_place_construction(this, in_place_type<error_type>, il, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_exception_constructor<Args...>))
constexpr explicit basic_outcome(in_place_type_t<exception_type_if_enabled> ,
Args &&... args) noexcept(detail::is_nothrow_constructible<exception_type, Args...>)
: base()
, _ptr(static_cast<Args &&>(args)...)
{
using namespace hooks;
this->_state._status.set_have_exception(true);
hook_outcome_in_place_construction(this, in_place_type<exception_type>, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class U, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_exception_constructor<std::initializer_list<U>, Args...>))
constexpr explicit basic_outcome(in_place_type_t<exception_type_if_enabled> , std::initializer_list<U> il,
Args &&... args) noexcept(detail::is_nothrow_constructible<exception_type, std::initializer_list<U>, Args...>)
: base()
, _ptr(il, static_cast<Args &&>(args)...)
{
using namespace hooks;
this->_state._status.set_have_exception(true);
hook_outcome_in_place_construction(this, in_place_type<exception_type>, il, static_cast<Args &&>(args)...);
}

BOOST_OUTCOME_TEMPLATE(class A1, class A2, class... Args)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(predicate::template enable_inplace_value_error_exception_constructor<A1, A2, Args...>))
constexpr basic_outcome(A1 &&a1, A2 &&a2, Args &&... args) noexcept(
noexcept(typename predicate::template choose_inplace_value_error_exception_constructor<A1, A2, Args...>(std::declval<A1>(), std::declval<A2>(),
std::declval<Args>()...)))
: basic_outcome(in_place_type<typename predicate::template choose_inplace_value_error_exception_constructor<A1, A2, Args...>>, static_cast<A1 &&>(a1),
static_cast<A2 &&>(a2), static_cast<Args &&>(args)...)
{
}


constexpr basic_outcome(const success_type<void> &o) noexcept(std::is_nothrow_default_constructible<value_type>::value)  
: base{in_place_type<typename base::_value_type>}
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void, void>))
constexpr basic_outcome(const success_type<T> &o) noexcept(detail::is_nothrow_constructible<value_type, T>)  
: base{in_place_type<typename base::_value_type>, detail::extract_value_from_success<value_type>(o)}
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void, void>))
constexpr basic_outcome(success_type<T> &&o) noexcept(detail::is_nothrow_constructible<value_type, T>)  
: base{in_place_type<typename base::_value_type>, detail::extract_value_from_success<value_type>(static_cast<success_type<T> &&>(o))}
{
using namespace hooks;
hook_outcome_move_construction(this, static_cast<success_type<T> &&>(o));
}


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, T, void, void>))
constexpr basic_outcome(const failure_type<T> &o,
error_failure_tag  = error_failure_tag()) noexcept(detail::is_nothrow_constructible<error_type, T>)  
: base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(o)}
, _ptr()
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, void, T, void>))
constexpr basic_outcome(const failure_type<T> &o,
exception_failure_tag  = exception_failure_tag()) noexcept(detail::is_nothrow_constructible<exception_type, T>)  
: base()
, _ptr(detail::extract_exception_from_failure<exception_type>(o))
{
this->_state._status.set_have_exception(true);
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_make_error_code_compatible_conversion<void, T, void, void>))
constexpr basic_outcome(const failure_type<T> &o,
explicit_make_error_code_compatible_copy_conversion_tag  =
explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>())))  
: base{in_place_type<typename base::_error_type>, make_error_code(detail::extract_error_from_failure<error_type>(o))}
, _ptr()
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<U>::value && predicate::template enable_compatible_conversion<void, T, U, void>))
constexpr basic_outcome(const failure_type<T, U> &o, explicit_compatible_copy_conversion_tag  = explicit_compatible_copy_conversion_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>)  
: base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(o)}
, _ptr(detail::extract_exception_from_failure<exception_type>(o))
{
if(!o.has_error())
{
this->_state._status.set_have_error(false);
}
if(o.has_exception())
{
this->_state._status.set_have_exception(true);
}
using namespace hooks;
hook_outcome_copy_construction(this, o);
}


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, T, void, void>))
constexpr basic_outcome(failure_type<T> &&o,
error_failure_tag  = error_failure_tag()) noexcept(detail::is_nothrow_constructible<error_type, T>)  
: base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o))}
, _ptr()
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, void, T, void>))
constexpr basic_outcome(failure_type<T> &&o,
exception_failure_tag  = exception_failure_tag()) noexcept(detail::is_nothrow_constructible<exception_type, T>)  
: base()
, _ptr(detail::extract_exception_from_failure<exception_type>(static_cast<failure_type<T> &&>(o)))
{
this->_state._status.set_have_exception(true);
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_make_error_code_compatible_conversion<void, T, void, void>))
constexpr basic_outcome(failure_type<T> &&o,
explicit_make_error_code_compatible_move_conversion_tag  =
explicit_make_error_code_compatible_move_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>())))  
: base{in_place_type<typename base::_error_type>, make_error_code(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
, _ptr()
{
using namespace hooks;
hook_outcome_copy_construction(this, o);
}

BOOST_OUTCOME_TEMPLATE(class T, class U)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!std::is_void<U>::value && predicate::template enable_compatible_conversion<void, T, U, void>))
constexpr basic_outcome(failure_type<T, U> &&o, explicit_compatible_move_conversion_tag  = explicit_compatible_move_conversion_tag()) noexcept(
detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>)  
: base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T, U> &&>(o))}
, _ptr(detail::extract_exception_from_failure<exception_type>(static_cast<failure_type<T, U> &&>(o)))
{
if(!o.has_error())
{
this->_state._status.set_have_error(false);
}
if(o.has_exception())
{
this->_state._status.set_have_exception(true);
}
using namespace hooks;
hook_outcome_move_construction(this, static_cast<failure_type<T, U> &&>(o));
}


using base::operator==;
using base::operator!=;

BOOST_OUTCOME_TEMPLATE(class T, class U, class V, class W)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<detail::devoid<value_type>>() == std::declval<detail::devoid<T>>()),  
BOOST_OUTCOME_TEXPR(std::declval<detail::devoid<error_type>>() == std::declval<detail::devoid<U>>()),  
BOOST_OUTCOME_TEXPR(std::declval<detail::devoid<exception_type>>() == std::declval<detail::devoid<V>>()))
constexpr bool operator==(const basic_outcome<T, U, V, W> &o) const noexcept(                 
noexcept(std::declval<detail::devoid<value_type>>() == std::declval<detail::devoid<T>>())     
&& noexcept(std::declval<detail::devoid<error_type>>() == std::declval<detail::devoid<U>>())  
&& noexcept(std::declval<detail::devoid<exception_type>>() == std::declval<detail::devoid<V>>()))
{
if(this->_state._status.have_value() && o._state._status.have_value())
{
return this->_state._value == o._state._value;  
}
if(this->_state._status.have_error() && o._state._status.have_error()  
&& this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_error == o._error && this->_ptr == o._ptr;
}
if(this->_state._status.have_error() && o._state._status.have_error())
{
return this->_error == o._error;
}
if(this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_ptr == o._ptr;
}
return false;
}

BOOST_OUTCOME_TEMPLATE(class T, class U)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<error_type>() == std::declval<T>()),  
BOOST_OUTCOME_TEXPR(std::declval<exception_type>() == std::declval<U>()))
constexpr bool operator==(const failure_type<T, U> &o) const noexcept(  
noexcept(std::declval<error_type>() == std::declval<T>()) && noexcept(std::declval<exception_type>() == std::declval<U>()))
{
if(this->_state._status.have_error() && o._state._status.have_error()  
&& this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_error == o.error() && this->_ptr == o.exception();
}
if(this->_state._status.have_error() && o._state._status.have_error())
{
return this->_error == o.error();
}
if(this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_ptr == o.exception();
}
return false;
}

BOOST_OUTCOME_TEMPLATE(class T, class U, class V, class W)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<detail::devoid<value_type>>() != std::declval<detail::devoid<T>>()),  
BOOST_OUTCOME_TEXPR(std::declval<detail::devoid<error_type>>() != std::declval<detail::devoid<U>>()),  
BOOST_OUTCOME_TEXPR(std::declval<detail::devoid<exception_type>>() != std::declval<detail::devoid<V>>()))
constexpr bool operator!=(const basic_outcome<T, U, V, W> &o) const noexcept(                 
noexcept(std::declval<detail::devoid<value_type>>() != std::declval<detail::devoid<T>>())     
&& noexcept(std::declval<detail::devoid<error_type>>() != std::declval<detail::devoid<U>>())  
&& noexcept(std::declval<detail::devoid<exception_type>>() != std::declval<detail::devoid<V>>()))
{
if(this->_state._status.have_value() && o._state._status.have_value())
{
return this->_state._value != o._state._value;  
}
if(this->_state._status.have_error() && o._state._status.have_error()  
&& this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_error != o._error || this->_ptr != o._ptr;
}
if(this->_state._status.have_error() && o._state._status.have_error())
{
return this->_error != o._error;
}
if(this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_ptr != o._ptr;
}
return true;
}

BOOST_OUTCOME_TEMPLATE(class T, class U)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<error_type>() != std::declval<T>()),  
BOOST_OUTCOME_TEXPR(std::declval<exception_type>() != std::declval<U>()))
constexpr bool operator!=(const failure_type<T, U> &o) const noexcept(  
noexcept(std::declval<error_type>() == std::declval<T>()) && noexcept(std::declval<exception_type>() == std::declval<U>()))
{
if(this->_state._status.have_error() && o._state._status.have_error()  
&& this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_error != o.error() || this->_ptr != o.exception();
}
if(this->_state._status.have_error() && o._state._status.have_error())
{
return this->_error != o.error();
}
if(this->_state._status.have_exception() && o._state._status.have_exception())
{
return this->_ptr != o.exception();
}
return true;
}


constexpr void swap(basic_outcome &o) noexcept((std::is_void<value_type>::value || detail::is_nothrow_swappable<value_type>::value)     
&& (std::is_void<error_type>::value || detail::is_nothrow_swappable<error_type>::value)  
&& (std::is_void<exception_type>::value || detail::is_nothrow_swappable<exception_type>::value))
{
#ifndef BOOST_NO_EXCEPTIONS
constexpr bool value_throws = !std::is_void<value_type>::value && !detail::is_nothrow_swappable<value_type>::value;
constexpr bool error_throws = !std::is_void<error_type>::value && !detail::is_nothrow_swappable<error_type>::value;
constexpr bool exception_throws = !std::is_void<exception_type>::value && !detail::is_nothrow_swappable<exception_type>::value;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)  
#endif
if(!exception_throws && !value_throws && !error_throws)
{
detail::basic_result_storage_swap<value_throws, error_throws>(*this, o);
using std::swap;
swap(this->_ptr, o._ptr);
return;
}
struct _
{
basic_outcome &a, &b;
bool exceptioned{false};
bool all_good{false};
~_()
{
if(!all_good)
{
a._state._status.set_have_lost_consistency(true);
b._state._status.set_have_lost_consistency(true);
return;
}
if(exceptioned)
{
try
{
strong_swap(all_good, a._ptr, b._ptr);
}
catch(...)
{
a._state._status.set_have_lost_consistency(true);
b._state._status.set_have_lost_consistency(true);
}

auto check = [](basic_outcome *t) {
if(t->has_value() && (t->has_error() || t->has_exception()))
{
t->_state._status.set_have_error(false).set_have_exception(false);
t->_state._status.set_have_lost_consistency(true);
}
if(!t->has_value() && !(t->has_error() || t->has_exception()))
{
t->_state._status.set_have_error(true).set_have_lost_consistency(true);
}
};
check(&a);
check(&b);
}
}
} _{*this, o};
strong_swap(_.all_good, this->_ptr, o._ptr);
_.exceptioned = true;
detail::basic_result_storage_swap<value_throws, error_throws>(*this, o);
_.exceptioned = false;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#else
detail::basic_result_storage_swap<false, false>(*this, o);
using std::swap;
swap(this->_ptr, o._ptr);
#endif
}


failure_type<error_type, exception_type> as_failure() const &
{
if(this->has_error() && this->has_exception())
{
return failure_type<error_type, exception_type>(this->assume_error(), this->assume_exception());
}
if(this->has_exception())
{
return failure_type<error_type, exception_type>(in_place_type<exception_type>, this->assume_exception());
}
return failure_type<error_type, exception_type>(in_place_type<error_type>, this->assume_error());
}


failure_type<error_type, exception_type> as_failure() &&
{
if(this->has_error() && this->has_exception())
{
return failure_type<error_type, exception_type>(static_cast<S &&>(this->assume_error()), static_cast<P &&>(this->assume_exception()));
}
if(this->has_exception())
{
return failure_type<error_type, exception_type>(in_place_type<exception_type>, static_cast<P &&>(this->assume_exception()));
}
return failure_type<error_type, exception_type>(in_place_type<error_type>, static_cast<S &&>(this->assume_error()));
}

#ifdef __APPLE__
failure_type<error_type, exception_type> _xcode_workaround_as_failure() &&;
#endif
};

#if __cplusplus < 202000L

BOOST_OUTCOME_TEMPLATE(class T, class U, class V,  
class R, class S, class P, class N)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<basic_outcome<R, S, P, N>>() == std::declval<basic_result<T, U, V>>()))
constexpr inline bool operator==(const basic_result<T, U, V> &a, const basic_outcome<R, S, P, N> &b) noexcept(  
noexcept(std::declval<basic_outcome<R, S, P, N>>() == std::declval<basic_result<T, U, V>>()))
{
return b == a;
}
#endif

BOOST_OUTCOME_TEMPLATE(class T, class U, class V,  
class R, class S, class P, class N)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<basic_outcome<R, S, P, N>>() != std::declval<basic_result<T, U, V>>()))
constexpr inline bool operator!=(const basic_result<T, U, V> &a, const basic_outcome<R, S, P, N> &b) noexcept(  
noexcept(std::declval<basic_outcome<R, S, P, N>>() != std::declval<basic_result<T, U, V>>()))
{
return b != a;
}

template <class R, class S, class P, class N> inline void swap(basic_outcome<R, S, P, N> &a, basic_outcome<R, S, P, N> &b) noexcept(noexcept(a.swap(b)))
{
a.swap(b);
}

namespace hooks
{

template <class R, class S, class P, class NoValuePolicy, class U>
constexpr inline void override_outcome_exception(basic_outcome<R, S, P, NoValuePolicy> *o, U &&v) noexcept
{
o->_ptr = static_cast<U &&>(v);  
o->_state._status.set_have_exception(true);
}
}  

BOOST_OUTCOME_V2_NAMESPACE_END

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "detail/basic_outcome_exception_observers_impl.hpp"

#if !defined(NDEBUG)
BOOST_OUTCOME_V2_NAMESPACE_BEGIN
static_assert(std::is_trivially_copyable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially copyable!");
static_assert(std::is_trivially_assignable<basic_outcome<int, long, double, policy::all_narrow>, basic_outcome<int, long, double, policy::all_narrow>>::value,
"outcome<int> is not trivially assignable!");
static_assert(std::is_trivially_destructible<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially destructible!");
static_assert(std::is_trivially_copy_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value,
"outcome<int> is not trivially copy constructible!");
static_assert(std::is_trivially_move_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value,
"outcome<int> is not trivially move constructible!");
static_assert(std::is_trivially_copy_assignable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially copy assignable!");
static_assert(std::is_trivially_move_assignable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially move assignable!");
BOOST_OUTCOME_V2_NAMESPACE_END
#endif

#endif
