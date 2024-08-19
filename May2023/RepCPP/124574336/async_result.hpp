
#ifndef BOOST_ASIO_ASYNC_RESULT_HPP
#define BOOST_ASIO_ASYNC_RESULT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/detail/variadic_templates.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_HAS_CONCEPTS) \
&& defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES) \
&& defined(BOOST_ASIO_HAS_DECLTYPE)

namespace detail {

template <typename T>
struct is_completion_signature : false_type
{
};

template <typename R, typename... Args>
struct is_completion_signature<R(Args...)> : true_type
{
};

template <typename T, typename... Args>
BOOST_ASIO_CONCEPT callable_with = requires(T t, Args&&... args)
{
t(static_cast<Args&&>(args)...);
};

template <typename T, typename Signature>
struct is_completion_handler_for : false_type
{
};

template <typename T, typename R, typename... Args>
struct is_completion_handler_for<T, R(Args...)>
: integral_constant<bool, (callable_with<T, Args...>)>
{
};

} 

template <typename T>
BOOST_ASIO_CONCEPT completion_signature =
detail::is_completion_signature<T>::value;

#define BOOST_ASIO_COMPLETION_SIGNATURE \
::boost::asio::completion_signature

template <typename T, typename Signature>
BOOST_ASIO_CONCEPT completion_handler_for =
detail::is_completion_signature<Signature>::value
&& detail::is_completion_handler_for<T, Signature>::value;

#define BOOST_ASIO_COMPLETION_HANDLER_FOR(s) \
::boost::asio::completion_handler_for<s>

#else 

#define BOOST_ASIO_COMPLETION_SIGNATURE typename
#define BOOST_ASIO_COMPLETION_HANDLER_FOR(s) typename

#endif 


template <typename CompletionToken, BOOST_ASIO_COMPLETION_SIGNATURE Signature>
class async_result
{
public:
typedef CompletionToken completion_handler_type;

typedef void return_type;


explicit async_result(completion_handler_type& h)
{
(void)h;
}

return_type get()
{
}

#if defined(GENERATING_DOCUMENTATION)

template <typename Initiation, typename RawCompletionToken, typename... Args>
static return_type initiate(
BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_MOVE_ARG(RawCompletionToken) token,
BOOST_ASIO_MOVE_ARG(Args)... args);

#elif defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Initiation,
BOOST_ASIO_COMPLETION_HANDLER_FOR(Signature) RawCompletionToken,
typename... Args>
static return_type initiate(
BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_MOVE_ARG(RawCompletionToken) token,
BOOST_ASIO_MOVE_ARG(Args)... args)
{
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)(
BOOST_ASIO_MOVE_CAST(RawCompletionToken)(token),
BOOST_ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Initiation,
BOOST_ASIO_COMPLETION_HANDLER_FOR(Signature) RawCompletionToken>
static return_type initiate(
BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_MOVE_ARG(RawCompletionToken) token)
{
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)(
BOOST_ASIO_MOVE_CAST(RawCompletionToken)(token));
}

#define BOOST_ASIO_PRIVATE_INITIATE_DEF(n) \
template <typename Initiation, \
BOOST_ASIO_COMPLETION_HANDLER_FOR(Signature) RawCompletionToken, \
BOOST_ASIO_VARIADIC_TPARAMS(n)> \
static return_type initiate( \
BOOST_ASIO_MOVE_ARG(Initiation) initiation, \
BOOST_ASIO_MOVE_ARG(RawCompletionToken) token, \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)( \
BOOST_ASIO_MOVE_CAST(RawCompletionToken)(token), \
BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_INITIATE_DEF)
#undef BOOST_ASIO_PRIVATE_INITIATE_DEF

#endif 

private:
async_result(const async_result&) BOOST_ASIO_DELETED;
async_result& operator=(const async_result&) BOOST_ASIO_DELETED;
};

#if !defined(GENERATING_DOCUMENTATION)

template <BOOST_ASIO_COMPLETION_SIGNATURE Signature>
class async_result<void, Signature>
{
};

#endif 

template <typename CompletionToken, BOOST_ASIO_COMPLETION_SIGNATURE Signature>
struct async_completion
{
typedef typename boost::asio::async_result<
typename decay<CompletionToken>::type,
Signature>::completion_handler_type completion_handler_type;

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

explicit async_completion(CompletionToken& token)
: completion_handler(static_cast<typename conditional<
is_same<CompletionToken, completion_handler_type>::value,
completion_handler_type&, CompletionToken&&>::type>(token)),
result(completion_handler)
{
}
#else 
explicit async_completion(typename decay<CompletionToken>::type& token)
: completion_handler(token),
result(completion_handler)
{
}

explicit async_completion(const typename decay<CompletionToken>::type& token)
: completion_handler(token),
result(completion_handler)
{
}
#endif 

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
typename conditional<
is_same<CompletionToken, completion_handler_type>::value,
completion_handler_type&, completion_handler_type>::type completion_handler;
#else 
completion_handler_type completion_handler;
#endif 

async_result<typename decay<CompletionToken>::type, Signature> result;
};

namespace detail {

template <typename CompletionToken, typename Signature>
struct async_result_helper
: async_result<typename decay<CompletionToken>::type, Signature>
{
};

struct async_result_memfns_base
{
void initiate();
};

template <typename T>
struct async_result_memfns_derived
: T, async_result_memfns_base
{
};

template <typename T, T>
struct async_result_memfns_check
{
};

template <typename>
char (&async_result_initiate_memfn_helper(...))[2];

template <typename T>
char async_result_initiate_memfn_helper(
async_result_memfns_check<
void (async_result_memfns_base::*)(),
&async_result_memfns_derived<T>::initiate>*);

template <typename CompletionToken, typename Signature>
struct async_result_has_initiate_memfn
: integral_constant<bool, sizeof(async_result_initiate_memfn_helper<
async_result<typename decay<CompletionToken>::type, Signature>
>(0)) != 1>
{
};

} 

#if defined(GENERATING_DOCUMENTATION)
# define BOOST_ASIO_INITFN_RESULT_TYPE(ct, sig) \
void_or_deduced
#elif defined(_MSC_VER) && (_MSC_VER < 1500)
# define BOOST_ASIO_INITFN_RESULT_TYPE(ct, sig) \
typename ::boost::asio::detail::async_result_helper< \
ct, sig>::return_type
#define BOOST_ASIO_HANDLER_TYPE(ct, sig) \
typename ::boost::asio::detail::async_result_helper< \
ct, sig>::completion_handler_type
#else
# define BOOST_ASIO_INITFN_RESULT_TYPE(ct, sig) \
typename ::boost::asio::async_result< \
typename ::boost::asio::decay<ct>::type, sig>::return_type
#define BOOST_ASIO_HANDLER_TYPE(ct, sig) \
typename ::boost::asio::async_result< \
typename ::boost::asio::decay<ct>::type, sig>::completion_handler_type
#endif

#if defined(GENERATING_DOCUMENTATION)
# define BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ct, sig) \
auto
#elif defined(BOOST_ASIO_HAS_RETURN_TYPE_DEDUCTION)
# define BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ct, sig) \
auto
#else
# define BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ct, sig) \
BOOST_ASIO_INITFN_RESULT_TYPE(ct, sig)
#endif

#if defined(GENERATING_DOCUMENTATION)
# define BOOST_ASIO_INITFN_DEDUCED_RESULT_TYPE(ct, sig, expr) \
void_or_deduced
#elif defined(BOOST_ASIO_HAS_DECLTYPE)
# define BOOST_ASIO_INITFN_DEDUCED_RESULT_TYPE(ct, sig, expr) \
decltype expr
#else
# define BOOST_ASIO_INITFN_DEDUCED_RESULT_TYPE(ct, sig, expr) \
BOOST_ASIO_INITFN_RESULT_TYPE(ct, sig)
#endif

#if defined(GENERATING_DOCUMENTATION)

template <typename CompletionToken,
completion_signature Signature,
typename Initiation, typename... Args>
void_or_deduced async_initiate(
BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken),
BOOST_ASIO_MOVE_ARG(Args)... args);

#elif defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename CompletionToken,
BOOST_ASIO_COMPLETION_SIGNATURE Signature,
typename Initiation, typename... Args>
inline typename enable_if<
detail::async_result_has_initiate_memfn<CompletionToken, Signature>::value,
BOOST_ASIO_INITFN_DEDUCED_RESULT_TYPE(CompletionToken, Signature,
(async_result<typename decay<CompletionToken>::type,
Signature>::initiate(declval<BOOST_ASIO_MOVE_ARG(Initiation)>(),
declval<BOOST_ASIO_MOVE_ARG(CompletionToken)>(),
declval<BOOST_ASIO_MOVE_ARG(Args)>()...)))>::type
async_initiate(BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token,
BOOST_ASIO_MOVE_ARG(Args)... args)
{
return async_result<typename decay<CompletionToken>::type,
Signature>::initiate(BOOST_ASIO_MOVE_CAST(Initiation)(initiation),
BOOST_ASIO_MOVE_CAST(CompletionToken)(token),
BOOST_ASIO_MOVE_CAST(Args)(args)...);
}

template <typename CompletionToken,
BOOST_ASIO_COMPLETION_SIGNATURE Signature,
typename Initiation, typename... Args>
inline typename enable_if<
!detail::async_result_has_initiate_memfn<CompletionToken, Signature>::value,
BOOST_ASIO_INITFN_RESULT_TYPE(CompletionToken, Signature)>::type
async_initiate(BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token,
BOOST_ASIO_MOVE_ARG(Args)... args)
{
async_completion<CompletionToken, Signature> completion(token);

BOOST_ASIO_MOVE_CAST(Initiation)(initiation)(
BOOST_ASIO_MOVE_CAST(BOOST_ASIO_HANDLER_TYPE(CompletionToken,
Signature))(completion.completion_handler),
BOOST_ASIO_MOVE_CAST(Args)(args)...);

return completion.result.get();
}

#else 

template <typename CompletionToken,
BOOST_ASIO_COMPLETION_SIGNATURE Signature,
typename Initiation>
inline typename enable_if<
detail::async_result_has_initiate_memfn<CompletionToken, Signature>::value,
BOOST_ASIO_INITFN_DEDUCED_RESULT_TYPE(CompletionToken, Signature,
(async_result<typename decay<CompletionToken>::type,
Signature>::initiate(declval<BOOST_ASIO_MOVE_ARG(Initiation)>(),
declval<BOOST_ASIO_MOVE_ARG(CompletionToken)>())))>::type
async_initiate(BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token)
{
return async_result<typename decay<CompletionToken>::type,
Signature>::initiate(BOOST_ASIO_MOVE_CAST(Initiation)(initiation),
BOOST_ASIO_MOVE_CAST(CompletionToken)(token));
}

template <typename CompletionToken,
BOOST_ASIO_COMPLETION_SIGNATURE Signature,
typename Initiation>
inline typename enable_if<
!detail::async_result_has_initiate_memfn<CompletionToken, Signature>::value,
BOOST_ASIO_INITFN_RESULT_TYPE(CompletionToken, Signature)>::type
async_initiate(BOOST_ASIO_MOVE_ARG(Initiation) initiation,
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token)
{
async_completion<CompletionToken, Signature> completion(token);

BOOST_ASIO_MOVE_CAST(Initiation)(initiation)(
BOOST_ASIO_MOVE_CAST(BOOST_ASIO_HANDLER_TYPE(CompletionToken,
Signature))(completion.completion_handler));

return completion.result.get();
}

#define BOOST_ASIO_PRIVATE_INITIATE_DEF(n) \
template <typename CompletionToken, \
BOOST_ASIO_COMPLETION_SIGNATURE Signature, \
typename Initiation, BOOST_ASIO_VARIADIC_TPARAMS(n)> \
inline typename enable_if< \
detail::async_result_has_initiate_memfn< \
CompletionToken, Signature>::value, \
BOOST_ASIO_INITFN_DEDUCED_RESULT_TYPE(CompletionToken, Signature, \
(async_result<typename decay<CompletionToken>::type, \
Signature>::initiate(declval<BOOST_ASIO_MOVE_ARG(Initiation)>(), \
declval<BOOST_ASIO_MOVE_ARG(CompletionToken)>(), \
BOOST_ASIO_VARIADIC_MOVE_DECLVAL(n))))>::type \
async_initiate(BOOST_ASIO_MOVE_ARG(Initiation) initiation, \
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token, \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return async_result<typename decay<CompletionToken>::type, \
Signature>::initiate(BOOST_ASIO_MOVE_CAST(Initiation)(initiation), \
BOOST_ASIO_MOVE_CAST(CompletionToken)(token), \
BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <typename CompletionToken, \
BOOST_ASIO_COMPLETION_SIGNATURE Signature, \
typename Initiation, BOOST_ASIO_VARIADIC_TPARAMS(n)> \
inline typename enable_if< \
!detail::async_result_has_initiate_memfn< \
CompletionToken, Signature>::value, \
BOOST_ASIO_INITFN_RESULT_TYPE(CompletionToken, Signature)>::type \
async_initiate(BOOST_ASIO_MOVE_ARG(Initiation) initiation, \
BOOST_ASIO_NONDEDUCED_MOVE_ARG(CompletionToken) token, \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
async_completion<CompletionToken, Signature> completion(token); \
\
BOOST_ASIO_MOVE_CAST(Initiation)(initiation)( \
BOOST_ASIO_MOVE_CAST(BOOST_ASIO_HANDLER_TYPE(CompletionToken, \
Signature))(completion.completion_handler), \
BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
\
return completion.result.get(); \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_INITIATE_DEF)
#undef BOOST_ASIO_PRIVATE_INITIATE_DEF

#endif 

#if defined(BOOST_ASIO_HAS_CONCEPTS) \
&& defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES) \
&& defined(BOOST_ASIO_HAS_DECLTYPE)

namespace detail {

template <typename Signature>
struct initiation_archetype
{
template <completion_handler_for<Signature> CompletionHandler>
void operator()(CompletionHandler&&) const
{
}
};

} 

template <typename T, typename Signature>
BOOST_ASIO_CONCEPT completion_token_for =
detail::is_completion_signature<Signature>::value
&&
requires(T&& t)
{
async_initiate<T, Signature>(detail::initiation_archetype<Signature>{}, t);
};

#define BOOST_ASIO_COMPLETION_TOKEN_FOR(s) \
::boost::asio::completion_token_for<s>

#else 

#define BOOST_ASIO_COMPLETION_TOKEN_FOR(s) typename

#endif 

namespace detail {

template <typename T, typename = void>
struct default_completion_token_impl
{
typedef void type;
};

template <typename T>
struct default_completion_token_impl<T,
typename void_type<typename T::default_completion_token_type>::type>
{
typedef typename T::default_completion_token_type type;
};

} 

#if defined(GENERATING_DOCUMENTATION)


template <typename T>
struct default_completion_token
{
typedef see_below type;
};
#else
template <typename T>
struct default_completion_token
: detail::default_completion_token_impl<T>
{
};
#endif

#if defined(BOOST_ASIO_HAS_ALIAS_TEMPLATES)

template <typename T>
using default_completion_token_t = typename default_completion_token<T>::type;

#endif 

#if defined(BOOST_ASIO_HAS_DEFAULT_FUNCTION_TEMPLATE_ARGUMENTS)

#define BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(e) \
= typename ::boost::asio::default_completion_token<e>::type
#define BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(e) \
= typename ::boost::asio::default_completion_token<e>::type()

#else 

#define BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(e)
#define BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(e)

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
