
#ifndef BOOST_XPRESSIVE_ACTIONS_HPP_EAN_03_22_2007
#define BOOST_XPRESSIVE_ACTIONS_HPP_EAN_03_22_2007

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/ref.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/noncopyable.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/decay.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/core/matcher/attr_matcher.hpp>
#include <boost/xpressive/detail/core/matcher/attr_end_matcher.hpp>
#include <boost/xpressive/detail/core/matcher/attr_begin_matcher.hpp>
#include <boost/xpressive/detail/core/matcher/predicate_matcher.hpp>
#include <boost/xpressive/detail/utility/ignore_unused.hpp>
#include <boost/xpressive/detail/static/type_traits.hpp>

#include <boost/typeof/std/map.hpp>
#include <boost/typeof/std/string.hpp>

#ifndef BOOST_XPRESSIVE_DOXYGEN_INVOKED
# include <boost/proto/core.hpp>
# include <boost/proto/transform.hpp>
# include <boost/xpressive/detail/core/matcher/action_matcher.hpp>
#endif

#if BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 4510) 
#pragma warning(disable : 4512) 
#pragma warning(disable : 4610) 
#endif

namespace boost { namespace xpressive
{

namespace detail
{
template<typename T, typename U>
struct action_arg
{
typedef T type;
typedef typename add_reference<T>::type reference;

reference cast(void *pv) const
{
return *static_cast<typename remove_reference<T>::type *>(pv);
}
};

template<typename T>
struct value_wrapper
: private noncopyable
{
value_wrapper()
: value()
{}

value_wrapper(T const &t)
: value(t)
{}

T value;
};

struct check_tag
{};

struct BindArg
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename MatchResults, typename Expr>
struct result<This(MatchResults, Expr)>
{
typedef Expr type;
};

template<typename MatchResults, typename Expr>
Expr const & operator ()(MatchResults &what, Expr const &expr) const
{
what.let(expr);
return expr;
}
};

struct let_tag
{};

struct BindArgs
: proto::function<
proto::terminal<let_tag>
, proto::vararg<
proto::when<
proto::assign<proto::_, proto::_>
, proto::call<BindArg(proto::_data, proto::_)>
>
>
>
{};

struct let_domain
: boost::proto::domain<boost::proto::pod_generator<let_> >
{};

template<typename Expr>
struct let_
{
BOOST_PROTO_BASIC_EXTENDS(Expr, let_<Expr>, let_domain)
BOOST_PROTO_EXTENDS_FUNCTION()
};

template<typename Args, typename BidiIter>
void bind_args(let_<Args> const &args, match_results<BidiIter> &what)
{
BindArgs()(args, 0, what);
}

typedef boost::proto::functional::make_expr<proto::tag::function, proto::default_domain> make_function;
}

namespace op
{
struct at
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Cont, typename Idx>
struct result<This(Cont &, Idx)>
{
typedef typename Cont::reference type;
};

template<typename This, typename Cont, typename Idx>
struct result<This(Cont const &, Idx)>
{
typedef typename Cont::const_reference type;
};

template<typename This, typename Cont, typename Idx>
struct result<This(Cont, Idx)>
{
typedef typename Cont::const_reference type;
};

template<typename Cont, typename Idx>
typename Cont::reference operator()(Cont &c, Idx idx BOOST_PROTO_DISABLE_IF_IS_CONST(Cont)) const
{
return c[idx];
}

template<typename Cont, typename Idx>
typename Cont::const_reference operator()(Cont const &c, Idx idx) const
{
return c[idx];
}
};

struct push
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

template<typename Sequence, typename Value>
void operator()(Sequence &seq, Value const &val) const
{
seq.push(val);
}
};

struct push_back
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

template<typename Sequence, typename Value>
void operator()(Sequence &seq, Value const &val) const
{
seq.push_back(val);
}
};

struct push_front
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

template<typename Sequence, typename Value>
void operator()(Sequence &seq, Value const &val) const
{
seq.push_front(val);
}
};

struct pop
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

template<typename Sequence>
void operator()(Sequence &seq) const
{
seq.pop();
}
};

struct pop_back
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

template<typename Sequence>
void operator()(Sequence &seq) const
{
seq.pop_back();
}
};

struct pop_front
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

template<typename Sequence>
void operator()(Sequence &seq) const
{
seq.pop_front();
}
};

struct front
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Sequence>
struct result<This(Sequence)>
{
typedef typename remove_reference<Sequence>::type sequence_type;
typedef
typename mpl::if_c<
is_const<sequence_type>::value
, typename sequence_type::const_reference
, typename sequence_type::reference
>::type
type;
};

template<typename Sequence>
typename result<front(Sequence &)>::type operator()(Sequence &seq) const
{
return seq.front();
}
};

struct back
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Sequence>
struct result<This(Sequence)>
{
typedef typename remove_reference<Sequence>::type sequence_type;
typedef
typename mpl::if_c<
is_const<sequence_type>::value
, typename sequence_type::const_reference
, typename sequence_type::reference
>::type
type;
};

template<typename Sequence>
typename result<back(Sequence &)>::type operator()(Sequence &seq) const
{
return seq.back();
}
};

struct top
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Sequence>
struct result<This(Sequence)>
{
typedef typename remove_reference<Sequence>::type sequence_type;
typedef
typename mpl::if_c<
is_const<sequence_type>::value
, typename sequence_type::value_type const &
, typename sequence_type::value_type &
>::type
type;
};

template<typename Sequence>
typename result<top(Sequence &)>::type operator()(Sequence &seq) const
{
return seq.top();
}
};

struct first
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Pair>
struct result<This(Pair)>
{
typedef typename remove_reference<Pair>::type::first_type type;
};

template<typename Pair>
typename Pair::first_type operator()(Pair const &p) const
{
return p.first;
}
};

struct second
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Pair>
struct result<This(Pair)>
{
typedef typename remove_reference<Pair>::type::second_type type;
};

template<typename Pair>
typename Pair::second_type operator()(Pair const &p) const
{
return p.second;
}
};

struct matched
{
BOOST_PROTO_CALLABLE()
typedef bool result_type;

template<typename Sub>
bool operator()(Sub const &sub) const
{
return sub.matched;
}
};

struct length
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Sub>
struct result<This(Sub)>
{
typedef typename remove_reference<Sub>::type::difference_type type;
};

template<typename Sub>
typename Sub::difference_type operator()(Sub const &sub) const
{
return sub.length();
}
};

struct str
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Sub>
struct result<This(Sub)>
{
typedef typename remove_reference<Sub>::type::string_type type;
};

template<typename Sub>
typename Sub::string_type operator()(Sub const &sub) const
{
return sub.str();
}
};

struct insert
{
BOOST_PROTO_CALLABLE()

struct detail
{
template<typename Sig, typename EnableIf = void>
struct result_detail
{};

template<typename This, typename Cont, typename Value>
struct result_detail<This(Cont, Value), void>
{
typedef typename remove_reference<Cont>::type cont_type;
typedef typename remove_reference<Value>::type value_type;
static cont_type &scont_;
static value_type &svalue_;
typedef char yes_type;
typedef char (&no_type)[2];
static yes_type check_insert_return(typename cont_type::iterator);
static no_type check_insert_return(std::pair<typename cont_type::iterator, bool>);
BOOST_STATIC_CONSTANT(bool, is_iterator = (sizeof(yes_type) == sizeof(check_insert_return(scont_.insert(svalue_)))));
typedef
typename mpl::if_c<
is_iterator
, typename cont_type::iterator
, std::pair<typename cont_type::iterator, bool>
>::type
type;
};

template<typename This, typename Cont, typename It, typename Value>
struct result_detail<This(Cont, It, Value),
typename disable_if<
mpl::or_<
is_integral<typename remove_cv<typename remove_reference<It>::type>::type>
, is_same<
typename remove_cv<typename remove_reference<It>::type>::type
, typename remove_cv<typename remove_reference<Value>::type>::type
>
>
>::type
>
{
typedef typename remove_reference<Cont>::type::iterator type;
};

template<typename This, typename Cont, typename Size, typename T>
struct result_detail<This(Cont, Size, T),
typename enable_if<
is_integral<typename remove_cv<typename remove_reference<Size>::type>::type>
>::type
>
{
typedef typename remove_reference<Cont>::type &type;
};

template<typename This, typename Cont, typename It>
struct result_detail<This(Cont, It, It), void>
{
typedef void type;
};

template<typename This, typename Cont, typename It, typename Size, typename Value>
struct result_detail<This(Cont, It, Size, Value),
typename disable_if<
is_integral<typename remove_cv<typename remove_reference<It>::type>::type>
>::type
>
{
typedef void type;
};

template<typename This, typename Cont, typename Size, typename A0, typename A1>
struct result_detail<This(Cont, Size, A0, A1),
typename enable_if<
is_integral<typename remove_cv<typename remove_reference<Size>::type>::type>
>::type
>
{
typedef typename remove_reference<Cont>::type &type;
};

template<typename This, typename Cont, typename Pos0, typename String, typename Pos1, typename Length>
struct result_detail<This(Cont, Pos0, String, Pos1, Length)>
{
typedef typename remove_reference<Cont>::type &type;
};
};

template<typename Sig>
struct result
{
typedef typename detail::result_detail<Sig>::type type;
};

template<typename Cont, typename A0>
typename result<insert(Cont &, A0 const &)>::type
operator()(Cont &cont, A0 const &a0) const
{
return cont.insert(a0);
}

template<typename Cont, typename A0, typename A1>
typename result<insert(Cont &, A0 const &, A1 const &)>::type
operator()(Cont &cont, A0 const &a0, A1 const &a1) const
{
return cont.insert(a0, a1);
}

template<typename Cont, typename A0, typename A1, typename A2>
typename result<insert(Cont &, A0 const &, A1 const &, A2 const &)>::type
operator()(Cont &cont, A0 const &a0, A1 const &a1, A2 const &a2) const
{
return cont.insert(a0, a1, a2);
}

template<typename Cont, typename A0, typename A1, typename A2, typename A3>
typename result<insert(Cont &, A0 const &, A1 const &, A2 const &, A3 const &)>::type
operator()(Cont &cont, A0 const &a0, A1 const &a1, A2 const &a2, A3 const &a3) const
{
return cont.insert(a0, a1, a2, a3);
}
};

struct make_pair
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename First, typename Second>
struct result<This(First, Second)>
{
typedef typename decay<First>::type first_type;
typedef typename decay<Second>::type second_type;
typedef std::pair<first_type, second_type> type;
};

template<typename First, typename Second>
std::pair<First, Second> operator()(First const &first, Second const &second) const
{
return std::make_pair(first, second);
}
};

template<typename T>
struct as
{
BOOST_PROTO_CALLABLE()
typedef T result_type;

template<typename Value>
T operator()(Value const &val) const
{
return boost::lexical_cast<T>(val);
}

T operator()(csub_match const &val) const
{
return val.matched
? boost::lexical_cast<T>(boost::make_iterator_range(val.first, val.second))
: boost::lexical_cast<T>("");
}

#ifndef BOOST_XPRESSIVE_NO_WREGEX
T operator()(wcsub_match const &val) const
{
return val.matched
? boost::lexical_cast<T>(boost::make_iterator_range(val.first, val.second))
: boost::lexical_cast<T>("");
}
#endif

template<typename BidiIter>
T operator()(sub_match<BidiIter> const &val) const
{
typedef typename iterator_value<BidiIter>::type char_type;
BOOST_MPL_ASSERT_MSG(
(xpressive::detail::is_char<char_type>::value)
, CAN_ONLY_CONVERT_FROM_CHARACTER_SEQUENCES
, (char_type)
);
return this->impl(val, xpressive::detail::is_string_iterator<BidiIter>());
}

private:
template<typename RandIter>
T impl(sub_match<RandIter> const &val, mpl::true_) const
{
return val.matched
? boost::lexical_cast<T>(boost::make_iterator_range(&*val.first, &*val.first + (val.second - val.first)))
: boost::lexical_cast<T>("");
}

template<typename BidiIter>
T impl(sub_match<BidiIter> const &val, mpl::false_) const
{
return boost::lexical_cast<T>(val.str());
}
};

template<typename T>
struct static_cast_
{
BOOST_PROTO_CALLABLE()
typedef T result_type;

template<typename Value>
T operator()(Value const &val) const
{
return static_cast<T>(val);
}
};

template<typename T>
struct dynamic_cast_
{
BOOST_PROTO_CALLABLE()
typedef T result_type;

template<typename Value>
T operator()(Value const &val) const
{
return dynamic_cast<T>(val);
}
};

template<typename T>
struct const_cast_
{
BOOST_PROTO_CALLABLE()
typedef T result_type;

template<typename Value>
T operator()(Value const &val) const
{
return const_cast<T>(val);
}
};

template<typename T>
struct construct
{
BOOST_PROTO_CALLABLE()
typedef T result_type;

T operator()() const
{
return T();
}

template<typename A0>
T operator()(A0 const &a0) const
{
return T(a0);
}

template<typename A0, typename A1>
T operator()(A0 const &a0, A1 const &a1) const
{
return T(a0, a1);
}

template<typename A0, typename A1, typename A2>
T operator()(A0 const &a0, A1 const &a1, A2 const &a2) const
{
return T(a0, a1, a2);
}
};

template<typename Except>
struct throw_
{
BOOST_PROTO_CALLABLE()
typedef void result_type;

void operator()() const
{
BOOST_THROW_EXCEPTION(Except());
}

template<typename A0>
void operator()(A0 const &a0) const
{
BOOST_THROW_EXCEPTION(Except(a0));
}

template<typename A0, typename A1>
void operator()(A0 const &a0, A1 const &a1) const
{
BOOST_THROW_EXCEPTION(Except(a0, a1));
}

template<typename A0, typename A1, typename A2>
void operator()(A0 const &a0, A1 const &a1, A2 const &a2) const
{
BOOST_THROW_EXCEPTION(Except(a0, a1, a2));
}
};

struct unwrap_reference
{
BOOST_PROTO_CALLABLE()
template<typename Sig>
struct result {};

template<typename This, typename Ref>
struct result<This(Ref)>
{
typedef typename boost::unwrap_reference<Ref>::type &type;
};

template<typename This, typename Ref>
struct result<This(Ref &)>
{
typedef typename boost::unwrap_reference<Ref>::type &type;
};

template<typename T>
T &operator()(boost::reference_wrapper<T> r) const
{
return static_cast<T &>(r);
}
};
}



template<typename PolymorphicFunctionObject>
struct function
{
typedef typename proto::terminal<PolymorphicFunctionObject>::type type;
};

function<op::at>::type const at = {{}};

function<op::push>::type const push = {{}};

function<op::push_back>::type const push_back = {{}};

function<op::push_front>::type const push_front = {{}};

function<op::pop>::type const pop = {{}};

function<op::pop_back>::type const pop_back = {{}};

function<op::pop_front>::type const pop_front = {{}};

function<op::top>::type const top = {{}};

function<op::back>::type const back = {{}};

function<op::front>::type const front = {{}};

function<op::first>::type const first = {{}};

function<op::second>::type const second = {{}};

function<op::matched>::type const matched = {{}};

function<op::length>::type const length = {{}};

function<op::str>::type const str = {{}};

function<op::insert>::type const insert = {{}};

function<op::make_pair>::type const make_pair = {{}};

function<op::unwrap_reference>::type const unwrap_reference = {{}};


template<typename T>
struct value
: proto::extends<typename proto::terminal<T>::type, value<T> >
{
typedef proto::extends<typename proto::terminal<T>::type, value<T> > base_type;

value()
: base_type()
{}

explicit value(T const &t)
: base_type(base_type::proto_base_expr::make(t))
{}

using base_type::operator=;

T &get()
{
return proto::value(*this);
}

T const &get() const
{
return proto::value(*this);
}
};


template<typename T>
struct reference
: proto::extends<typename proto::terminal<reference_wrapper<T> >::type, reference<T> >
{
typedef proto::extends<typename proto::terminal<reference_wrapper<T> >::type, reference<T> > base_type;

explicit reference(T &t)
: base_type(base_type::proto_base_expr::make(boost::ref(t)))
{}

using base_type::operator=;

T &get() const
{
return proto::value(*this).get();
}
};


template<typename T>
struct local
: detail::value_wrapper<T>
, proto::terminal<reference_wrapper<T> >::type
{
typedef typename proto::terminal<reference_wrapper<T> >::type base_type;

local()
: detail::value_wrapper<T>()
, base_type(base_type::make(boost::ref(detail::value_wrapper<T>::value)))
{}

explicit local(T const &t)
: detail::value_wrapper<T>(t)
, base_type(base_type::make(boost::ref(detail::value_wrapper<T>::value)))
{}

using base_type::operator=;

T &get()
{
return proto::value(*this);
}

T const &get() const
{
return proto::value(*this);
}
};

template<typename T, typename A>
typename detail::make_function::impl<op::as<T> const, A const &>::result_type const
as(A const &a)
{
return detail::make_function::impl<op::as<T> const, A const &>()((op::as<T>()), a);
}

template<typename T, typename A>
typename detail::make_function::impl<op::static_cast_<T> const, A const &>::result_type const
static_cast_(A const &a)
{
return detail::make_function::impl<op::static_cast_<T> const, A const &>()((op::static_cast_<T>()), a);
}

template<typename T, typename A>
typename detail::make_function::impl<op::dynamic_cast_<T> const, A const &>::result_type const
dynamic_cast_(A const &a)
{
return detail::make_function::impl<op::dynamic_cast_<T> const, A const &>()((op::dynamic_cast_<T>()), a);
}

template<typename T, typename A>
typename detail::make_function::impl<op::const_cast_<T> const, A const &>::result_type const
const_cast_(A const &a)
{
return detail::make_function::impl<op::const_cast_<T> const, A const &>()((op::const_cast_<T>()), a);
}

template<typename T>
value<T> const val(T const &t)
{
return value<T>(t);
}

template<typename T>
reference<T> const ref(T &t)
{
return reference<T>(t);
}

template<typename T>
reference<T const> const cref(T const &t)
{
return reference<T const>(t);
}




#ifdef BOOST_XPRESSIVE_DOXYGEN_INVOKED 
template<typename T>
detail::unspecified check(T const &t);
#else
proto::terminal<detail::check_tag>::type const check = {{}};
#endif



#ifdef BOOST_XPRESSIVE_DOXYGEN_INVOKED 
template<typename...ArgBindings>
detail::unspecified let(ArgBindings const &...args);
#else
detail::let_<proto::terminal<detail::let_tag>::type> const let = {{{}}};
#endif




#ifdef BOOST_XPRESSIVE_DOXYGEN_INVOKED 
template<typename T, int I = 0>
struct placeholder
{
detail::unspecified operator=(T &t) const;
detail::unspecified operator=(T const &t) const;
};
#else
template<typename T, int I, typename Dummy>
struct placeholder
{
typedef placeholder<T, I, Dummy> this_type;
typedef
typename proto::terminal<detail::action_arg<T, mpl::int_<I> > >::type
action_arg_type;

BOOST_PROTO_EXTENDS(action_arg_type, this_type, proto::default_domain)
};
#endif

#ifdef BOOST_XPRESSIVE_DOXYGEN_INVOKED 
template<typename T, typename ...Args>
detail::unspecified construct(Args const &...args);
#else
#define BOOST_PROTO_LOCAL_MACRO(N, typename_A, A_const_ref, A_const_ref_a, a)                       \
template<typename X2_0 BOOST_PP_COMMA_IF(N) typename_A(N)>                                      \
typename detail::make_function::impl<                                                           \
op::construct<X2_0> const                                                                   \
BOOST_PP_COMMA_IF(N) A_const_ref(N)                                                         \
>::result_type const                                                                            \
construct(A_const_ref_a(N))                                                                     \
{                                                                                               \
return detail::make_function::impl<                                                         \
op::construct<X2_0> const                                                               \
BOOST_PP_COMMA_IF(N) A_const_ref(N)                                                     \
>()((op::construct<X2_0>()) BOOST_PP_COMMA_IF(N) a(N));                                     \
}                                                                                               \
\
template<typename X2_0 BOOST_PP_COMMA_IF(N) typename_A(N)>                                      \
typename detail::make_function::impl<                                                           \
op::throw_<X2_0> const                                                                      \
BOOST_PP_COMMA_IF(N) A_const_ref(N)                                                         \
>::result_type const                                                                            \
throw_(A_const_ref_a(N))                                                                        \
{                                                                                               \
return detail::make_function::impl<                                                         \
op::throw_<X2_0> const                                                                  \
BOOST_PP_COMMA_IF(N) A_const_ref(N)                                                     \
>()((op::throw_<X2_0>()) BOOST_PP_COMMA_IF(N) a(N));                                        \
}                                                                                               \


#define BOOST_PROTO_LOCAL_a         BOOST_PROTO_a                               
#define BOOST_PROTO_LOCAL_LIMITS    (0, BOOST_PP_DEC(BOOST_PROTO_MAX_ARITY))    
#include BOOST_PROTO_LOCAL_ITERATE()
#endif

namespace detail
{
inline void ignore_unused_regex_actions()
{
detail::ignore_unused(xpressive::at);
detail::ignore_unused(xpressive::push);
detail::ignore_unused(xpressive::push_back);
detail::ignore_unused(xpressive::push_front);
detail::ignore_unused(xpressive::pop);
detail::ignore_unused(xpressive::pop_back);
detail::ignore_unused(xpressive::pop_front);
detail::ignore_unused(xpressive::top);
detail::ignore_unused(xpressive::back);
detail::ignore_unused(xpressive::front);
detail::ignore_unused(xpressive::first);
detail::ignore_unused(xpressive::second);
detail::ignore_unused(xpressive::matched);
detail::ignore_unused(xpressive::length);
detail::ignore_unused(xpressive::str);
detail::ignore_unused(xpressive::insert);
detail::ignore_unused(xpressive::make_pair);
detail::ignore_unused(xpressive::unwrap_reference);
detail::ignore_unused(xpressive::check);
detail::ignore_unused(xpressive::let);
}

struct mark_nbr
{
BOOST_PROTO_CALLABLE()
typedef int result_type;

int operator()(mark_placeholder m) const
{
return m.mark_number_;
}
};

struct ReplaceAlgo
: proto::or_<
proto::when<
proto::terminal<mark_placeholder>
, op::at(proto::_data, proto::call<mark_nbr(proto::_value)>)
>
, proto::when<
proto::terminal<any_matcher>
, op::at(proto::_data, proto::size_t<0>)
>
, proto::when<
proto::terminal<reference_wrapper<proto::_> >
, op::unwrap_reference(proto::_value)
>
, proto::_default<ReplaceAlgo>
>
{};
}
}}

#if BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
