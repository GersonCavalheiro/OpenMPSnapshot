

#ifndef BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_HPP
#define BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_HPP

#include "boost/type_traits/add_reference.hpp"
#include "boost/type_traits/add_const.hpp"
#include "boost/type_traits/remove_const.hpp"
#include "boost/lambda/detail/lambda_fwd.hpp"
#include "boost/lambda/detail/lambda_traits.hpp"

namespace boost { 
namespace lambda {

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

template <class T>
class identity {

T elem;
public:

typedef T element_t;

typedef typename boost::add_reference<
typename boost::add_const<T>::type
>::type par_t;

explicit identity(par_t t) : elem(t) {}

template <typename SigArgs> 
struct sig { typedef typename boost::remove_const<element_t>::type type; };

template<class RET, CALL_TEMPLATE_ARGS>
RET call(CALL_FORMAL_ARGS) const { CALL_USE_ARGS; return elem; }
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

template <class T> 
inline lambda_functor<identity<T&> > var(T& t) { return identity<T&>(t); }

template <class T>
lambda_functor<T> var(const lambda_functor<T>& t) { return t; }

template <class T> struct var_type {
typedef lambda_functor<identity<T&> > type;
};


template <class T> 
inline 
lambda_functor<identity<typename bound_argument_conversion<const T>::type> >
constant(const T& t) { 
return identity<typename bound_argument_conversion<const T>::type>(t); 
}
template <class T>
lambda_functor<T> constant(const lambda_functor<T>& t) { return t; }

template <class T> struct constant_type {
typedef 
lambda_functor<
identity<typename bound_argument_conversion<const T>::type> 
> type;
};



template <class T> 
inline lambda_functor<identity<const T&> > constant_ref(const T& t) { 
return identity<const T&>(t); 
}
template <class T>
lambda_functor<T> constant_ref(const lambda_functor<T>& t) { return t; }

template <class T> struct constant_ref_type {
typedef 
lambda_functor<identity<const T&> > type;
};



template <class T>
struct as_lambda_functor { 
typedef typename 
detail::remove_reference_and_cv<T>::type plain_T;
typedef typename 
detail::IF<is_lambda_functor<plain_T>::value, 
plain_T,
lambda_functor<
identity<typename bound_argument_conversion<T>::type> 
>
>::RET type; 
};

template <class T> 
inline 
lambda_functor<identity<typename bound_argument_conversion<const T>::type> > 
to_lambda_functor(const T& t) { 
return identity<typename bound_argument_conversion<const T>::type>(t);
}

template <class T> 
inline lambda_functor<T> 
to_lambda_functor(const lambda_functor<T>& t) { 
return t;
}

namespace detail {   



template <class T> struct constify_rvals {
template<class U>
static inline const U& go(const U& u) { return u; }
};

template <class T> struct constify_rvals<T&> {
template<class U>
static inline U& go(U& u) { return u; }
};

template <class T> struct is_null_type 
{ BOOST_STATIC_CONSTANT(bool, value = false); };

template <> struct is_null_type<null_type> 
{ BOOST_STATIC_CONSTANT(bool, value = true); };

template<class Tuple> struct has_null_type {
BOOST_STATIC_CONSTANT(bool, value = (is_null_type<typename Tuple::head_type>::value || has_null_type<typename Tuple::tail_type>::value));
};
template<> struct has_null_type<null_type> {
BOOST_STATIC_CONSTANT(bool, value = false);
};




template<class Args, class SigArgs>
class deduce_argument_types_ {
typedef typename as_lambda_functor<typename Args::head_type>::type lf_t;
typedef typename lf_t::inherited::template sig<SigArgs>::type el_t;  
public:
typedef
boost::tuples::cons<
el_t, 
typename deduce_argument_types_<typename Args::tail_type, SigArgs>::type
> type;
};

template<class SigArgs>
class deduce_argument_types_<null_type, SigArgs> {
public:
typedef null_type type; 
};



template<class Args, class SigArgs>
class deduce_non_ref_argument_types_ {
typedef typename as_lambda_functor<typename Args::head_type>::type lf_t;
typedef typename lf_t::inherited::template sig<SigArgs>::type el_t;  
public:
typedef
boost::tuples::cons<
typename boost::remove_reference<el_t>::type, 
typename deduce_non_ref_argument_types_<typename Args::tail_type, SigArgs>::type
> type;
};

template<class SigArgs>
class deduce_non_ref_argument_types_<null_type, SigArgs> {
public:
typedef null_type type; 
};


template<class Args, class SigArgs>
class deduce_argument_types {
typedef typename deduce_argument_types_<Args, SigArgs>::type t1;
public:
typedef typename detail::IF<
has_null_type<t1>::value, null_type, t1
>::RET type; 
};


template<class Args, class SigArgs>
class deduce_non_ref_argument_types {
typedef typename deduce_non_ref_argument_types_<Args, SigArgs>::type t1;
public:
typedef typename detail::IF<
has_null_type<t1>::value, null_type, t1
>::RET type; 
};

template <int N, class Args, class SigArgs>
struct nth_return_type_sig {
typedef typename 
as_lambda_functor<
typename boost::tuples::element<N, Args>::type 
>::type lf_type;

typedef typename lf_type::inherited::template sig<SigArgs>::type type;  
};

template<int N, class Tuple> struct element_or_null {
typedef typename boost::tuples::element<N, Tuple>::type type;
};

template<int N> struct element_or_null<N, null_type> {
typedef null_type type;
};




} 


template<class RET, class Args>
class lambda_functor_base<explicit_return_type_action<RET>, Args> 
{
public:
Args args;

typedef RET result_type;

explicit lambda_functor_base(const Args& a) : args(a) {}

template <class SigArgs> struct sig { typedef RET type; };

template<class RET_, CALL_TEMPLATE_ARGS>
RET call(CALL_FORMAL_ARGS) const 
{
return detail::constify_rvals<RET>::go(
detail::r_select<RET>::go(boost::tuples::get<0>(args), CALL_ACTUAL_ARGS));
}
};

template<class Args>
class lambda_functor_base<protect_action, Args>
{
public:
Args args;
public:

explicit lambda_functor_base(const Args& a) : args(a) {}


template<class RET, CALL_TEMPLATE_ARGS>
RET call(CALL_FORMAL_ARGS) const 
{
CALL_USE_ARGS;
return boost::tuples::get<0>(args);
}

template<class SigArgs> struct sig { 
typedef typename boost::tuples::element<0, Args>::type type;
};
};

class do_nothing_action {};

template<class Args>
class lambda_functor_base<do_nothing_action, Args> {
public:
lambda_functor_base() {}


template<class RET, CALL_TEMPLATE_ARGS> RET call(CALL_FORMAL_ARGS) const {
return CALL_USE_ARGS;
}

template<class SigArgs> struct sig { typedef void type; };
};  






template<class Act, class Args>  
class lambda_functor_base<action<0, Act>, Args>           
{  
public:  
explicit lambda_functor_base(const Args& ) {}  

template<class SigArgs> struct sig {  
typedef typename return_type_N<Act, null_type>::type type;
};

template<class RET, CALL_TEMPLATE_ARGS>  
RET call(CALL_FORMAL_ARGS) const {  
CALL_USE_ARGS;
return Act::template apply<RET>();
}
};


#if defined BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART  
#error "Multiple defines of BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART"  
#endif  


#define BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(ARITY)             \
template<class Act, class Args>                                        \
class lambda_functor_base<action<ARITY, Act>, Args>                    \
{                                                                      \
public:                                                                \
Args args;                                                           \
\
explicit lambda_functor_base(const Args& a) : args(a) {}             \
\
template<class SigArgs> struct sig {                                 \
typedef typename                                                   \
detail::deduce_argument_types<Args, SigArgs>::type rets_t;         \
public:                                                              \
typedef typename                                                   \
return_type_N_prot<Act, rets_t>::type type;                      \
};                                                                   \
\
\
template<class RET, CALL_TEMPLATE_ARGS>                              \
RET call(CALL_FORMAL_ARGS) const {                                   \
using boost::tuples::get;                                          \
using detail::constify_rvals;                                      \
using detail::r_select;                                            \
using detail::element_or_null;                                     \
using detail::deduce_argument_types;                                

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(1)

typedef typename
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS))
);
}
};


BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(2)

typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(3)

typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;

typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(4)
typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(5)
typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;
typedef typename element_or_null<4, rets_t>::type rt4;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt4>::go(r_select<rt4>::go(get<4>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(6)

typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;
typedef typename element_or_null<4, rets_t>::type rt4;
typedef typename element_or_null<5, rets_t>::type rt5;


return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt4>::go(r_select<rt4>::go(get<4>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt5>::go(r_select<rt5>::go(get<5>(args), CALL_ACTUAL_ARGS)) 
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(7)
typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;
typedef typename element_or_null<4, rets_t>::type rt4;
typedef typename element_or_null<5, rets_t>::type rt5;
typedef typename element_or_null<6, rets_t>::type rt6;


return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt4>::go(r_select<rt4>::go(get<4>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt5>::go(r_select<rt5>::go(get<5>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt6>::go(r_select<rt6>::go(get<6>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(8)
typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;
typedef typename element_or_null<4, rets_t>::type rt4;
typedef typename element_or_null<5, rets_t>::type rt5;
typedef typename element_or_null<6, rets_t>::type rt6;
typedef typename element_or_null<7, rets_t>::type rt7;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt4>::go(r_select<rt4>::go(get<4>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt5>::go(r_select<rt5>::go(get<5>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt6>::go(r_select<rt6>::go(get<6>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt7>::go(r_select<rt7>::go(get<7>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(9)
typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;
typedef typename element_or_null<4, rets_t>::type rt4;
typedef typename element_or_null<5, rets_t>::type rt5;
typedef typename element_or_null<6, rets_t>::type rt6;
typedef typename element_or_null<7, rets_t>::type rt7;
typedef typename element_or_null<8, rets_t>::type rt8;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt4>::go(r_select<rt4>::go(get<4>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt5>::go(r_select<rt5>::go(get<5>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt6>::go(r_select<rt6>::go(get<6>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt7>::go(r_select<rt7>::go(get<7>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt8>::go(r_select<rt8>::go(get<8>(args), CALL_ACTUAL_ARGS))
);
}
};

BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART(10) 
typedef typename 
deduce_argument_types<Args, tuple<CALL_REFERENCE_TYPES> >::type rets_t;
typedef typename element_or_null<0, rets_t>::type rt0;
typedef typename element_or_null<1, rets_t>::type rt1;
typedef typename element_or_null<2, rets_t>::type rt2;
typedef typename element_or_null<3, rets_t>::type rt3;
typedef typename element_or_null<4, rets_t>::type rt4;
typedef typename element_or_null<5, rets_t>::type rt5;
typedef typename element_or_null<6, rets_t>::type rt6;
typedef typename element_or_null<7, rets_t>::type rt7;
typedef typename element_or_null<8, rets_t>::type rt8;
typedef typename element_or_null<9, rets_t>::type rt9;

return Act::template apply<RET>(
constify_rvals<rt0>::go(r_select<rt0>::go(get<0>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt1>::go(r_select<rt1>::go(get<1>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt2>::go(r_select<rt2>::go(get<2>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt3>::go(r_select<rt3>::go(get<3>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt4>::go(r_select<rt4>::go(get<4>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt5>::go(r_select<rt5>::go(get<5>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt6>::go(r_select<rt6>::go(get<6>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt7>::go(r_select<rt7>::go(get<7>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt8>::go(r_select<rt8>::go(get<8>(args), CALL_ACTUAL_ARGS)),
constify_rvals<rt9>::go(r_select<rt9>::go(get<9>(args), CALL_ACTUAL_ARGS)) 
);
}
};

#undef BOOST_LAMBDA_LAMBDA_FUNCTOR_BASE_FIRST_PART


} 
} 

#endif
