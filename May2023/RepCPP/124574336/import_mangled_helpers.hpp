
#ifndef BOOST_DLL_DETAIL_IMPORT_MANGLED_HELPERS_HPP_
#define BOOST_DLL_DETAIL_IMPORT_MANGLED_HELPERS_HPP_


#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/remove_cv.hpp>


#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace dll { namespace experimental { namespace detail {

template<class ...Args>
struct sequence {};

template<class Value, class Seq> struct push_front;
template<class Value, class ...Args>
struct push_front<Value, sequence<Args...>>
{
typedef sequence<Value, Args...> type;
};

template<class Lhs, class Rhs>
struct unqalified_is_same :
boost::is_same<
typename boost::remove_cv<Lhs>::type,
typename boost::remove_cv<Rhs>::type
>
{
};



template<class T> struct is_function_seq;

template<class Class, class...Args> struct is_function_seq<sequence<Class, Args...>>
: boost::conditional<
boost::is_function<Class>::value,
is_function_seq<sequence<Args...>>,
boost::false_type>::type
{};

template<class Class>
struct is_function_seq<sequence<Class>> : boost::is_function<Class>
{
};

template<>
struct is_function_seq<sequence<>> : boost::false_type
{
};



template <class ...Ts>
struct function_tuple;

template <class Return, class...Args, class T2, class ...Ts>
struct function_tuple<Return(Args...), T2, Ts...>
: function_tuple<T2, Ts...>
{
Return(*f_)(Args...);

constexpr function_tuple(Return(* t)(Args...), T2* t2, Ts* ... ts)
: function_tuple<T2, Ts...>(t2, ts...)
, f_(t)
{}

Return operator()(Args...args) const {
return (*f_)(static_cast<Args>(args)...);
}
using function_tuple<T2, Ts...>::operator();
};

template <class Return, class...Args>
struct function_tuple<Return(Args...)> {
Return(*f_)(Args...);

constexpr function_tuple(Return(* t)(Args...))
: f_(t)
{}

Return operator()(Args...args) const {
return (*f_)(static_cast<Args>(args)...);
}
};




template<class Class, class Func>
struct mem_fn_def
{
typedef Class class_type;
typedef Func  func_type;
typedef typename boost::dll::detail::get_mem_fn_type<Class, Func>::mem_fn mem_fn;
};

template<class ...Args>
struct make_mem_fn_seq;

template<bool, class T0, class T1, class T2>
struct make_mem_fn_seq_getter;

template<class T0, class T1, class T2>
struct make_mem_fn_seq_getter<true, T0, T1, T2>
{
typedef mem_fn_def<T1, T2> type;
};

template<class T0, class T1, class T2>
struct make_mem_fn_seq_getter<false, T0, T1, T2>
{
typedef mem_fn_def<T0, T1> type;
};

template<class Class, class Signature>
struct make_mem_fn_seq<Class, Signature>
{
typedef mem_fn_def<Class, Signature> mem_fn;
typedef sequence<mem_fn>   type;
};

template<class Class>
struct make_mem_fn_seq<Class>
{
typedef sequence<>   type;
};

template<class T0, class T1, class T2, class ... Args>
struct make_mem_fn_seq<T0, T1, T2, Args...>
{

static_assert(boost::is_object<T0>::value, "");
typedef typename make_mem_fn_seq_getter<
unqalified_is_same<T0, T1>::value, T0, T1, T2>::type mem_fn_type;

typedef typename boost::conditional<
unqalified_is_same<T0, T1>::value,
make_mem_fn_seq<T1, Args...>,
make_mem_fn_seq<T0, T2, Args...>> ::type next;

typedef typename push_front<mem_fn_type, typename next::type>::type type;
};








template<class T, class U, class ...Args>
struct is_mem_fn_seq_impl
{
typedef typename boost::conditional<
boost::is_function<U>::value || boost::dll::experimental::detail::unqalified_is_same<T, U>::value,
typename is_mem_fn_seq_impl<T, Args...>::type,
boost::false_type>::type type;
};

template<class T, class U>
struct is_mem_fn_seq_impl<T, U>
{
typedef typename boost::conditional<
boost::is_function<U>::value && boost::is_object<T>::value,
boost::true_type, boost::false_type>::type type;
};

template<class T, class U, class Last>
struct is_mem_fn_seq_impl<T, U, Last>
{
typedef typename boost::conditional<
(boost::is_function<U>::value || boost::dll::experimental::detail::unqalified_is_same<T, U>::value)
&& boost::is_function<Last>::value,
boost::true_type, boost::false_type>::type type;
};

template<class T> struct is_mem_fn_seq : boost::false_type {};

template<class T, class U>
struct is_mem_fn_seq<sequence<T, U>> : boost::conditional<
boost::is_object<T>::value && boost::is_function<U>::value,
boost::true_type, boost::false_type>::type
{
};


template<class T, class Func, class ...Args>
struct is_mem_fn_seq<sequence<T, Func, Args...>> :
boost::conditional<
boost::is_class<T>::value && boost::is_function<Func>::value,
typename is_mem_fn_seq_impl<T, Args...>::type,
boost::false_type>::type {};





template <class ...Ts>
struct mem_fn_tuple;

template <class Class, class Return, class...Args, class T2, class ...Ts>
struct mem_fn_tuple<mem_fn_def<Class, Return(Args...)>, T2, Ts...>
: mem_fn_tuple<T2, Ts...>
{
typedef typename boost::dll::detail::get_mem_fn_type<Class, Return(Args...)>::mem_fn mem_fn;
mem_fn f_;

constexpr mem_fn_tuple(mem_fn f, typename T2::mem_fn t2, typename Ts::mem_fn ... ts)
: mem_fn_tuple<T2, Ts...>(t2, ts...)
, f_(f)
{}

Return operator()(Class* const cl, Args...args) const {
return (cl->*f_)(static_cast<Args>(args)...);
}
using mem_fn_tuple<T2, Ts...>::operator();

};

template <class Class, class Return, class...Args>
struct mem_fn_tuple<mem_fn_def<Class, Return(Args...)>> {
typedef typename boost::dll::detail::get_mem_fn_type<Class, Return(Args...)>::mem_fn mem_fn;
mem_fn f_;

constexpr mem_fn_tuple(mem_fn f)
: f_(f)
{}

Return operator()(Class * const cl, Args...args) const {
return (cl->*f_)(static_cast<Args>(args)...);
}
};

}}}}
#endif 
