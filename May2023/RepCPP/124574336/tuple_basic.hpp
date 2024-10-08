




#ifndef BOOST_TUPLE_BASIC_HPP
#define BOOST_TUPLE_BASIC_HPP


#include <utility> 

#include <boost/type_traits/cv_traits.hpp>
#include <boost/type_traits/function_traits.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/utility/swap.hpp>

#include <boost/detail/workaround.hpp> 

#if defined(BOOST_GCC) && (BOOST_GCC >= 40700)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace boost {
namespace tuples {

struct null_type {};

namespace detail {
inline const null_type cnull() { return null_type(); }



template <bool If, class Then, class Else> struct IF { typedef Then RET; };

template <class Then, class Else> struct IF<false, Then, Else> {
typedef Else RET;
};

} 

template <class HT, class TT> struct cons;


template <
class T0 = null_type, class T1 = null_type, class T2 = null_type,
class T3 = null_type, class T4 = null_type, class T5 = null_type,
class T6 = null_type, class T7 = null_type, class T8 = null_type,
class T9 = null_type>
class tuple;

template<class T> struct length;



namespace detail {

template<class T>
class generate_error;

template<int N>
struct drop_front {
template<class Tuple>
struct apply {
typedef BOOST_DEDUCED_TYPENAME drop_front<N-1>::BOOST_NESTED_TEMPLATE
apply<Tuple> next;
typedef BOOST_DEDUCED_TYPENAME next::type::tail_type type;
static const type& call(const Tuple& tup) {
return next::call(tup).tail;
}
};
};

template<>
struct drop_front<0> {
template<class Tuple>
struct apply {
typedef Tuple type;
static const type& call(const Tuple& tup) {
return tup;
}
};
};

} 



#ifndef BOOST_NO_CV_SPECIALIZATIONS

template<int N, class T>
struct element
{
typedef BOOST_DEDUCED_TYPENAME detail::drop_front<N>::BOOST_NESTED_TEMPLATE
apply<T>::type::head_type type;
};

template<int N, class T>
struct element<N, const T>
{
private:
typedef BOOST_DEDUCED_TYPENAME detail::drop_front<N>::BOOST_NESTED_TEMPLATE
apply<T>::type::head_type unqualified_type;
public:
#if BOOST_WORKAROUND(BOOST_BORLANDC,<0x600)
typedef const unqualified_type type;
#else
typedef BOOST_DEDUCED_TYPENAME boost::add_const<unqualified_type>::type type;
#endif
};
#else 

namespace detail {

template<int N, class T, bool IsConst>
struct element_impl
{
typedef BOOST_DEDUCED_TYPENAME detail::drop_front<N>::BOOST_NESTED_TEMPLATE
apply<T>::type::head_type type;
};

template<int N, class T>
struct element_impl<N, T, true >
{
typedef BOOST_DEDUCED_TYPENAME detail::drop_front<N>::BOOST_NESTED_TEMPLATE
apply<T>::type::head_type unqualified_type;
typedef const unqualified_type type;
};

} 


template<int N, class T>
struct element:
public detail::element_impl<N, T, ::boost::is_const<T>::value>
{
};

#endif





template <class T> struct access_traits {
typedef const T& const_type;
typedef T& non_const_type;

typedef const typename boost::remove_cv<T>::type& parameter_type;

};

template <class T> struct access_traits<T&> {

typedef T& const_type;
typedef T& non_const_type;

typedef T& parameter_type;
};


template<int N, class HT, class TT>
inline typename access_traits<
typename element<N, cons<HT, TT> >::type
>::non_const_type
get(cons<HT, TT>& c) {
typedef BOOST_DEDUCED_TYPENAME detail::drop_front<N>::BOOST_NESTED_TEMPLATE
apply<cons<HT, TT> > impl;
typedef BOOST_DEDUCED_TYPENAME impl::type cons_element;
return const_cast<cons_element&>(impl::call(c)).head;
}

template<int N, class HT, class TT>
inline typename access_traits<
typename element<N, cons<HT, TT> >::type
>::const_type
get(const cons<HT, TT>& c) {
typedef BOOST_DEDUCED_TYPENAME detail::drop_front<N>::BOOST_NESTED_TEMPLATE
apply<cons<HT, TT> > impl;
return impl::call(c).head;
}

namespace detail {


template <class T> class non_storeable_type {
non_storeable_type();
};

template <class T> struct wrap_non_storeable_type {
typedef typename IF<
::boost::is_function<T>::value, non_storeable_type<T>, T
>::RET type;
};
template <> struct wrap_non_storeable_type<void> {
typedef non_storeable_type<void> type;
};

} 

template <class HT, class TT>
struct cons {

typedef HT head_type;
typedef TT tail_type;

typedef typename
detail::wrap_non_storeable_type<head_type>::type stored_head_type;

stored_head_type head;
tail_type tail;

typename access_traits<stored_head_type>::non_const_type
get_head() { return head; }

typename access_traits<tail_type>::non_const_type
get_tail() { return tail; }

typename access_traits<stored_head_type>::const_type
get_head() const { return head; }

typename access_traits<tail_type>::const_type
get_tail() const { return tail; }

cons() : head(), tail() {}


cons(typename access_traits<stored_head_type>::parameter_type h,
const tail_type& t)
: head (h), tail(t) {}

template <class T1, class T2, class T3, class T4, class T5,
class T6, class T7, class T8, class T9, class T10>
cons( T1& t1, T2& t2, T3& t3, T4& t4, T5& t5,
T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 )
: head (t1),
tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, detail::cnull())
{}

template <class T2, class T3, class T4, class T5,
class T6, class T7, class T8, class T9, class T10>
cons( const null_type& , T2& t2, T3& t3, T4& t4, T5& t5,
T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 )
: head (),
tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, detail::cnull())
{}

cons( const cons& u ) : head(u.head), tail(u.tail) {}

template <class HT2, class TT2>
cons( const cons<HT2, TT2>& u ) : head(u.head), tail(u.tail) {}

template <class HT2, class TT2>
cons& operator=( const cons<HT2, TT2>& u ) {
head=u.head; tail=u.tail; return *this;
}

cons& operator=(const cons& u) {
head = u.head; tail = u.tail;  return *this;
}

template <class T1, class T2>
cons& operator=( const std::pair<T1, T2>& u ) {
BOOST_STATIC_ASSERT(length<cons>::value == 2); 
head = u.first; tail.head = u.second; return *this;
}

template <int N>
typename access_traits<
typename element<N, cons<HT, TT> >::type
>::non_const_type
get() {
return boost::tuples::get<N>(*this); 
}

template <int N>
typename access_traits<
typename element<N, cons<HT, TT> >::type
>::const_type
get() const {
return boost::tuples::get<N>(*this); 
}
};

template <class HT>
struct cons<HT, null_type> {

typedef HT head_type;
typedef null_type tail_type;
typedef cons<HT, null_type> self_type;

typedef typename
detail::wrap_non_storeable_type<head_type>::type stored_head_type;
stored_head_type head;

typename access_traits<stored_head_type>::non_const_type
get_head() { return head; }

null_type get_tail() { return null_type(); }

typename access_traits<stored_head_type>::const_type
get_head() const { return head; }

const null_type get_tail() const { return null_type(); }

cons() : head() {}

cons(typename access_traits<stored_head_type>::parameter_type h,
const null_type& = null_type())
: head (h) {}

template<class T1>
cons(T1& t1, const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&)
: head (t1) {}

cons(const null_type&,
const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&)
: head () {}

cons( const cons& u ) : head(u.head) {}

template <class HT2>
cons( const cons<HT2, null_type>& u ) : head(u.head) {}

template <class HT2>
cons& operator=(const cons<HT2, null_type>& u )
{ head = u.head; return *this; }

cons& operator=(const cons& u) { head = u.head; return *this; }

template <int N>
typename access_traits<
typename element<N, self_type>::type
>::non_const_type
get() {
return boost::tuples::get<N>(*this);
}

template <int N>
typename access_traits<
typename element<N, self_type>::type
>::const_type
get() const {
return boost::tuples::get<N>(*this);
}

};


template<class T>
struct length: boost::integral_constant<int, 1 + length<typename T::tail_type>::value>
{
};

template<>
struct length<tuple<> >: boost::integral_constant<int, 0>
{
};

template<>
struct length<tuple<> const>: boost::integral_constant<int, 0>
{
};

template<>
struct length<null_type>: boost::integral_constant<int, 0>
{
};

template<>
struct length<null_type const>: boost::integral_constant<int, 0>
{
};

namespace detail {

template <class T0, class T1, class T2, class T3, class T4,
class T5, class T6, class T7, class T8, class T9>
struct map_tuple_to_cons
{
typedef cons<T0,
typename map_tuple_to_cons<T1, T2, T3, T4, T5,
T6, T7, T8, T9, null_type>::type
> type;
};

template <>
struct map_tuple_to_cons<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>
{
typedef null_type type;
};

} 

template <class T0, class T1, class T2, class T3, class T4,
class T5, class T6, class T7, class T8, class T9>

class tuple :
public detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
{
public:
typedef typename
detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type inherited;
typedef typename inherited::head_type head_type;
typedef typename inherited::tail_type tail_type;


tuple() {}

explicit tuple(typename access_traits<T0>::parameter_type t0)
: inherited(t0, detail::cnull(), detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1)
: inherited(t0, t1, detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2)
: inherited(t0, t1, t2, detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3)
: inherited(t0, t1, t2, t3, detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull(),
detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3,
typename access_traits<T4>::parameter_type t4)
: inherited(t0, t1, t2, t3, t4, detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull(), detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3,
typename access_traits<T4>::parameter_type t4,
typename access_traits<T5>::parameter_type t5)
: inherited(t0, t1, t2, t3, t4, t5, detail::cnull(), detail::cnull(),
detail::cnull(), detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3,
typename access_traits<T4>::parameter_type t4,
typename access_traits<T5>::parameter_type t5,
typename access_traits<T6>::parameter_type t6)
: inherited(t0, t1, t2, t3, t4, t5, t6, detail::cnull(),
detail::cnull(), detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3,
typename access_traits<T4>::parameter_type t4,
typename access_traits<T5>::parameter_type t5,
typename access_traits<T6>::parameter_type t6,
typename access_traits<T7>::parameter_type t7)
: inherited(t0, t1, t2, t3, t4, t5, t6, t7, detail::cnull(),
detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3,
typename access_traits<T4>::parameter_type t4,
typename access_traits<T5>::parameter_type t5,
typename access_traits<T6>::parameter_type t6,
typename access_traits<T7>::parameter_type t7,
typename access_traits<T8>::parameter_type t8)
: inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8, detail::cnull()) {}

tuple(typename access_traits<T0>::parameter_type t0,
typename access_traits<T1>::parameter_type t1,
typename access_traits<T2>::parameter_type t2,
typename access_traits<T3>::parameter_type t3,
typename access_traits<T4>::parameter_type t4,
typename access_traits<T5>::parameter_type t5,
typename access_traits<T6>::parameter_type t6,
typename access_traits<T7>::parameter_type t7,
typename access_traits<T8>::parameter_type t8,
typename access_traits<T9>::parameter_type t9)
: inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9) {}


template<class U1, class U2>
tuple(const cons<U1, U2>& p) : inherited(p) {}

template <class U1, class U2>
tuple& operator=(const cons<U1, U2>& k) {
inherited::operator=(k);
return *this;
}

template <class U1, class U2>
tuple& operator=(const std::pair<U1, U2>& k) {
BOOST_STATIC_ASSERT(length<tuple>::value == 2);
this->head = k.first;
this->tail.head = k.second;
return *this;
}

};

template <>
class tuple<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>  :
public null_type
{
public:
typedef null_type inherited;
};


namespace detail {

struct swallow_assign;
typedef void (detail::swallow_assign::*ignore_t)();
struct swallow_assign {
swallow_assign(ignore_t(*)(ignore_t)) {}
template<typename T>
swallow_assign const& operator=(const T&) const {
return *this;
}
};


} 

inline detail::ignore_t ignore(detail::ignore_t) { return 0; }






template<class T>
struct make_tuple_traits {
typedef T type;


};


template<class T>
struct make_tuple_traits<T&> {
typedef typename
detail::generate_error<T&>::
do_not_use_with_reference_type error;
};

template<class T, int n>  struct make_tuple_traits <T[n]> {
typedef const T (&type)[n];
};

template<class T, int n>
struct make_tuple_traits<const T[n]> {
typedef const T (&type)[n];
};

template<class T, int n>  struct make_tuple_traits<volatile T[n]> {
typedef const volatile T (&type)[n];
};

template<class T, int n>
struct make_tuple_traits<const volatile T[n]> {
typedef const volatile T (&type)[n];
};

template<class T>
struct make_tuple_traits<reference_wrapper<T> >{
typedef T& type;
};

template<class T>
struct make_tuple_traits<const reference_wrapper<T> >{
typedef T& type;
};

template<>
struct make_tuple_traits<detail::ignore_t(detail::ignore_t)> {
typedef detail::swallow_assign type;
};



namespace detail {

template <
class T0 = null_type, class T1 = null_type, class T2 = null_type,
class T3 = null_type, class T4 = null_type, class T5 = null_type,
class T6 = null_type, class T7 = null_type, class T8 = null_type,
class T9 = null_type
>
struct make_tuple_mapper {
typedef
tuple<typename make_tuple_traits<T0>::type,
typename make_tuple_traits<T1>::type,
typename make_tuple_traits<T2>::type,
typename make_tuple_traits<T3>::type,
typename make_tuple_traits<T4>::type,
typename make_tuple_traits<T5>::type,
typename make_tuple_traits<T6>::type,
typename make_tuple_traits<T7>::type,
typename make_tuple_traits<T8>::type,
typename make_tuple_traits<T9>::type> type;
};

} 

inline tuple<> make_tuple() {
return tuple<>();
}

template<class T0>
inline typename detail::make_tuple_mapper<T0>::type
make_tuple(const T0& t0) {
typedef typename detail::make_tuple_mapper<T0>::type t;
return t(t0);
}

template<class T0, class T1>
inline typename detail::make_tuple_mapper<T0, T1>::type
make_tuple(const T0& t0, const T1& t1) {
typedef typename detail::make_tuple_mapper<T0, T1>::type t;
return t(t0, t1);
}

template<class T0, class T1, class T2>
inline typename detail::make_tuple_mapper<T0, T1, T2>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2) {
typedef typename detail::make_tuple_mapper<T0, T1, T2>::type t;
return t(t0, t1, t2);
}

template<class T0, class T1, class T2, class T3>
inline typename detail::make_tuple_mapper<T0, T1, T2, T3>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3) {
typedef typename detail::make_tuple_mapper<T0, T1, T2, T3>::type t;
return t(t0, t1, t2, t3);
}

template<class T0, class T1, class T2, class T3, class T4>
inline typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
const T4& t4) {
typedef typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type t;
return t(t0, t1, t2, t3, t4);
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
inline typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
const T4& t4, const T5& t5) {
typedef typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type t;
return t(t0, t1, t2, t3, t4, t5);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
inline typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
const T4& t4, const T5& t5, const T6& t6) {
typedef typename detail::make_tuple_mapper
<T0, T1, T2, T3, T4, T5, T6>::type t;
return t(t0, t1, t2, t3, t4, t5, t6);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6,
class T7>
inline typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
const T4& t4, const T5& t5, const T6& t6, const T7& t7) {
typedef typename detail::make_tuple_mapper
<T0, T1, T2, T3, T4, T5, T6, T7>::type t;
return t(t0, t1, t2, t3, t4, t5, t6, t7);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6,
class T7, class T8>
inline typename detail::make_tuple_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
const T4& t4, const T5& t5, const T6& t6, const T7& t7,
const T8& t8) {
typedef typename detail::make_tuple_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type t;
return t(t0, t1, t2, t3, t4, t5, t6, t7, t8);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6,
class T7, class T8, class T9>
inline typename detail::make_tuple_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
const T4& t4, const T5& t5, const T6& t6, const T7& t7,
const T8& t8, const T9& t9) {
typedef typename detail::make_tuple_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type t;
return t(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9);
}

namespace detail {

template<class T>
struct tie_traits {
typedef T& type;
};

template<>
struct tie_traits<ignore_t(ignore_t)> {
typedef swallow_assign type;
};

template<>
struct tie_traits<void> {
typedef null_type type;
};

template <
class T0 = void, class T1 = void, class T2 = void,
class T3 = void, class T4 = void, class T5 = void,
class T6 = void, class T7 = void, class T8 = void,
class T9 = void
>
struct tie_mapper {
typedef
tuple<typename tie_traits<T0>::type,
typename tie_traits<T1>::type,
typename tie_traits<T2>::type,
typename tie_traits<T3>::type,
typename tie_traits<T4>::type,
typename tie_traits<T5>::type,
typename tie_traits<T6>::type,
typename tie_traits<T7>::type,
typename tie_traits<T8>::type,
typename tie_traits<T9>::type> type;
};

}

template<class T0>
inline typename detail::tie_mapper<T0>::type
tie(T0& t0) {
typedef typename detail::tie_mapper<T0>::type t;
return t(t0);
}

template<class T0, class T1>
inline typename detail::tie_mapper<T0, T1>::type
tie(T0& t0, T1& t1) {
typedef typename detail::tie_mapper<T0, T1>::type t;
return t(t0, t1);
}

template<class T0, class T1, class T2>
inline typename detail::tie_mapper<T0, T1, T2>::type
tie(T0& t0, T1& t1, T2& t2) {
typedef typename detail::tie_mapper<T0, T1, T2>::type t;
return t(t0, t1, t2);
}

template<class T0, class T1, class T2, class T3>
inline typename detail::tie_mapper<T0, T1, T2, T3>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3) {
typedef typename detail::tie_mapper<T0, T1, T2, T3>::type t;
return t(t0, t1, t2, t3);
}

template<class T0, class T1, class T2, class T3, class T4>
inline typename detail::tie_mapper<T0, T1, T2, T3, T4>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3,
T4& t4) {
typedef typename detail::tie_mapper<T0, T1, T2, T3, T4>::type t;
return t(t0, t1, t2, t3, t4);
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
inline typename detail::tie_mapper<T0, T1, T2, T3, T4, T5>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3,
T4& t4, T5& t5) {
typedef typename detail::tie_mapper<T0, T1, T2, T3, T4, T5>::type t;
return t(t0, t1, t2, t3, t4, t5);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
inline typename detail::tie_mapper<T0, T1, T2, T3, T4, T5, T6>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3,
T4& t4, T5& t5, T6& t6) {
typedef typename detail::tie_mapper
<T0, T1, T2, T3, T4, T5, T6>::type t;
return t(t0, t1, t2, t3, t4, t5, t6);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6,
class T7>
inline typename detail::tie_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3,
T4& t4, T5& t5, T6& t6, T7& t7) {
typedef typename detail::tie_mapper
<T0, T1, T2, T3, T4, T5, T6, T7>::type t;
return t(t0, t1, t2, t3, t4, t5, t6, t7);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6,
class T7, class T8>
inline typename detail::tie_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3,
T4& t4, T5& t5, T6& t6, T7& t7,
T8& t8) {
typedef typename detail::tie_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type t;
return t(t0, t1, t2, t3, t4, t5, t6, t7, t8);
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6,
class T7, class T8, class T9>
inline typename detail::tie_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
tie(T0& t0, T1& t1, T2& t2, T3& t3,
T4& t4, T5& t5, T6& t6, T7& t7,
T8& t8, T9& t9) {
typedef typename detail::tie_mapper
<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type t;
return t(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9);
}

template <class T0, class T1, class T2, class T3, class T4,
class T5, class T6, class T7, class T8, class T9>
void swap(tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& lhs,
tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& rhs);
inline void swap(null_type&, null_type&) {}
template<class HH>
inline void swap(cons<HH, null_type>& lhs, cons<HH, null_type>& rhs) {
::boost::swap(lhs.head, rhs.head);
}
template<class HH, class TT>
inline void swap(cons<HH, TT>& lhs, cons<HH, TT>& rhs) {
::boost::swap(lhs.head, rhs.head);
::boost::tuples::swap(lhs.tail, rhs.tail);
}
template <class T0, class T1, class T2, class T3, class T4,
class T5, class T6, class T7, class T8, class T9>
inline void swap(tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& lhs,
tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& rhs) {
typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> tuple_type;
typedef typename tuple_type::inherited base;
::boost::tuples::swap(static_cast<base&>(lhs), static_cast<base&>(rhs));
}

} 
} 


#if defined(BOOST_GCC) && (BOOST_GCC >= 40700)
#pragma GCC diagnostic pop
#endif


#endif 
