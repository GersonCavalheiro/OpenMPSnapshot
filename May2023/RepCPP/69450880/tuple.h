

#pragma once

#include <cstddef> 
#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/swap.h>

namespace hydra_thrust
{

struct null_type {};

#ifndef HYDRA_THRUST_VARIADIC_TUPLE

__host__ __device__ inline
bool operator==(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator<(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator>(const null_type&, const null_type&) { return false; }

template <
class T0 = null_type, class T1 = null_type, class T2 = null_type,
class T3 = null_type, class T4 = null_type, class T5 = null_type,
class T6 = null_type, class T7 = null_type, class T8 = null_type,
class T9 = null_type>
class tuple;

template<size_t i, typename T> struct tuple_element;

template<class T>
struct tuple_element<0,T>
{
typedef typename T::head_type type;
}; 

template<size_t N, class T>
struct tuple_element<N, const T>
{
private:
typedef typename T::tail_type Next;
typedef typename tuple_element<N-1, Next>::type unqualified_type;

public:
typedef typename hydra_thrust::detail::add_const<unqualified_type>::type type;
}; 

template<class T>
struct tuple_element<0,const T>
{
typedef typename hydra_thrust::detail::add_const<typename T::head_type>::type type;
}; 



template<class T> struct tuple_size;

template<>
struct tuple_size< tuple<> >
{
static const int value = 0;
}; 

template<>
struct tuple_size<null_type>
{
static const int value = 0;
}; 



namespace detail
{

template <class HT, class TT> struct cons;

} 

#endif 

template <class T> struct access_traits
{
typedef const T& const_type;
typedef T& non_const_type;

typedef const typename hydra_thrust::detail::remove_cv<T>::type& parameter_type;

}; 

template <class T> struct access_traits<T&>
{
typedef T& const_type;
typedef T& non_const_type;

typedef T& parameter_type;
}; 

#ifndef HYDRA_THRUST_VARIADIC_TUPLE

template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
typename tuple_element<N, detail::cons<HT, TT> >::type
>::non_const_type
get(detail::cons<HT, TT>& c);

template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
typename tuple_element<N, detail::cons<HT, TT> >::type
>::const_type
get(const detail::cons<HT, TT>& c);

namespace detail
{

template<class T>
class generate_error;


template< int N >
struct get_class
{
template<class RET, class HT, class TT >
__host__ __device__
inline static RET get(const cons<HT, TT>& t)
{
return get_class<N-1>::template get<RET>(t.tail);

}

template<class RET, class HT, class TT >
__host__ __device__
inline static RET get(cons<HT, TT>& t)
{
return get_class<N-1>::template get<RET>(t.tail);

}
}; 

template<>
struct get_class<0>
{
template<class RET, class HT, class TT>
__host__ __device__
inline static RET get(const cons<HT, TT>& t)
{
return t.head;
}

template<class RET, class HT, class TT>
__host__ __device__
inline static RET get(cons<HT, TT>& t)
{
return t.head;
}
}; 


template <bool If, class Then, class Else> struct IF
{
typedef Then RET;
};

template <class Then, class Else> struct IF<false, Then, Else>
{
typedef Else RET;
};


template <class T> class non_storeable_type
{
__host__ __device__
non_storeable_type();
};

template <class T> struct wrap_non_storeable_type
{

typedef T type;
};

template <> struct wrap_non_storeable_type<void>
{
typedef non_storeable_type<void> type;
};


template <class HT, class TT>
struct cons
{
typedef HT head_type;
typedef TT tail_type;

typedef typename
wrap_non_storeable_type<head_type>::type stored_head_type;

stored_head_type head;
tail_type tail;

inline __host__ __device__
typename access_traits<stored_head_type>::non_const_type
get_head() { return head; }

inline __host__ __device__
typename access_traits<tail_type>::non_const_type
get_tail() { return tail; }

inline __host__ __device__
typename access_traits<stored_head_type>::const_type
get_head() const { return head; }

inline __host__ __device__
typename access_traits<tail_type>::const_type
get_tail() const { return tail; }

inline __host__ __device__
cons(void) : head(), tail() {}


inline __host__ __device__
cons(typename access_traits<stored_head_type>::parameter_type h,
const tail_type& t)
: head (h), tail(t) {}

template <class T1, class T2, class T3, class T4, class T5,
class T6, class T7, class T8, class T9, class T10>
inline __host__ __device__
cons( T1& t1, T2& t2, T3& t3, T4& t4, T5& t5,
T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 )
: head (t1),
tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, static_cast<const null_type&>(null_type()))
{}

template <class T2, class T3, class T4, class T5,
class T6, class T7, class T8, class T9, class T10>
inline __host__ __device__
cons( const null_type& , T2& t2, T3& t3, T4& t4, T5& t5,
T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 )
: head (),
tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, static_cast<const null_type&>(null_type()))
{}


template <class HT2, class TT2>
inline __host__ __device__
cons( const cons<HT2, TT2>& u ) : head(u.head), tail(u.tail) {}

__hydra_thrust_exec_check_disable__
template <class HT2, class TT2>
inline __host__ __device__
cons& operator=( const cons<HT2, TT2>& u ) {
head=u.head; tail=u.tail; return *this;
}

__hydra_thrust_exec_check_disable__
inline __host__ __device__
cons& operator=(const cons& u) {
head = u.head; tail = u.tail;  return *this;
}


template <int N>
__host__ __device__
typename access_traits<
typename tuple_element<N, cons<HT, TT> >::type
>::non_const_type
get() {
return hydra_thrust::get<N>(*this); 
}

template <int N>
__host__ __device__
typename access_traits<
typename tuple_element<N, cons<HT, TT> >::type
>::const_type
get() const {
return hydra_thrust::get<N>(*this); 
}

inline __host__ __device__
void swap(cons &c)
{
using hydra_thrust::swap;

swap(head, c.head);
tail.swap(c.tail);
}
};

template <class HT>
struct cons<HT, null_type>
{
typedef HT head_type;
typedef null_type tail_type;
typedef cons<HT, null_type> self_type;

typedef typename
wrap_non_storeable_type<head_type>::type stored_head_type;
stored_head_type head;

typename access_traits<stored_head_type>::non_const_type
inline __host__ __device__
get_head() { return head; }

inline __host__ __device__
null_type get_tail() { return null_type(); }

inline __host__ __device__
typename access_traits<stored_head_type>::const_type
get_head() const { return head; }

inline __host__ __device__
null_type get_tail() const { return null_type(); }

inline __host__ __device__
cons() : head() {}

inline __host__ __device__
cons(typename access_traits<stored_head_type>::parameter_type h,
const null_type& = null_type())
: head (h) {}

template<class T1>
inline __host__ __device__
cons(T1& t1, const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&)
: head (t1) {}

inline __host__ __device__
cons(const null_type&,
const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&,
const null_type&, const null_type&, const null_type&)
: head () {}

template <class HT2>
inline __host__ __device__
cons( const cons<HT2, null_type>& u ) : head(u.head) {}

__hydra_thrust_exec_check_disable__
template <class HT2>
inline __host__ __device__
cons& operator=(const cons<HT2, null_type>& u )
{
head = u.head;
return *this;
}

inline __host__ __device__
cons& operator=(const cons& u) { head = u.head; return *this; }

template <int N>
inline __host__ __device__
typename access_traits<
typename tuple_element<N, self_type>::type
>::non_const_type
get(void)
{
return hydra_thrust::get<N>(*this);
}

template <int N>
inline __host__ __device__
typename access_traits<
typename tuple_element<N, self_type>::type
>::const_type
get(void) const
{
return hydra_thrust::get<N>(*this);
}

inline __host__ __device__
void swap(cons &c)
{
using hydra_thrust::swap;

swap(head, c.head);
}
}; 

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


template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
typename tuple_element<N, detail::cons<HT, TT> >::type
>::non_const_type
get(detail::cons<HT, TT>& c)
{


return detail::get_class<N>::template
get<
typename access_traits<
typename tuple_element<N, detail::cons<HT, TT> >::type
>::non_const_type,
HT,TT
>(c);
}


template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
typename tuple_element<N, detail::cons<HT, TT> >::type
>::const_type
get(const detail::cons<HT, TT>& c)
{


return detail::get_class<N>::template
get<
typename access_traits<
typename tuple_element<N, detail::cons<HT, TT> >::type
>::const_type,
HT,TT
>(c);
}


template<class T0>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0>::type
make_tuple(const T0& t0)
{
typedef typename detail::make_tuple_mapper<T0>::type t;
return t(t0);
} 

template<class T0, class T1>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1>::type
make_tuple(const T0& t0, const T1& t1)
{
typedef typename detail::make_tuple_mapper<T0,T1>::type t;
return t(t0,t1);
} 

template<class T0, class T1, class T2>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2>::type t;
return t(t0,t1,t2);
} 

template<class T0, class T1, class T2, class T3>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3>::type t;
return t(t0,t1,t2,t3);
} 

template<class T0, class T1, class T2, class T3, class T4>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4>::type t;
return t(t0,t1,t2,t3,t4);
} 

template<class T0, class T1, class T2, class T3, class T4, class T5>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5>::type t;
return t(t0,t1,t2,t3,t4,t5);
} 

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6>::type t;
return t(t0,t1,t2,t3,t4,t5,t6);
} 

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6,T7>::type t;
return t(t0,t1,t2,t3,t4,t5,t6,t7);
} 

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6,T7,T8>::type t;
return t(t0,t1,t2,t3,t4,t5,t6,t7,t8);
} 

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
__host__ __device__ inline
typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9)
{
typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::type t;
return t(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9);
} 


template<typename T0>
__host__ __device__ inline
tuple<T0&> tie(T0 &t0)
{
return tuple<T0&>(t0);
}

template<typename T0,typename T1>
__host__ __device__ inline
tuple<T0&,T1&> tie(T0 &t0, T1 &t1)
{
return tuple<T0&,T1&>(t0,t1);
}

template<typename T0,typename T1, typename T2>
__host__ __device__ inline
tuple<T0&,T1&,T2&> tie(T0 &t0, T1 &t1, T2 &t2)
{
return tuple<T0&,T1&,T2&>(t0,t1,t2);
}

template<typename T0,typename T1, typename T2, typename T3>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3)
{
return tuple<T0&,T1&,T2&,T3&>(t0,t1,t2,t3);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4)
{
return tuple<T0&,T1&,T2&,T3&,T4&>(t0,t1,t2,t3,t4);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5)
{
return tuple<T0&,T1&,T2&,T3&,T4&,T5&>(t0,t1,t2,t3,t4,t5);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6)
{
return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&>(t0,t1,t2,t3,t4,t5,t6);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7)
{
return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&>(t0,t1,t2,t3,t4,t5,t6,t7);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8)
{
return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&>(t0,t1,t2,t3,t4,t5,t6,t7,t8);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9)
{
return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&>(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9);
}

template<
typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9,
typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9
>
__host__ __device__ inline
void swap(hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> &x,
hydra_thrust::tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9> &y)
{
return x.swap(y);
}



namespace detail
{

template<class T1, class T2>
__host__ __device__
inline bool eq(const T1& lhs, const T2& rhs) {
return lhs.get_head() == rhs.get_head() &&
eq(lhs.get_tail(), rhs.get_tail());
}
template<>
inline bool eq<null_type,null_type>(const null_type&, const null_type&) { return true; }

template<class T1, class T2>
__host__ __device__
inline bool neq(const T1& lhs, const T2& rhs) {
return lhs.get_head() != rhs.get_head()  ||
neq(lhs.get_tail(), rhs.get_tail());
}
template<>
__host__ __device__
inline bool neq<null_type,null_type>(const null_type&, const null_type&) { return false; }

template<class T1, class T2>
__host__ __device__
inline bool lt(const T1& lhs, const T2& rhs) {
return (lhs.get_head() < rhs.get_head())  ||
(!(rhs.get_head() < lhs.get_head()) &&
lt(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool lt<null_type,null_type>(const null_type&, const null_type&) { return false; }

template<class T1, class T2>
__host__ __device__
inline bool gt(const T1& lhs, const T2& rhs) {
return (lhs.get_head() > rhs.get_head())  ||
(!(rhs.get_head() > lhs.get_head()) &&
gt(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool gt<null_type,null_type>(const null_type&, const null_type&) { return false; }

template<class T1, class T2>
__host__ __device__
inline bool lte(const T1& lhs, const T2& rhs) {
return lhs.get_head() <= rhs.get_head()  &&
( !(rhs.get_head() <= lhs.get_head()) ||
lte(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool lte<null_type,null_type>(const null_type&, const null_type&) { return true; }

template<class T1, class T2>
__host__ __device__
inline bool gte(const T1& lhs, const T2& rhs) {
return lhs.get_head() >= rhs.get_head()  &&
( !(rhs.get_head() >= lhs.get_head()) ||
gte(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool gte<null_type,null_type>(const null_type&, const null_type&) { return true; }

} 




template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator==(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{

return  detail::eq(lhs, rhs);
} 


template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator!=(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{

return detail::neq(lhs, rhs);
} 

template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator<(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{

return detail::lt(lhs, rhs);
} 

template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator>(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{

return detail::gt(lhs, rhs);
} 

template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator<=(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{

return detail::lte(lhs, rhs);
} 

template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator>=(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{

return detail::gte(lhs, rhs);
} 

#endif 

} 

