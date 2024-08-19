
#ifndef _BOOST_UBLAS_VECTOR_ASSIGN_
#define _BOOST_UBLAS_VECTOR_ASSIGN_

#include <boost/numeric/ublas/functional.hpp> 
#include <vector>


namespace boost { namespace numeric { namespace ublas {
namespace detail {

template<class E1, class E2, class S>
BOOST_UBLAS_INLINE
bool equals (const vector_expression<E1> &e1, const vector_expression<E2> &e2, S epsilon, S min_norm) {
return norm_inf (e1 - e2) <= epsilon *
std::max<S> (std::max<S> (norm_inf (e1), norm_inf (e2)), min_norm);
}

template<class E1, class E2>
BOOST_UBLAS_INLINE
bool expression_type_check (const vector_expression<E1> &e1, const vector_expression<E2> &e2) {
typedef typename type_traits<typename promote_traits<typename E1::value_type,
typename E2::value_type>::promote_type>::real_type real_type;
return equals (e1, e2, BOOST_UBLAS_TYPE_CHECK_EPSILON, BOOST_UBLAS_TYPE_CHECK_MIN);
}


template<class V, class E>
void make_conformant (V &v, const vector_expression<E> &e) {
BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
typedef typename V::size_type size_type;
typedef typename V::difference_type difference_type;
typedef typename V::value_type value_type;
std::vector<size_type> index;
typename V::iterator it (v.begin ());
typename V::iterator it_end (v.end ());
typename E::const_iterator ite (e ().begin ());
typename E::const_iterator ite_end (e ().end ());
if (it != it_end && ite != ite_end) {
size_type it_index = it.index (), ite_index = ite.index ();
for (;;) {
difference_type compare = it_index - ite_index;
if (compare == 0) {
++ it, ++ ite;
if (it != it_end && ite != ite_end) {
it_index = it.index ();
ite_index = ite.index ();
} else
break;
} else if (compare < 0) {
increment (it, it_end, - compare);
if (it != it_end)
it_index = it.index ();
else
break;
} else if (compare > 0) {
if (*ite != value_type())
index.push_back (ite.index ());
++ ite;
if (ite != ite_end)
ite_index = ite.index ();
else
break;
}
}
}

while (ite != ite_end) {
if (*ite != value_type())
index.push_back (ite.index ());
++ ite;
}
for (size_type k = 0; k < index.size (); ++ k)
v (index [k]) = value_type();
}

}


template<template <class T1, class T2> class F, class V, class T>
void iterating_vector_assign_scalar (V &v, const T &t) {
typedef F<typename V::iterator::reference, T> functor_type;
typedef typename V::difference_type difference_type;
difference_type size (v.size ());
typename V::iterator it (v.begin ());
BOOST_UBLAS_CHECK (v.end () - it == size, bad_size ());
#ifndef BOOST_UBLAS_USE_DUFF_DEVICE
while (-- size >= 0)
functor_type::apply (*it, t), ++ it;
#else
DD (size, 4, r, (functor_type::apply (*it, t), ++ it));
#endif
}
template<template <class T1, class T2> class F, class V, class T>
void indexing_vector_assign_scalar (V &v, const T &t) {
typedef F<typename V::reference, T> functor_type;
typedef typename V::size_type size_type;
size_type size (v.size ());
#ifndef BOOST_UBLAS_USE_DUFF_DEVICE
for (size_type i = 0; i < size; ++ i)
functor_type::apply (v (i), t);
#else
size_type i (0);
DD (size, 4, r, (functor_type::apply (v (i), t), ++ i));
#endif
}

template<template <class T1, class T2> class F, class V, class T>
void vector_assign_scalar (V &v, const T &t, dense_proxy_tag) {
#ifdef BOOST_UBLAS_USE_INDEXING
indexing_vector_assign_scalar<F> (v, t);
#elif BOOST_UBLAS_USE_ITERATING
iterating_vector_assign_scalar<F> (v, t);
#else
typedef typename V::size_type size_type;
size_type size (v.size ());
if (size >= BOOST_UBLAS_ITERATOR_THRESHOLD)
iterating_vector_assign_scalar<F> (v, t);
else
indexing_vector_assign_scalar<F> (v, t);
#endif
}
template<template <class T1, class T2> class F, class V, class T>
void vector_assign_scalar (V &v, const T &t, packed_proxy_tag) {
typedef F<typename V::iterator::reference, T> functor_type;
typedef typename V::difference_type difference_type;
typename V::iterator it (v.begin ());
difference_type size (v.end () - it);
while (-- size >= 0)
functor_type::apply (*it, t), ++ it;
}
template<template <class T1, class T2> class F, class V, class T>
void vector_assign_scalar (V &v, const T &t, sparse_proxy_tag) {
typedef F<typename V::iterator::reference, T> functor_type;
typename V::iterator it (v.begin ());
typename V::iterator it_end (v.end ());
while (it != it_end)
functor_type::apply (*it, t), ++ it;
}

template<template <class T1, class T2> class F, class V, class T>
BOOST_UBLAS_INLINE
void vector_assign_scalar (V &v, const T &t) {
typedef typename V::storage_category storage_category;
vector_assign_scalar<F> (v, t, storage_category ());
}

template<class SC, bool COMPUTED, class RI>
struct vector_assign_traits {
typedef SC storage_category;
};

template<bool COMPUTED>
struct vector_assign_traits<dense_tag, COMPUTED, packed_random_access_iterator_tag> {
typedef packed_tag storage_category;
};
template<>
struct vector_assign_traits<dense_tag, false, sparse_bidirectional_iterator_tag> {
typedef sparse_tag storage_category;
};
template<>
struct vector_assign_traits<dense_tag, true, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<bool COMPUTED>
struct vector_assign_traits<dense_proxy_tag, COMPUTED, packed_random_access_iterator_tag> {
typedef packed_proxy_tag storage_category;
};
template<>
struct vector_assign_traits<dense_proxy_tag, false, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};
template<>
struct vector_assign_traits<dense_proxy_tag, true, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<>
struct vector_assign_traits<packed_tag, false, sparse_bidirectional_iterator_tag> {
typedef sparse_tag storage_category;
};
template<>
struct vector_assign_traits<packed_tag, true, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<bool COMPUTED>
struct vector_assign_traits<packed_proxy_tag, COMPUTED, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<>
struct vector_assign_traits<sparse_tag, true, dense_random_access_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};
template<>
struct vector_assign_traits<sparse_tag, true, packed_random_access_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};
template<>
struct vector_assign_traits<sparse_tag, true, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<template <class T1, class T2> class F, class V, class E>
void iterating_vector_assign (V &v, const vector_expression<E> &e) {
typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
typedef typename V::difference_type difference_type;
difference_type size (BOOST_UBLAS_SAME (v.size (), e ().size ()));
typename V::iterator it (v.begin ());
BOOST_UBLAS_CHECK (v.end () - it == size, bad_size ());
typename E::const_iterator ite (e ().begin ());
BOOST_UBLAS_CHECK (e ().end () - ite == size, bad_size ());
#ifndef BOOST_UBLAS_USE_DUFF_DEVICE
while (-- size >= 0)
functor_type::apply (*it, *ite), ++ it, ++ ite;
#else
DD (size, 2, r, (functor_type::apply (*it, *ite), ++ it, ++ ite));
#endif
}
template<template <class T1, class T2> class F, class V, class E>
void indexing_vector_assign (V &v, const vector_expression<E> &e) {
typedef F<typename V::reference, typename E::value_type> functor_type;
typedef typename V::size_type size_type;
size_type size (BOOST_UBLAS_SAME (v.size (), e ().size ()));
#ifndef BOOST_UBLAS_USE_DUFF_DEVICE
for (size_type i = 0; i < size; ++ i)
functor_type::apply (v (i), e () (i));
#else
size_type i (0);
DD (size, 2, r, (functor_type::apply (v (i), e () (i)), ++ i));
#endif
}

template<template <class T1, class T2> class F, class V, class E>
void vector_assign (V &v, const vector_expression<E> &e, dense_proxy_tag) {
#ifdef BOOST_UBLAS_USE_INDEXING
indexing_vector_assign<F> (v, e);
#elif BOOST_UBLAS_USE_ITERATING
iterating_vector_assign<F> (v, e);
#else
typedef typename V::size_type size_type;
size_type size (BOOST_UBLAS_SAME (v.size (), e ().size ()));
if (size >= BOOST_UBLAS_ITERATOR_THRESHOLD)
iterating_vector_assign<F> (v, e);
else
indexing_vector_assign<F> (v, e);
#endif
}
template<template <class T1, class T2> class F, class V, class E>
void vector_assign (V &v, const vector_expression<E> &e, packed_proxy_tag) {
BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
typedef typename V::difference_type difference_type;
typedef typename V::value_type value_type;
#if BOOST_UBLAS_TYPE_CHECK
vector<value_type> cv (v.size ());
indexing_vector_assign<scalar_assign> (cv, v);
indexing_vector_assign<F> (cv, e);
#endif
typename V::iterator it (v.begin ());
typename V::iterator it_end (v.end ());
typename E::const_iterator ite (e ().begin ());
typename E::const_iterator ite_end (e ().end ());
difference_type it_size (it_end - it);
difference_type ite_size (ite_end - ite);
if (it_size > 0 && ite_size > 0) {
difference_type size ((std::min) (difference_type (it.index () - ite.index ()), ite_size));
if (size > 0) {
ite += size;
ite_size -= size;
}
}
if (it_size > 0 && ite_size > 0) {
difference_type size ((std::min) (difference_type (ite.index () - it.index ()), it_size));
if (size > 0) {
it_size -= size;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
if (!functor_type::computed) {
#ifdef _MSC_VER
#pragma warning(pop)
#endif
while (-- size >= 0)    
functor_type::apply (*it, value_type()), ++ it;
} else {
it += size;
}
}
}
difference_type size ((std::min) (it_size, ite_size));
it_size -= size;
ite_size -= size;
while (-- size >= 0)
functor_type::apply (*it, *ite), ++ it, ++ ite;
size = it_size;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
if (!functor_type::computed) {
#ifdef _MSC_VER
#pragma warning(pop)
#endif
while (-- size >= 0)    
functor_type::apply (*it, value_type()), ++ it;
} else {
it += size;
}
#if BOOST_UBLAS_TYPE_CHECK
if (! disable_type_check<bool>::value) 
BOOST_UBLAS_CHECK (detail::expression_type_check (v, cv), 
external_logic ("external logic or bad condition of inputs"));
#endif
}
template<template <class T1, class T2> class F, class V, class E>
void vector_assign (V &v, const vector_expression<E> &e, sparse_tag) {
BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
BOOST_STATIC_ASSERT ((!functor_type::computed));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
typedef typename V::value_type value_type;
#if BOOST_UBLAS_TYPE_CHECK
vector<value_type> cv (v.size ());
indexing_vector_assign<scalar_assign> (cv, v);
indexing_vector_assign<F> (cv, e);
#endif
v.clear ();
typename E::const_iterator ite (e ().begin ());
typename E::const_iterator ite_end (e ().end ());
while (ite != ite_end) {
value_type t (*ite);
if (t != value_type())
v.insert_element (ite.index (), t);
++ ite;
}
#if BOOST_UBLAS_TYPE_CHECK
if (! disable_type_check<bool>::value) 
BOOST_UBLAS_CHECK (detail::expression_type_check (v, cv), 
external_logic ("external logic or bad condition of inputs"));
#endif
}
template<template <class T1, class T2> class F, class V, class E>
void vector_assign (V &v, const vector_expression<E> &e, sparse_proxy_tag) {
BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
typedef typename V::size_type size_type;
typedef typename V::difference_type difference_type;
typedef typename V::value_type value_type;

#if BOOST_UBLAS_TYPE_CHECK
vector<value_type> cv (v.size ());
indexing_vector_assign<scalar_assign> (cv, v);
indexing_vector_assign<F> (cv, e);
#endif
detail::make_conformant (v, e);

typename V::iterator it (v.begin ());
typename V::iterator it_end (v.end ());
typename E::const_iterator ite (e ().begin ());
typename E::const_iterator ite_end (e ().end ());
if (it != it_end && ite != ite_end) {
size_type it_index = it.index (), ite_index = ite.index ();
for (;;) {
difference_type compare = it_index - ite_index;
if (compare == 0) {
functor_type::apply (*it, *ite);
++ it, ++ ite;
if (it != it_end && ite != ite_end) {
it_index = it.index ();
ite_index = ite.index ();
} else
break;
} else if (compare < 0) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
if (!functor_type::computed) {
#ifdef _MSC_VER
#pragma warning(pop)
#endif
functor_type::apply (*it, value_type());
++ it;
} else
increment (it, it_end, - compare);
if (it != it_end)
it_index = it.index ();
else
break;
} else if (compare > 0) {
increment (ite, ite_end, compare);
if (ite != ite_end)
ite_index = ite.index ();
else
break;
}
}
}
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
if (!functor_type::computed) {
#ifdef _MSC_VER
#pragma warning(pop)
#endif
while (it != it_end) {  
functor_type::apply (*it, value_type());
++ it;
}
} else {
it = it_end;
}
#if BOOST_UBLAS_TYPE_CHECK
if (! disable_type_check<bool>::value)
BOOST_UBLAS_CHECK (detail::expression_type_check (v, cv), 
external_logic ("external logic or bad condition of inputs"));
#endif
}

template<template <class T1, class T2> class F, class V, class E>
BOOST_UBLAS_INLINE
void vector_assign (V &v, const vector_expression<E> &e) {
typedef typename vector_assign_traits<typename V::storage_category,
F<typename V::reference, typename E::value_type>::computed,
typename E::const_iterator::iterator_category>::storage_category storage_category;
vector_assign<F> (v, e, storage_category ());
}

template<class SC, class RI>
struct vector_swap_traits {
typedef SC storage_category;
};

template<>
struct vector_swap_traits<dense_proxy_tag, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<>
struct vector_swap_traits<packed_proxy_tag, sparse_bidirectional_iterator_tag> {
typedef sparse_proxy_tag storage_category;
};

template<template <class T1, class T2> class F, class V, class E>
void vector_swap (V &v, vector_expression<E> &e, dense_proxy_tag) {
typedef F<typename V::iterator::reference, typename E::iterator::reference> functor_type;
typedef typename V::difference_type difference_type;
difference_type size (BOOST_UBLAS_SAME (v.size (), e ().size ()));
typename V::iterator it (v.begin ());
typename E::iterator ite (e ().begin ());
while (-- size >= 0)
functor_type::apply (*it, *ite), ++ it, ++ ite;
}
template<template <class T1, class T2> class F, class V, class E>
void vector_swap (V &v, vector_expression<E> &e, packed_proxy_tag) {
typedef F<typename V::iterator::reference, typename E::iterator::reference> functor_type;
typedef typename V::difference_type difference_type;
typename V::iterator it (v.begin ());
typename V::iterator it_end (v.end ());
typename E::iterator ite (e ().begin ());
typename E::iterator ite_end (e ().end ());
difference_type it_size (it_end - it);
difference_type ite_size (ite_end - ite);
if (it_size > 0 && ite_size > 0) {
difference_type size ((std::min) (difference_type (it.index () - ite.index ()), ite_size));
if (size > 0) {
ite += size;
ite_size -= size;
}
}
if (it_size > 0 && ite_size > 0) {
difference_type size ((std::min) (difference_type (ite.index () - it.index ()), it_size));
if (size > 0)
it_size -= size;
}
difference_type size ((std::min) (it_size, ite_size));
it_size -= size;
ite_size -= size;
while (-- size >= 0)
functor_type::apply (*it, *ite), ++ it, ++ ite;
}
template<template <class T1, class T2> class F, class V, class E>
void vector_swap (V &v, vector_expression<E> &e, sparse_proxy_tag) {
BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
typedef F<typename V::iterator::reference, typename E::iterator::reference> functor_type;
typedef typename V::size_type size_type;
typedef typename V::difference_type difference_type;

detail::make_conformant (v, e);
detail::make_conformant (e (), v);

typename V::iterator it (v.begin ());
typename V::iterator it_end (v.end ());
typename E::iterator ite (e ().begin ());
typename E::iterator ite_end (e ().end ());
if (it != it_end && ite != ite_end) {
size_type it_index = it.index (), ite_index = ite.index ();
for (;;) {
difference_type compare = it_index - ite_index;
if (compare == 0) {
functor_type::apply (*it, *ite);
++ it, ++ ite;
if (it != it_end && ite != ite_end) {
it_index = it.index ();
ite_index = ite.index ();
} else
break;
} else if (compare < 0) {
increment (it, it_end, - compare);
if (it != it_end)
it_index = it.index ();
else
break;
} else if (compare > 0) {
increment (ite, ite_end, compare);
if (ite != ite_end)
ite_index = ite.index ();
else
break;
}
}
}

#if BOOST_UBLAS_TYPE_CHECK
increment (ite, ite_end);
increment (it, it_end);
#endif
}

template<template <class T1, class T2> class F, class V, class E>
BOOST_UBLAS_INLINE
void vector_swap (V &v, vector_expression<E> &e) {
typedef typename vector_swap_traits<typename V::storage_category,
typename E::const_iterator::iterator_category>::storage_category storage_category;
vector_swap<F> (v, e, storage_category ());
}

}}}

#endif
