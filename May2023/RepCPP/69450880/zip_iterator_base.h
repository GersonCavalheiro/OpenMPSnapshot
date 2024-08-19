

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_category.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/tuple/tuple_meta_transform.h>
#include <hydra/detail/external/hydra_thrust/detail/tuple/tuple_transform.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/tuple_of_iterator_references.h>

namespace hydra_thrust
{

template<typename IteratorTuple> class zip_iterator;

namespace detail
{


template<typename DiffType>
class advance_iterator
{
public:
inline __host__ __device__
advance_iterator(DiffType step) : m_step(step) {}

__hydra_thrust_exec_check_disable__
template<typename Iterator>
inline __host__ __device__
void operator()(Iterator& it) const
{ it += m_step; }

private:
DiffType m_step;
}; 


struct increment_iterator
{
__hydra_thrust_exec_check_disable__
template<typename Iterator>
inline __host__ __device__
void operator()(Iterator& it)
{ ++it; }
}; 


struct decrement_iterator
{
__hydra_thrust_exec_check_disable__
template<typename Iterator>
inline __host__ __device__
void operator()(Iterator& it)
{ --it; }
}; 


struct dereference_iterator
{
template<typename Iterator>
struct apply
{
typedef typename
iterator_traits<Iterator>::reference
type;
}; 

__hydra_thrust_exec_check_disable__
template<typename Iterator>
__host__ __device__
typename apply<Iterator>::type operator()(Iterator const& it)
{
return *it;
}
}; 


namespace tuple_impl_specific
{

template<typename UnaryMetaFunctionClass, class Arg>
struct apply1
: UnaryMetaFunctionClass::template apply<Arg>
{
}; 


template<typename UnaryMetaFunctionClass, class Arg1, class Arg2>
struct apply2
: UnaryMetaFunctionClass::template apply<Arg1,Arg2>
{
}; 


template<class Tuple, class BinaryMetaFun, class StartType>
struct tuple_meta_accumulate;

#ifdef HYDRA_THRUST_VARIADIC_TUPLE

template<
class BinaryMetaFun
, typename StartType
>
struct tuple_meta_accumulate<hydra_thrust::tuple<>,BinaryMetaFun,StartType>
{
typedef typename hydra_thrust::detail::identity_<StartType>::type type;
};


template<
class BinaryMetaFun
, typename StartType
, typename    T
, typename... Types
>
struct tuple_meta_accumulate<hydra_thrust::tuple<T,Types...>,BinaryMetaFun,StartType>
{
typedef typename apply2<
BinaryMetaFun
, T
, typename tuple_meta_accumulate<
hydra_thrust::tuple<Types...>
, BinaryMetaFun
, StartType
>::type
>::type type;
};

#else

template<
typename Tuple
, class BinaryMetaFun
, typename StartType
>
struct tuple_meta_accumulate_impl
{
typedef typename apply2<
BinaryMetaFun
, typename Tuple::head_type
, typename tuple_meta_accumulate<
typename Tuple::tail_type
, BinaryMetaFun
, StartType 
>::type
>::type type;
};


template<
typename Tuple
, class BinaryMetaFun
, typename StartType
>
struct tuple_meta_accumulate
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_same<Tuple, hydra_thrust::null_type>::value
, hydra_thrust::detail::identity_<StartType>
, tuple_meta_accumulate_impl<
Tuple
, BinaryMetaFun
, StartType
>
> 
{
}; 

#endif

#ifdef HYDRA_THRUST_VARIADIC_TUPLE

template<typename Fun>
inline __host__ __device__
Fun tuple_for_each_helper(Fun f)
{
return f;
}

template<typename Fun, typename T, typename... Types>
inline __host__ __device__
Fun tuple_for_each_helper(Fun f, T& t, Types&... ts)
{
f(t);
return tuple_for_each_helper(f, ts...);
};

template<typename Fun, typename... Types, size_t... I>
inline __host__ __device__
Fun tuple_for_each(hydra_thrust::tuple<Types...>& t, Fun f, hydra_thrust::__index_sequence<I...>)
{
return tuple_for_each_helper(f, hydra_thrust::get<I>(t)...);
};

template<typename Fun, typename... Types>
inline __host__ __device__
Fun tuple_for_each(hydra_thrust::tuple<Types...>& t, Fun f)
{
return tuple_for_each(t, f, hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<hydra_thrust::tuple<Types...>>::value>{});    
}

#else

template<typename Fun>
inline __host__ __device__
Fun tuple_for_each(hydra_thrust::null_type, Fun f)
{
return f;
} 


template<typename Tuple, typename Fun>
inline __host__ __device__
Fun tuple_for_each(Tuple& t, Fun f)
{ 
f( t.get_head() );
return tuple_for_each(t.get_tail(), f);
} 
#endif

#ifdef HYDRA_THRUST_VARIADIC_TUPLE
template<typename Tuple1, typename Tuple2>
__host__ __device__
bool tuple_equal(Tuple1 const& t1, Tuple2 const& t2)
{ 
return t1 == t2 ;
} 


#else
__host__ __device__
inline bool tuple_equal(hydra_thrust::null_type, hydra_thrust::null_type)
{ return true; }


template<typename Tuple1, typename Tuple2>
__host__ __device__
bool tuple_equal(Tuple1 const& t1, Tuple2 const& t2)
{ 
return t1.get_head() == t2.get_head() && 
tuple_equal(t1.get_tail(), t2.get_tail());
} 

#endif

} 


template<typename IteratorTuple>
struct tuple_of_value_types
: tuple_meta_transform<
IteratorTuple,
iterator_value
>
{
}; 


struct minimum_category_lambda
{
template<typename T1, typename T2>
struct apply : minimum_category<T1,T2>
{};
};



template<typename IteratorTuple>
struct minimum_traversal_category_in_iterator_tuple
{
typedef typename tuple_meta_transform<
IteratorTuple
, hydra_thrust::iterator_traversal
>::type tuple_of_traversal_tags;

typedef typename tuple_impl_specific::tuple_meta_accumulate<
tuple_of_traversal_tags
, minimum_category_lambda
, hydra_thrust::random_access_traversal_tag
>::type type;
};


struct minimum_system_lambda
{
template<typename T1, typename T2>
struct apply : minimum_system<T1,T2>
{};
};



template<typename IteratorTuple>
struct minimum_system_in_iterator_tuple
{
typedef typename hydra_thrust::detail::tuple_meta_transform<
IteratorTuple,
hydra_thrust::iterator_system
>::type tuple_of_system_tags;

typedef typename tuple_impl_specific::tuple_meta_accumulate<
tuple_of_system_tags,
minimum_system_lambda,
hydra_thrust::any_system_tag
>::type type;
};

namespace zip_iterator_base_ns
{


#ifdef HYDRA_THRUST_VARIADIC_TUPLE
template<typename Tuple, typename IndexSequence>
struct tuple_of_iterator_references_helper;

template<typename Tuple, size_t... I>
struct tuple_of_iterator_references_helper<Tuple, hydra_thrust::__index_sequence<I...>>
{
typedef hydra_thrust::detail::tuple_of_iterator_references<
typename hydra_thrust::tuple_element<I,Tuple>::type...
> type;
};
#else
template<int i, typename Tuple>
struct tuple_elements_helper
: eval_if<
(i < tuple_size<Tuple>::value),
tuple_element<i,Tuple>,
identity_<hydra_thrust::null_type>
>
{};


template<typename Tuple>
struct tuple_elements
{
typedef typename tuple_elements_helper<0,Tuple>::type T0;
typedef typename tuple_elements_helper<1,Tuple>::type T1;
typedef typename tuple_elements_helper<2,Tuple>::type T2;
typedef typename tuple_elements_helper<3,Tuple>::type T3;
typedef typename tuple_elements_helper<4,Tuple>::type T4;
typedef typename tuple_elements_helper<5,Tuple>::type T5;
typedef typename tuple_elements_helper<6,Tuple>::type T6;
typedef typename tuple_elements_helper<7,Tuple>::type T7;
typedef typename tuple_elements_helper<8,Tuple>::type T8;
typedef typename tuple_elements_helper<9,Tuple>::type T9;
};
#endif


template<typename IteratorTuple>
struct tuple_of_iterator_references
{
typedef typename tuple_meta_transform<
IteratorTuple,
iterator_reference
>::type tuple_of_references;

#ifdef HYDRA_THRUST_VARIADIC_TUPLE
typedef typename tuple_of_iterator_references_helper<
tuple_of_references,
hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<tuple_of_references>::value>
>::type type;
#else
typedef tuple_elements<tuple_of_references> elements;

typedef hydra_thrust::detail::tuple_of_iterator_references<
typename elements::T0,
typename elements::T1,
typename elements::T2,
typename elements::T3,
typename elements::T4,
typename elements::T5,
typename elements::T6,
typename elements::T7,
typename elements::T8,
typename elements::T9
> type;
#endif
};


} 

template<typename IteratorTuple>
struct zip_iterator_base
{
typedef typename zip_iterator_base_ns::tuple_of_iterator_references<IteratorTuple>::type reference;

typedef typename tuple_of_value_types<IteratorTuple>::type value_type;

typedef typename hydra_thrust::iterator_traits<
typename hydra_thrust::tuple_element<0, IteratorTuple>::type
>::difference_type difference_type;

typedef typename
minimum_system_in_iterator_tuple<IteratorTuple>::type system;

typedef typename
minimum_traversal_category_in_iterator_tuple<IteratorTuple>::type traversal_category;

public:

typedef hydra_thrust::iterator_facade<
zip_iterator<IteratorTuple>,
value_type,
system,
traversal_category,
reference,
difference_type
> type;
}; 

} 

} 


