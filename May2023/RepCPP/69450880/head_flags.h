

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/functional.h>


namespace hydra_thrust
{
namespace detail
{


template<typename RandomAccessIterator,
typename BinaryPredicate = hydra_thrust::equal_to<typename hydra_thrust::iterator_value<RandomAccessIterator>::type>,
typename ValueType = bool,
typename IndexType = typename hydra_thrust::iterator_difference<RandomAccessIterator>::type>
class head_flags_with_init
{
typedef typename hydra_thrust::iterator_value<RandomAccessIterator>::type init_type;

public:
struct head_flag_functor
{
BinaryPredicate binary_pred; 
init_type init;
IndexType n;

typedef ValueType result_type;

__host__ __device__
head_flag_functor(init_type init, IndexType n)
: binary_pred(), init(init), n(n)
{}

__host__ __device__
head_flag_functor(init_type init, IndexType n, BinaryPredicate binary_pred)
: binary_pred(binary_pred), init(init), n(n)
{}

template<typename Tuple>
__host__ __device__ __hydra_thrust_forceinline__
result_type operator()(const Tuple &t)
{
const IndexType i = hydra_thrust::get<0>(t);

if(i == 0)
{
return !binary_pred(init, hydra_thrust::get<1>(t));
}

return !binary_pred(hydra_thrust::get<1>(t), hydra_thrust::get<2>(t));
}
};

typedef hydra_thrust::counting_iterator<IndexType> counting_iterator;

public:
typedef hydra_thrust::transform_iterator<
head_flag_functor,
hydra_thrust::zip_iterator<hydra_thrust::tuple<counting_iterator,RandomAccessIterator,RandomAccessIterator> >
> iterator;

__hydra_thrust_exec_check_disable__
__host__ __device__
head_flags_with_init(RandomAccessIterator first, RandomAccessIterator last, init_type init)
: m_begin(hydra_thrust::make_transform_iterator(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(hydra_thrust::counting_iterator<IndexType>(0), first, first - 1)),
head_flag_functor(init, last - first))),
m_end(m_begin + (last - first))
{}

__hydra_thrust_exec_check_disable__
__host__ __device__
head_flags_with_init(RandomAccessIterator first, RandomAccessIterator last, init_type init, BinaryPredicate binary_pred)
: m_begin(hydra_thrust::make_transform_iterator(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(hydra_thrust::counting_iterator<IndexType>(0), first, first - 1)),
head_flag_functor(init, last - first, binary_pred))),
m_end(m_begin + (last - first))
{}

__host__ __device__
iterator begin() const
{
return m_begin;
}

__host__ __device__
iterator end() const
{
return m_end;
}

template<typename OtherIndex>
__host__ __device__
typename iterator::reference operator[](OtherIndex i)
{
return *(begin() + i);
}

private:
iterator m_begin, m_end;
};



template<typename RandomAccessIterator,
typename BinaryPredicate = hydra_thrust::equal_to<typename hydra_thrust::iterator_value<RandomAccessIterator>::type>,
typename ValueType = bool,
typename IndexType = typename hydra_thrust::iterator_difference<RandomAccessIterator>::type>
class head_flags
{
public:
struct head_flag_functor
{
BinaryPredicate binary_pred; 
IndexType n;

typedef ValueType result_type;

__host__ __device__
head_flag_functor(IndexType n)
: binary_pred(), n(n)
{}

__host__ __device__
head_flag_functor(IndexType n, BinaryPredicate binary_pred)
: binary_pred(binary_pred), n(n)
{}

template<typename Tuple>
__host__ __device__ __hydra_thrust_forceinline__
result_type operator()(const Tuple &t)
{
const IndexType i = hydra_thrust::get<0>(t);

return (i == 0 || !binary_pred(hydra_thrust::get<1>(t), hydra_thrust::get<2>(t)));
}
};

typedef hydra_thrust::counting_iterator<IndexType> counting_iterator;

public:
typedef hydra_thrust::transform_iterator<
head_flag_functor,
hydra_thrust::zip_iterator<hydra_thrust::tuple<counting_iterator,RandomAccessIterator,RandomAccessIterator> >
> iterator;

__host__ __device__
head_flags(RandomAccessIterator first, RandomAccessIterator last)
: m_begin(hydra_thrust::make_transform_iterator(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(hydra_thrust::counting_iterator<IndexType>(0), first, first - 1)),
head_flag_functor(last - first))),
m_end(m_begin + (last - first))
{}

__host__ __device__
head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
: m_begin(hydra_thrust::make_transform_iterator(hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(hydra_thrust::counting_iterator<IndexType>(0), first, first - 1)),
head_flag_functor(last - first, binary_pred))),
m_end(m_begin + (last - first))
{}

__host__ __device__
iterator begin() const
{
return m_begin;
}

__host__ __device__
iterator end() const
{
return m_end;
}

template<typename OtherIndex>
__host__ __device__
typename iterator::reference operator[](OtherIndex i)
{
return *(begin() + i);
}

private:
iterator m_begin, m_end;
};


template<typename RandomAccessIterator, typename BinaryPredicate>
__host__ __device__
head_flags<RandomAccessIterator, BinaryPredicate>
make_head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
return head_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}


template<typename RandomAccessIterator>
__host__ __device__
head_flags<RandomAccessIterator>
make_head_flags(RandomAccessIterator first, RandomAccessIterator last)
{
return head_flags<RandomAccessIterator>(first, last);
}


} 
} 

