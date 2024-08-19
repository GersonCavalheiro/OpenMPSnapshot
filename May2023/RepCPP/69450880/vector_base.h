




#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/detail/normal_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/reverse_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/contiguous_storage.h>
#include <vector>

namespace hydra_thrust
{

namespace detail
{

template<typename T, typename Alloc>
class vector_base
{
private:
typedef hydra_thrust::detail::contiguous_storage<T,Alloc> storage_type;

public:
typedef typename storage_type::value_type      value_type;
typedef typename storage_type::pointer         pointer;
typedef typename storage_type::const_pointer   const_pointer;
typedef typename storage_type::reference       reference;
typedef typename storage_type::const_reference const_reference;
typedef typename storage_type::size_type       size_type;
typedef typename storage_type::difference_type difference_type;
typedef typename storage_type::allocator_type  allocator_type;

typedef typename storage_type::iterator        iterator;
typedef typename storage_type::const_iterator  const_iterator;

typedef hydra_thrust::reverse_iterator<iterator>       reverse_iterator;
typedef hydra_thrust::reverse_iterator<const_iterator> const_reverse_iterator;


vector_base(void);


explicit vector_base(const Alloc &alloc);


explicit vector_base(size_type n);


explicit vector_base(size_type n, const Alloc &alloc);


explicit vector_base(size_type n, const value_type &value);


explicit vector_base(size_type n, const value_type &value, const Alloc &alloc);


vector_base(const vector_base &v);


vector_base(const vector_base &v, const Alloc &alloc);

#if __cplusplus >= 201103L

vector_base(vector_base &&v);

#endif


vector_base &operator=(const vector_base &v);

#if __cplusplus >= 201103L

vector_base &operator=(vector_base &&v);
#endif


template<typename OtherT, typename OtherAlloc>
vector_base(const vector_base<OtherT, OtherAlloc> &v);


template<typename OtherT, typename OtherAlloc>
vector_base &operator=(const vector_base<OtherT,OtherAlloc> &v);


template<typename OtherT, typename OtherAlloc>
vector_base(const std::vector<OtherT, OtherAlloc> &v);


template<typename OtherT, typename OtherAlloc>
vector_base &operator=(const std::vector<OtherT,OtherAlloc> &v);


template<typename InputIterator>
vector_base(InputIterator first, InputIterator last);


template<typename InputIterator>
vector_base(InputIterator first, InputIterator last, const Alloc &alloc);


~vector_base(void);


void resize(size_type new_size);


void resize(size_type new_size, const value_type &x);


size_type size(void) const;


size_type max_size(void) const;


void reserve(size_type n);


size_type capacity(void) const;


void shrink_to_fit(void);


reference operator[](size_type n);


const_reference operator[](size_type n) const;


iterator begin(void);


const_iterator begin(void) const;


const_iterator cbegin(void) const;


reverse_iterator rbegin(void);


const_reverse_iterator rbegin(void) const;


const_reverse_iterator crbegin(void) const;


iterator end(void);


const_iterator end(void) const;


const_iterator cend(void) const;


reverse_iterator rend(void);


const_reverse_iterator rend(void) const;


const_reverse_iterator crend(void) const;


const_reference front(void) const;


reference front(void);


const_reference back(void) const;


reference back(void);


pointer data(void);


const_pointer data(void) const;


void clear(void);


bool empty(void) const;


void push_back(const value_type &x);


void pop_back(void);


void swap(vector_base &v);


iterator erase(iterator pos);


iterator erase(iterator first, iterator last);


iterator insert(iterator position, const T &x); 


void insert(iterator position, size_type n, const T &x);


template<typename InputIterator>
void insert(iterator position, InputIterator first, InputIterator last);


void assign(size_type n, const T &x);


template<typename InputIterator>
void assign(InputIterator first, InputIterator last);


allocator_type get_allocator(void) const;

protected:
storage_type m_storage;

size_type m_size;

private:
template<typename IteratorOrIntegralType>
void init_dispatch(IteratorOrIntegralType begin, IteratorOrIntegralType end, false_type); 

template<typename IteratorOrIntegralType>
void init_dispatch(IteratorOrIntegralType n, IteratorOrIntegralType value, true_type); 

template<typename InputIterator>
void range_init(InputIterator first, InputIterator last);

template<typename InputIterator>
void range_init(InputIterator first, InputIterator last, hydra_thrust::incrementable_traversal_tag);

template<typename ForwardIterator>
void range_init(ForwardIterator first, ForwardIterator last, hydra_thrust::random_access_traversal_tag);

void default_init(size_type n);

void fill_init(size_type n, const T &x);

template<typename InputIteratorOrIntegralType>
void insert_dispatch(iterator position, InputIteratorOrIntegralType first, InputIteratorOrIntegralType last, false_type);

template<typename InputIteratorOrIntegralType>
void insert_dispatch(iterator position, InputIteratorOrIntegralType n, InputIteratorOrIntegralType x, true_type);

void append(size_type n);

void fill_insert(iterator position, size_type n, const T &x);

template<typename InputIterator>
void copy_insert(iterator position, InputIterator first, InputIterator last);

template<typename InputIterator>
void assign_dispatch(InputIterator first, InputIterator last, false_type);

template<typename Integral>
void assign_dispatch(Integral n, Integral x, true_type);

template<typename InputIterator>
void range_assign(InputIterator first, InputIterator last);

template<typename RandomAccessIterator>
void range_assign(RandomAccessIterator first, RandomAccessIterator last, hydra_thrust::random_access_traversal_tag);

template<typename InputIterator>
void range_assign(InputIterator first, InputIterator last, hydra_thrust::incrementable_traversal_tag);

void fill_assign(size_type n, const T &x);

template<typename ForwardIterator>
void allocate_and_copy(size_type requested_size,
ForwardIterator first, ForwardIterator last,
storage_type &new_storage);
}; 

} 


template<typename T, typename Alloc>
void swap(detail::vector_base<T,Alloc> &a,
detail::vector_base<T,Alloc> &b);



template<typename T1, typename Alloc1,
typename T2, typename Alloc2>
bool operator==(const detail::vector_base<T1,Alloc1>& lhs,
const detail::vector_base<T2,Alloc2>& rhs);

template<typename T1, typename Alloc1,
typename T2, typename Alloc2>
bool operator==(const detail::vector_base<T1,Alloc1>& lhs,
const std::vector<T2,Alloc2>&         rhs);

template<typename T1, typename Alloc1,
typename T2, typename Alloc2>
bool operator==(const std::vector<T1,Alloc1>&         lhs,
const detail::vector_base<T2,Alloc2>& rhs);


template<typename T1, typename Alloc1,
typename T2, typename Alloc2>
bool operator!=(const detail::vector_base<T1,Alloc1>& lhs,
const detail::vector_base<T2,Alloc2>& rhs);

template<typename T1, typename Alloc1,
typename T2, typename Alloc2>
bool operator!=(const detail::vector_base<T1,Alloc1>& lhs,
const std::vector<T2,Alloc2>&         rhs);

template<typename T1, typename Alloc1,
typename T2, typename Alloc2>
bool operator!=(const std::vector<T1,Alloc1>&         lhs,
const detail::vector_base<T2,Alloc2>& rhs);

} 

#include <hydra/detail/external/hydra_thrust/detail/vector_base.inl>

