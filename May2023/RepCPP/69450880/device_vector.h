




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/vector_base.h>
#include <hydra/detail/external/hydra_thrust/device_allocator.h>

#include <vector>
#include <utility>

namespace hydra_thrust
{

template<typename T, typename Alloc> class host_vector;




template<typename T, typename Alloc = hydra_thrust::device_allocator<T> >
class device_vector
: public detail::vector_base<T,Alloc>
{
private:
typedef detail::vector_base<T,Alloc> Parent;

public:

typedef typename Parent::size_type  size_type;
typedef typename Parent::value_type value_type;



__host__
device_vector(void)
:Parent() {}


__host__
device_vector(const Alloc &alloc)
:Parent(alloc) {}


__host__
~device_vector(void) {}


__host__
explicit device_vector(size_type n)
:Parent(n) {}


__host__
explicit device_vector(size_type n, const Alloc &alloc)
:Parent(n,alloc) {}


__host__
explicit device_vector(size_type n, const value_type &value)
:Parent(n,value) {}


__host__
explicit device_vector(size_type n, const value_type &value, const Alloc &alloc)
:Parent(n,value,alloc) {}


__host__
device_vector(const device_vector &v)
:Parent(v) {}


__host__
device_vector(const device_vector &v, const Alloc &alloc)
:Parent(v,alloc) {}

#if HYDRA_THRUST_CPP_DIALECT >= 2011

__host__
device_vector(device_vector &&v)
:Parent(std::move(v)) {}


__host__
device_vector(device_vector &&v, const Alloc &alloc)
:Parent(std::move(v), alloc) {}
#endif 


__host__
device_vector &operator=(const device_vector &v)
{ Parent::operator=(v); return *this; }

#if HYDRA_THRUST_CPP_DIALECT >= 2011

__host__
device_vector &operator=(device_vector &&v)
{ Parent::operator=(std::move(v)); return *this; }
#endif 


template<typename OtherT, typename OtherAlloc>
__host__ explicit
__device__
device_vector(const device_vector<OtherT,OtherAlloc> &v)
:Parent(v) {}


template<typename OtherT, typename OtherAlloc>
__host__
device_vector &operator=(const device_vector<OtherT,OtherAlloc> &v)
{ Parent::operator=(v); return *this; }


template<typename OtherT, typename OtherAlloc>
__host__
device_vector(const std::vector<OtherT,OtherAlloc> &v)
:Parent(v) {}


template<typename OtherT, typename OtherAlloc>
__host__
device_vector &operator=(const std::vector<OtherT,OtherAlloc> &v)
{ Parent::operator=(v); return *this;}


template<typename OtherT, typename OtherAlloc>
__host__
device_vector(const host_vector<OtherT,OtherAlloc> &v);


template<typename OtherT, typename OtherAlloc>
__host__
device_vector &operator=(const host_vector<OtherT,OtherAlloc> &v)
{ Parent::operator=(v); return *this; }


template<typename InputIterator>
__host__
device_vector(InputIterator first, InputIterator last)
:Parent(first,last) {}


template<typename InputIterator>
__host__
device_vector(InputIterator first, InputIterator last, const Alloc &alloc)
:Parent(first,last,alloc) {}

#if 0

void resize(size_type new_size, const value_type &x = value_type());


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


void swap(device_vector &v);


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
#endif 
}; 


template<typename T, typename Alloc>
void swap(device_vector<T,Alloc> &a, device_vector<T,Alloc> &b)
{
a.swap(b);
} 



} 

#include <hydra/detail/external/hydra_thrust/detail/device_vector.inl>


