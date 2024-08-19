
#ifndef aligned_allocator_hh_
#define aligned_allocator_hh_

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <limits>
#include <climits>
#include <new>

#include "madthreading/macros.hh"

#ifndef do_pragma
#   define do_pragma(x) _Pragma(#x)
#endif

#if defined(__GNUC__) && defined(__x86_64__)
#define attrib_assume_aligned __attribute__((assume_aligned(64)))
#define attrib_aligned __attribute__((aligned (64)))
#else
#define attrib_assume_aligned
#define attrib_aligned
#endif

namespace mad
{

static size_t const SIMD_WIDTH = 64;

void* aligned_alloc(size_t size, size_t align);
void  aligned_free(void* ptr);

template <typename T>
class simd_allocator
{
public:
typedef T               value_type;
typedef T*              pointer;
typedef const T*        const_pointer;
typedef T&              reference;
typedef const T&        const_reference;
typedef std::size_t     size_type;
typedef std::ptrdiff_t  difference_type;

template <typename U>
struct rebind
{
typedef simd_allocator<U> other;
};

pointer address(reference value) const
{
return &value;
}

const_pointer address(const_reference value) const
{
return &value;
}

simd_allocator() throw() { }

simd_allocator(const simd_allocator&) throw() { }

template <typename U>
simd_allocator(const simd_allocator<U>&) throw() { }

~simd_allocator() throw() { }

size_type max_size() const throw()
{
#if defined(CXX11)
return std::numeric_limits<std::size_t>::max() / sizeof(T);
#else
return SIZE_MAX / sizeof(T);
#endif
}

pointer allocate(const size_type num, const void* hint = 0)
{
return static_cast<pointer>(aligned_alloc(num*sizeof(T), SIMD_WIDTH));
}

void construct(pointer p, const T& value)
{
new (static_cast<void*>(p)) T(value);
}

void destroy(pointer p)
{
p->~T();
}

void deallocate (pointer p, size_type num)
{
aligned_free(static_cast<void*>(p));
}
};

template <typename T1, typename T2>
bool operator ==(const simd_allocator<T1>&,
const simd_allocator<T2>&) throw()
{
return true;
}
template <typename T1, typename T2>
bool operator !=(const simd_allocator<T1>&,
const simd_allocator<T2>&) throw()
{
return false;
}


template <typename _Tp>
_Tp* simd_alloc(size_t n)
{
return static_cast<_Tp*>(mad::aligned_alloc(n * sizeof(_Tp),
mad::SIMD_WIDTH));
}

template <typename _Tp>
class simd_array
{
public:
typedef std::size_t size_type;

public:
simd_array()
: m_data(nullptr)
{ }

simd_array(size_type _n)
: m_data(mad::simd_alloc<_Tp>(_n))
{ }

simd_array(size_type _n, const _Tp& _init)
: m_data(mad::simd_alloc<_Tp>(_n))
{
for(size_type i = 0; i < _n; ++i)
m_data[i] = _init;
}

~simd_array()
{
mad::aligned_free(m_data);
}

operator const _Tp*() const attrib_assume_aligned { return m_data; }
operator _Tp*() attrib_assume_aligned { return m_data; }


simd_array& operator=(const simd_array& rhs)
{
if(this != &rhs)
{
if(m_data)
mad::aligned_free(m_data);
m_data = static_cast<_Tp*>(rhs.m_data);
const_cast<simd_array&>(rhs).m_data = nullptr;
}
return *this;
}

private:
_Tp* m_data;

} attrib_aligned;


}



#endif
