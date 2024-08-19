
#ifndef BOOST_CONTAINER_STATIC_VECTOR_HPP
#define BOOST_CONTAINER_STATIC_VECTOR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/detail/type_traits.hpp>
#include <boost/container/vector.hpp>

#include <cstddef>
#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
#include <initializer_list>
#endif

namespace boost { namespace container {

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

namespace dtl {

template<class T, std::size_t N, std::size_t InplaceAlignment, bool ThrowOnOverflow>
class static_storage_allocator
{
typedef bool_<ThrowOnOverflow> throw_on_overflow_t;

static BOOST_NORETURN BOOST_CONTAINER_FORCEINLINE void on_capacity_overflow(true_type)
{
(throw_bad_alloc)();
}

static BOOST_CONTAINER_FORCEINLINE void on_capacity_overflow(false_type)
{
BOOST_ASSERT_MSG(false, "ERROR: static vector capacity overflow");
}

public:
typedef T value_type;

BOOST_CONTAINER_FORCEINLINE static_storage_allocator() BOOST_NOEXCEPT_OR_NOTHROW
{}

BOOST_CONTAINER_FORCEINLINE static_storage_allocator(const static_storage_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{}

BOOST_CONTAINER_FORCEINLINE static_storage_allocator & operator=(const static_storage_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return *this;  }

BOOST_CONTAINER_FORCEINLINE T* internal_storage() const BOOST_NOEXCEPT_OR_NOTHROW
{  return const_cast<T*>(static_cast<const T*>(static_cast<const void*>(storage.data)));  }

BOOST_CONTAINER_FORCEINLINE T* internal_storage() BOOST_NOEXCEPT_OR_NOTHROW
{  return static_cast<T*>(static_cast<void*>(storage.data));  }

static const std::size_t internal_capacity = N;

std::size_t max_size() const
{  return N;   }

static BOOST_CONTAINER_FORCEINLINE void on_capacity_overflow()
{
(on_capacity_overflow)(throw_on_overflow_t());
}

typedef boost::container::dtl::version_type<static_storage_allocator, 0>   version;

BOOST_CONTAINER_FORCEINLINE friend bool operator==(const static_storage_allocator &, const static_storage_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return false;  }

BOOST_CONTAINER_FORCEINLINE friend bool operator!=(const static_storage_allocator &, const static_storage_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return true;  }

private:
BOOST_STATIC_ASSERT_MSG(!InplaceAlignment || (InplaceAlignment & (InplaceAlignment-1)) == 0, "Alignment option must be zero or power of two");
static const std::size_t final_alignment = InplaceAlignment ? InplaceAlignment : dtl::alignment_of<T>::value;
typename dtl::aligned_storage<sizeof(T)*N, final_alignment>::type storage;
};

template<class Options>
struct get_static_vector_opt
{
typedef Options type;
};

template<>
struct get_static_vector_opt<void>
{
typedef static_vector_null_opt type;
};

template <typename T, std::size_t Capacity, class Options>
struct get_static_vector_allocator
{
typedef typename  get_static_vector_opt<Options>::type options_t;
typedef dtl::static_storage_allocator
< T
, Capacity
, options_t::inplace_alignment
, options_t::throw_on_overflow
> type;
};


}  

#endif   

template <typename T, std::size_t Capacity, class Options BOOST_CONTAINER_DOCONLY(= void) >
class static_vector
: public vector<T, typename dtl::get_static_vector_allocator< T, Capacity, Options>::type>
{
public:
#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
typedef typename dtl::get_static_vector_allocator< T, Capacity, Options>::type allocator_type;
typedef vector<T, allocator_type > base_t;

BOOST_COPYABLE_AND_MOVABLE(static_vector)

template<class U, std::size_t OtherCapacity, class OtherOptions>
friend class static_vector;

public:
#endif   

public:
typedef typename base_t::value_type value_type;
typedef typename base_t::size_type size_type;
typedef typename base_t::difference_type difference_type;
typedef typename base_t::pointer pointer;
typedef typename base_t::const_pointer const_pointer;
typedef typename base_t::reference reference;
typedef typename base_t::const_reference const_reference;
typedef typename base_t::iterator iterator;
typedef typename base_t::const_iterator const_iterator;
typedef typename base_t::reverse_iterator reverse_iterator;
typedef typename base_t::const_reverse_iterator const_reverse_iterator;

static const size_type static_capacity = Capacity;

BOOST_CONTAINER_FORCEINLINE static_vector() BOOST_NOEXCEPT_OR_NOTHROW
: base_t()
{}

BOOST_CONTAINER_FORCEINLINE explicit static_vector(size_type count)
: base_t(count)
{}

BOOST_CONTAINER_FORCEINLINE static_vector(size_type count, default_init_t)
: base_t(count, default_init_t())
{}

BOOST_CONTAINER_FORCEINLINE static_vector(size_type count, value_type const& value)
: base_t(count, value)
{}

template <typename Iterator>
BOOST_CONTAINER_FORCEINLINE static_vector(Iterator first, Iterator last)
: base_t(first, last)
{}

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
BOOST_CONTAINER_FORCEINLINE static_vector(std::initializer_list<value_type> il)
: base_t(il)
{}
#endif

BOOST_CONTAINER_FORCEINLINE static_vector(static_vector const& other)
: base_t(other)
{}

BOOST_CONTAINER_FORCEINLINE static_vector(static_vector const& other, const allocator_type &)
: base_t(other)
{}

BOOST_CONTAINER_FORCEINLINE static_vector(BOOST_RV_REF(static_vector) other,  const allocator_type &)
BOOST_NOEXCEPT_IF(boost::container::dtl::is_nothrow_move_constructible<value_type>::value)
: base_t(BOOST_MOVE_BASE(base_t, other))
{}

BOOST_CONTAINER_FORCEINLINE explicit static_vector(const allocator_type &)
: base_t()
{}

template <std::size_t C, class O>
BOOST_CONTAINER_FORCEINLINE static_vector(static_vector<T, C, O> const& other)
: base_t(other)
{}

BOOST_CONTAINER_FORCEINLINE static_vector(BOOST_RV_REF(static_vector) other)
BOOST_NOEXCEPT_IF(boost::container::dtl::is_nothrow_move_constructible<value_type>::value)
: base_t(BOOST_MOVE_BASE(base_t, other))
{}

template <std::size_t C, class O>
BOOST_CONTAINER_FORCEINLINE static_vector(BOOST_RV_REF_BEG static_vector<T, C, O> BOOST_RV_REF_END other)
: base_t(BOOST_MOVE_BASE(typename static_vector<T BOOST_MOVE_I C>::base_t, other))
{}

BOOST_CONTAINER_FORCEINLINE static_vector & operator=(BOOST_COPY_ASSIGN_REF(static_vector) other)
{
return static_cast<static_vector&>(base_t::operator=(static_cast<base_t const&>(other)));
}

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)
BOOST_CONTAINER_FORCEINLINE static_vector & operator=(std::initializer_list<value_type> il)
{ return static_cast<static_vector&>(base_t::operator=(il));  }
#endif

template <std::size_t C, class O>
BOOST_CONTAINER_FORCEINLINE static_vector & operator=(static_vector<T, C, O> const& other)
{
return static_cast<static_vector&>(base_t::operator=
(static_cast<typename static_vector<T, C, O>::base_t const&>(other)));
}

BOOST_CONTAINER_FORCEINLINE static_vector & operator=(BOOST_RV_REF(static_vector) other)
BOOST_NOEXCEPT_IF(boost::container::dtl::is_nothrow_move_assignable<value_type>::value)
{
return static_cast<static_vector&>(base_t::operator=(BOOST_MOVE_BASE(base_t, other)));
}

template <std::size_t C, class O>
BOOST_CONTAINER_FORCEINLINE static_vector & operator=(BOOST_RV_REF_BEG static_vector<T, C, O> BOOST_RV_REF_END other)
{
return static_cast<static_vector&>(base_t::operator=
(BOOST_MOVE_BASE(typename static_vector<T BOOST_MOVE_I C>::base_t, other)));
}

#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED

~static_vector();

void swap(static_vector & other);

template <std::size_t C, class O>
void swap(static_vector<T, C, O> & other);

void resize(size_type count);

void resize(size_type count, default_init_t);

void resize(size_type count, value_type const& value);

void reserve(size_type count)  BOOST_NOEXCEPT_OR_NOTHROW;

void push_back(value_type const& value);

void push_back(BOOST_RV_REF(value_type) value);

void pop_back();

iterator insert(const_iterator p, value_type const& value);

iterator insert(const_iterator p, BOOST_RV_REF(value_type) value);

iterator insert(const_iterator p, size_type count, value_type const& value);

template <typename Iterator>
iterator insert(const_iterator p, Iterator first, Iterator last);

iterator insert(const_iterator p, std::initializer_list<value_type> il);

iterator erase(const_iterator p);

iterator erase(const_iterator first, const_iterator last);

template <typename Iterator>
void assign(Iterator first, Iterator last);

void assign(std::initializer_list<value_type> il);

void assign(size_type count, value_type const& value);

template<class ...Args>
reference emplace_back(Args &&...args);

template<class ...Args>
iterator emplace(const_iterator p, Args &&...args);

void clear()  BOOST_NOEXCEPT_OR_NOTHROW;

reference at(size_type i);

const_reference at(size_type i) const;

reference operator[](size_type i);

const_reference operator[](size_type i) const;

iterator nth(size_type i);

const_iterator nth(size_type i) const;

size_type index_of(iterator p);

size_type index_of(const_iterator p) const;

reference front();

const_reference front() const;

reference back();

const_reference back() const;

T * data() BOOST_NOEXCEPT_OR_NOTHROW;

const T * data() const BOOST_NOEXCEPT_OR_NOTHROW;

iterator begin() BOOST_NOEXCEPT_OR_NOTHROW;

const_iterator begin() const BOOST_NOEXCEPT_OR_NOTHROW;

const_iterator cbegin() const BOOST_NOEXCEPT_OR_NOTHROW;

iterator end() BOOST_NOEXCEPT_OR_NOTHROW;

const_iterator end() const BOOST_NOEXCEPT_OR_NOTHROW;

const_iterator cend() const BOOST_NOEXCEPT_OR_NOTHROW;

reverse_iterator rbegin() BOOST_NOEXCEPT_OR_NOTHROW;

const_reverse_iterator rbegin() const BOOST_NOEXCEPT_OR_NOTHROW;

const_reverse_iterator crbegin() const BOOST_NOEXCEPT_OR_NOTHROW;

reverse_iterator rend() BOOST_NOEXCEPT_OR_NOTHROW;

const_reverse_iterator rend() const BOOST_NOEXCEPT_OR_NOTHROW;

const_reverse_iterator crend() const BOOST_NOEXCEPT_OR_NOTHROW;

#endif   

BOOST_CONTAINER_FORCEINLINE static size_type capacity() BOOST_NOEXCEPT_OR_NOTHROW
{ return static_capacity; }

BOOST_CONTAINER_FORCEINLINE static size_type max_size() BOOST_NOEXCEPT_OR_NOTHROW
{ return static_capacity; }

#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED

size_type size() const BOOST_NOEXCEPT_OR_NOTHROW;

bool empty() const BOOST_NOEXCEPT_OR_NOTHROW;
#else

BOOST_CONTAINER_FORCEINLINE friend void swap(static_vector &x, static_vector &y)
{
x.swap(y);
}

#endif 

};

#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
bool operator== (static_vector<V, C1, O1> const& x, static_vector<V, C2, O2> const& y);

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
bool operator!= (static_vector<V, C1, O1> const& x, static_vector<V, C2, O2> const& y);

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
bool operator< (static_vector<V, C1, O1> const& x, static_vector<V, C2, O2> const& y);

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
bool operator> (static_vector<V, C1, O1> const& x, static_vector<V, C2, O2> const& y);

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
bool operator<= (static_vector<V, C1, O1> const& x, static_vector<V, C2, O2> const& y);

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
bool operator>= (static_vector<V, C1, O1> const& x, static_vector<V, C2, O2> const& y);

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
inline void swap(static_vector<V, C1, O1> & x, static_vector<V, C2, O2> & y);

#else

template<typename V, std::size_t C1, std::size_t C2, class O1, class O2>
inline void swap(static_vector<V, C1, O1> & x, static_vector<V, C2, O2> & y
, typename dtl::enable_if_c< C1 != C2>::type * = 0)
{
x.swap(y);
}

#endif 

}} 

#include <boost/container/detail/config_end.hpp>

#endif 
