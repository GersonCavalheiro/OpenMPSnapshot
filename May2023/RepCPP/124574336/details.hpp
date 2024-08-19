



#if !defined(BOOST_CIRCULAR_BUFFER_DETAILS_HPP)
#define BOOST_CIRCULAR_BUFFER_DETAILS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/throw_exception.hpp>
#include <boost/core/allocator_access.hpp>
#include <boost/core/pointer_traits.hpp>
#include <boost/move/move.hpp>
#include <boost/type_traits/is_nothrow_move_constructible.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <iterator>

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable:4913)
#endif

namespace boost {

namespace cb_details {

template <class Alloc> struct nonconst_traits;

template<class ForwardIterator, class Diff, class T, class Alloc>
void uninitialized_fill_n_with_alloc(
ForwardIterator first, Diff n, const T& item, Alloc& alloc);

template<class InputIterator, class ForwardIterator, class Alloc>
ForwardIterator uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator dest, Alloc& a);

template<class InputIterator, class ForwardIterator, class Alloc>
ForwardIterator uninitialized_move_if_noexcept(InputIterator first, InputIterator last, ForwardIterator dest, Alloc& a);


template <class Alloc>
struct const_traits {
typedef typename Alloc::value_type value_type;
typedef typename boost::allocator_const_pointer<Alloc>::type pointer;
typedef const value_type& reference;
typedef typename boost::allocator_size_type<Alloc>::type size_type;
typedef typename boost::allocator_difference_type<Alloc>::type difference_type;

typedef nonconst_traits<Alloc> nonconst_self;
};


template <class Alloc>
struct nonconst_traits {
typedef typename Alloc::value_type value_type;
typedef typename boost::allocator_pointer<Alloc>::type pointer;
typedef value_type& reference;
typedef typename boost::allocator_size_type<Alloc>::type size_type;
typedef typename boost::allocator_difference_type<Alloc>::type difference_type;

typedef nonconst_traits<Alloc> nonconst_self;
};


template <class Iterator>
struct iterator_wrapper {
mutable Iterator m_it;
explicit iterator_wrapper(Iterator it) : m_it(it) {}
Iterator operator () () const { return m_it++; }
private:
iterator_wrapper<Iterator>& operator = (const iterator_wrapper<Iterator>&); 
};


template <class Pointer, class Value>
struct item_wrapper {
Value m_item;
explicit item_wrapper(Value item) : m_item(item) {}
Pointer operator () () const { return &m_item; }
private:
item_wrapper<Pointer, Value>& operator = (const item_wrapper<Pointer, Value>&); 
};


template <class Value, class Alloc>
struct assign_n {
typedef typename boost::allocator_size_type<Alloc>::type size_type;
size_type m_n;
Value m_item;
Alloc& m_alloc;
assign_n(size_type n, Value item, Alloc& alloc) : m_n(n), m_item(item), m_alloc(alloc) {}
template <class Pointer>
void operator () (Pointer p) const {
uninitialized_fill_n_with_alloc(p, m_n, m_item, m_alloc);
}
private:
assign_n<Value, Alloc>& operator = (const assign_n<Value, Alloc>&); 
};


template <class Iterator, class Alloc>
struct assign_range {
Iterator m_first;
Iterator m_last;
Alloc&   m_alloc;

assign_range(const Iterator& first, const Iterator& last, Alloc& alloc)
: m_first(first), m_last(last), m_alloc(alloc) {}

template <class Pointer>
void operator () (Pointer p) const {
boost::cb_details::uninitialized_copy(m_first, m_last, p, m_alloc);
}
};

template <class Iterator, class Alloc>
inline assign_range<Iterator, Alloc> make_assign_range(const Iterator& first, const Iterator& last, Alloc& a) {
return assign_range<Iterator, Alloc>(first, last, a);
}


template <class Size>
class capacity_control {

Size m_capacity;

Size m_min_capacity;

public:

capacity_control(Size buffer_capacity, Size min_buffer_capacity = 0)
: m_capacity(buffer_capacity), m_min_capacity(min_buffer_capacity)
{ 
BOOST_CB_ASSERT(buffer_capacity >= min_buffer_capacity);
}



Size capacity() const { return m_capacity; }

Size min_capacity() const { return m_min_capacity; }

operator Size() const { return m_capacity; }
};


template <class Buff, class Traits>
struct iterator
#if BOOST_CB_ENABLE_DEBUG
: public debug_iterator_base
#endif 
{

typedef iterator<Buff, typename Traits::nonconst_self> nonconst_self;

typedef std::random_access_iterator_tag iterator_category;

typedef typename Traits::value_type value_type;

typedef typename Traits::pointer pointer;

typedef typename Traits::reference reference;

typedef typename Traits::size_type size_type;

typedef typename Traits::difference_type difference_type;


const Buff* m_buff;

pointer m_it;



iterator() : m_buff(0), m_it(0) {}

#if BOOST_CB_ENABLE_DEBUG

iterator(const nonconst_self& it) : debug_iterator_base(it), m_buff(it.m_buff), m_it(it.m_it) {}


iterator(const Buff* cb, const pointer p) : debug_iterator_base(cb), m_buff(cb), m_it(p) {}

#else

iterator(const nonconst_self& it) : m_buff(it.m_buff), m_it(it.m_it) {}

iterator(const Buff* cb, const pointer p) : m_buff(cb), m_it(p) {}

#endif 

iterator& operator = (const iterator& it) {
if (this == &it)
return *this;
#if BOOST_CB_ENABLE_DEBUG
debug_iterator_base::operator =(it);
#endif 
m_buff = it.m_buff;
m_it = it.m_it;
return *this;
}


reference operator * () const {
BOOST_CB_ASSERT(is_valid(m_buff)); 
BOOST_CB_ASSERT(m_it != 0);        
return *m_it;
}

pointer operator -> () const { return &(operator*()); }

template <class Traits0>
difference_type operator - (const iterator<Buff, Traits0>& it) const {
BOOST_CB_ASSERT(is_valid(m_buff));    
BOOST_CB_ASSERT(it.is_valid(m_buff)); 
return linearize_pointer(*this) - linearize_pointer(it);
}

iterator& operator ++ () {
BOOST_CB_ASSERT(is_valid(m_buff)); 
BOOST_CB_ASSERT(m_it != 0);        
m_buff->increment(m_it);
if (m_it == m_buff->m_last)
m_it = 0;
return *this;
}

iterator operator ++ (int) {
iterator<Buff, Traits> tmp = *this;
++*this;
return tmp;
}

iterator& operator -- () {
BOOST_CB_ASSERT(is_valid(m_buff));        
BOOST_CB_ASSERT(m_it != m_buff->m_first); 
if (m_it == 0)
m_it = m_buff->m_last;
m_buff->decrement(m_it);
return *this;
}

iterator operator -- (int) {
iterator<Buff, Traits> tmp = *this;
--*this;
return tmp;
}

iterator& operator += (difference_type n) {
BOOST_CB_ASSERT(is_valid(m_buff)); 
if (n > 0) {
BOOST_CB_ASSERT(m_buff->end() - *this >= n); 
m_it = m_buff->add(m_it, n);
if (m_it == m_buff->m_last)
m_it = 0;
} else if (n < 0) {
*this -= -n;
}
return *this;
}

iterator operator + (difference_type n) const { return iterator<Buff, Traits>(*this) += n; }

iterator& operator -= (difference_type n) {
BOOST_CB_ASSERT(is_valid(m_buff)); 
if (n > 0) {
BOOST_CB_ASSERT(*this - m_buff->begin() >= n); 
m_it = m_buff->sub(m_it == 0 ? m_buff->m_last : m_it, n);
} else if (n < 0) {
*this += -n;
}
return *this;
}

iterator operator - (difference_type n) const { return iterator<Buff, Traits>(*this) -= n; }

reference operator [] (difference_type n) const { return *(*this + n); }


template <class Traits0>
bool operator == (const iterator<Buff, Traits0>& it) const {
BOOST_CB_ASSERT(is_valid(m_buff));    
BOOST_CB_ASSERT(it.is_valid(m_buff)); 
return m_it == it.m_it;
}

template <class Traits0>
bool operator != (const iterator<Buff, Traits0>& it) const {
BOOST_CB_ASSERT(is_valid(m_buff));    
BOOST_CB_ASSERT(it.is_valid(m_buff)); 
return m_it != it.m_it;
}

template <class Traits0>
bool operator < (const iterator<Buff, Traits0>& it) const {
BOOST_CB_ASSERT(is_valid(m_buff));    
BOOST_CB_ASSERT(it.is_valid(m_buff)); 
return linearize_pointer(*this) < linearize_pointer(it);
}

template <class Traits0>
bool operator > (const iterator<Buff, Traits0>& it) const { return it < *this; }

template <class Traits0>
bool operator <= (const iterator<Buff, Traits0>& it) const { return !(it < *this); }

template <class Traits0>
bool operator >= (const iterator<Buff, Traits0>& it) const { return !(*this < it); }


template <class Traits0>
typename Traits0::pointer linearize_pointer(const iterator<Buff, Traits0>& it) const {
return it.m_it == 0 ? m_buff->m_buff + m_buff->size() :
(it.m_it < m_buff->m_first ? it.m_it + (m_buff->m_end - m_buff->m_first)
: m_buff->m_buff + (it.m_it - m_buff->m_first));
}
};

template <class Buff, class Traits>
inline iterator<Buff, Traits>
operator + (typename Traits::difference_type n, const iterator<Buff, Traits>& it) {
return it + n;
}


template<class InputIterator, class ForwardIterator, class Alloc>
inline ForwardIterator uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator dest, Alloc& a) {
ForwardIterator next = dest;
BOOST_TRY {
for (; first != last; ++first, ++dest)
boost::allocator_construct(a, boost::to_address(dest), *first);
} BOOST_CATCH(...) {
for (; next != dest; ++next)
boost::allocator_destroy(a, boost::to_address(next));
BOOST_RETHROW
}
BOOST_CATCH_END
return dest;
}

template<class InputIterator, class ForwardIterator, class Alloc>
ForwardIterator uninitialized_move_if_noexcept_impl(InputIterator first, InputIterator last, ForwardIterator dest, Alloc& a,
true_type) {
for (; first != last; ++first, ++dest)
boost::allocator_construct(a, boost::to_address(dest), boost::move(*first));
return dest;
}

template<class InputIterator, class ForwardIterator, class Alloc>
ForwardIterator uninitialized_move_if_noexcept_impl(InputIterator first, InputIterator last, ForwardIterator dest, Alloc& a,
false_type) {
return uninitialized_copy(first, last, dest, a);
}


template<class InputIterator, class ForwardIterator, class Alloc>
ForwardIterator uninitialized_move_if_noexcept(InputIterator first, InputIterator last, ForwardIterator dest, Alloc& a) {
typedef typename boost::is_nothrow_move_constructible<typename Alloc::value_type>::type tag_t;
return uninitialized_move_if_noexcept_impl(first, last, dest, a, tag_t());
}


template<class ForwardIterator, class Diff, class T, class Alloc>
inline void uninitialized_fill_n_with_alloc(ForwardIterator first, Diff n, const T& item, Alloc& alloc) {
ForwardIterator next = first;
BOOST_TRY {
for (; n > 0; ++first, --n)
boost::allocator_construct(alloc, boost::to_address(first), item);
} BOOST_CATCH(...) {
for (; next != first; ++next)
boost::allocator_destroy(alloc, boost::to_address(next));
BOOST_RETHROW
}
BOOST_CATCH_END
}

} 

} 

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

#endif 
