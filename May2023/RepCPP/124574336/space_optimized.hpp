


#if !defined(BOOST_CIRCULAR_BUFFER_SPACE_OPTIMIZED_HPP)
#define BOOST_CIRCULAR_BUFFER_SPACE_OPTIMIZED_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/type_traits/is_same.hpp>
#include <boost/config/workaround.hpp>

namespace boost {


template <class T, class Alloc>
class circular_buffer_space_optimized :

#if BOOST_CB_ENABLE_DEBUG
public
#endif

circular_buffer<T, Alloc> {
public:

typedef typename circular_buffer<T, Alloc>::value_type value_type;
typedef typename circular_buffer<T, Alloc>::pointer pointer;
typedef typename circular_buffer<T, Alloc>::const_pointer const_pointer;
typedef typename circular_buffer<T, Alloc>::reference reference;
typedef typename circular_buffer<T, Alloc>::const_reference const_reference;
typedef typename circular_buffer<T, Alloc>::size_type size_type;
typedef typename circular_buffer<T, Alloc>::difference_type difference_type;
typedef typename circular_buffer<T, Alloc>::allocator_type allocator_type;
typedef typename circular_buffer<T, Alloc>::const_iterator const_iterator;
typedef typename circular_buffer<T, Alloc>::iterator iterator;
typedef typename circular_buffer<T, Alloc>::const_reverse_iterator const_reverse_iterator;
typedef typename circular_buffer<T, Alloc>::reverse_iterator reverse_iterator;
typedef typename circular_buffer<T, Alloc>::array_range array_range;
typedef typename circular_buffer<T, Alloc>::const_array_range const_array_range;
typedef typename circular_buffer<T, Alloc>::param_value_type param_value_type;
typedef typename circular_buffer<T, Alloc>::rvalue_type rvalue_type;




typedef cb_details::capacity_control<size_type> capacity_type;


using circular_buffer<T, Alloc>::get_allocator;
using circular_buffer<T, Alloc>::begin;
using circular_buffer<T, Alloc>::end;
using circular_buffer<T, Alloc>::rbegin;
using circular_buffer<T, Alloc>::rend;
using circular_buffer<T, Alloc>::at;
using circular_buffer<T, Alloc>::front;
using circular_buffer<T, Alloc>::back;
using circular_buffer<T, Alloc>::array_one;
using circular_buffer<T, Alloc>::array_two;
using circular_buffer<T, Alloc>::linearize;
using circular_buffer<T, Alloc>::is_linearized;
using circular_buffer<T, Alloc>::rotate;
using circular_buffer<T, Alloc>::size;
using circular_buffer<T, Alloc>::max_size;
using circular_buffer<T, Alloc>::empty;

#if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
reference operator [] (size_type n) { return circular_buffer<T, Alloc>::operator[](n); }
const_reference operator [] (size_type n) const { return circular_buffer<T, Alloc>::operator[](n); }
#else
using circular_buffer<T, Alloc>::operator[];
#endif

private:

capacity_type m_capacity_ctrl;

public:


bool full() const BOOST_NOEXCEPT { return m_capacity_ctrl == size(); }


size_type reserve() const BOOST_NOEXCEPT { return m_capacity_ctrl - size(); }


const capacity_type& capacity() const BOOST_NOEXCEPT { return m_capacity_ctrl; }

#if defined(BOOST_CB_TEST)


size_type internal_capacity() const BOOST_NOEXCEPT { return circular_buffer<T, Alloc>::capacity(); }

#endif 


void set_capacity(const capacity_type& capacity_ctrl) {
m_capacity_ctrl = capacity_ctrl;
if (capacity_ctrl < size()) {
iterator e = end();
circular_buffer<T, Alloc>::erase(e - (size() - capacity_ctrl), e);
}
adjust_min_capacity();
}


void resize(size_type new_size, param_value_type item = value_type()) {
if (new_size > size()) {
if (new_size > m_capacity_ctrl)
m_capacity_ctrl = capacity_type(new_size, m_capacity_ctrl.min_capacity());
insert(end(), new_size - size(), item);
} else {
iterator e = end();
erase(e - (size() - new_size), e);
}
}


void rset_capacity(const capacity_type& capacity_ctrl) {
m_capacity_ctrl = capacity_ctrl;
if (capacity_ctrl < size()) {
iterator b = begin();
circular_buffer<T, Alloc>::rerase(b, b + (size() - capacity_ctrl));
}
adjust_min_capacity();
}


void rresize(size_type new_size, param_value_type item = value_type()) {
if (new_size > size()) {
if (new_size > m_capacity_ctrl)
m_capacity_ctrl = capacity_type(new_size, m_capacity_ctrl.min_capacity());
rinsert(begin(), new_size - size(), item);
} else {
rerase(begin(), end() - new_size);
}
}


explicit circular_buffer_space_optimized(const allocator_type& alloc = allocator_type()) BOOST_NOEXCEPT
: circular_buffer<T, Alloc>(0, alloc)
, m_capacity_ctrl(0) {}


explicit circular_buffer_space_optimized(capacity_type capacity_ctrl,
const allocator_type& alloc = allocator_type())
: circular_buffer<T, Alloc>(capacity_ctrl.min_capacity(), alloc)
, m_capacity_ctrl(capacity_ctrl) {}


circular_buffer_space_optimized(capacity_type capacity_ctrl, param_value_type item,
const allocator_type& alloc = allocator_type())
: circular_buffer<T, Alloc>(capacity_ctrl.capacity(), item, alloc)
, m_capacity_ctrl(capacity_ctrl) {}


circular_buffer_space_optimized(capacity_type capacity_ctrl, size_type n, param_value_type item,
const allocator_type& alloc = allocator_type())
: circular_buffer<T, Alloc>(init_capacity(capacity_ctrl, n), n, item, alloc)
, m_capacity_ctrl(capacity_ctrl) {}


circular_buffer_space_optimized(const circular_buffer_space_optimized<T, Alloc>& cb)
: circular_buffer<T, Alloc>(cb.begin(), cb.end(), cb.get_allocator())
, m_capacity_ctrl(cb.m_capacity_ctrl) {}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

circular_buffer_space_optimized(circular_buffer_space_optimized<T, Alloc>&& cb) BOOST_NOEXCEPT
: circular_buffer<T, Alloc>()
, m_capacity_ctrl(0) {
cb.swap(*this);
}
#endif 


template <class InputIterator>
circular_buffer_space_optimized(InputIterator first, InputIterator last,
const allocator_type& alloc = allocator_type())
: circular_buffer<T, Alloc>(first, last, alloc)
, m_capacity_ctrl(circular_buffer<T, Alloc>::capacity()) {}


template <class InputIterator>
circular_buffer_space_optimized(capacity_type capacity_ctrl, InputIterator first, InputIterator last,
const allocator_type& alloc = allocator_type())
: circular_buffer<T, Alloc>(
init_capacity(capacity_ctrl, first, last, is_integral<InputIterator>()),
first, last, alloc)
, m_capacity_ctrl(capacity_ctrl) {
reduce_capacity(
is_same< BOOST_DEDUCED_TYPENAME std::iterator_traits<InputIterator>::iterator_category, std::input_iterator_tag >());
}

#if defined(BOOST_CB_NEVER_DEFINED)


~circular_buffer_space_optimized();

void erase_begin(size_type n);

void erase_end(size_type n);

#endif 


circular_buffer_space_optimized<T, Alloc>& operator = (const circular_buffer_space_optimized<T, Alloc>& cb) {
if (this == &cb)
return *this;
circular_buffer<T, Alloc>::assign(cb.begin(), cb.end());
m_capacity_ctrl = cb.m_capacity_ctrl;
return *this;
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

circular_buffer_space_optimized<T, Alloc>& operator = (circular_buffer_space_optimized<T, Alloc>&& cb) BOOST_NOEXCEPT {
cb.swap(*this); 
circular_buffer<T, Alloc>(get_allocator()) 
.swap(cb); 
return *this;
}
#endif 



void assign(size_type n, param_value_type item) {
circular_buffer<T, Alloc>::assign(n, item);
m_capacity_ctrl = capacity_type(n);
}


void assign(capacity_type capacity_ctrl, size_type n, param_value_type item) {
BOOST_CB_ASSERT(capacity_ctrl.capacity() >= n); 
circular_buffer<T, Alloc>::assign((std::max)(capacity_ctrl.min_capacity(), n), n, item);
m_capacity_ctrl = capacity_ctrl;
}


template <class InputIterator>
void assign(InputIterator first, InputIterator last) {
circular_buffer<T, Alloc>::assign(first, last);
m_capacity_ctrl = capacity_type(circular_buffer<T, Alloc>::capacity());
}


template <class InputIterator>
void assign(capacity_type capacity_ctrl, InputIterator first, InputIterator last) {
m_capacity_ctrl = capacity_ctrl;
circular_buffer<T, Alloc>::assign(capacity_ctrl, first, last);
}



void swap(circular_buffer_space_optimized<T, Alloc>& cb) BOOST_NOEXCEPT {
std::swap(m_capacity_ctrl, cb.m_capacity_ctrl);
circular_buffer<T, Alloc>::swap(cb);
}


void push_back(param_value_type item) {
check_low_capacity();
circular_buffer<T, Alloc>::push_back(item);
}


void push_back(rvalue_type item) {
check_low_capacity();
circular_buffer<T, Alloc>::push_back(boost::move(item));
}


void push_back() {
check_low_capacity();
circular_buffer<T, Alloc>::push_back();
}


void push_front(param_value_type item) {
check_low_capacity();
circular_buffer<T, Alloc>::push_front(item);
}


void push_front(rvalue_type item) {
check_low_capacity();
circular_buffer<T, Alloc>::push_front(boost::move(item));
}


void push_front() {
check_low_capacity();
circular_buffer<T, Alloc>::push_front();
}


void pop_back() {
circular_buffer<T, Alloc>::pop_back();
check_high_capacity();
}


void pop_front() {
circular_buffer<T, Alloc>::pop_front();
check_high_capacity();
}


iterator insert(iterator pos, param_value_type item) {
size_type index = pos - begin();
check_low_capacity();
return circular_buffer<T, Alloc>::insert(begin() + index, item);
}


iterator insert(iterator pos, rvalue_type item) {
size_type index = pos - begin();
check_low_capacity();
return circular_buffer<T, Alloc>::insert(begin() + index, boost::move(item));
}


iterator insert(iterator pos) {
size_type index = pos - begin();
check_low_capacity();
return circular_buffer<T, Alloc>::insert(begin() + index);
}


void insert(iterator pos, size_type n, param_value_type item) {
size_type index = pos - begin();
check_low_capacity(n);
circular_buffer<T, Alloc>::insert(begin() + index, n, item);
}


template <class InputIterator>
void insert(iterator pos, InputIterator first, InputIterator last) {
insert(pos, first, last, is_integral<InputIterator>());
}


iterator rinsert(iterator pos, param_value_type item) {
size_type index = pos - begin();
check_low_capacity();
return circular_buffer<T, Alloc>::rinsert(begin() + index, item);
}


iterator rinsert(iterator pos, rvalue_type item) {
size_type index = pos - begin();
check_low_capacity();
return circular_buffer<T, Alloc>::rinsert(begin() + index, boost::move(item));
}


iterator rinsert(iterator pos) {
size_type index = pos - begin();
check_low_capacity();
return circular_buffer<T, Alloc>::rinsert(begin() + index);
}


void rinsert(iterator pos, size_type n, param_value_type item) {
size_type index = pos - begin();
check_low_capacity(n);
circular_buffer<T, Alloc>::rinsert(begin() + index, n, item);
}


template <class InputIterator>
void rinsert(iterator pos, InputIterator first, InputIterator last) {
rinsert(pos, first, last, is_integral<InputIterator>());
}


iterator erase(iterator pos) {
iterator it = circular_buffer<T, Alloc>::erase(pos);
size_type index = it - begin();
check_high_capacity();
return begin() + index;
}


iterator erase(iterator first, iterator last) {
iterator it = circular_buffer<T, Alloc>::erase(first, last);
size_type index = it - begin();
check_high_capacity();
return begin() + index;
}


iterator rerase(iterator pos) {
iterator it = circular_buffer<T, Alloc>::rerase(pos);
size_type index = it - begin();
check_high_capacity();
return begin() + index;
}


iterator rerase(iterator first, iterator last) {
iterator it = circular_buffer<T, Alloc>::rerase(first, last);
size_type index = it - begin();
check_high_capacity();
return begin() + index;
}


void clear() { erase(begin(), end()); }

private:


void adjust_min_capacity() {
if (m_capacity_ctrl.min_capacity() > circular_buffer<T, Alloc>::capacity())
circular_buffer<T, Alloc>::set_capacity(m_capacity_ctrl.min_capacity());
else
check_high_capacity();
}


size_type ensure_reserve(size_type new_capacity, size_type buffer_size) const {
if (buffer_size + new_capacity / 5 >= new_capacity)
new_capacity *= 2; 
if (new_capacity > m_capacity_ctrl)
return m_capacity_ctrl;
return new_capacity;
}


void check_low_capacity(size_type n = 1) {
size_type new_size = size() + n;
size_type new_capacity = circular_buffer<T, Alloc>::capacity();
if (new_size > new_capacity) {
if (new_capacity == 0)
new_capacity = 1;
for (; new_size > new_capacity; new_capacity *= 2) {}
circular_buffer<T, Alloc>::set_capacity(
ensure_reserve(new_capacity, new_size));
}
#if BOOST_CB_ENABLE_DEBUG
this->invalidate_iterators_except(end());
#endif
}


void check_high_capacity() {
size_type new_capacity = circular_buffer<T, Alloc>::capacity();
while (new_capacity / 3 >= size()) { 
new_capacity /= 2;
if (new_capacity <= m_capacity_ctrl.min_capacity()) {
new_capacity = m_capacity_ctrl.min_capacity();
break;
}
}
circular_buffer<T, Alloc>::set_capacity(
ensure_reserve(new_capacity, size()));
#if BOOST_CB_ENABLE_DEBUG
this->invalidate_iterators_except(end());
#endif
}


void reduce_capacity(const true_type&) {
circular_buffer<T, Alloc>::set_capacity((std::max)(m_capacity_ctrl.min_capacity(), size()));
}


void reduce_capacity(const false_type&) {}


static size_type init_capacity(const capacity_type& capacity_ctrl, size_type n) {
BOOST_CB_ASSERT(capacity_ctrl.capacity() >= n); 
return (std::max)(capacity_ctrl.min_capacity(), n);
}


template <class IntegralType>
static size_type init_capacity(const capacity_type& capacity_ctrl, IntegralType n, IntegralType,
const true_type&) {
return init_capacity(capacity_ctrl, static_cast<size_type>(n));
}


template <class Iterator>
static size_type init_capacity(const capacity_type& capacity_ctrl, Iterator first, Iterator last,
const false_type&) {
BOOST_CB_IS_CONVERTIBLE(Iterator, value_type); 
#if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x581))
return init_capacity(capacity_ctrl, first, last, std::iterator_traits<Iterator>::iterator_category());
#else
return init_capacity(
capacity_ctrl, first, last, BOOST_DEDUCED_TYPENAME std::iterator_traits<Iterator>::iterator_category());
#endif
}


template <class InputIterator>
static size_type init_capacity(const capacity_type& capacity_ctrl, InputIterator, InputIterator,
const std::input_iterator_tag&) {
return capacity_ctrl.capacity();
}


template <class ForwardIterator>
static size_type init_capacity(const capacity_type& capacity_ctrl, ForwardIterator first, ForwardIterator last,
const std::forward_iterator_tag&) {
BOOST_CB_ASSERT(std::distance(first, last) >= 0); 
return (std::max)(capacity_ctrl.min_capacity(),
(std::min)(capacity_ctrl.capacity(), static_cast<size_type>(std::distance(first, last))));
}


template <class IntegralType>
void insert(const iterator& pos, IntegralType n, IntegralType item, const true_type&) {
insert(pos, static_cast<size_type>(n), static_cast<value_type>(item));
}


template <class Iterator>
void insert(const iterator& pos, Iterator first, Iterator last, const false_type&) {
size_type index = pos - begin();
check_low_capacity(std::distance(first, last));
circular_buffer<T, Alloc>::insert(begin() + index, first, last);
}


template <class IntegralType>
void rinsert(const iterator& pos, IntegralType n, IntegralType item, const true_type&) {
rinsert(pos, static_cast<size_type>(n), static_cast<value_type>(item));
}


template <class Iterator>
void rinsert(const iterator& pos, Iterator first, Iterator last, const false_type&) {
size_type index = pos - begin();
check_low_capacity(std::distance(first, last));
circular_buffer<T, Alloc>::rinsert(begin() + index, first, last);
}
};


template <class T, class Alloc>
inline bool operator == (const circular_buffer_space_optimized<T, Alloc>& lhs,
const circular_buffer_space_optimized<T, Alloc>& rhs) {
return lhs.size() == rhs.size() &&
std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <class T, class Alloc>
inline bool operator < (const circular_buffer_space_optimized<T, Alloc>& lhs,
const circular_buffer_space_optimized<T, Alloc>& rhs) {
return std::lexicographical_compare(
lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

#if !defined(BOOST_NO_FUNCTION_TEMPLATE_ORDERING) || BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1310))

template <class T, class Alloc>
inline bool operator != (const circular_buffer_space_optimized<T, Alloc>& lhs,
const circular_buffer_space_optimized<T, Alloc>& rhs) {
return !(lhs == rhs);
}

template <class T, class Alloc>
inline bool operator > (const circular_buffer_space_optimized<T, Alloc>& lhs,
const circular_buffer_space_optimized<T, Alloc>& rhs) {
return rhs < lhs;
}

template <class T, class Alloc>
inline bool operator <= (const circular_buffer_space_optimized<T, Alloc>& lhs,
const circular_buffer_space_optimized<T, Alloc>& rhs) {
return !(rhs < lhs);
}

template <class T, class Alloc>
inline bool operator >= (const circular_buffer_space_optimized<T, Alloc>& lhs,
const circular_buffer_space_optimized<T, Alloc>& rhs) {
return !(lhs < rhs);
}

template <class T, class Alloc>
inline void swap(circular_buffer_space_optimized<T, Alloc>& lhs,
circular_buffer_space_optimized<T, Alloc>& rhs) BOOST_NOEXCEPT {
lhs.swap(rhs);
}

#endif 

} 

#endif 
