
#ifndef BOOST_CONTAINER_DETAIL_VARRAY_HPP
#define BOOST_CONTAINER_DETAIL_VARRAY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>

#include <boost/container/detail/addressof.hpp>
#include <boost/container/detail/algorithm.hpp> 
#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)
#include <boost/move/detail/fwd_macros.hpp>
#endif
#include <boost/container/detail/iterator.hpp>
#include <boost/container/detail/iterators.hpp>
#include <boost/container/detail/mpl.hpp>
#include <boost/container/detail/type_traits.hpp>
#include <boost/move/adl_move_swap.hpp> 


#include "varray_util.hpp"

#include <boost/assert.hpp>
#include <boost/config.hpp>

#include <boost/static_assert.hpp>

#ifndef BOOST_NO_EXCEPTIONS
#include <stdexcept>
#endif 




namespace boost { namespace container { namespace dtl {

template <typename Value, std::size_t Capacity, typename Strategy>
class varray;

namespace strategy {


template <typename Value>
struct def
{
typedef Value value_type;
typedef std::size_t size_type;
typedef std::ptrdiff_t difference_type;
typedef Value* pointer;
typedef const Value* const_pointer;
typedef Value& reference;
typedef const Value& const_reference;

static void allocate_failed()
{
BOOST_ASSERT_MSG(false, "size can't exceed the capacity");
}
};

template <typename Allocator>
struct allocator_adaptor
{
typedef typename Allocator::value_type value_type;
typedef typename Allocator::size_type size_type;
typedef typename Allocator::difference_type difference_type;
typedef typename Allocator::pointer pointer;
typedef typename Allocator::const_pointer const_pointer;
typedef typename Allocator::reference reference;
typedef typename Allocator::const_reference const_reference;

static void allocate_failed()
{
BOOST_ASSERT_MSG(false, "size can't exceed the capacity");
}
};

} 

struct varray_error_handler
{
template <typename V, std::size_t Capacity, typename S>
static void check_capacity(varray<V, Capacity, S> const&, std::size_t s)
{
if ( Capacity < s )
S::allocate_failed();
}

template <typename V, std::size_t C, typename S>
static void check_at(varray<V, C, S> const& v,
typename varray<V, C, S>::size_type i)
{
(void)v;
(void)i;
#ifndef BOOST_NO_EXCEPTIONS
if ( v.size() <= i )
throw std::out_of_range("index out of bounds");
#else 
BOOST_ASSERT_MSG(i < v.size(), "index out of bounds");
#endif 
}

template <typename V, std::size_t C, typename S>
static void check_operator_brackets(varray<V, C, S> const& v,
typename varray<V, C, S>::size_type i)
{
(void)v;
(void)i;
BOOST_ASSERT_MSG(i < v.size(), "index out of bounds");
}

template <typename V, std::size_t C, typename S>
static void check_empty(varray<V, C, S> const& v)
{
(void)v;
BOOST_ASSERT_MSG(0 < v.size(), "the container is empty");
}

template <typename V, std::size_t C, typename S>
static void check_iterator_end_neq(varray<V, C, S> const& v,
typename varray<V, C, S>::const_iterator position)
{
(void)v;
(void)position;
BOOST_ASSERT_MSG(v.begin() <= position && position < v.end(), "iterator out of bounds");
}

template <typename V, std::size_t C, typename S>
static void check_iterator_end_eq(varray<V, C, S> const& v,
typename varray<V, C, S>::const_iterator position)
{
(void)v;
(void)position;
BOOST_ASSERT_MSG(v.begin() <= position && position <= v.end(), "iterator out of bounds");
}
};

template <typename Value, std::size_t Capacity, typename Strategy>
struct varray_traits
{
typedef typename Strategy::value_type value_type;
typedef typename Strategy::size_type size_type;
typedef typename Strategy::difference_type difference_type;
typedef typename Strategy::pointer pointer;
typedef typename Strategy::const_pointer const_pointer;
typedef typename Strategy::reference reference;
typedef typename Strategy::const_reference const_reference;

typedef varray_error_handler error_handler;

typedef false_type use_memop_in_swap_and_move;
typedef false_type use_optimized_swap;
typedef false_type disable_trivial_init;
};


template <typename Value, std::size_t Capacity, typename Strategy = strategy::def<Value> >
class varray
{
typedef dtl::varray_traits<
Value, Capacity, Strategy
> vt;

typedef typename vt::error_handler errh;
typedef typename aligned_storage<
sizeof(Value[Capacity]),
boost::container::dtl::alignment_of<Value[Capacity]>::value
>::type aligned_storage_type;

template <typename V, std::size_t C, typename S>
friend class varray;

BOOST_COPYABLE_AND_MOVABLE(varray)

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
public:
template <std::size_t C, typename S>
varray & operator=(varray<Value, C, S> & sv)
{
typedef varray<Value, C, S> other;
this->operator=(static_cast<const ::boost::rv<other> &>(const_cast<const other &>(sv)));
return *this;
}
#endif

public:
typedef typename vt::value_type value_type;
typedef typename vt::size_type size_type;
typedef typename vt::difference_type difference_type;
typedef typename vt::pointer pointer;
typedef typename vt::const_pointer const_pointer;
typedef typename vt::reference reference;
typedef typename vt::const_reference const_reference;

typedef pointer iterator;
typedef const_pointer const_iterator;
typedef boost::container::reverse_iterator<iterator> reverse_iterator;
typedef boost::container::reverse_iterator<const_iterator> const_reverse_iterator;

typedef Strategy strategy_type;

varray()
: m_size(0)
{}

explicit varray(size_type count)
: m_size(0)
{
this->resize(count);                                                        
}

varray(size_type count, value_type const& value)
: m_size(0)
{
this->resize(count, value);                                                 
}

template <typename Iterator>
varray(Iterator first, Iterator last)
: m_size(0)
{
this->assign(first, last);                                                    
}

varray(varray const& other)
: m_size(other.size())
{
namespace sv = varray_detail;
sv::uninitialized_copy(other.begin(), other.end(), this->begin());          
}

template <std::size_t C, typename S>
varray(varray<value_type, C, S> const& other)
: m_size(other.size())
{
errh::check_capacity(*this, other.size());                                  

namespace sv = varray_detail;
sv::uninitialized_copy(other.begin(), other.end(), this->begin());          
}

varray & operator=(BOOST_COPY_ASSIGN_REF(varray) other)
{
this->assign(other.begin(), other.end());                                     

return *this;
}

template <std::size_t C, typename S>
#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
varray & operator=(::boost::rv< varray<value_type, C, S> > const& other)
#else
varray & operator=(varray<value_type, C, S> const& other)
#endif
{
this->assign(other.begin(), other.end());                                     

return *this;
}

varray(BOOST_RV_REF(varray) other)
{
typedef typename
vt::use_memop_in_swap_and_move use_memop_in_swap_and_move;

this->move_ctor_dispatch(other, use_memop_in_swap_and_move());
}

template <std::size_t C, typename S>
varray(BOOST_RV_REF_3_TEMPL_ARGS(varray, value_type, C, S) other)
: m_size(other.m_size)
{
errh::check_capacity(*this, other.size());                                  

typedef typename
vt::use_memop_in_swap_and_move use_memop_in_swap_and_move;

this->move_ctor_dispatch(other, use_memop_in_swap_and_move());
}

varray & operator=(BOOST_RV_REF(varray) other)
{
if ( &other == this )
return *this;

typedef typename
vt::use_memop_in_swap_and_move use_memop_in_swap_and_move;

this->move_assign_dispatch(other, use_memop_in_swap_and_move());

return *this;
}

template <std::size_t C, typename S>
varray & operator=(BOOST_RV_REF_3_TEMPL_ARGS(varray, value_type, C, S) other)
{
errh::check_capacity(*this, other.size());                                  

typedef typename
vt::use_memop_in_swap_and_move use_memop_in_swap_and_move;

this->move_assign_dispatch(other, use_memop_in_swap_and_move());

return *this;
}

~varray()
{
namespace sv = varray_detail;
sv::destroy(this->begin(), this->end());
}

void swap(varray & other)
{
typedef typename
vt::use_optimized_swap use_optimized_swap;

this->swap_dispatch(other, use_optimized_swap());
}

template <std::size_t C, typename S>
void swap(varray<value_type, C, S> & other)
{
errh::check_capacity(*this, other.size());
errh::check_capacity(other, this->size());

typedef typename
vt::use_optimized_swap use_optimized_swap;

this->swap_dispatch(other, use_optimized_swap());
}

void resize(size_type count)
{
namespace sv = varray_detail;
typedef typename vt::disable_trivial_init dti;

if ( count < m_size )
{
sv::destroy(this->begin() + count, this->end());
}
else
{
errh::check_capacity(*this, count);                                     

sv::uninitialized_fill(this->end(), this->begin() + count, dti()); 
}
m_size = count; 
}

void resize(size_type count, value_type const& value)
{
if ( count < m_size )
{
namespace sv = varray_detail;
sv::destroy(this->begin() + count, this->end());
}
else
{
errh::check_capacity(*this, count);                                     

std::uninitialized_fill(this->end(), this->begin() + count, value);     
}
m_size = count; 
}

void reserve(size_type count)
{
errh::check_capacity(*this, count);                                         
}

void push_back(value_type const& value)
{
typedef typename vt::disable_trivial_init dti;

errh::check_capacity(*this, m_size + 1);                                    

namespace sv = varray_detail;
sv::construct(dti(), this->end(), value);                                          
++m_size; 
}

void push_back(BOOST_RV_REF(value_type) value)
{
typedef typename vt::disable_trivial_init dti;

errh::check_capacity(*this, m_size + 1);                                    

namespace sv = varray_detail;
sv::construct(dti(), this->end(), ::boost::move(value));                                          
++m_size; 
}

void pop_back()
{
errh::check_empty(*this);

namespace sv = varray_detail;
sv::destroy(this->end() - 1);
--m_size; 
}

iterator insert(iterator position, value_type const& value)
{
return this->priv_insert(position, value);
}

iterator insert(iterator position, BOOST_RV_REF(value_type) value)
{
return this->priv_insert(position, boost::move(value));
}

iterator insert(iterator position, size_type count, value_type const& value)
{
errh::check_iterator_end_eq(*this, position);
errh::check_capacity(*this, m_size + count);                                

if ( position == this->end() )
{
std::uninitialized_fill(position, position + count, value);             
m_size += count; 
}
else
{
namespace sv = varray_detail;

difference_type to_move = boost::container::iterator_distance(position, this->end());


if ( count < static_cast<size_type>(to_move) )
{
sv::uninitialized_move(this->end() - count, this->end(), this->end());          
m_size += count; 
sv::move_backward(position, position + to_move - count, this->end() - count);   
std::fill_n(position, count, value);                                            
}
else
{
std::uninitialized_fill(this->end(), position + count, value);                  
m_size += count - to_move; 
sv::uninitialized_move(position, position + to_move, position + count);         
m_size += to_move; 
std::fill_n(position, to_move, value);                                          
}
}

return position;
}

template <typename Iterator>
iterator insert(iterator position, Iterator first, Iterator last)
{
this->insert_dispatch(position, first, last);
return position;
}

iterator erase(iterator position)
{
namespace sv = varray_detail;

errh::check_iterator_end_neq(*this, position);


sv::move(position + 1, this->end(), position);                              
sv::destroy(this->end() - 1);
--m_size;

return position;
}

iterator erase(iterator first, iterator last)
{
namespace sv = varray_detail;

errh::check_iterator_end_eq(*this, first);
errh::check_iterator_end_eq(*this, last);

difference_type n = boost::container::iterator_distance(first, last);


sv::move(last, this->end(), first);                                         
sv::destroy(this->end() - n, this->end());
m_size -= n;

return first;
}

template <typename Iterator>
void assign(Iterator first, Iterator last)
{
this->assign_dispatch(first, last);                            
}

void assign(size_type count, value_type const& value)
{
if ( count < m_size )
{
namespace sv = varray_detail;

std::fill_n(this->begin(), count, value);                               
sv::destroy(this->begin() + count, this->end());
}
else
{
errh::check_capacity(*this, count);                                     

std::fill_n(this->begin(), m_size, value);                              
std::uninitialized_fill(this->end(), this->begin() + count, value);     
}
m_size = count; 
}

#if !defined(BOOST_CONTAINER_VARRAY_DISABLE_EMPLACE)
#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
template<class ...Args>
void emplace_back(BOOST_FWD_REF(Args) ...args)
{
typedef typename vt::disable_trivial_init dti;

errh::check_capacity(*this, m_size + 1);                                    

namespace sv = varray_detail;
sv::construct(dti(), this->end(), ::boost::forward<Args>(args)...);                
++m_size; 
}

template<class ...Args>
iterator emplace(iterator position, BOOST_FWD_REF(Args) ...args)
{
typedef typename vt::disable_trivial_init dti;

namespace sv = varray_detail;

errh::check_iterator_end_eq(*this, position);
errh::check_capacity(*this, m_size + 1);                                    

if ( position == this->end() )
{
sv::construct(dti(), position, ::boost::forward<Args>(args)...);               
++m_size; 
}
else
{

value_type & r = *(this->end() - 1);
sv::construct(dti(), this->end(), boost::move(r));                             
++m_size; 
sv::move_backward(position, this->end() - 2, this->end() - 1);          

typename aligned_storage
<sizeof(value_type), alignment_of<value_type>::value>::type temp_storage;
value_type * val_p = static_cast<value_type*>(static_cast<void*>(&temp_storage));
sv::construct(dti(), val_p, ::boost::forward<Args>(args)...);                  
sv::scoped_destructor<value_type> d(val_p);
sv::assign(position, ::boost::move(*val_p));                            
}

return position;
}

#else 

#define BOOST_CONTAINER_VARRAY_EMPLACE_CODE(N) \
BOOST_MOVE_TMPL_LT##N BOOST_MOVE_CLASS##N BOOST_MOVE_GT##N \
void emplace_back(BOOST_MOVE_UREF##N)\
{\
typedef typename vt::disable_trivial_init dti;\
errh::check_capacity(*this, m_size + 1);\
\
namespace sv = varray_detail;\
sv::construct(dti(), this->end() BOOST_MOVE_I##N BOOST_MOVE_FWD##N ); \
++m_size; \
}\
\
BOOST_MOVE_TMPL_LT##N BOOST_MOVE_CLASS##N BOOST_MOVE_GT##N \
iterator emplace(iterator position BOOST_MOVE_I##N BOOST_MOVE_UREF##N)\
{\
typedef typename vt::disable_trivial_init dti;\
namespace sv = varray_detail;\
errh::check_iterator_end_eq(*this, position);\
errh::check_capacity(*this, m_size + 1); \
if ( position == this->end() ){\
sv::construct(dti(), position BOOST_MOVE_I##N BOOST_MOVE_FWD##N ); \
++m_size; \
}\
else{\
\
\
value_type & r = *(this->end() - 1);\
sv::construct(dti(), this->end(), boost::move(r));\
++m_size; \
sv::move_backward(position, this->end() - 2, this->end() - 1);\
typename aligned_storage\
<sizeof(value_type), alignment_of<value_type>::value>::type temp_storage;\
value_type * val_p = static_cast<value_type*>(static_cast<void*>(&temp_storage));\
sv::construct(dti(), val_p BOOST_MOVE_I##N BOOST_MOVE_FWD##N ); \
sv::scoped_destructor<value_type> d(val_p);\
sv::assign(position, ::boost::move(*val_p));\
}\
return position;\
}\
BOOST_MOVE_ITERATE_0TO9(BOOST_CONTAINER_VARRAY_EMPLACE_CODE)
#undef BOOST_CONTAINER_VARRAY_EMPLACE_CODE

#endif 
#endif 

void clear()
{
namespace sv = varray_detail;
sv::destroy(this->begin(), this->end());
m_size = 0; 
}

reference at(size_type i)
{
errh::check_at(*this, i);                                   
return *(this->begin() + i);
}

const_reference at(size_type i) const
{
errh::check_at(*this, i);                                   
return *(this->begin() + i);
}

reference operator[](size_type i)
{
errh::check_operator_brackets(*this, i);
return *(this->begin() + i);
}

const_reference operator[](size_type i) const
{
errh::check_operator_brackets(*this, i);
return *(this->begin() + i);
}

reference front()
{
errh::check_empty(*this);
return *(this->begin());
}

const_reference front() const
{
errh::check_empty(*this);
return *(this->begin());
}

reference back()
{
errh::check_empty(*this);
return *(this->end() - 1);
}

const_reference back() const
{
errh::check_empty(*this);
return *(this->end() - 1);
}

Value * data()
{
return (addressof)(*(this->ptr()));
}

const Value * data() const
{
return (addressof)(*(this->ptr()));
}


iterator begin() { return this->ptr(); }

const_iterator begin() const { return this->ptr(); }

const_iterator cbegin() const { return this->ptr(); }

iterator end() { return this->begin() + m_size; }

const_iterator end() const { return this->begin() + m_size; }

const_iterator cend() const { return this->cbegin() + m_size; }

reverse_iterator rbegin() { return reverse_iterator(this->end()); }

const_reverse_iterator rbegin() const { return reverse_iterator(this->end()); }

const_reverse_iterator crbegin() const { return reverse_iterator(this->end()); }

reverse_iterator rend() { return reverse_iterator(this->begin()); }

const_reverse_iterator rend() const { return reverse_iterator(this->begin()); }

const_reverse_iterator crend() const { return reverse_iterator(this->begin()); }

static size_type capacity() { return Capacity; }

static size_type max_size() { return Capacity; }

size_type size() const { return m_size; }

bool empty() const { return 0 == m_size; }

private:

template <std::size_t C, typename S>
void move_ctor_dispatch(varray<value_type, C, S> & other, true_type )
{
::memcpy(this->data(), other.data(), sizeof(Value) * other.m_size);
m_size = other.m_size;
}

template <std::size_t C, typename S>
void move_ctor_dispatch(varray<value_type, C, S> & other, false_type )
{
namespace sv = varray_detail;
sv::uninitialized_move_if_noexcept(other.begin(), other.end(), this->begin());                  
m_size = other.m_size;
}

template <std::size_t C, typename S>
void move_assign_dispatch(varray<value_type, C, S> & other, true_type )
{
this->clear();

::memcpy(this->data(), other.data(), sizeof(Value) * other.m_size);
boost::adl_move_swap(m_size, other.m_size);
}

template <std::size_t C, typename S>
void move_assign_dispatch(varray<value_type, C, S> & other, false_type )
{
namespace sv = varray_detail;
if ( m_size <= static_cast<size_type>(other.size()) )
{
sv::move_if_noexcept(other.begin(), other.begin() + m_size, this->begin());             
sv::uninitialized_move_if_noexcept(other.begin() + m_size, other.end(), this->end());   
}
else
{
sv::move_if_noexcept(other.begin(), other.end(), this->begin());                        
sv::destroy(this->begin() + other.size(), this->end());
}
m_size = other.size(); 
}

template <std::size_t C, typename S>
void swap_dispatch(varray<value_type, C, S> & other, true_type const& )
{
typedef typename
if_c<
Capacity < C,
aligned_storage_type,
typename varray<value_type, C, S>::aligned_storage_type
>::type
storage_type;

storage_type temp_storage;
value_type * temp_ptr = static_cast<value_type*>(static_cast<void*>(&temp_storage));

::memcpy(temp_ptr, this->data(), sizeof(Value) * this->size());
::memcpy(this->data(), other.data(), sizeof(Value) * other.size());
::memcpy(other.data(), temp_ptr, sizeof(Value) * this->size());

boost::adl_move_swap(m_size, other.m_size);
}

template <std::size_t C, typename S>
void swap_dispatch(varray<value_type, C, S> & other, false_type const& )
{
namespace sv = varray_detail;

typedef typename
vt::use_memop_in_swap_and_move use_memop_in_swap_and_move;

if ( this->size() < other.size() )
swap_dispatch_impl(this->begin(), this->end(), other.begin(), other.end(), use_memop_in_swap_and_move()); 
else
swap_dispatch_impl(other.begin(), other.end(), this->begin(), this->end(), use_memop_in_swap_and_move()); 
boost::adl_move_swap(m_size, other.m_size);
}

void swap_dispatch_impl(iterator first_sm, iterator last_sm, iterator first_la, iterator last_la, true_type const& )
{

namespace sv = varray_detail;
for (; first_sm != last_sm ; ++first_sm, ++first_la)
{
typename aligned_storage<
sizeof(value_type),
alignment_of<value_type>::value
>::type temp_storage;
value_type * temp_ptr = static_cast<value_type*>(static_cast<void*>(&temp_storage));
::memcpy(temp_ptr, (addressof)(*first_sm), sizeof(value_type));
::memcpy((addressof)(*first_sm), (addressof)(*first_la), sizeof(value_type));
::memcpy((addressof)(*first_la), temp_ptr, sizeof(value_type));
}

::memcpy(first_sm, first_la, sizeof(value_type) * boost::container::iterator_distance(first_la, last_la));
}

void swap_dispatch_impl(iterator first_sm, iterator last_sm, iterator first_la, iterator last_la, false_type const& )
{

namespace sv = varray_detail;
for (; first_sm != last_sm ; ++first_sm, ++first_la)
{
value_type temp(boost::move(*first_sm));                                
*first_sm = boost::move(*first_la);                                     
*first_la = boost::move(temp);                                          
}
sv::uninitialized_move(first_la, last_la, first_sm);                        
sv::destroy(first_la, last_la);
}


template <typename V>
iterator priv_insert(iterator position, V & value)
{
typedef typename vt::disable_trivial_init dti;
namespace sv = varray_detail;

errh::check_iterator_end_eq(*this, position);
errh::check_capacity(*this, m_size + 1);                                    

if ( position == this->end() )
{
sv::construct(dti(), position, value);                                         
++m_size; 
}
else
{

value_type & r = *(this->end() - 1);
sv::construct(dti(), this->end(), boost::move(r));                             
++m_size; 
sv::move_backward(position, this->end() - 2, this->end() - 1);          
sv::assign(position, value);                                            
}

return position;
}


template <typename Iterator>
typename iterator_enable_if_tag<Iterator, std::random_access_iterator_tag>::type
insert_dispatch(iterator position, Iterator first, Iterator last)
{
errh::check_iterator_end_eq(*this, position);

size_type count = boost::container::iterator_distance(first, last);

errh::check_capacity(*this, m_size + count);                                             

if ( position == this->end() )
{
namespace sv = varray_detail;

sv::uninitialized_copy(first, last, position);                                      
m_size += count; 
}
else
{
this->insert_in_the_middle(position, first, last, count);                           
}
}

template <typename Iterator, typename Category>
typename iterator_disable_if_tag<Iterator, std::random_access_iterator_tag>::type
insert_dispatch(iterator position, Iterator first, Iterator last)
{
errh::check_iterator_end_eq(*this, position);

if ( position == this->end() )
{
namespace sv = varray_detail;

std::ptrdiff_t d = boost::container::iterator_distance(position, this->begin() + Capacity);
std::size_t count = sv::uninitialized_copy_s(first, last, position, d);                     

errh::check_capacity(*this, count <= static_cast<std::size_t>(d) ? m_size + count : Capacity + 1);  

m_size += count;
}
else
{
size_type count = boost::container::iterator_distance(first, last);

errh::check_capacity(*this, m_size + count);                                                

this->insert_in_the_middle(position, first, last, count);                                   
}
}

template <typename Iterator>
void insert_in_the_middle(iterator position, Iterator first, Iterator last, difference_type count)
{
namespace sv = varray_detail;

difference_type to_move = boost::container::iterator_distance(position, this->end());


if ( count < to_move )
{
sv::uninitialized_move(this->end() - count, this->end(), this->end());              
m_size += count; 
sv::move_backward(position, position + to_move - count, this->end() - count);       
sv::copy(first, last, position);                                                    
}
else
{
Iterator middle_iter = first;
boost::container::iterator_advance(middle_iter, to_move);

sv::uninitialized_copy(middle_iter, last, this->end());                             
m_size += count - to_move; 
sv::uninitialized_move(position, position + to_move, position + count);             
m_size += to_move; 
sv::copy(first, middle_iter, position);                                             
}
}


template <typename Iterator>
typename iterator_enable_if_tag<Iterator, std::random_access_iterator_tag>::type
assign_dispatch(Iterator first, Iterator last)
{
namespace sv = varray_detail;

size_type s = boost::container::iterator_distance(first, last);

errh::check_capacity(*this, s);                                     

if ( m_size <= static_cast<size_type>(s) )
{
sv::copy(first, first + m_size, this->begin());                 
sv::uninitialized_copy(first + m_size, last, this->end());      
}
else
{
sv::copy(first, last, this->begin());                           
sv::destroy(this->begin() + s, this->end());
}
m_size = s; 
}

template <typename Iterator, typename Category>
typename iterator_disable_if_tag<Iterator, std::random_access_iterator_tag>::type
assign_dispatch(Iterator first, Iterator last)
{
namespace sv = varray_detail;

size_type s = 0;
iterator it = this->begin();

for ( ; it != this->end() && first != last ; ++it, ++first, ++s )
*it = *first;                                                                                   

sv::destroy(it, this->end());

std::ptrdiff_t d = boost::container::iterator_distance(it, this->begin() + Capacity);
std::size_t count = sv::uninitialized_copy_s(first, last, it, d);                                   
s += count;

errh::check_capacity(*this, count <= static_cast<std::size_t>(d) ? s : Capacity + 1);               

m_size = s; 
}

pointer ptr()
{
return pointer(static_cast<Value*>(static_cast<void*>(&m_storage)));
}

const_pointer ptr() const
{
return pointer(static_cast<const Value*>(static_cast<const void*>(&m_storage)));
}

size_type m_size;
aligned_storage_type m_storage;
};

#if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

template<typename Value, typename Strategy>
class varray<Value, 0, Strategy>
{
typedef varray_traits<
Value, 0, Strategy
> vt;

typedef typename vt::size_type stored_size_type;
typedef typename vt::error_handler errh;

public:
typedef typename vt::value_type value_type;
typedef stored_size_type size_type;
typedef typename vt::difference_type difference_type;
typedef typename vt::pointer pointer;
typedef typename vt::const_pointer const_pointer;
typedef typename vt::reference reference;
typedef typename vt::const_reference const_reference;

typedef pointer iterator;
typedef const_pointer const_iterator;
typedef boost::container::reverse_iterator<iterator> reverse_iterator;
typedef boost::container::reverse_iterator<const_iterator> const_reverse_iterator;

varray() {}

explicit varray(size_type count)
{
errh::check_capacity(*this, count);                                         
}

varray(size_type count, value_type const&)
{
errh::check_capacity(*this, count);                                         
}

varray(varray const& other)
{
errh::check_capacity(*this, other.size());
}

template <size_t C, typename S>
varray(varray<value_type, C, S> const& other)
{
errh::check_capacity(*this, other.size());                                  
}

template <typename Iterator>
varray(Iterator first, Iterator last)
{
errh::check_capacity(*this, boost::container::iterator_distance(first, last));                    
}

varray & operator=(varray const& other)
{
errh::check_capacity(*this, other.size());
return *this;
}

template <size_t C, typename S>
varray & operator=(varray<value_type, C, S> const& other)
{
errh::check_capacity(*this, other.size());                                  
return *this;
}

~varray() {}

void resize(size_type count)
{
errh::check_capacity(*this, count);                                         
}

void resize(size_type count, value_type const&)
{
errh::check_capacity(*this, count);                                         
}


void reserve(size_type count)
{
errh::check_capacity(*this, count);                                         
}

void push_back(value_type const&)
{
errh::check_capacity(*this, 1);                                             
}

void pop_back()
{
errh::check_empty(*this);
}

void insert(iterator position, value_type const&)
{
errh::check_iterator_end_eq(*this, position);
errh::check_capacity(*this, 1);                                             
}

void insert(iterator position, size_type count, value_type const&)
{
errh::check_iterator_end_eq(*this, position);
errh::check_capacity(*this, count);                                         
}

template <typename Iterator>
void insert(iterator, Iterator first, Iterator last)
{
errh::check_capacity(*this, boost::container::iterator_distance(first, last));                    
}

void erase(iterator position)
{
errh::check_iterator_end_neq(*this, position);
}

void erase(iterator first, iterator last)
{
errh::check_iterator_end_eq(*this, first);
errh::check_iterator_end_eq(*this, last);

}

template <typename Iterator>
void assign(Iterator first, Iterator last)
{
errh::check_capacity(*this, boost::container::iterator_distance(first, last));                    
}

void assign(size_type count, value_type const&)
{
errh::check_capacity(*this, count);                                     
}

void clear() {}

reference at(size_type i)
{
errh::check_at(*this, i);                                   
return *(this->begin() + i);
}

const_reference at(size_type i) const
{
errh::check_at(*this, i);                                   
return *(this->begin() + i);
}

reference operator[](size_type i)
{
errh::check_operator_brackets(*this, i);
return *(this->begin() + i);
}

const_reference operator[](size_type i) const
{
errh::check_operator_brackets(*this, i);
return *(this->begin() + i);
}

reference front()
{
errh::check_empty(*this);
return *(this->begin());
}

const_reference front() const
{
errh::check_empty(*this);
return *(this->begin());
}

reference back()
{
errh::check_empty(*this);
return *(this->end() - 1);
}

const_reference back() const
{
errh::check_empty(*this);
return *(this->end() - 1);
}

Value * data() { return (addressof)(*(this->ptr())); }
const Value * data() const { return (addressof)(*(this->ptr())); }

iterator begin() { return this->ptr(); }
const_iterator begin() const { return this->ptr(); }
const_iterator cbegin() const { return this->ptr(); }
iterator end() { return this->begin(); }
const_iterator end() const { return this->begin(); }
const_iterator cend() const { return this->cbegin(); }
reverse_iterator rbegin() { return reverse_iterator(this->end()); }
const_reverse_iterator rbegin() const { return reverse_iterator(this->end()); }
const_reverse_iterator crbegin() const { return reverse_iterator(this->end()); }
reverse_iterator rend() { return reverse_iterator(this->begin()); }
const_reverse_iterator rend() const { return reverse_iterator(this->begin()); }
const_reverse_iterator crend() const { return reverse_iterator(this->begin()); }

size_type capacity() const { return 0; }
size_type max_size() const { return 0; }
size_type size() const { return 0; }
bool empty() const { return true; }

private:

pointer ptr()
{
return pointer(reinterpret_cast<Value*>(this));
}

const_pointer ptr() const
{
return const_pointer(reinterpret_cast<const Value*>(this));
}
};

#endif 

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
bool operator== (varray<V, C1, S1> const& x, varray<V, C2, S2> const& y)
{
return x.size() == y.size() && ::boost::container::algo_equal(x.begin(), x.end(), y.begin());
}

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
bool operator!= (varray<V, C1, S1> const& x, varray<V, C2, S2> const& y)
{
return !(x==y);
}

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
bool operator< (varray<V, C1, S1> const& x, varray<V, C2, S2> const& y)
{
return ::boost::container::algo_lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
bool operator> (varray<V, C1, S1> const& x, varray<V, C2, S2> const& y)
{
return y<x;
}

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
bool operator<= (varray<V, C1, S1> const& x, varray<V, C2, S2> const& y)
{
return !(y<x);
}

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
bool operator>= (varray<V, C1, S1> const& x, varray<V, C2, S2> const& y)
{
return !(x<y);
}

template<typename V, std::size_t C1, typename S1, std::size_t C2, typename S2>
inline void swap(varray<V, C1, S1> & x, varray<V, C2, S2> & y)
{
x.swap(y);
}

}}} 

#include <boost/container/detail/config_end.hpp>

#endif 
