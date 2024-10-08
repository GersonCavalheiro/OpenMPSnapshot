


#ifndef tbb_test_harness_allocator_H
#define tbb_test_harness_allocator_H

#include "harness_defs.h"

#if __linux__ || __APPLE__ || __sun
#include <unistd.h>
#elif _WIN32
#include "tbb/machine/windows_api.h"
#endif 
#include <memory>
#include <new>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include __TBB_STD_SWAP_HEADER

#include "tbb/atomic.h"
#include "tbb/tbb_allocator.h"

#if __SUNPRO_CC
using std::printf;
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#if defined(_Wp64)
#pragma warning (disable: 4267)
#endif
#if _MSC_VER <= 1600
#pragma warning (disable: 4355)
#endif
#if _MSC_VER <= 1800
#pragma warning (disable: 4512)
#endif
#endif

#if TBB_INTERFACE_VERSION >= 7005
namespace Harness {
#if __TBB_ALLOCATOR_TRAITS_PRESENT
using std::true_type;
using std::false_type;
#else
using tbb::internal::true_type;
using tbb::internal::false_type;
#endif 
}
#endif

template<typename counter_type = size_t>
struct arena_data {
char * const my_buffer;
size_t const my_size; 
counter_type my_allocated; 

template<typename T>
arena_data(T * a_buffer, size_t a_size) __TBB_NOEXCEPT(true)
:   my_buffer(reinterpret_cast<char*>(a_buffer))
,   my_size(a_size * sizeof(T))
{
my_allocated =0;
}
private:
void operator=( const arena_data& ); 
};

template<typename T, typename pocma = Harness::false_type, typename counter_type = size_t>
struct arena {
typedef arena_data<counter_type> arena_data_t;
private:
arena_data_t * my_data;
public:
typedef T value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef size_t size_type;
typedef ptrdiff_t difference_type;
template<typename U> struct rebind {
typedef arena<U, pocma, counter_type> other;
};

typedef pocma propagate_on_container_move_assignment;

arena(arena_data_t & data) __TBB_NOEXCEPT(true) : my_data(&data) {}

template<typename U1, typename U2, typename U3>
friend struct arena;

template<typename U1, typename U2 >
arena(arena<U1, U2, counter_type> const& other) __TBB_NOEXCEPT(true) : my_data(other.my_data) {}

friend void swap(arena & lhs ,arena & rhs){
std::swap(lhs.my_data, rhs.my_data);
}

pointer address(reference x) const {return &x;}
const_pointer address(const_reference x) const {return &x;}

pointer allocate( size_type n, const void* =0) {
size_t new_size = (my_data->my_allocated += n*sizeof(T));
ASSERT(my_data->my_allocated <= my_data->my_size,"trying to allocate more than was reserved");
char* result =  &(my_data->my_buffer[new_size - n*sizeof(T)]);
return reinterpret_cast<pointer>(result);
}

void deallocate( pointer p_arg, size_type n) {
char* p = reinterpret_cast<char*>(p_arg);
ASSERT(p >=my_data->my_buffer && p <= my_data->my_buffer + my_data->my_size, "trying to deallocate pointer not from arena ?");
ASSERT(p + n*sizeof(T) <= my_data->my_buffer + my_data->my_size, "trying to deallocate incorrect number of items?");
tbb::internal::suppress_unused_warning(p, n);
}

size_type max_size() const throw() {
return my_data->my_size / sizeof(T);
}

#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
template<typename U, typename... Args>
void construct(U *p, Args&&... args)
{ ::new((void *)p) U(std::forward<Args>(args)...); }
#else 
#if __TBB_CPP11_RVALUE_REF_PRESENT
void construct( pointer p, value_type&& value ) {::new((void*)(p)) value_type(std::move(value));}
#endif
void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif 

void destroy( pointer p ) {
p->~value_type();
tbb::internal::suppress_unused_warning(p);
}

friend bool operator==(arena const& lhs, arena const& rhs){
return lhs.my_data == rhs.my_data;
}

friend bool operator!=(arena const& lhs, arena const& rhs){
return !(lhs== rhs);
}
};

template <typename count_t = tbb::atomic<size_t> >
struct allocator_counters {
count_t items_allocated;
count_t items_freed;
count_t allocations;
count_t frees;

friend bool operator==(allocator_counters const & lhs, allocator_counters const & rhs){
return     lhs.items_allocated == rhs.items_allocated
&& lhs.items_freed == rhs.items_freed
&& lhs.allocations == rhs.allocations
&& lhs.frees == rhs.frees
;
}
};

template <typename base_alloc_t, typename count_t = tbb::atomic<size_t> >
class static_counting_allocator : public base_alloc_t
{
public:
typedef typename base_alloc_t::pointer pointer;
typedef typename base_alloc_t::const_pointer const_pointer;
typedef typename base_alloc_t::reference reference;
typedef typename base_alloc_t::const_reference const_reference;
typedef typename base_alloc_t::value_type value_type;
typedef typename base_alloc_t::size_type size_type;
typedef typename base_alloc_t::difference_type difference_type;
template<typename U> struct rebind {
typedef static_counting_allocator<typename base_alloc_t::template rebind<U>::other,count_t> other;
};

typedef allocator_counters<count_t> counters_t;

static size_t max_items;
static count_t items_allocated;
static count_t items_freed;
static count_t allocations;
static count_t frees;
static bool verbose, throwing;

static_counting_allocator() throw() { }

static_counting_allocator(const base_alloc_t& src) throw()
: base_alloc_t(src) { }

static_counting_allocator(const static_counting_allocator& src) throw()
: base_alloc_t(src) { }

template<typename U, typename C>
static_counting_allocator(const static_counting_allocator<U, C>& src) throw()
: base_alloc_t(src) { }

pointer allocate(const size_type n)
{
if(verbose) printf("\t+%d|", int(n));
if(max_items && items_allocated + n >= max_items) {
if(verbose) printf("items limit hits!");
if(throwing)
__TBB_THROW( std::bad_alloc() );
return NULL;
}
pointer p = base_alloc_t::allocate(n, pointer(0));
allocations++;
items_allocated += n;
return p;
}

pointer allocate(const size_type n, const void * const)
{   return allocate(n); }

void deallocate(const pointer ptr, const size_type n)
{
if(verbose) printf("\t-%d|", int(n));
frees++;
items_freed += n;
base_alloc_t::deallocate(ptr, n);
}

static counters_t counters(){
counters_t c = {items_allocated, items_freed, allocations, frees} ;
return c;
}

static void init_counters(bool v = false) {
verbose = v;
if(verbose) printf("\n------------------------------------------- Allocations:\n");
items_allocated = 0;
items_freed = 0;
allocations = 0;
frees = 0;
max_items = 0;
}

static void set_limits(size_type max = 0, bool do_throw = true) {
max_items = max;
throwing = do_throw;
}
};

template <typename base_alloc_t, typename count_t>
size_t static_counting_allocator<base_alloc_t, count_t>::max_items;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::items_allocated;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::items_freed;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::allocations;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::frees;
template <typename base_alloc_t, typename count_t>
bool static_counting_allocator<base_alloc_t, count_t>::verbose;
template <typename base_alloc_t, typename count_t>
bool static_counting_allocator<base_alloc_t, count_t>::throwing;


template <typename tag, typename count_t = tbb::atomic<size_t> >
class static_shared_counting_allocator_base
{
public:
typedef allocator_counters<count_t> counters_t;

static size_t max_items;
static count_t items_allocated;
static count_t items_freed;
static count_t allocations;
static count_t frees;
static bool verbose, throwing;

static counters_t counters(){
counters_t c = {items_allocated, items_freed, allocations, frees} ;
return c;
}

static void init_counters(bool v = false) {
verbose = v;
if(verbose) printf("\n------------------------------------------- Allocations:\n");
items_allocated = 0;
items_freed = 0;
allocations = 0;
frees = 0;
max_items = 0;
}

static void set_limits(size_t max = 0, bool do_throw = true) {
max_items = max;
throwing = do_throw;
}
};

template <typename tag, typename count_t>
size_t static_shared_counting_allocator_base<tag, count_t>::max_items;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::items_allocated;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::items_freed;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::allocations;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::frees;

template <typename tag, typename count_t>
bool static_shared_counting_allocator_base<tag, count_t>::verbose;

template <typename tag, typename count_t>
bool static_shared_counting_allocator_base<tag, count_t>::throwing;

template <typename tag, typename base_alloc_t, typename count_t = tbb::atomic<size_t> >
class static_shared_counting_allocator : public static_shared_counting_allocator_base<tag, count_t>, public base_alloc_t
{
typedef static_shared_counting_allocator_base<tag, count_t> base_t;
public:
typedef typename base_alloc_t::pointer pointer;
typedef typename base_alloc_t::const_pointer const_pointer;
typedef typename base_alloc_t::reference reference;
typedef typename base_alloc_t::const_reference const_reference;
typedef typename base_alloc_t::value_type value_type;
typedef typename base_alloc_t::size_type size_type;
typedef typename base_alloc_t::difference_type difference_type;
template<typename U> struct rebind {
typedef static_shared_counting_allocator<tag, typename base_alloc_t::template rebind<U>::other, count_t> other;
};

static_shared_counting_allocator() throw() { }

static_shared_counting_allocator(const base_alloc_t& src) throw()
: base_alloc_t(src) { }

static_shared_counting_allocator(const static_shared_counting_allocator& src) throw()
: base_alloc_t(src) { }

template<typename U, typename C>
static_shared_counting_allocator(const static_shared_counting_allocator<tag, U, C>& src) throw()
: base_alloc_t(src) { }

pointer allocate(const size_type n)
{
if(base_t::verbose) printf("\t+%d|", int(n));
if(base_t::max_items && base_t::items_allocated + n >= base_t::max_items) {
if(base_t::verbose) printf("items limit hits!");
if(base_t::throwing)
__TBB_THROW( std::bad_alloc() );
return NULL;
}
base_t::allocations++;
base_t::items_allocated += n;
return base_alloc_t::allocate(n, pointer(0));
}

pointer allocate(const size_type n, const void * const)
{   return allocate(n); }

void deallocate(const pointer ptr, const size_type n)
{
if(base_t::verbose) printf("\t-%d|", int(n));
base_t::frees++;
base_t::items_freed += n;
base_alloc_t::deallocate(ptr, n);
}
};

template <typename base_alloc_t, typename count_t = tbb::atomic<size_t> >
class local_counting_allocator : public base_alloc_t
{
public:
typedef typename base_alloc_t::pointer pointer;
typedef typename base_alloc_t::const_pointer const_pointer;
typedef typename base_alloc_t::reference reference;
typedef typename base_alloc_t::const_reference const_reference;
typedef typename base_alloc_t::value_type value_type;
typedef typename base_alloc_t::size_type size_type;
typedef typename base_alloc_t::difference_type difference_type;
template<typename U> struct rebind {
typedef local_counting_allocator<typename base_alloc_t::template rebind<U>::other,count_t> other;
};

count_t items_allocated;
count_t items_freed;
count_t allocations;
count_t frees;
size_t max_items;

void set_counters(const count_t & a_items_allocated, const count_t & a_items_freed, const count_t & a_allocations, const count_t & a_frees, const count_t & a_max_items){
items_allocated = a_items_allocated;
items_freed = a_items_freed;
allocations = a_allocations;
frees = a_frees;
max_items = a_max_items;
}

template< typename allocator_t>
void set_counters(const allocator_t & a){
this->set_counters(a.items_allocated, a.items_freed, a.allocations, a.frees, a.max_items);
}

void clear_counters(){
count_t zero;
zero = 0;
this->set_counters(zero,zero,zero,zero,zero);
}

local_counting_allocator() throw() {
this->clear_counters();
}

local_counting_allocator(const local_counting_allocator &a) throw()
: base_alloc_t(a)
, items_allocated(a.items_allocated)
, items_freed(a.items_freed)
, allocations(a.allocations)
, frees(a.frees)
, max_items(a.max_items)
{ }

template<typename U, typename C>
local_counting_allocator(const static_counting_allocator<U,C> & a) throw() {
this->set_counters(a);
}

template<typename U, typename C>
local_counting_allocator(const local_counting_allocator<U,C> &a) throw()
: items_allocated(a.items_allocated)
, items_freed(a.items_freed)
, allocations(a.allocations)
, frees(a.frees)
, max_items(a.max_items)
{ }

bool operator==(const local_counting_allocator &a) const
{ return static_cast<const base_alloc_t&>(a) == *this; }

pointer allocate(const size_type n)
{
if(max_items && items_allocated + n >= max_items)
__TBB_THROW( std::bad_alloc() );
pointer p = base_alloc_t::allocate(n, pointer(0));
++allocations;
items_allocated += n;
return p;
}

pointer allocate(const size_type n, const void * const)
{ return allocate(n); }

void deallocate(const pointer ptr, const size_type n)
{
++frees;
items_freed += n;
base_alloc_t::deallocate(ptr, n);
}

void set_limits(size_type max = 0) {
max_items = max;
}
};

template <typename T, template<typename X> class Allocator = std::allocator>
class debug_allocator : public Allocator<T>
{
public:
typedef Allocator<T> base_allocator_type;
typedef typename base_allocator_type::value_type value_type;
typedef typename base_allocator_type::pointer pointer;
typedef typename base_allocator_type::const_pointer const_pointer;
typedef typename base_allocator_type::reference reference;
typedef typename base_allocator_type::const_reference const_reference;
typedef typename base_allocator_type::size_type size_type;
typedef typename base_allocator_type::difference_type difference_type;
template<typename U> struct rebind {
typedef debug_allocator<U, Allocator> other;
};

debug_allocator() throw() { }
debug_allocator(const debug_allocator &a) throw() : base_allocator_type( a ) { }
template<typename U>
debug_allocator(const debug_allocator<U> &a) throw() : base_allocator_type( Allocator<U>( a ) ) { }

pointer allocate(const size_type n, const void *hint = 0 ) {
pointer ptr = base_allocator_type::allocate( n, hint );
std::memset( (void*)ptr, 0xE3E3E3E3, n * sizeof(value_type) );
return ptr;
}
};


template<template<typename T> class Allocator>
class debug_allocator<void, Allocator> : public Allocator<void> {
public:
typedef Allocator<void> base_allocator_type;
typedef typename base_allocator_type::value_type value_type;
typedef typename base_allocator_type::pointer pointer;
typedef typename base_allocator_type::const_pointer const_pointer;
template<typename U> struct rebind {
typedef debug_allocator<U, Allocator> other;
};
};

template<typename T1, template<typename X1> class B1, typename T2, template<typename X2> class B2>
inline bool operator==( const debug_allocator<T1,B1> &a, const debug_allocator<T2,B2> &b) {
return static_cast< B1<T1> >(a) == static_cast< B2<T2> >(b);
}
template<typename T1, template<typename X1> class B1, typename T2, template<typename X2> class B2>
inline bool operator!=( const debug_allocator<T1,B1> &a, const debug_allocator<T2,B2> &b) {
return static_cast< B1<T1> >(a) != static_cast< B2<T2> >(b);
}

template <typename T, typename pocma = Harness::false_type, template<typename X> class Allocator = std::allocator>
class stateful_allocator : public Allocator<T>
{
void* unique_pointer;

template<typename T1, typename pocma1, template<typename X1> class Allocator1>
friend class  stateful_allocator;
public:
typedef Allocator<T> base_allocator_type;
typedef typename base_allocator_type::value_type value_type;
typedef typename base_allocator_type::pointer pointer;
typedef typename base_allocator_type::const_pointer const_pointer;
typedef typename base_allocator_type::reference reference;
typedef typename base_allocator_type::const_reference const_reference;
typedef typename base_allocator_type::size_type size_type;
typedef typename base_allocator_type::difference_type difference_type;
template<typename U> struct rebind {
typedef stateful_allocator<U, pocma, Allocator> other;
};
typedef pocma propagate_on_container_move_assignment;

stateful_allocator() throw() : unique_pointer(this) { }

template<typename U>
stateful_allocator(const stateful_allocator<U, pocma> &a) throw() : base_allocator_type( Allocator<U>( a ) ),  unique_pointer(a.uniqe_pointer) { }

friend bool operator==(stateful_allocator const& lhs, stateful_allocator const& rhs){
return lhs.unique_pointer == rhs.unique_pointer;
}

friend bool operator!=(stateful_allocator const& rhs, stateful_allocator const& lhs){
return !(lhs == rhs);
}

};

template <typename T>
class pmr_stateful_allocator
{
private:
pmr_stateful_allocator& operator=(const pmr_stateful_allocator&); 
public:
typedef T value_type;
typedef Harness::false_type propagate_on_container_move_assignment;
typedef Harness::false_type propagate_on_container_copy_assignment;
typedef Harness::false_type propagate_on_container_swap;

#if !__TBB_ALLOCATOR_TRAITS_PRESENT
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef size_t size_type;
typedef ptrdiff_t difference_type;
template<class U> struct rebind {
typedef pmr_stateful_allocator<U> other;
};
#endif

pmr_stateful_allocator() throw() : unique_pointer(this) {}

pmr_stateful_allocator(const pmr_stateful_allocator &a) : unique_pointer(a.unique_pointer) {}

template<typename U>
pmr_stateful_allocator(const pmr_stateful_allocator<U> &a) throw() : unique_pointer(a.unique_pointer) {}

value_type* allocate( size_t n, const void*  = 0 ) {
return static_cast<value_type*>( malloc( n * sizeof(value_type) ) );
}

void deallocate( value_type* p, size_t ) {
free( p );
}

#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
template<typename U, typename... Args>
void construct(U *p, Args&&... args)
{
::new((void *)p) U(std::forward<Args>(args)...);
}
#else 
#if __TBB_CPP11_RVALUE_REF_PRESENT
void construct(value_type* p, value_type&& value) { ::new((void*)(p)) value_type(std::move(value)); }
#endif
void construct(value_type* p, const value_type& value) { ::new((void*)(p)) value_type(value); }
#endif 

void destroy(value_type* p) {
p->~value_type();
tbb::internal::suppress_unused_warning(p);
}

friend bool operator==(pmr_stateful_allocator const& lhs, pmr_stateful_allocator const& rhs){
return lhs.unique_pointer == rhs.unique_pointer;
}

friend bool operator!=(pmr_stateful_allocator const& rhs, pmr_stateful_allocator const& lhs){
return !(lhs == rhs);
}

void* unique_pointer;
};

#if __TBB_ALLOCATOR_TRAITS_PRESENT
#include "tbb/internal/_allocator_traits.h" 

template <typename Allocator, typename POCMA = tbb::internal::traits_false_type,
typename POCCA = tbb::internal::traits_false_type, typename POCS = tbb::internal::traits_false_type>
struct propagating_allocator : Allocator {
typedef POCMA propagate_on_container_move_assignment;
typedef POCCA propagate_on_container_copy_assignment;
typedef POCS propagate_on_container_swap;
bool* propagated_on_copy_assignment;
bool* propagated_on_move_assignment;
bool* propagated_on_swap;
bool* selected_on_copy_construction;

template <typename U>
struct rebind {
typedef propagating_allocator<typename tbb::internal::allocator_rebind<Allocator, U>::type,
POCMA, POCCA, POCS> other;
};

propagating_allocator() : propagated_on_copy_assignment(NULL),
propagated_on_move_assignment(NULL),
propagated_on_swap(NULL),
selected_on_copy_construction(NULL) {}

propagating_allocator(bool& poca, bool& poma, bool& pos, bool& soc)
: propagated_on_copy_assignment(&poca),
propagated_on_move_assignment(&poma),
propagated_on_swap(&pos),
selected_on_copy_construction(&soc) {}

propagating_allocator(const propagating_allocator& other)
: Allocator(other),
propagated_on_copy_assignment(other.propagated_on_copy_assignment),
propagated_on_move_assignment(other.propagated_on_move_assignment),
propagated_on_swap(other.propagated_on_swap),
selected_on_copy_construction(other.selected_on_copy_construction) {}

template <typename Allocator2>
propagating_allocator(const propagating_allocator<Allocator2, POCMA, POCCA, POCS>& other)
: Allocator(other),
propagated_on_copy_assignment(other.propagated_on_copy_assignment),
propagated_on_move_assignment(other.propagated_on_move_assignment),
propagated_on_swap(other.propagated_on_swap),
selected_on_copy_construction(other.selected_on_copy_construction) {}

propagating_allocator& operator=(const propagating_allocator&) {
ASSERT(POCCA::value, "Allocator should not copy assign if pocca is false");
if (propagated_on_copy_assignment)
*propagated_on_copy_assignment = true;
return *this;
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
propagating_allocator& operator=(propagating_allocator&&) {
ASSERT(POCMA::value, "Allocator should not move assign if pocma is false");
if (propagated_on_move_assignment)
*propagated_on_move_assignment = true;
return *this;
}
#endif

propagating_allocator select_on_container_copy_construction() const {
if (selected_on_copy_construction)
*selected_on_copy_construction = true;
return *this;
}
};

namespace propagating_allocators {
typedef tbb::tbb_allocator<int> base_allocator;
typedef tbb::internal::traits_true_type true_type;
typedef tbb::internal::traits_false_type false_type;

typedef propagating_allocator<base_allocator, true_type, true_type,
true_type> always_propagating_allocator;
typedef propagating_allocator<base_allocator, false_type, false_type, false_type> never_propagating_allocator;
typedef propagating_allocator<base_allocator, true_type, false_type, false_type> pocma_allocator;
typedef propagating_allocator<base_allocator, false_type, true_type, false_type> pocca_allocator;
typedef propagating_allocator<base_allocator, false_type, false_type, true_type> pocs_allocator;
}

template <typename Allocator, typename POCMA, typename POCCA, typename POCS>
void swap(propagating_allocator<Allocator, POCMA, POCCA, POCS>& lhs,
propagating_allocator<Allocator, POCMA, POCCA, POCS>&) {
ASSERT(POCS::value, "Allocator should not swap if pocs is false");
if (lhs.propagated_on_swap)
*lhs.propagated_on_swap = true;
}

template <typename ContainerType>
void test_allocator_traits_support() {
typedef typename ContainerType::allocator_type allocator_type;
typedef std::allocator_traits<allocator_type> allocator_traits;
typedef typename allocator_traits::propagate_on_container_copy_assignment pocca_type;
#if __TBB_CPP11_RVALUE_REF_PRESENT
typedef typename allocator_traits::propagate_on_container_move_assignment pocma_type;
#endif
typedef typename allocator_traits::propagate_on_container_swap pocs_type;

bool propagated_on_copy = false;
bool propagated_on_move = false;
bool propagated_on_swap = false;
bool selected_on_copy = false;

allocator_type alloc(propagated_on_copy, propagated_on_move, propagated_on_swap, selected_on_copy);

ContainerType c1(alloc), c2(c1);
ASSERT(selected_on_copy, "select_on_container_copy_construction function was not called");

c1 = c2;
ASSERT(propagated_on_copy == pocca_type::value, "Unexpected allocator propagation on copy assignment");

#if __TBB_CPP11_RVALUE_REF_PRESENT
c2 = std::move(c1);
ASSERT(propagated_on_move == pocma_type::value, "Unexpected allocator propagation on move assignment");
#endif

c1.swap(c2);
ASSERT(propagated_on_swap == pocs_type::value, "Unexpected allocator propagation on swap");
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
class non_movable_object {
non_movable_object() {}
private:
non_movable_object(non_movable_object&&);
non_movable_object& operator=(non_movable_object&&);
};

template <typename ContainerType>
void test_allocator_traits_with_non_movable_value_type() {
typedef typename ContainerType::allocator_type allocator_type;
typedef std::allocator_traits<allocator_type> allocator_traits;
typedef typename allocator_traits::propagate_on_container_move_assignment pocma_type;
ASSERT(pocma_type::value, "Allocator POCMA must be true for this test");
allocator_type alloc;
ContainerType container1(alloc), container2(alloc);
container1 = std::move(container2);
}
#endif 

#endif 

#if __TBB_CPP11_RVALUE_REF_PRESENT

template<typename Allocator>
class allocator_aware_data {
public:
static bool assert_on_constructions;
typedef Allocator allocator_type;

allocator_aware_data(const allocator_type& allocator = allocator_type())
: my_allocator(allocator), my_value(0) {}
allocator_aware_data(int v, const allocator_type& allocator = allocator_type())
: my_allocator(allocator), my_value(v) {}
allocator_aware_data(const allocator_aware_data&) {
ASSERT(!assert_on_constructions, "Allocator should propagate to the data during copy construction");
}
allocator_aware_data(allocator_aware_data&&) {
ASSERT(!assert_on_constructions, "Allocator should propagate to the data during move construction");
}
allocator_aware_data(const allocator_aware_data& rhs, const allocator_type& allocator)
: my_allocator(allocator), my_value(rhs.my_value) {}
allocator_aware_data(allocator_aware_data&& rhs, const allocator_type& allocator)
: my_allocator(allocator), my_value(rhs.my_value) {}

int value() const { return my_value; }
private:
allocator_type my_allocator;
int my_value;
};

template<typename Allocator>
bool allocator_aware_data<Allocator>::assert_on_constructions = false;

#endif 

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif 

namespace Harness {

struct IsEqual {
#if __TBB_CPP11_SMART_POINTERS_PRESENT
template <typename T>
static bool compare( const std::weak_ptr<T> &t1, const std::weak_ptr<T> &t2 ) {
return t1.lock().get() == t2.lock().get();
}
template <typename T>
static bool compare( const std::unique_ptr<T> &t1, const std::unique_ptr<T> &t2 ) {
return *t1 == *t2;
}
template <typename T1, typename T2>
static bool compare( const std::pair< const std::weak_ptr<T1>, std::weak_ptr<T2> > &t1,
const std::pair< const std::weak_ptr<T1>, std::weak_ptr<T2> > &t2 ) {
return t1.first.lock().get() == t2.first.lock().get() &&
t1.second.lock().get() == t2.second.lock().get();
}
#endif 
template <typename T1, typename T2>
static bool compare( const T1 &t1, const T2 &t2 ) {
return t1 == t2;
}
template <typename T1, typename T2>
bool operator()( T1 &t1, T2 &t2) const {
return compare( (const T1&)t1, (const T2&)t2 );
}
};

} 
#endif 
