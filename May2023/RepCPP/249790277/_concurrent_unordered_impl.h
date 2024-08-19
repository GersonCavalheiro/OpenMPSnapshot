



#ifndef __TBB__concurrent_unordered_impl_H
#define __TBB__concurrent_unordered_impl_H
#if !defined(__TBB_concurrent_unordered_map_H) && !defined(__TBB_concurrent_unordered_set_H) && !defined(__TBB_concurrent_hash_map_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#include "../tbb_stddef.h"

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <iterator>
#include <utility>      
#include <functional>   
#include <string>       
#include <cstring>      
#include __TBB_STD_SWAP_HEADER

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#include "../atomic.h"
#include "../tbb_exception.h"
#include "../tbb_allocator.h"

#if __TBB_INITIALIZER_LISTS_PRESENT
#include <initializer_list>
#endif

#include "_tbb_hash_compare_impl.h"

namespace tbb {
namespace interface5 {
namespace internal {

template <typename T, typename Allocator>
class split_ordered_list;
template <typename Traits>
class concurrent_unordered_base;

template<class Solist, typename Value>
class flist_iterator : public std::iterator<std::forward_iterator_tag, Value>
{
template <typename T, typename Allocator>
friend class split_ordered_list;
template <typename Traits>
friend class concurrent_unordered_base;
template<class M, typename V>
friend class flist_iterator;

typedef typename Solist::nodeptr_t nodeptr_t;
public:
typedef typename Solist::value_type value_type;
typedef typename Solist::difference_type difference_type;
typedef typename Solist::pointer pointer;
typedef typename Solist::reference reference;

flist_iterator() : my_node_ptr(0) {}
flist_iterator( const flist_iterator<Solist, typename Solist::value_type> &other )
: my_node_ptr(other.my_node_ptr) {}

reference operator*() const { return my_node_ptr->my_element; }
pointer operator->() const { return &**this; }

flist_iterator& operator++() {
my_node_ptr = my_node_ptr->my_next;
return *this;
}

flist_iterator operator++(int) {
flist_iterator tmp = *this;
++*this;
return tmp;
}

protected:
flist_iterator(nodeptr_t pnode) : my_node_ptr(pnode) {}
nodeptr_t get_node_ptr() const { return my_node_ptr; }

nodeptr_t my_node_ptr;

template<typename M, typename T, typename U>
friend bool operator==( const flist_iterator<M,T> &i, const flist_iterator<M,U> &j );
template<typename M, typename T, typename U>
friend bool operator!=( const flist_iterator<M,T>& i, const flist_iterator<M,U>& j );
};

template<typename Solist, typename T, typename U>
bool operator==( const flist_iterator<Solist,T> &i, const flist_iterator<Solist,U> &j ) {
return i.my_node_ptr == j.my_node_ptr;
}
template<typename Solist, typename T, typename U>
bool operator!=( const flist_iterator<Solist,T>& i, const flist_iterator<Solist,U>& j ) {
return i.my_node_ptr != j.my_node_ptr;
}

template<class Solist, typename Value>
class solist_iterator : public flist_iterator<Solist, Value>
{
typedef flist_iterator<Solist, Value> base_type;
typedef typename Solist::nodeptr_t nodeptr_t;
using base_type::get_node_ptr;
template <typename T, typename Allocator>
friend class split_ordered_list;
template<class M, typename V>
friend class solist_iterator;
template<typename M, typename T, typename U>
friend bool operator==( const solist_iterator<M,T> &i, const solist_iterator<M,U> &j );
template<typename M, typename T, typename U>
friend bool operator!=( const solist_iterator<M,T>& i, const solist_iterator<M,U>& j );

const Solist *my_list_ptr;
solist_iterator(nodeptr_t pnode, const Solist *plist) : base_type(pnode), my_list_ptr(plist) {}

public:
typedef typename Solist::value_type value_type;
typedef typename Solist::difference_type difference_type;
typedef typename Solist::pointer pointer;
typedef typename Solist::reference reference;

solist_iterator() {}
solist_iterator(const solist_iterator<Solist, typename Solist::value_type> &other )
: base_type(other), my_list_ptr(other.my_list_ptr) {}

reference operator*() const {
return this->base_type::operator*();
}

pointer operator->() const {
return (&**this);
}

solist_iterator& operator++() {
do ++(*(base_type *)this);
while (get_node_ptr() != NULL && get_node_ptr()->is_dummy());

return (*this);
}

solist_iterator operator++(int) {
solist_iterator tmp = *this;
do ++*this;
while (get_node_ptr() != NULL && get_node_ptr()->is_dummy());

return (tmp);
}
};

template<typename Solist, typename T, typename U>
bool operator==( const solist_iterator<Solist,T> &i, const solist_iterator<Solist,U> &j ) {
return i.my_node_ptr == j.my_node_ptr && i.my_list_ptr == j.my_list_ptr;
}
template<typename Solist, typename T, typename U>
bool operator!=( const solist_iterator<Solist,T>& i, const solist_iterator<Solist,U>& j ) {
return i.my_node_ptr != j.my_node_ptr || i.my_list_ptr != j.my_list_ptr;
}

typedef size_t sokey_t;


template <typename T, typename Allocator>
class split_ordered_list
{
public:
typedef split_ordered_list<T, Allocator> self_type;
typedef typename Allocator::template rebind<T>::other allocator_type;
struct node;
typedef node *nodeptr_t;

typedef typename allocator_type::size_type size_type;
typedef typename allocator_type::difference_type difference_type;
typedef typename allocator_type::pointer pointer;
typedef typename allocator_type::const_pointer const_pointer;
typedef typename allocator_type::reference reference;
typedef typename allocator_type::const_reference const_reference;
typedef typename allocator_type::value_type value_type;

typedef solist_iterator<self_type, const value_type> const_iterator;
typedef solist_iterator<self_type, value_type> iterator;
typedef flist_iterator<self_type, const value_type> raw_const_iterator;
typedef flist_iterator<self_type, value_type> raw_iterator;

struct node : tbb::internal::no_assign
{
private:
node();  
public:
void init(sokey_t order_key) {
my_order_key = order_key;
my_next = NULL;
}

sokey_t get_order_key() const { 
return my_order_key;
}

nodeptr_t atomic_set_next(nodeptr_t new_node, nodeptr_t current_node)
{
nodeptr_t exchange_node = tbb::internal::as_atomic(my_next).compare_and_swap(new_node, current_node);

if (exchange_node == current_node) 
{
return new_node;
}
else
{
return exchange_node;
}
}

bool is_dummy() const {
return (my_order_key & 0x1) == 0;
}


nodeptr_t  my_next;      
value_type my_element;   
sokey_t    my_order_key; 
};

nodeptr_t create_node(sokey_t order_key) {
nodeptr_t pnode = my_node_allocator.allocate(1);
pnode->init(order_key);
return (pnode);
}

template<typename Arg>
nodeptr_t create_node(sokey_t order_key, __TBB_FORWARDING_REF(Arg) t,
tbb::internal::true_type=tbb::internal::true_type()){
nodeptr_t pnode = my_node_allocator.allocate(1);

__TBB_TRY {
new(static_cast<void*>(&pnode->my_element)) T(tbb::internal::forward<Arg>(t));
pnode->init(order_key);
} __TBB_CATCH(...) {
my_node_allocator.deallocate(pnode, 1);
__TBB_RETHROW();
}

return (pnode);
}

template<typename Arg>
nodeptr_t create_node(sokey_t, __TBB_FORWARDING_REF(Arg),
tbb::internal::false_type){
__TBB_ASSERT(false, "This compile-time helper should never get called");
return nodeptr_t();
}

template<typename __TBB_PARAMETER_PACK Args>
nodeptr_t create_node_v( __TBB_FORWARDING_REF(Args) __TBB_PARAMETER_PACK args){
nodeptr_t pnode = my_node_allocator.allocate(1);

__TBB_TRY {
new(static_cast<void*>(&pnode->my_element)) T(__TBB_PACK_EXPANSION(tbb::internal::forward<Args>(args)));
} __TBB_CATCH(...) {
my_node_allocator.deallocate(pnode, 1);
__TBB_RETHROW();
}

return (pnode);
}

split_ordered_list(allocator_type a = allocator_type())
: my_node_allocator(a), my_element_count(0)
{
my_head = create_node(sokey_t(0));
}

~split_ordered_list()
{
clear();

nodeptr_t pnode = my_head;
my_head = NULL;

__TBB_ASSERT(pnode != NULL && pnode->my_next == NULL, "Invalid head list node");

destroy_node(pnode);
}


allocator_type get_allocator() const {
return (my_node_allocator);
}

void clear() {
nodeptr_t pnext;
nodeptr_t pnode = my_head;

__TBB_ASSERT(my_head != NULL, "Invalid head list node");
pnext = pnode->my_next;
pnode->my_next = NULL;
pnode = pnext;

while (pnode != NULL)
{
pnext = pnode->my_next;
destroy_node(pnode);
pnode = pnext;
}

my_element_count = 0;
}

iterator begin() {
return first_real_iterator(raw_begin());
}

const_iterator begin() const {
return first_real_iterator(raw_begin());
}

iterator end() {
return (iterator(0, this));
}

const_iterator end() const {
return (const_iterator(0, this));
}

const_iterator cbegin() const {
return (((const self_type *)this)->begin());
}

const_iterator cend() const {
return (((const self_type *)this)->end());
}

bool empty() const {
return (my_element_count == 0);
}

size_type size() const {
return my_element_count;
}

size_type max_size() const {
return my_node_allocator.max_size();
}

void swap(self_type& other)
{
if (this == &other)
{
return;
}

std::swap(my_element_count, other.my_element_count);
std::swap(my_head, other.my_head);
}


raw_iterator raw_begin() {
return raw_iterator(my_head);
}

raw_const_iterator raw_begin() const {
return raw_const_iterator(my_head);
}

raw_iterator raw_end() {
return raw_iterator(0);
}

raw_const_iterator raw_end() const {
return raw_const_iterator(0);
}

static sokey_t get_order_key(const raw_const_iterator& it) {
return it.get_node_ptr()->get_order_key();
}

static sokey_t get_safe_order_key(const raw_const_iterator& it) {
if( !it.get_node_ptr() )  return ~sokey_t(0);
return it.get_node_ptr()->get_order_key();
}

iterator get_iterator(raw_iterator it) {
__TBB_ASSERT(it.get_node_ptr() == NULL || !it.get_node_ptr()->is_dummy(), "Invalid user node (dummy)");
return iterator(it.get_node_ptr(), this);
}

const_iterator get_iterator(raw_const_iterator it) const {
__TBB_ASSERT(it.get_node_ptr() == NULL || !it.get_node_ptr()->is_dummy(), "Invalid user node (dummy)");
return const_iterator(it.get_node_ptr(), this);
}

raw_iterator get_iterator(raw_const_iterator it) {
return raw_iterator(it.get_node_ptr());
}

static iterator get_iterator(const_iterator it) {
return iterator(it.my_node_ptr, it.my_list_ptr);
}

iterator first_real_iterator(raw_iterator it)
{
while (it != raw_end() && it.get_node_ptr()->is_dummy())
++it;

return iterator(it.get_node_ptr(), this);
}

const_iterator first_real_iterator(raw_const_iterator it) const
{
while (it != raw_end() && it.get_node_ptr()->is_dummy())
++it;

return const_iterator(it.get_node_ptr(), this);
}

void destroy_node(nodeptr_t pnode) {
if (!pnode->is_dummy()) my_node_allocator.destroy(pnode);
my_node_allocator.deallocate(pnode, 1);
}

static nodeptr_t try_insert_atomic(nodeptr_t previous, nodeptr_t new_node, nodeptr_t current_node) {
new_node->my_next = current_node;
return previous->atomic_set_next(new_node, current_node);
}

std::pair<iterator, bool> try_insert(raw_iterator it, raw_iterator next, nodeptr_t pnode, size_type *new_count)
{
nodeptr_t inserted_node = try_insert_atomic(it.get_node_ptr(), pnode, next.get_node_ptr());

if (inserted_node == pnode)
{
check_range(it, next);
*new_count = tbb::internal::as_atomic(my_element_count).fetch_and_increment();
return std::pair<iterator, bool>(iterator(pnode, this), true);
}
else
{
return std::pair<iterator, bool>(end(), false);
}
}

raw_iterator insert_dummy(raw_iterator it, sokey_t order_key)
{
raw_iterator last = raw_end();
raw_iterator where = it;

__TBB_ASSERT(where != last, "Invalid head node");

++where;

nodeptr_t dummy_node = create_node(order_key);

for (;;)
{
__TBB_ASSERT(it != last, "Invalid head list node");

if (where == last || get_order_key(where) > order_key)
{
__TBB_ASSERT(get_order_key(it) < order_key, "Invalid node order in the list");

nodeptr_t inserted_node = try_insert_atomic(it.get_node_ptr(), dummy_node, where.get_node_ptr());

if (inserted_node == dummy_node)
{
check_range(it, where);
return raw_iterator(dummy_node);
}
else
{
where = it;
++where;
continue;
}
}
else if (get_order_key(where) == order_key)
{
destroy_node(dummy_node);
return where;
}

it = where;
++where;
}

}

void erase_node(raw_iterator previous, raw_const_iterator& where)
{
nodeptr_t pnode = (where++).get_node_ptr();
nodeptr_t prevnode = previous.get_node_ptr();
__TBB_ASSERT(prevnode->my_next == pnode, "Erase must take consecutive iterators");
prevnode->my_next = pnode->my_next;

destroy_node(pnode);
}

iterator erase_node(raw_iterator previous, const_iterator where)
{
raw_const_iterator it = where;
erase_node(previous, it);
my_element_count--;

return get_iterator(first_real_iterator(it));
}

void move_all(self_type& source)
{
raw_const_iterator first = source.raw_begin();
raw_const_iterator last = source.raw_end();

if (first == last)
return;

nodeptr_t previous_node = my_head;
raw_const_iterator begin_iterator = first++;

for (raw_const_iterator it = first; it != last;)
{
nodeptr_t pnode = it.get_node_ptr();

nodeptr_t dummy_node = pnode->is_dummy() ? create_node(pnode->get_order_key()) : create_node(pnode->get_order_key(), pnode->my_element);
previous_node = try_insert_atomic(previous_node, dummy_node, NULL);
__TBB_ASSERT(previous_node != NULL, "Insertion must succeed");
raw_const_iterator where = it++;
source.erase_node(get_iterator(begin_iterator), where);
}
check_range();
}


private:
template <typename Traits>
friend class concurrent_unordered_base;

void check_range( raw_iterator first, raw_iterator last )
{
#if TBB_USE_ASSERT
for (raw_iterator it = first; it != last; ++it)
{
raw_iterator next = it;
++next;

__TBB_ASSERT(next == raw_end() || get_order_key(next) >= get_order_key(it), "!!! List order inconsistency !!!");
}
#else
tbb::internal::suppress_unused_warning(first, last);
#endif
}
void check_range()
{
#if TBB_USE_ASSERT
check_range( raw_begin(), raw_end() );
#endif
}

typename allocator_type::template rebind<node>::other my_node_allocator;  
size_type                                             my_element_count;   
nodeptr_t                                             my_head;            
};

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable: 4127) 
#endif

template <typename Traits>
class concurrent_unordered_base : public Traits
{
protected:
typedef concurrent_unordered_base<Traits> self_type;
typedef typename Traits::value_type value_type;
typedef typename Traits::key_type key_type;
typedef typename Traits::hash_compare hash_compare;
typedef typename Traits::allocator_type allocator_type;
typedef typename hash_compare::hasher hasher;
typedef typename hash_compare::key_equal key_equal;
typedef typename allocator_type::pointer pointer;
typedef typename allocator_type::const_pointer const_pointer;
typedef typename allocator_type::reference reference;
typedef typename allocator_type::const_reference const_reference;
typedef typename allocator_type::size_type size_type;
typedef typename allocator_type::difference_type difference_type;
typedef split_ordered_list<value_type, typename Traits::allocator_type> solist_t;
typedef typename solist_t::nodeptr_t nodeptr_t;
typedef typename solist_t::raw_iterator raw_iterator;
typedef typename solist_t::raw_const_iterator raw_const_iterator;
typedef typename solist_t::iterator iterator; 
typedef typename solist_t::const_iterator const_iterator;
typedef iterator local_iterator;
typedef const_iterator const_local_iterator;
using Traits::my_hash_compare;
using Traits::get_key;
using Traits::allow_multimapping;

static const size_type initial_bucket_number = 8;                               
private:
typedef std::pair<iterator, iterator> pairii_t;
typedef std::pair<const_iterator, const_iterator> paircc_t;

static size_type const pointers_per_table = sizeof(size_type) * 8;              
static const size_type initial_bucket_load = 4;                                

struct call_internal_clear_on_exit{
concurrent_unordered_base* my_instance;
call_internal_clear_on_exit(concurrent_unordered_base* instance) : my_instance(instance) {}
void dismiss(){ my_instance = NULL;}
~call_internal_clear_on_exit(){
if (my_instance){
my_instance->internal_clear();
}
}
};
protected:
concurrent_unordered_base(size_type n_of_buckets = initial_bucket_number,
const hash_compare& hc = hash_compare(), const allocator_type& a = allocator_type())
: Traits(hc), my_solist(a),
my_allocator(a), my_maximum_bucket_size((float) initial_bucket_load)
{
if( n_of_buckets == 0) ++n_of_buckets;
my_number_of_buckets = size_type(1)<<__TBB_Log2((uintptr_t)n_of_buckets*2-1); 
internal_init();
}

concurrent_unordered_base(const concurrent_unordered_base& right, const allocator_type& a)
: Traits(right.my_hash_compare), my_solist(a), my_allocator(a)
{
internal_init();
internal_copy(right);
}

concurrent_unordered_base(const concurrent_unordered_base& right)
: Traits(right.my_hash_compare), my_solist(right.get_allocator()), my_allocator(right.get_allocator())
{
internal_init();
internal_copy(right);
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
concurrent_unordered_base(concurrent_unordered_base&& right)
: Traits(right.my_hash_compare), my_solist(right.get_allocator()), my_allocator(right.get_allocator())
{
internal_init();
swap(right);
}

concurrent_unordered_base(concurrent_unordered_base&& right, const allocator_type& a)
: Traits(right.my_hash_compare), my_solist(a), my_allocator(a)
{
call_internal_clear_on_exit clear_buckets_on_exception(this);

internal_init();
if (a == right.get_allocator()){
this->swap(right);
}else{
my_maximum_bucket_size = right.my_maximum_bucket_size;
my_number_of_buckets = right.my_number_of_buckets;
my_solist.my_element_count = right.my_solist.my_element_count;

if (! right.my_solist.empty()){
nodeptr_t previous_node = my_solist.my_head;

for (raw_const_iterator it = ++(right.my_solist.raw_begin()), last = right.my_solist.raw_end(); it != last; ++it)
{
const nodeptr_t pnode = it.get_node_ptr();
nodeptr_t node;
if (pnode->is_dummy()) {
node = my_solist.create_node(pnode->get_order_key());
size_type bucket = __TBB_ReverseBits(pnode->get_order_key()) % my_number_of_buckets;
set_bucket(bucket, node);
}else{
node = my_solist.create_node(pnode->get_order_key(), std::move(pnode->my_element));
}

previous_node = my_solist.try_insert_atomic(previous_node, node, NULL);
__TBB_ASSERT(previous_node != NULL, "Insertion of node failed. Concurrent inserts in constructor ?");
}
my_solist.check_range();
}
}

clear_buckets_on_exception.dismiss();
}

#endif 

concurrent_unordered_base& operator=(const concurrent_unordered_base& right) {
if (this != &right)
internal_copy(right);
return (*this);
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
concurrent_unordered_base& operator=(concurrent_unordered_base&& other)
{
if(this != &other){
typedef typename tbb::internal::allocator_traits<allocator_type>::propagate_on_container_move_assignment pocma_t;
if(pocma_t::value || this->my_allocator == other.my_allocator) {
concurrent_unordered_base trash (std::move(*this));
swap(other);
if (pocma_t::value) {
using std::swap;
swap(this->my_solist.my_node_allocator, other.my_solist.my_node_allocator);
swap(this->my_allocator, other.my_allocator);
}
} else {
concurrent_unordered_base moved_copy(std::move(other),this->my_allocator);
this->swap(moved_copy);
}
}
return *this;
}

#endif 

#if __TBB_INITIALIZER_LISTS_PRESENT
concurrent_unordered_base& operator=(std::initializer_list<value_type> il)
{
this->clear();
this->insert(il.begin(),il.end());
return (*this);
}
#endif 


~concurrent_unordered_base() {
internal_clear();
}

public:
allocator_type get_allocator() const {
return my_solist.get_allocator();
}

bool empty() const {
return my_solist.empty();
}

size_type size() const {
return my_solist.size();
}

size_type max_size() const {
return my_solist.max_size();
}

iterator begin() {
return my_solist.begin();
}

const_iterator begin() const {
return my_solist.begin();
}

iterator end() {
return my_solist.end();
}

const_iterator end() const {
return my_solist.end();
}

const_iterator cbegin() const {
return my_solist.cbegin();
}

const_iterator cend() const {
return my_solist.cend();
}

class const_range_type : tbb::internal::no_assign {
const concurrent_unordered_base &my_table;
raw_const_iterator my_begin_node;
raw_const_iterator my_end_node;
mutable raw_const_iterator my_midpoint_node;
public:
typedef typename concurrent_unordered_base::size_type size_type;
typedef typename concurrent_unordered_base::value_type value_type;
typedef typename concurrent_unordered_base::reference reference;
typedef typename concurrent_unordered_base::difference_type difference_type;
typedef typename concurrent_unordered_base::const_iterator iterator;

bool empty() const {return my_begin_node == my_end_node;}

bool is_divisible() const {
return my_midpoint_node != my_end_node;
}
const_range_type( const_range_type &r, split ) :
my_table(r.my_table), my_end_node(r.my_end_node)
{
r.my_end_node = my_begin_node = r.my_midpoint_node;
__TBB_ASSERT( !empty(), "Splitting despite the range is not divisible" );
__TBB_ASSERT( !r.empty(), "Splitting despite the range is not divisible" );
set_midpoint();
r.set_midpoint();
}
const_range_type( const concurrent_unordered_base &a_table ) :
my_table(a_table), my_begin_node(a_table.my_solist.begin()),
my_end_node(a_table.my_solist.end())
{
set_midpoint();
}
iterator begin() const { return my_table.my_solist.get_iterator(my_begin_node); }
iterator end() const { return my_table.my_solist.get_iterator(my_end_node); }
size_type grainsize() const { return 1; }

void set_midpoint() const {
if( my_begin_node == my_end_node ) 
my_midpoint_node = my_end_node;
else {
sokey_t begin_key = solist_t::get_safe_order_key(my_begin_node);
sokey_t end_key = solist_t::get_safe_order_key(my_end_node);
size_t mid_bucket = __TBB_ReverseBits( begin_key + (end_key-begin_key)/2 ) % my_table.my_number_of_buckets;
while ( !my_table.is_initialized(mid_bucket) ) mid_bucket = my_table.get_parent(mid_bucket);
if(__TBB_ReverseBits(mid_bucket) > begin_key) {
my_midpoint_node = my_table.my_solist.first_real_iterator(my_table.get_bucket( mid_bucket ));
}
else {
my_midpoint_node = my_end_node;
}
#if TBB_USE_ASSERT
{
sokey_t mid_key = solist_t::get_safe_order_key(my_midpoint_node);
__TBB_ASSERT( begin_key < mid_key, "my_begin_node is after my_midpoint_node" );
__TBB_ASSERT( mid_key <= end_key, "my_midpoint_node is after my_end_node" );
}
#endif 
}
}
};

class range_type : public const_range_type {
public:
typedef typename concurrent_unordered_base::iterator iterator;
range_type( range_type &r, split ) : const_range_type( r, split() ) {}
range_type( const concurrent_unordered_base &a_table ) : const_range_type(a_table) {}

iterator begin() const { return solist_t::get_iterator( const_range_type::begin() ); }
iterator end() const { return solist_t::get_iterator( const_range_type::end() ); }
};

range_type range() {
return range_type( *this );
}

const_range_type range() const {
return const_range_type( *this );
}

std::pair<iterator, bool> insert(const value_type& value) {
return internal_insert<tbb::internal::true_type>(value);
}

iterator insert(const_iterator, const value_type& value) {
return insert(value).first;
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
std::pair<iterator, bool> insert(value_type&& value) {
return internal_insert<tbb::internal::true_type>(std::move(value));
}

iterator insert(const_iterator, value_type&& value) {
return insert(std::move(value)).first;
}

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
template<typename... Args>
std::pair<iterator, bool> emplace(Args&&... args) {
nodeptr_t pnode = my_solist.create_node_v(tbb::internal::forward<Args>(args)...);
const sokey_t hashed_element_key = (sokey_t) my_hash_compare(get_key(pnode->my_element));
const sokey_t order_key = split_order_key_regular(hashed_element_key);
pnode->init(order_key);

return internal_insert<tbb::internal::false_type>(pnode->my_element, pnode);
}

template<typename... Args>
iterator emplace_hint(const_iterator, Args&&... args) {
return emplace(tbb::internal::forward<Args>(args)...).first;
}

#endif 
#endif 

template<class Iterator>
void insert(Iterator first, Iterator last) {
for (Iterator it = first; it != last; ++it)
insert(*it);
}

#if __TBB_INITIALIZER_LISTS_PRESENT
void insert(std::initializer_list<value_type> il) {
insert(il.begin(), il.end());
}
#endif

iterator unsafe_erase(const_iterator where) {
return internal_erase(where);
}

iterator unsafe_erase(const_iterator first, const_iterator last) {
while (first != last)
unsafe_erase(first++);
return my_solist.get_iterator(first);
}

size_type unsafe_erase(const key_type& key) {
pairii_t where = equal_range(key);
size_type item_count = internal_distance(where.first, where.second);
unsafe_erase(where.first, where.second);
return item_count;
}

void swap(concurrent_unordered_base& right) {
if (this != &right) {
std::swap(my_hash_compare, right.my_hash_compare); 
my_solist.swap(right.my_solist);
internal_swap_buckets(right);
std::swap(my_number_of_buckets, right.my_number_of_buckets);
std::swap(my_maximum_bucket_size, right.my_maximum_bucket_size);
}
}

hasher hash_function() const {
return my_hash_compare.my_hash_object;
}

key_equal key_eq() const {
return my_hash_compare.my_key_compare_object;
}

void clear() {
my_solist.clear();

internal_clear();

__TBB_ASSERT(my_buckets[0] == NULL, NULL);
raw_iterator dummy_node = my_solist.raw_begin();
set_bucket(0, dummy_node);
}

iterator find(const key_type& key) {
return internal_find(key);
}

const_iterator find(const key_type& key) const {
return const_cast<self_type*>(this)->internal_find(key);
}

size_type count(const key_type& key) const {
if(allow_multimapping) {
paircc_t answer = equal_range(key);
size_type item_count = internal_distance(answer.first, answer.second);
return item_count;
} else {
return const_cast<self_type*>(this)->internal_find(key) == end()?0:1;
}
}

std::pair<iterator, iterator> equal_range(const key_type& key) {
return internal_equal_range(key);
}

std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
return const_cast<self_type*>(this)->internal_equal_range(key);
}

size_type unsafe_bucket_count() const {
return my_number_of_buckets;
}

size_type unsafe_max_bucket_count() const {
return segment_size(pointers_per_table-1);
}

size_type unsafe_bucket_size(size_type bucket) {
size_type item_count = 0;
if (is_initialized(bucket)) {
raw_iterator it = get_bucket(bucket);
++it;
for (; it != my_solist.raw_end() && !it.get_node_ptr()->is_dummy(); ++it)
++item_count;
}
return item_count;
}

size_type unsafe_bucket(const key_type& key) const {
sokey_t order_key = (sokey_t) my_hash_compare(key);
size_type bucket = order_key % my_number_of_buckets;
return bucket;
}

local_iterator unsafe_begin(size_type bucket) {
if (!is_initialized(bucket))
return end();

raw_iterator it = get_bucket(bucket);
return my_solist.first_real_iterator(it);
}

const_local_iterator unsafe_begin(size_type bucket) const
{
if (!is_initialized(bucket))
return end();

raw_const_iterator it = get_bucket(bucket);
return my_solist.first_real_iterator(it);
}

local_iterator unsafe_end(size_type bucket)
{
if (!is_initialized(bucket))
return end();

raw_iterator it = get_bucket(bucket);

do ++it;
while(it != my_solist.raw_end() && !it.get_node_ptr()->is_dummy());

return my_solist.first_real_iterator(it);
}

const_local_iterator unsafe_end(size_type bucket) const
{
if (!is_initialized(bucket))
return end();

raw_const_iterator it = get_bucket(bucket);

do ++it;
while(it != my_solist.raw_end() && !it.get_node_ptr()->is_dummy());

return my_solist.first_real_iterator(it);
}

const_local_iterator unsafe_cbegin(size_type bucket) const {
return ((const self_type *) this)->unsafe_begin(bucket);
}

const_local_iterator unsafe_cend(size_type bucket) const {
return ((const self_type *) this)->unsafe_end(bucket);
}

float load_factor() const {
return (float) size() / (float) unsafe_bucket_count();
}

float max_load_factor() const {
return my_maximum_bucket_size;
}

void max_load_factor(float newmax) {
if (newmax != newmax || newmax < 0)
tbb::internal::throw_exception(tbb::internal::eid_invalid_load_factor);
my_maximum_bucket_size = newmax;
}

void rehash(size_type buckets) {
size_type current_buckets = my_number_of_buckets;
if (current_buckets >= buckets)
return;
my_number_of_buckets = size_type(1)<<__TBB_Log2((uintptr_t)buckets*2-1); 
}

private:

void internal_init() {
memset(my_buckets, 0, pointers_per_table * sizeof(void *));

raw_iterator dummy_node = my_solist.raw_begin();
set_bucket(0, dummy_node);
}

void internal_clear() {
for (size_type index = 0; index < pointers_per_table; ++index) {
if (my_buckets[index] != NULL) {
size_type sz = segment_size(index);
for (size_type index2 = 0; index2 < sz; ++index2)
my_allocator.destroy(&my_buckets[index][index2]);
my_allocator.deallocate(my_buckets[index], sz);
my_buckets[index] = 0;
}
}
}

void internal_copy(const self_type& right) {
clear();

my_maximum_bucket_size = right.my_maximum_bucket_size;
my_number_of_buckets = right.my_number_of_buckets;

__TBB_TRY {
insert(right.begin(), right.end());
my_hash_compare = right.my_hash_compare;
} __TBB_CATCH(...) {
my_solist.clear();
__TBB_RETHROW();
}
}

void internal_swap_buckets(concurrent_unordered_base& right)
{
for (size_type index = 0; index < pointers_per_table; ++index)
{
raw_iterator * iterator_pointer = my_buckets[index];
my_buckets[index] = right.my_buckets[index];
right.my_buckets[index] = iterator_pointer;
}
}

static size_type internal_distance(const_iterator first, const_iterator last)
{
size_type num = 0;

for (const_iterator it = first; it != last; ++it)
++num;

return num;
}

template<typename AllowCreate, typename ValueType>
std::pair<iterator, bool> internal_insert(__TBB_FORWARDING_REF(ValueType) value, nodeptr_t pnode = NULL)
{
const key_type *pkey = &get_key(value);
sokey_t hash_key = (sokey_t) my_hash_compare(*pkey);
size_type new_count = 0;
sokey_t order_key = split_order_key_regular(hash_key);
raw_iterator previous = prepare_bucket(hash_key);
raw_iterator last = my_solist.raw_end();
__TBB_ASSERT(previous != last, "Invalid head node");

for (raw_iterator where = previous;;)
{
++where;
if (where == last || solist_t::get_order_key(where) > order_key ||
(allow_multimapping && solist_t::get_order_key(where) == order_key &&
!my_hash_compare(get_key(*where), *pkey))) 
{
if (!pnode) {
pnode = my_solist.create_node(order_key, tbb::internal::forward<ValueType>(value), AllowCreate());
pkey = &get_key(pnode->my_element);
}

std::pair<iterator, bool> result = my_solist.try_insert(previous, where, pnode, &new_count);

if (result.second)
{
adjust_table_size(new_count, my_number_of_buckets);
return result;
}
else
{
where = previous;
continue;
}
}
else if (!allow_multimapping && solist_t::get_order_key(where) == order_key &&
!my_hash_compare(get_key(*where), *pkey)) 
{ 
if (pnode)
my_solist.destroy_node(pnode);
return std::pair<iterator, bool>(my_solist.get_iterator(where), false);
}
previous = where;
}
}

iterator internal_find(const key_type& key)
{
sokey_t hash_key = (sokey_t) my_hash_compare(key);
sokey_t order_key = split_order_key_regular(hash_key);
raw_iterator last = my_solist.raw_end();

for (raw_iterator it = prepare_bucket(hash_key); it != last; ++it)
{
if (solist_t::get_order_key(it) > order_key)
{
return end();
}
else if (solist_t::get_order_key(it) == order_key)
{
if (!my_hash_compare(get_key(*it), key)) 
return my_solist.get_iterator(it);
}
}

return end();
}

iterator internal_erase(const_iterator it)
{
sokey_t hash_key = (sokey_t) my_hash_compare(get_key(*it));
raw_iterator previous = prepare_bucket(hash_key);
raw_iterator last = my_solist.raw_end();
__TBB_ASSERT(previous != last, "Invalid head node");

for (raw_iterator where = previous; ; previous = where) {
++where;
if (where == last)
return end();
else if (my_solist.get_iterator(where) == it)
return my_solist.erase_node(previous, it);
}
}

pairii_t internal_equal_range(const key_type& key)
{
sokey_t hash_key = (sokey_t) my_hash_compare(key);
sokey_t order_key = split_order_key_regular(hash_key);
raw_iterator end_it = my_solist.raw_end();

for (raw_iterator it = prepare_bucket(hash_key); it != end_it; ++it)
{
if (solist_t::get_order_key(it) > order_key)
{
return pairii_t(end(), end());
}
else if (solist_t::get_order_key(it) == order_key &&
!my_hash_compare(get_key(*it), key)) 
{
iterator first = my_solist.get_iterator(it);
iterator last = first;
do ++last; while( allow_multimapping && last != end() && !my_hash_compare(get_key(*last), key) );
return pairii_t(first, last);
}
}

return pairii_t(end(), end());
}

void init_bucket(size_type bucket)
{
__TBB_ASSERT( bucket != 0, "The first bucket must always be initialized");

size_type parent_bucket = get_parent(bucket);

if (!is_initialized(parent_bucket))
init_bucket(parent_bucket);

raw_iterator parent = get_bucket(parent_bucket);

raw_iterator dummy_node = my_solist.insert_dummy(parent, split_order_key_dummy(bucket));
set_bucket(bucket, dummy_node);
}

void adjust_table_size(size_type total_elements, size_type current_size)
{
if ( ((float) total_elements / (float) current_size) > my_maximum_bucket_size )
{
my_number_of_buckets.compare_and_swap(2u*current_size, current_size);
}
}

size_type get_parent(size_type bucket) const
{
size_type msb = __TBB_Log2((uintptr_t)bucket);
return bucket & ~(size_type(1) << msb);
}


static size_type segment_index_of( size_type index ) {
return size_type( __TBB_Log2( uintptr_t(index|1) ) );
}

static size_type segment_base( size_type k ) {
return (size_type(1)<<k & ~size_type(1));
}

static size_type segment_size( size_type k ) {
return k? size_type(1)<<k : 2;
}

raw_iterator get_bucket(size_type bucket) const {
size_type segment = segment_index_of(bucket);
bucket -= segment_base(segment);
__TBB_ASSERT( my_buckets[segment], "bucket must be in an allocated segment" );
return my_buckets[segment][bucket];
}

raw_iterator prepare_bucket(sokey_t hash_key) {
size_type bucket = hash_key % my_number_of_buckets;
size_type segment = segment_index_of(bucket);
size_type index = bucket - segment_base(segment);
if (my_buckets[segment] == NULL || my_buckets[segment][index].get_node_ptr() == NULL)
init_bucket(bucket);
return my_buckets[segment][index];
}

void set_bucket(size_type bucket, raw_iterator dummy_head) {
size_type segment = segment_index_of(bucket);
bucket -= segment_base(segment);

if (my_buckets[segment] == NULL) {
size_type sz = segment_size(segment);
raw_iterator * new_segment = my_allocator.allocate(sz);
std::memset(new_segment, 0, sz*sizeof(raw_iterator));

if (my_buckets[segment].compare_and_swap( new_segment, NULL) != NULL)
my_allocator.deallocate(new_segment, sz);
}

my_buckets[segment][bucket] = dummy_head;
}

bool is_initialized(size_type bucket) const {
size_type segment = segment_index_of(bucket);
bucket -= segment_base(segment);

if (my_buckets[segment] == NULL)
return false;

raw_iterator it = my_buckets[segment][bucket];
return (it.get_node_ptr() != NULL);
}


sokey_t split_order_key_regular(sokey_t order_key) const {
return __TBB_ReverseBits(order_key) | 0x1;
}

sokey_t split_order_key_dummy(sokey_t order_key) const {
return __TBB_ReverseBits(order_key) & ~sokey_t(0x1);
}

atomic<size_type>                                             my_number_of_buckets;       
solist_t                                                      my_solist;                  
typename allocator_type::template rebind<raw_iterator>::other my_allocator;               
float                                                         my_maximum_bucket_size;     
atomic<raw_iterator*>                                         my_buckets[pointers_per_table]; 
};
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning(pop) 
#endif

} 
} 
} 
#endif 
