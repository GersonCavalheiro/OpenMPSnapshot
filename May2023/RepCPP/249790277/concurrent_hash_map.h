

#ifndef __TBB_concurrent_hash_map_H
#define __TBB_concurrent_hash_map_H

#include "tbb_stddef.h"

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <iterator>
#include <utility>      
#include <cstring>      
#include __TBB_STD_SWAP_HEADER

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#include "cache_aligned_allocator.h"
#include "tbb_allocator.h"
#include "spin_rw_mutex.h"
#include "atomic.h"
#include "tbb_exception.h"
#include "tbb_profiling.h"
#include "internal/_tbb_hash_compare_impl.h"
#if __TBB_INITIALIZER_LISTS_PRESENT
#include <initializer_list>
#endif
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
#include <typeinfo>
#endif
#if __TBB_STATISTICS
#include <stdio.h>
#endif

namespace tbb {

namespace interface5 {

template<typename Key, typename T, typename HashCompare = tbb_hash_compare<Key>, typename A = tbb_allocator<std::pair<Key, T> > >
class concurrent_hash_map;

namespace internal {
using namespace tbb::internal;


typedef size_t hashcode_t;
struct hash_map_node_base : tbb::internal::no_copy {
typedef spin_rw_mutex mutex_t;
typedef mutex_t::scoped_lock scoped_t;
hash_map_node_base *next;
mutex_t mutex;
};
static hash_map_node_base *const rehash_req = reinterpret_cast<hash_map_node_base*>(size_t(3));
static hash_map_node_base *const empty_rehashed = reinterpret_cast<hash_map_node_base*>(size_t(0));
class hash_map_base {
public:
typedef size_t size_type;
typedef size_t hashcode_t;
typedef size_t segment_index_t;
typedef hash_map_node_base node_base;
struct bucket : tbb::internal::no_copy {
typedef spin_rw_mutex mutex_t;
typedef mutex_t::scoped_lock scoped_t;
mutex_t mutex;
node_base *node_list;
};
static size_type const embedded_block = 1;
static size_type const embedded_buckets = 1<<embedded_block;
static size_type const first_block = 8; 
static size_type const pointers_per_table = sizeof(segment_index_t) * 8; 
typedef bucket *segment_ptr_t;
typedef segment_ptr_t segments_table_t[pointers_per_table];
atomic<hashcode_t> my_mask;
segments_table_t my_table;
atomic<size_type> my_size; 
bucket my_embedded_segment[embedded_buckets];
#if __TBB_STATISTICS
atomic<unsigned> my_info_resizes; 
mutable atomic<unsigned> my_info_restarts; 
atomic<unsigned> my_info_rehashes;  
#endif
hash_map_base() {
std::memset( this, 0, pointers_per_table*sizeof(segment_ptr_t) 
+ sizeof(my_size) + sizeof(my_mask)  
+ embedded_buckets*sizeof(bucket) ); 
for( size_type i = 0; i < embedded_block; i++ ) 
my_table[i] = my_embedded_segment + segment_base(i);
my_mask = embedded_buckets - 1;
__TBB_ASSERT( embedded_block <= first_block, "The first block number must include embedded blocks");
#if __TBB_STATISTICS
my_info_resizes = 0; 
my_info_restarts = 0; 
my_info_rehashes = 0;  
#endif
}

static segment_index_t segment_index_of( size_type index ) {
return segment_index_t( __TBB_Log2( index|1 ) );
}

static segment_index_t segment_base( segment_index_t k ) {
return (segment_index_t(1)<<k & ~segment_index_t(1));
}

static size_type segment_size( segment_index_t k ) {
return size_type(1)<<k; 
}

static bool is_valid( void *ptr ) {
return reinterpret_cast<uintptr_t>(ptr) > uintptr_t(63);
}

static void init_buckets( segment_ptr_t ptr, size_type sz, bool is_initial ) {
if( is_initial ) std::memset(ptr, 0, sz*sizeof(bucket) );
else for(size_type i = 0; i < sz; i++, ptr++) {
*reinterpret_cast<intptr_t*>(&ptr->mutex) = 0;
ptr->node_list = rehash_req;
}
}

static void add_to_bucket( bucket *b, node_base *n ) {
__TBB_ASSERT(b->node_list != rehash_req, NULL);
n->next = b->node_list;
b->node_list = n; 
}

struct enable_segment_failsafe : tbb::internal::no_copy {
segment_ptr_t *my_segment_ptr;
enable_segment_failsafe(segments_table_t &table, segment_index_t k) : my_segment_ptr(&table[k]) {}
~enable_segment_failsafe() {
if( my_segment_ptr ) *my_segment_ptr = 0; 
}
};

void enable_segment( segment_index_t k, bool is_initial = false ) {
__TBB_ASSERT( k, "Zero segment must be embedded" );
enable_segment_failsafe watchdog( my_table, k );
cache_aligned_allocator<bucket> alloc;
size_type sz;
__TBB_ASSERT( !is_valid(my_table[k]), "Wrong concurrent assignment");
if( k >= first_block ) {
sz = segment_size( k );
segment_ptr_t ptr = alloc.allocate( sz );
init_buckets( ptr, sz, is_initial );
itt_hide_store_word( my_table[k], ptr );
sz <<= 1;
} else { 
__TBB_ASSERT( k == embedded_block, "Wrong segment index" );
sz = segment_size( first_block );
segment_ptr_t ptr = alloc.allocate( sz - embedded_buckets );
init_buckets( ptr, sz - embedded_buckets, is_initial );
ptr -= segment_base(embedded_block);
for(segment_index_t i = embedded_block; i < first_block; i++) 
itt_hide_store_word( my_table[i], ptr + segment_base(i) );
}
itt_store_word_with_release( my_mask, sz-1 );
watchdog.my_segment_ptr = 0;
}

bucket *get_bucket( hashcode_t h ) const throw() { 
segment_index_t s = segment_index_of( h );
h -= segment_base(s);
segment_ptr_t seg = my_table[s];
__TBB_ASSERT( is_valid(seg), "hashcode must be cut by valid mask for allocated segments" );
return &seg[h];
}

void mark_rehashed_levels( hashcode_t h ) throw () {
segment_index_t s = segment_index_of( h );
while( segment_ptr_t seg = my_table[++s] )
if( seg[h].node_list == rehash_req ) {
seg[h].node_list = empty_rehashed;
mark_rehashed_levels( h + ((hashcode_t)1<<s) ); 
}
}

inline bool check_mask_race( const hashcode_t h, hashcode_t &m ) const {
hashcode_t m_now, m_old = m;
m_now = (hashcode_t) itt_load_word_with_acquire( my_mask );
if( m_old != m_now )
return check_rehashing_collision( h, m_old, m = m_now );
return false;
}

bool check_rehashing_collision( const hashcode_t h, hashcode_t m_old, hashcode_t m ) const {
__TBB_ASSERT(m_old != m, NULL); 
if( (h & m_old) != (h & m) ) { 
for( ++m_old; !(h & m_old); m_old <<= 1 ) 
;
m_old = (m_old<<1) - 1; 
__TBB_ASSERT((m_old&(m_old+1))==0 && m_old <= m, NULL);
if( itt_load_word_with_acquire(get_bucket(h & m_old)->node_list) != rehash_req )
{
#if __TBB_STATISTICS
my_info_restarts++; 
#endif
return true;
}
}
return false;
}

segment_index_t insert_new_node( bucket *b, node_base *n, hashcode_t mask ) {
size_type sz = ++my_size; 
add_to_bucket( b, n );
if( sz >= mask ) { 
segment_index_t new_seg = __TBB_Log2( mask+1 ); 
__TBB_ASSERT( is_valid(my_table[new_seg-1]), "new allocations must not publish new mask until segment has allocated");
static const segment_ptr_t is_allocating = (segment_ptr_t)2;
if( !itt_hide_load_word(my_table[new_seg])
&& as_atomic(my_table[new_seg]).compare_and_swap(is_allocating, NULL) == NULL )
return new_seg; 
}
return 0;
}

void reserve(size_type buckets) {
if( !buckets-- ) return;
bool is_initial = !my_size;
for( size_type m = my_mask; buckets > m; m = my_mask )
enable_segment( segment_index_of( m+1 ), is_initial );
}
void internal_swap(hash_map_base &table) {
using std::swap;
swap(this->my_mask, table.my_mask);
swap(this->my_size, table.my_size);
for(size_type i = 0; i < embedded_buckets; i++)
swap(this->my_embedded_segment[i].node_list, table.my_embedded_segment[i].node_list);
for(size_type i = embedded_block; i < pointers_per_table; i++)
swap(this->my_table[i], table.my_table[i]);
}
};

template<typename Iterator>
class hash_map_range;


template<typename Container, typename Value>
class hash_map_iterator
: public std::iterator<std::forward_iterator_tag,Value>
{
typedef Container map_type;
typedef typename Container::node node;
typedef hash_map_base::node_base node_base;
typedef hash_map_base::bucket bucket;

template<typename C, typename T, typename U>
friend bool operator==( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

template<typename C, typename T, typename U>
friend bool operator!=( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

template<typename C, typename T, typename U>
friend ptrdiff_t operator-( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

template<typename C, typename U>
friend class hash_map_iterator;

template<typename I>
friend class hash_map_range;

void advance_to_next_bucket() { 
size_t k = my_index+1;
__TBB_ASSERT( my_bucket, "advancing an invalid iterator?");
while( k <= my_map->my_mask ) {
if( k&(k-2) ) 
++my_bucket;
else my_bucket = my_map->get_bucket( k );
my_node = static_cast<node*>( my_bucket->node_list );
if( hash_map_base::is_valid(my_node) ) {
my_index = k; return;
}
++k;
}
my_bucket = 0; my_node = 0; my_index = k; 
}
#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
template<typename Key, typename T, typename HashCompare, typename A>
friend class interface5::concurrent_hash_map;
#else
public: 
#endif
const Container *my_map;

size_t my_index;

const bucket *my_bucket;

node *my_node;

hash_map_iterator( const Container &map, size_t index, const bucket *b, node_base *n );

public:
hash_map_iterator(): my_map(), my_index(), my_bucket(), my_node() {}
hash_map_iterator( const hash_map_iterator<Container,typename Container::value_type> &other ) :
my_map(other.my_map),
my_index(other.my_index),
my_bucket(other.my_bucket),
my_node(other.my_node)
{}
Value& operator*() const {
__TBB_ASSERT( hash_map_base::is_valid(my_node), "iterator uninitialized or at end of container?" );
return my_node->item;
}
Value* operator->() const {return &operator*();}
hash_map_iterator& operator++();

hash_map_iterator operator++(int) {
hash_map_iterator old(*this);
operator++();
return old;
}
};

template<typename Container, typename Value>
hash_map_iterator<Container,Value>::hash_map_iterator( const Container &map, size_t index, const bucket *b, node_base *n ) :
my_map(&map),
my_index(index),
my_bucket(b),
my_node( static_cast<node*>(n) )
{
if( b && !hash_map_base::is_valid(n) )
advance_to_next_bucket();
}

template<typename Container, typename Value>
hash_map_iterator<Container,Value>& hash_map_iterator<Container,Value>::operator++() {
my_node = static_cast<node*>( my_node->next );
if( !my_node ) advance_to_next_bucket();
return *this;
}

template<typename Container, typename T, typename U>
bool operator==( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
return i.my_node == j.my_node && i.my_map == j.my_map;
}

template<typename Container, typename T, typename U>
bool operator!=( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
return i.my_node != j.my_node || i.my_map != j.my_map;
}


template<typename Iterator>
class hash_map_range {
typedef typename Iterator::map_type map_type;
Iterator my_begin;
Iterator my_end;
mutable Iterator my_midpoint;
size_t my_grainsize;
void set_midpoint() const;
template<typename U> friend class hash_map_range;
public:
typedef std::size_t size_type;
typedef typename Iterator::value_type value_type;
typedef typename Iterator::reference reference;
typedef typename Iterator::difference_type difference_type;
typedef Iterator iterator;

bool empty() const {return my_begin==my_end;}

bool is_divisible() const {
return my_midpoint!=my_end;
}
hash_map_range( hash_map_range& r, split ) :
my_end(r.my_end),
my_grainsize(r.my_grainsize)
{
r.my_end = my_begin = r.my_midpoint;
__TBB_ASSERT( !empty(), "Splitting despite the range is not divisible" );
__TBB_ASSERT( !r.empty(), "Splitting despite the range is not divisible" );
set_midpoint();
r.set_midpoint();
}
template<typename U>
hash_map_range( hash_map_range<U>& r) :
my_begin(r.my_begin),
my_end(r.my_end),
my_midpoint(r.my_midpoint),
my_grainsize(r.my_grainsize)
{}
hash_map_range( const map_type &map, size_type grainsize_ = 1 ) :
my_begin( Iterator( map, 0, map.my_embedded_segment, map.my_embedded_segment->node_list ) ),
my_end( Iterator( map, map.my_mask + 1, 0, 0 ) ),
my_grainsize( grainsize_ )
{
__TBB_ASSERT( grainsize_>0, "grainsize must be positive" );
set_midpoint();
}
const Iterator& begin() const {return my_begin;}
const Iterator& end() const {return my_end;}
size_type grainsize() const {return my_grainsize;}
};

template<typename Iterator>
void hash_map_range<Iterator>::set_midpoint() const {
size_t m = my_end.my_index-my_begin.my_index;
if( m > my_grainsize ) {
m = my_begin.my_index + m/2u;
hash_map_base::bucket *b = my_begin.my_map->get_bucket(m);
my_midpoint = Iterator(*my_begin.my_map,m,b,b->node_list);
} else {
my_midpoint = my_end;
}
__TBB_ASSERT( my_begin.my_index <= my_midpoint.my_index,
"my_begin is after my_midpoint" );
__TBB_ASSERT( my_midpoint.my_index <= my_end.my_index,
"my_midpoint is after my_end" );
__TBB_ASSERT( my_begin != my_midpoint || my_begin == my_end,
"[my_begin, my_midpoint) range should not be empty" );
}

} 

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif


template<typename Key, typename T, typename HashCompare, typename Allocator>
class concurrent_hash_map : protected internal::hash_map_base {
template<typename Container, typename Value>
friend class internal::hash_map_iterator;

template<typename I>
friend class internal::hash_map_range;

public:
typedef Key key_type;
typedef T mapped_type;
typedef std::pair<const Key,T> value_type;
typedef hash_map_base::size_type size_type;
typedef ptrdiff_t difference_type;
typedef value_type *pointer;
typedef const value_type *const_pointer;
typedef value_type &reference;
typedef const value_type &const_reference;
typedef internal::hash_map_iterator<concurrent_hash_map,value_type> iterator;
typedef internal::hash_map_iterator<concurrent_hash_map,const value_type> const_iterator;
typedef internal::hash_map_range<iterator> range_type;
typedef internal::hash_map_range<const_iterator> const_range_type;
typedef Allocator allocator_type;

protected:
friend class const_accessor;
struct node;
typedef typename Allocator::template rebind<node>::other node_allocator_type;
node_allocator_type my_allocator;
HashCompare my_hash_compare;

struct node : public node_base {
value_type item;
node( const Key &key ) : item(key, T()) {}
node( const Key &key, const T &t ) : item(key, t) {}
#if __TBB_CPP11_RVALUE_REF_PRESENT
node( const Key &key, T &&t ) : item(key, std::move(t)) {}
node( value_type&& i ) : item(std::move(i)){}
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
template<typename... Args>
node( Args&&... args ) : item(std::forward<Args>(args)...) {}
#if __TBB_COPY_FROM_NON_CONST_REF_BROKEN
node( value_type& i ) : item(const_cast<const value_type&>(i)) {}
#endif 
#endif 
#endif 
node( const value_type& i ) : item(i) {}

void *operator new( size_t , node_allocator_type &a ) {
void *ptr = a.allocate(1);
if(!ptr)
tbb::internal::throw_exception(tbb::internal::eid_bad_alloc);
return ptr;
}
void operator delete( void *ptr, node_allocator_type &a ) { a.deallocate(static_cast<node*>(ptr),1); }
};

void delete_node( node_base *n ) {
my_allocator.destroy( static_cast<node*>(n) );
my_allocator.deallocate( static_cast<node*>(n), 1);
}

static node* allocate_node_copy_construct(node_allocator_type& allocator, const Key &key, const T * t){
return  new( allocator ) node(key, *t);
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
static node* allocate_node_move_construct(node_allocator_type& allocator, const Key &key, const T * t){
return  new( allocator ) node(key, std::move(*const_cast<T*>(t)));
}
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
template<typename... Args>
static node* allocate_node_emplace_construct(node_allocator_type& allocator, Args&&... args){
return  new( allocator ) node(std::forward<Args>(args)...);
}
#endif 
#endif

static node* allocate_node_default_construct(node_allocator_type& allocator, const Key &key, const T * ){
return  new( allocator ) node(key);
}

static node* do_not_allocate_node(node_allocator_type& , const Key &, const T * ){
__TBB_ASSERT(false,"this dummy function should not be called");
return NULL;
}

node *search_bucket( const key_type &key, bucket *b ) const {
node *n = static_cast<node*>( b->node_list );
while( is_valid(n) && !my_hash_compare.equal(key, n->item.first) )
n = static_cast<node*>( n->next );
__TBB_ASSERT(n != internal::rehash_req, "Search can be executed only for rehashed bucket");
return n;
}

class bucket_accessor : public bucket::scoped_t {
bucket *my_b;
public:
bucket_accessor( concurrent_hash_map *base, const hashcode_t h, bool writer = false ) { acquire( base, h, writer ); }
inline void acquire( concurrent_hash_map *base, const hashcode_t h, bool writer = false ) {
my_b = base->get_bucket( h );
if( itt_load_word_with_acquire(my_b->node_list) == internal::rehash_req
&& try_acquire( my_b->mutex, true ) )
{
if( my_b->node_list == internal::rehash_req ) base->rehash_bucket( my_b, h ); 
}
else bucket::scoped_t::acquire( my_b->mutex, writer );
__TBB_ASSERT( my_b->node_list != internal::rehash_req, NULL);
}
bool is_writer() { return bucket::scoped_t::is_writer; }
bucket *operator() () { return my_b; }
};

void rehash_bucket( bucket *b_new, const hashcode_t h ) {
__TBB_ASSERT( *(intptr_t*)(&b_new->mutex), "b_new must be locked (for write)");
__TBB_ASSERT( h > 1, "The lowermost buckets can't be rehashed" );
__TBB_store_with_release(b_new->node_list, internal::empty_rehashed); 
hashcode_t mask = ( 1u<<__TBB_Log2( h ) ) - 1; 
#if __TBB_STATISTICS
my_info_rehashes++; 
#endif

bucket_accessor b_old( this, h & mask );

mask = (mask<<1) | 1; 
__TBB_ASSERT( (mask&(mask+1))==0 && (h & mask) == h, NULL );
restart:
for( node_base **p = &b_old()->node_list, *n = __TBB_load_with_acquire(*p); is_valid(n); n = *p ) {
hashcode_t c = my_hash_compare.hash( static_cast<node*>(n)->item.first );
#if TBB_USE_ASSERT
hashcode_t bmask = h & (mask>>1);
bmask = bmask==0? 1 : ( 1u<<(__TBB_Log2( bmask )+1 ) ) - 1; 
__TBB_ASSERT( (c & bmask) == (h & bmask), "hash() function changed for key in table" );
#endif
if( (c & mask) == h ) {
if( !b_old.is_writer() )
if( !b_old.upgrade_to_writer() ) {
goto restart; 
}
*p = n->next; 
add_to_bucket( b_new, n );
} else p = &n->next; 
}
}

struct call_clear_on_leave {
concurrent_hash_map* my_ch_map;
call_clear_on_leave( concurrent_hash_map* a_ch_map ) : my_ch_map(a_ch_map) {}
void dismiss() {my_ch_map = 0;}
~call_clear_on_leave(){
if (my_ch_map){
my_ch_map->clear();
}
}
};
public:

class accessor;
class const_accessor : private node::scoped_t  {
friend class concurrent_hash_map<Key,T,HashCompare,Allocator>;
friend class accessor;
public:
typedef const typename concurrent_hash_map::value_type value_type;

bool empty() const { return !my_node; }

void release() {
if( my_node ) {
node::scoped_t::release();
my_node = 0;
}
}

const_reference operator*() const {
__TBB_ASSERT( my_node, "attempt to dereference empty accessor" );
return my_node->item;
}

const_pointer operator->() const {
return &operator*();
}

const_accessor() : my_node(NULL) {}

~const_accessor() {
my_node = NULL; 
}
protected:
bool is_writer() { return node::scoped_t::is_writer; }
node *my_node;
hashcode_t my_hash;
};

class accessor: public const_accessor {
public:
typedef typename concurrent_hash_map::value_type value_type;

reference operator*() const {
__TBB_ASSERT( this->my_node, "attempt to dereference empty accessor" );
return this->my_node->item;
}

pointer operator->() const {
return &operator*();
}
};

explicit concurrent_hash_map( const allocator_type &a = allocator_type() )
: internal::hash_map_base(), my_allocator(a)
{}

concurrent_hash_map( size_type n, const allocator_type &a = allocator_type() )
: my_allocator(a)
{
reserve( n );
}

concurrent_hash_map( const concurrent_hash_map &table, const allocator_type &a = allocator_type() )
: internal::hash_map_base(), my_allocator(a)
{
internal_copy(table);
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
concurrent_hash_map( concurrent_hash_map &&table )
: internal::hash_map_base(), my_allocator(std::move(table.get_allocator()))
{
swap(table);
}

concurrent_hash_map( concurrent_hash_map &&table, const allocator_type &a )
: internal::hash_map_base(), my_allocator(a)
{
if (a == table.get_allocator()){
this->swap(table);
}else{
call_clear_on_leave scope_guard(this);
internal_copy(std::make_move_iterator(table.begin()), std::make_move_iterator(table.end()));
scope_guard.dismiss();
}
}
#endif 

template<typename I>
concurrent_hash_map( I first, I last, const allocator_type &a = allocator_type() )
: my_allocator(a)
{
reserve( std::distance(first, last) ); 
internal_copy(first, last);
}

#if __TBB_INITIALIZER_LISTS_PRESENT
concurrent_hash_map( std::initializer_list<value_type> il, const allocator_type &a = allocator_type() )
: my_allocator(a)
{
reserve(il.size());
internal_copy(il.begin(), il.end());
}

#endif 

concurrent_hash_map& operator=( const concurrent_hash_map &table ) {
if( this!=&table ) {
clear();
internal_copy(table);
}
return *this;
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
concurrent_hash_map& operator=( concurrent_hash_map &&table ) {
if(this != &table){
typedef typename tbb::internal::allocator_traits<allocator_type>::propagate_on_container_move_assignment pocma_t;
if(pocma_t::value || this->my_allocator == table.my_allocator) {
concurrent_hash_map trash (std::move(*this));
this->swap(table);
} else {
concurrent_hash_map moved_copy(std::move(table), this->my_allocator);
this->swap(moved_copy);
}
}
return *this;
}
#endif 

#if __TBB_INITIALIZER_LISTS_PRESENT
concurrent_hash_map& operator=( std::initializer_list<value_type> il ) {
clear();
reserve(il.size());
internal_copy(il.begin(), il.end());
return *this;
}
#endif 



void rehash(size_type n = 0);

void clear();

~concurrent_hash_map() { clear(); }

range_type range( size_type grainsize=1 ) {
return range_type( *this, grainsize );
}
const_range_type range( size_type grainsize=1 ) const {
return const_range_type( *this, grainsize );
}

iterator begin() { return iterator( *this, 0, my_embedded_segment, my_embedded_segment->node_list ); }
iterator end() { return iterator( *this, 0, 0, 0 ); }
const_iterator begin() const { return const_iterator( *this, 0, my_embedded_segment, my_embedded_segment->node_list ); }
const_iterator end() const { return const_iterator( *this, 0, 0, 0 ); }
std::pair<iterator, iterator> equal_range( const Key& key ) { return internal_equal_range( key, end() ); }
std::pair<const_iterator, const_iterator> equal_range( const Key& key ) const { return internal_equal_range( key, end() ); }

size_type size() const { return my_size; }

bool empty() const { return my_size == 0; }

size_type max_size() const {return (~size_type(0))/sizeof(node);}

size_type bucket_count() const { return my_mask+1; }

allocator_type get_allocator() const { return this->my_allocator; }

void swap( concurrent_hash_map &table );


size_type count( const Key &key ) const {
return const_cast<concurrent_hash_map*>(this)->lookup(false, key, NULL, NULL, false, &do_not_allocate_node );
}


bool find( const_accessor &result, const Key &key ) const {
result.release();
return const_cast<concurrent_hash_map*>(this)->lookup(false, key, NULL, &result, false, &do_not_allocate_node );
}


bool find( accessor &result, const Key &key ) {
result.release();
return lookup(false, key, NULL, &result, true, &do_not_allocate_node );
}


bool insert( const_accessor &result, const Key &key ) {
result.release();
return lookup(true, key, NULL, &result, false, &allocate_node_default_construct );
}


bool insert( accessor &result, const Key &key ) {
result.release();
return lookup(true, key, NULL, &result, true, &allocate_node_default_construct );
}


bool insert( const_accessor &result, const value_type &value ) {
result.release();
return lookup(true, value.first, &value.second, &result, false, &allocate_node_copy_construct );
}


bool insert( accessor &result, const value_type &value ) {
result.release();
return lookup(true, value.first, &value.second, &result, true, &allocate_node_copy_construct );
}


bool insert( const value_type &value ) {
return lookup(true, value.first, &value.second, NULL, false, &allocate_node_copy_construct );
}

#if __TBB_CPP11_RVALUE_REF_PRESENT

bool insert( const_accessor &result, value_type && value ) {
return generic_move_insert(result, std::move(value));
}


bool insert( accessor &result, value_type && value ) {
return generic_move_insert(result, std::move(value));
}


bool insert( value_type && value ) {
return generic_move_insert(accessor_not_used(), std::move(value));
}

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT

template<typename... Args>
bool emplace( const_accessor &result, Args&&... args ) {
return generic_emplace(result, std::forward<Args>(args)...);
}


template<typename... Args>
bool emplace( accessor &result, Args&&... args ) {
return generic_emplace(result, std::forward<Args>(args)...);
}


template<typename... Args>
bool emplace( Args&&... args ) {
return generic_emplace(accessor_not_used(), std::forward<Args>(args)...);
}
#endif 
#endif 

template<typename I>
void insert( I first, I last ) {
for ( ; first != last; ++first )
insert( *first );
}

#if __TBB_INITIALIZER_LISTS_PRESENT
void insert( std::initializer_list<value_type> il ) {
insert( il.begin(), il.end() );
}
#endif 


bool erase( const Key& key );


bool erase( const_accessor& item_accessor ) {
return exclude( item_accessor );
}


bool erase( accessor& item_accessor ) {
return exclude( item_accessor );
}

protected:
bool lookup(bool op_insert, const Key &key, const T *t, const_accessor *result, bool write,  node* (*allocate_node)(node_allocator_type& ,  const Key &, const T * ), node *tmp_n = 0  ) ;

struct accessor_not_used { void release(){}};
friend const_accessor* accessor_location( accessor_not_used const& ){ return NULL;}
friend const_accessor* accessor_location( const_accessor & a )      { return &a;}

friend bool is_write_access_needed( accessor const& )           { return true;}
friend bool is_write_access_needed( const_accessor const& )     { return false;}
friend bool is_write_access_needed( accessor_not_used const& )  { return false;}

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename Accessor>
bool generic_move_insert( Accessor && result, value_type && value ) {
result.release();
return lookup(true, value.first, &value.second, accessor_location(result), is_write_access_needed(result), &allocate_node_move_construct );
}

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
template<typename Accessor, typename... Args>
bool generic_emplace( Accessor && result, Args &&... args ) {
result.release();
node * node_ptr = allocate_node_emplace_construct(my_allocator, std::forward<Args>(args)...);
return lookup(true, node_ptr->item.first, NULL, accessor_location(result), is_write_access_needed(result), &do_not_allocate_node, node_ptr );
}
#endif 
#endif 

bool exclude( const_accessor &item_accessor );

template<typename I>
std::pair<I, I> internal_equal_range( const Key& key, I end ) const;

void internal_copy( const concurrent_hash_map& source );

template<typename I>
void internal_copy( I first, I last );


const_pointer internal_fast_find( const Key& key ) const {
hashcode_t h = my_hash_compare.hash( key );
hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
node *n;
restart:
__TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
bucket *b = get_bucket( h & m );
if( itt_load_word_with_acquire(b->node_list) == internal::rehash_req )
{
bucket::scoped_t lock;
if( lock.try_acquire( b->mutex, true ) ) {
if( b->node_list == internal::rehash_req)
const_cast<concurrent_hash_map*>(this)->rehash_bucket( b, h & m ); 
}
else lock.acquire( b->mutex, false );
__TBB_ASSERT(b->node_list!=internal::rehash_req,NULL);
}
n = search_bucket( key, b );
if( n )
return &n->item;
else if( check_mask_race( h, m ) )
goto restart;
return 0;
}
};

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::lookup( bool op_insert, const Key &key, const T *t, const_accessor *result, bool write, node* (*allocate_node)(node_allocator_type& , const Key&, const T*), node *tmp_n ) {
__TBB_ASSERT( !result || !result->my_node, NULL );
bool return_value;
hashcode_t const h = my_hash_compare.hash( key );
hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
segment_index_t grow_segment = 0;
node *n;
restart:
{
__TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
return_value = false;
bucket_accessor b( this, h & m );

n = search_bucket( key, b() );
if( op_insert ) {
if( !n ) {
if( !tmp_n ) {
tmp_n = allocate_node(my_allocator, key, t);
}
if( !b.is_writer() && !b.upgrade_to_writer() ) { 
n = search_bucket( key, b() );
if( is_valid(n) ) { 
b.downgrade_to_reader();
goto exists;
}
}
if( check_mask_race(h, m) )
goto restart; 
grow_segment = insert_new_node( b(), n = tmp_n, m );
tmp_n = 0;
return_value = true;
}
} else { 
if( !n ) {
if( check_mask_race( h, m ) )
goto restart; 
return false;
}
return_value = true;
}
exists:
if( !result ) goto check_growth;
if( !result->try_acquire( n->mutex, write ) ) {
for( tbb::internal::atomic_backoff backoff(true);; ) {
if( result->try_acquire( n->mutex, write ) ) break;
if( !backoff.bounded_pause() ) {
b.release();
__TBB_ASSERT( !op_insert || !return_value, "Can't acquire new item in locked bucket?" );
__TBB_Yield();
m = (hashcode_t) itt_load_word_with_acquire( my_mask );
goto restart;
}
}
}
}
result->my_node = n;
result->my_hash = h;
check_growth:
if( grow_segment ) {
#if __TBB_STATISTICS
my_info_resizes++; 
#endif
enable_segment( grow_segment );
}
if( tmp_n ) 
delete_node( tmp_n );
return return_value;
}

template<typename Key, typename T, typename HashCompare, typename A>
template<typename I>
std::pair<I, I> concurrent_hash_map<Key,T,HashCompare,A>::internal_equal_range( const Key& key, I end_ ) const {
hashcode_t h = my_hash_compare.hash( key );
hashcode_t m = my_mask;
__TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
h &= m;
bucket *b = get_bucket( h );
while( b->node_list == internal::rehash_req ) {
m = ( 1u<<__TBB_Log2( h ) ) - 1; 
b = get_bucket( h &= m );
}
node *n = search_bucket( key, b );
if( !n )
return std::make_pair(end_, end_);
iterator lower(*this, h, b, n), upper(lower);
return std::make_pair(lower, ++upper);
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::exclude( const_accessor &item_accessor ) {
__TBB_ASSERT( item_accessor.my_node, NULL );
node_base *const n = item_accessor.my_node;
hashcode_t const h = item_accessor.my_hash;
hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
do {
bucket_accessor b( this, h & m, true );
node_base **p = &b()->node_list;
while( *p && *p != n )
p = &(*p)->next;
if( !*p ) { 
if( check_mask_race( h, m ) )
continue;
item_accessor.release();
return false;
}
__TBB_ASSERT( *p == n, NULL );
*p = n->next; 
my_size--;
break;
} while(true);
if( !item_accessor.is_writer() ) 
item_accessor.upgrade_to_writer(); 
item_accessor.release();
delete_node( n ); 
return true;
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::erase( const Key &key ) {
node_base *n;
hashcode_t const h = my_hash_compare.hash( key );
hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
restart:
{
bucket_accessor b( this, h & m );
search:
node_base **p = &b()->node_list;
n = *p;
while( is_valid(n) && !my_hash_compare.equal(key, static_cast<node*>(n)->item.first ) ) {
p = &n->next;
n = *p;
}
if( !n ) { 
if( check_mask_race( h, m ) )
goto restart;
return false;
}
else if( !b.is_writer() && !b.upgrade_to_writer() ) {
if( check_mask_race( h, m ) ) 
goto restart;
goto search;
}
*p = n->next;
my_size--;
}
{
typename node::scoped_t item_locker( n->mutex, true );
}
delete_node( n ); 
return true;
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::swap(concurrent_hash_map<Key,T,HashCompare,A> &table) {
using std::swap;
swap(this->my_allocator, table.my_allocator);
swap(this->my_hash_compare, table.my_hash_compare);
internal_swap(table);
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::rehash(size_type sz) {
reserve( sz ); 
hashcode_t mask = my_mask;
hashcode_t b = (mask+1)>>1; 
__TBB_ASSERT((b&(b-1))==0, NULL); 
bucket *bp = get_bucket( b ); 
for(; b <= mask; b++, bp++ ) {
node_base *n = bp->node_list;
__TBB_ASSERT( is_valid(n) || n == internal::empty_rehashed || n == internal::rehash_req, "Broken internal structure" );
__TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during rehash() execution" );
if( n == internal::rehash_req ) { 
hashcode_t h = b; bucket *b_old = bp;
do {
__TBB_ASSERT( h > 1, "The lowermost buckets can't be rehashed" );
hashcode_t m = ( 1u<<__TBB_Log2( h ) ) - 1; 
b_old = get_bucket( h &= m );
} while( b_old->node_list == internal::rehash_req );
mark_rehashed_levels( h ); 
for( node_base **p = &b_old->node_list, *q = *p; is_valid(q); q = *p ) {
hashcode_t c = my_hash_compare.hash( static_cast<node*>(q)->item.first );
if( (c & mask) != h ) { 
*p = q->next; 
bucket *b_new = get_bucket( c & mask );
__TBB_ASSERT( b_new->node_list != internal::rehash_req, "hash() function changed for key in table or internal error" );
add_to_bucket( b_new, q );
} else p = &q->next; 
}
}
}
#if TBB_USE_PERFORMANCE_WARNINGS
int current_size = int(my_size), buckets = int(mask)+1, empty_buckets = 0, overpopulated_buckets = 0; 
static bool reported = false;
#endif
#if TBB_USE_ASSERT || TBB_USE_PERFORMANCE_WARNINGS
for( b = 0; b <= mask; b++ ) {
if( b & (b-2) ) ++bp; 
else bp = get_bucket( b );
node_base *n = bp->node_list;
__TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during rehash() execution" );
__TBB_ASSERT( is_valid(n) || n == internal::empty_rehashed, "Broken internal structure" );
#if TBB_USE_PERFORMANCE_WARNINGS
if( n == internal::empty_rehashed ) empty_buckets++;
else if( n->next ) overpopulated_buckets++;
#endif
#if TBB_USE_ASSERT
for( ; is_valid(n); n = n->next ) {
hashcode_t h = my_hash_compare.hash( static_cast<node*>(n)->item.first ) & mask;
__TBB_ASSERT( h == b, "hash() function changed for key in table or internal error" );
}
#endif
}
#endif 
#if TBB_USE_PERFORMANCE_WARNINGS
if( buckets > current_size) empty_buckets -= buckets - current_size;
else overpopulated_buckets -= current_size - buckets; 
if( !reported && buckets >= 512 && ( 2*empty_buckets > current_size || 2*overpopulated_buckets > current_size ) ) {
tbb::internal::runtime_warning(
"Performance is not optimal because the hash function produces bad randomness in lower bits in %s.\nSize: %d  Empties: %d  Overlaps: %d",
#if __TBB_USE_OPTIONAL_RTTI
typeid(*this).name(),
#else
"concurrent_hash_map",
#endif
current_size, empty_buckets, overpopulated_buckets );
reported = true;
}
#endif
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::clear() {
hashcode_t m = my_mask;
__TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
#if TBB_USE_ASSERT || TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
int current_size = int(my_size), buckets = int(m)+1, empty_buckets = 0, overpopulated_buckets = 0; 
static bool reported = false;
#endif
bucket *bp = 0;
for( segment_index_t b = 0; b <= m; b++ ) {
if( b & (b-2) ) ++bp; 
else bp = get_bucket( b );
node_base *n = bp->node_list;
__TBB_ASSERT( is_valid(n) || n == internal::empty_rehashed || n == internal::rehash_req, "Broken internal structure" );
__TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during clear() execution" );
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
if( n == internal::empty_rehashed ) empty_buckets++;
else if( n == internal::rehash_req ) buckets--;
else if( n->next ) overpopulated_buckets++;
#endif
#if __TBB_EXTRA_DEBUG
for(; is_valid(n); n = n->next ) {
hashcode_t h = my_hash_compare.hash( static_cast<node*>(n)->item.first );
h &= m;
__TBB_ASSERT( h == b || get_bucket(h)->node_list == internal::rehash_req, "hash() function changed for key in table or internal error" );
}
#endif
}
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
#if __TBB_STATISTICS
printf( "items=%d buckets: capacity=%d rehashed=%d empty=%d overpopulated=%d"
" concurrent: resizes=%u rehashes=%u restarts=%u\n",
current_size, int(m+1), buckets, empty_buckets, overpopulated_buckets,
unsigned(my_info_resizes), unsigned(my_info_rehashes), unsigned(my_info_restarts) );
my_info_resizes = 0; 
my_info_restarts = 0; 
my_info_rehashes = 0;  
#endif
if( buckets > current_size) empty_buckets -= buckets - current_size;
else overpopulated_buckets -= current_size - buckets; 
if( !reported && buckets >= 512 && ( 2*empty_buckets > current_size || 2*overpopulated_buckets > current_size ) ) {
tbb::internal::runtime_warning(
"Performance is not optimal because the hash function produces bad randomness in lower bits in %s.\nSize: %d  Empties: %d  Overlaps: %d",
#if __TBB_USE_OPTIONAL_RTTI
typeid(*this).name(),
#else
"concurrent_hash_map",
#endif
current_size, empty_buckets, overpopulated_buckets );
reported = true;
}
#endif
#endif
my_size = 0;
segment_index_t s = segment_index_of( m );
__TBB_ASSERT( s+1 == pointers_per_table || !my_table[s+1], "wrong mask or concurrent grow" );
cache_aligned_allocator<bucket> alloc;
do {
__TBB_ASSERT( is_valid( my_table[s] ), "wrong mask or concurrent grow" );
segment_ptr_t buckets_ptr = my_table[s];
size_type sz = segment_size( s ? s : 1 );
for( segment_index_t i = 0; i < sz; i++ )
for( node_base *n = buckets_ptr[i].node_list; is_valid(n); n = buckets_ptr[i].node_list ) {
buckets_ptr[i].node_list = n->next;
delete_node( n );
}
if( s >= first_block) 
alloc.deallocate( buckets_ptr, sz );
else if( s == embedded_block && embedded_block != first_block )
alloc.deallocate( buckets_ptr, segment_size(first_block)-embedded_buckets );
if( s >= embedded_block ) my_table[s] = 0;
} while(s-- > 0);
my_mask = embedded_buckets - 1;
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::internal_copy( const concurrent_hash_map& source ) {
reserve( source.my_size ); 
hashcode_t mask = source.my_mask;
if( my_mask == mask ) { 
bucket *dst = 0, *src = 0;
bool rehash_required = false;
for( hashcode_t k = 0; k <= mask; k++ ) {
if( k & (k-2) ) ++dst,src++; 
else { dst = get_bucket( k ); src = source.get_bucket( k ); }
__TBB_ASSERT( dst->node_list != internal::rehash_req, "Invalid bucket in destination table");
node *n = static_cast<node*>( src->node_list );
if( n == internal::rehash_req ) { 
rehash_required = true;
dst->node_list = internal::rehash_req;
} else for(; n; n = static_cast<node*>( n->next ) ) {
add_to_bucket( dst, new( my_allocator ) node(n->item.first, n->item.second) );
++my_size; 
}
}
if( rehash_required ) rehash();
} else internal_copy( source.begin(), source.end() );
}

template<typename Key, typename T, typename HashCompare, typename A>
template<typename I>
void concurrent_hash_map<Key,T,HashCompare,A>::internal_copy(I first, I last) {
hashcode_t m = my_mask;
for(; first != last; ++first) {
hashcode_t h = my_hash_compare.hash( (*first).first );
bucket *b = get_bucket( h & m );
__TBB_ASSERT( b->node_list != internal::rehash_req, "Invalid bucket in destination table");
node *n = new( my_allocator ) node(*first);
add_to_bucket( b, n );
++my_size; 
}
}

} 

using interface5::concurrent_hash_map;


template<typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator==(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b) {
if(a.size() != b.size()) return false;
typename concurrent_hash_map<Key, T, HashCompare, A1>::const_iterator i(a.begin()), i_end(a.end());
typename concurrent_hash_map<Key, T, HashCompare, A2>::const_iterator j, j_end(b.end());
for(; i != i_end; ++i) {
j = b.equal_range(i->first).first;
if( j == j_end || !(i->second == j->second) ) return false;
}
return true;
}

template<typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator!=(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b)
{    return !(a == b); }

template<typename Key, typename T, typename HashCompare, typename A>
inline void swap(concurrent_hash_map<Key, T, HashCompare, A> &a, concurrent_hash_map<Key, T, HashCompare, A> &b)
{    a.swap( b ); }

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif 

} 

#endif 
