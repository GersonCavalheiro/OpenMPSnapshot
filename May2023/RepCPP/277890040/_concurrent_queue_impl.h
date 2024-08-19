

#ifndef __TBB__concurrent_queue_impl_H
#define __TBB__concurrent_queue_impl_H

#ifndef __TBB_concurrent_queue_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#include "../tbb_stddef.h"
#include "../tbb_machine.h"
#include "../atomic.h"
#include "../spin_mutex.h"
#include "../cache_aligned_allocator.h"
#include "../tbb_exception.h"
#include "../tbb_profiling.h"
#include <new>
#include __TBB_STD_SWAP_HEADER
#include <iterator>

namespace tbb {

#if !__TBB_TEMPLATE_FRIENDS_BROKEN

namespace strict_ppl {
template<typename T, typename A> class concurrent_queue;
}

template<typename T, typename A> class concurrent_bounded_queue;

#endif

namespace strict_ppl {

namespace internal {

using namespace tbb::internal;

typedef size_t ticket;

template<typename T> class micro_queue ;
template<typename T> class micro_queue_pop_finalizer ;
template<typename T> class concurrent_queue_base_v3;
template<typename T> struct concurrent_queue_rep;


struct concurrent_queue_rep_base : no_copy {
template<typename T> friend class micro_queue;
template<typename T> friend class concurrent_queue_base_v3;

protected:
static const size_t phi = 3;

public:
static const size_t n_queue = 8;

struct page {
page* next;
uintptr_t mask;
};

atomic<ticket> head_counter;
char pad1[NFS_MaxLineSize-sizeof(atomic<ticket>)];
atomic<ticket> tail_counter;
char pad2[NFS_MaxLineSize-sizeof(atomic<ticket>)];

size_t items_per_page;

size_t item_size;

atomic<size_t> n_invalid_entries;

char pad3[NFS_MaxLineSize-sizeof(size_t)-sizeof(size_t)-sizeof(atomic<size_t>)];
} ;

inline bool is_valid_page(const concurrent_queue_rep_base::page* p) {
return uintptr_t(p)>1;
}


class concurrent_queue_page_allocator
{
template<typename T> friend class micro_queue ;
template<typename T> friend class micro_queue_pop_finalizer ;
protected:
virtual ~concurrent_queue_page_allocator() {}
private:
virtual concurrent_queue_rep_base::page* allocate_page() = 0;
virtual void deallocate_page( concurrent_queue_rep_base::page* p ) = 0;
} ;

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4146 )
#endif


template<typename T>
class micro_queue : no_copy {
public:
typedef void (*item_constructor_t)(T* location, const void* src);
private:
typedef concurrent_queue_rep_base::page page;

class destroyer: no_copy {
T& my_value;
public:
destroyer( T& value ) : my_value(value) {}
~destroyer() {my_value.~T();}
};

void copy_item( page& dst, size_t dindex, const void* src, item_constructor_t construct_item ) {
construct_item( &get_ref(dst, dindex), src );
}

void copy_item( page& dst, size_t dindex, const page& src, size_t sindex,
item_constructor_t construct_item )
{
T& src_item = get_ref( const_cast<page&>(src), sindex );
construct_item( &get_ref(dst, dindex), static_cast<const void*>(&src_item) );
}

void assign_and_destroy_item( void* dst, page& src, size_t index ) {
T& from = get_ref(src,index);
destroyer d(from);
*static_cast<T*>(dst) = tbb::internal::move( from );
}

void spin_wait_until_my_turn( atomic<ticket>& counter, ticket k, concurrent_queue_rep_base& rb ) const ;

public:
friend class micro_queue_pop_finalizer<T>;

struct padded_page: page {
padded_page();
void operator=( const padded_page& );
T last;
};

static T& get_ref( page& p, size_t index ) {
return (&static_cast<padded_page*>(static_cast<void*>(&p))->last)[index];
}

atomic<page*> head_page;
atomic<ticket> head_counter;

atomic<page*> tail_page;
atomic<ticket> tail_counter;

spin_mutex page_mutex;

void push( const void* item, ticket k, concurrent_queue_base_v3<T>& base,
item_constructor_t construct_item ) ;

bool pop( void* dst, ticket k, concurrent_queue_base_v3<T>& base ) ;

micro_queue& assign( const micro_queue& src, concurrent_queue_base_v3<T>& base,
item_constructor_t construct_item ) ;

page* make_copy( concurrent_queue_base_v3<T>& base, const page* src_page, size_t begin_in_page,
size_t end_in_page, ticket& g_index, item_constructor_t construct_item ) ;

void invalidate_page_and_rethrow( ticket k ) ;
};

template<typename T>
void micro_queue<T>::spin_wait_until_my_turn( atomic<ticket>& counter, ticket k, concurrent_queue_rep_base& rb ) const {
for( atomic_backoff b(true);;b.pause() ) {
ticket c = counter;
if( c==k ) return;
else if( c&1 ) {
++rb.n_invalid_entries;
throw_exception( eid_bad_last_alloc );
}
}
}

template<typename T>
void micro_queue<T>::push( const void* item, ticket k, concurrent_queue_base_v3<T>& base,
item_constructor_t construct_item )
{
k &= -concurrent_queue_rep_base::n_queue;
page* p = NULL;
size_t index = modulo_power_of_two( k/concurrent_queue_rep_base::n_queue, base.my_rep->items_per_page);
if( !index ) {
__TBB_TRY {
concurrent_queue_page_allocator& pa = base;
p = pa.allocate_page();
} __TBB_CATCH (...) {
++base.my_rep->n_invalid_entries;
invalidate_page_and_rethrow( k );
}
p->mask = 0;
p->next = NULL;
}

if( tail_counter != k ) spin_wait_until_my_turn( tail_counter, k, *base.my_rep );
call_itt_notify(acquired, &tail_counter);

if( p ) {
spin_mutex::scoped_lock lock( page_mutex );
page* q = tail_page;
if( is_valid_page(q) )
q->next = p;
else
head_page = p;
tail_page = p;
} else {
p = tail_page;
}

__TBB_TRY {
copy_item( *p, index, item, construct_item );
itt_hide_store_word(p->mask,  p->mask | uintptr_t(1)<<index);
call_itt_notify(releasing, &tail_counter);
tail_counter += concurrent_queue_rep_base::n_queue;
} __TBB_CATCH (...) {
++base.my_rep->n_invalid_entries;
call_itt_notify(releasing, &tail_counter);
tail_counter += concurrent_queue_rep_base::n_queue;
__TBB_RETHROW();
}
}

template<typename T>
bool micro_queue<T>::pop( void* dst, ticket k, concurrent_queue_base_v3<T>& base ) {
k &= -concurrent_queue_rep_base::n_queue;
if( head_counter!=k ) spin_wait_until_eq( head_counter, k );
call_itt_notify(acquired, &head_counter);
if( tail_counter==k ) spin_wait_while_eq( tail_counter, k );
call_itt_notify(acquired, &tail_counter);
page *p = head_page;
__TBB_ASSERT( p, NULL );
size_t index = modulo_power_of_two( k/concurrent_queue_rep_base::n_queue, base.my_rep->items_per_page );
bool success = false;
{
micro_queue_pop_finalizer<T> finalizer( *this, base, k+concurrent_queue_rep_base::n_queue, index==base.my_rep->items_per_page-1 ? p : NULL );
if( p->mask & uintptr_t(1)<<index ) {
success = true;
assign_and_destroy_item( dst, *p, index );
} else {
--base.my_rep->n_invalid_entries;
}
}
return success;
}

template<typename T>
micro_queue<T>& micro_queue<T>::assign( const micro_queue<T>& src, concurrent_queue_base_v3<T>& base,
item_constructor_t construct_item )
{
head_counter = src.head_counter;
tail_counter = src.tail_counter;

const page* srcp = src.head_page;
if( is_valid_page(srcp) ) {
ticket g_index = head_counter;
__TBB_TRY {
size_t n_items  = (tail_counter-head_counter)/concurrent_queue_rep_base::n_queue;
size_t index = modulo_power_of_two( head_counter/concurrent_queue_rep_base::n_queue, base.my_rep->items_per_page );
size_t end_in_first_page = (index+n_items<base.my_rep->items_per_page)?(index+n_items):base.my_rep->items_per_page;

head_page = make_copy( base, srcp, index, end_in_first_page, g_index, construct_item );
page* cur_page = head_page;

if( srcp != src.tail_page ) {
for( srcp = srcp->next; srcp!=src.tail_page; srcp=srcp->next ) {
cur_page->next = make_copy( base, srcp, 0, base.my_rep->items_per_page, g_index, construct_item );
cur_page = cur_page->next;
}

__TBB_ASSERT( srcp==src.tail_page, NULL );
size_t last_index = modulo_power_of_two( tail_counter/concurrent_queue_rep_base::n_queue, base.my_rep->items_per_page );
if( last_index==0 ) last_index = base.my_rep->items_per_page;

cur_page->next = make_copy( base, srcp, 0, last_index, g_index, construct_item );
cur_page = cur_page->next;
}
tail_page = cur_page;
} __TBB_CATCH (...) {
invalidate_page_and_rethrow( g_index );
}
} else {
head_page = tail_page = NULL;
}
return *this;
}

template<typename T>
void micro_queue<T>::invalidate_page_and_rethrow( ticket k ) {
page* invalid_page = (page*)uintptr_t(1);
{
spin_mutex::scoped_lock lock( page_mutex );
itt_store_word_with_release(tail_counter, k+concurrent_queue_rep_base::n_queue+1);
page* q = tail_page;
if( is_valid_page(q) )
q->next = invalid_page;
else
head_page = invalid_page;
tail_page = invalid_page;
}
__TBB_RETHROW();
}

template<typename T>
concurrent_queue_rep_base::page* micro_queue<T>::make_copy( concurrent_queue_base_v3<T>& base,
const concurrent_queue_rep_base::page* src_page, size_t begin_in_page, size_t end_in_page,
ticket& g_index, item_constructor_t construct_item )
{
concurrent_queue_page_allocator& pa = base;
page* new_page = pa.allocate_page();
new_page->next = NULL;
new_page->mask = src_page->mask;
for( ; begin_in_page!=end_in_page; ++begin_in_page, ++g_index )
if( new_page->mask & uintptr_t(1)<<begin_in_page )
copy_item( *new_page, begin_in_page, *src_page, begin_in_page, construct_item );
return new_page;
}

template<typename T>
class micro_queue_pop_finalizer: no_copy {
typedef concurrent_queue_rep_base::page page;
ticket my_ticket;
micro_queue<T>& my_queue;
page* my_page;
concurrent_queue_page_allocator& allocator;
public:
micro_queue_pop_finalizer( micro_queue<T>& queue, concurrent_queue_base_v3<T>& b, ticket k, page* p ) :
my_ticket(k), my_queue(queue), my_page(p), allocator(b)
{}
~micro_queue_pop_finalizer() ;
};

template<typename T>
micro_queue_pop_finalizer<T>::~micro_queue_pop_finalizer() {
page* p = my_page;
if( is_valid_page(p) ) {
spin_mutex::scoped_lock lock( my_queue.page_mutex );
page* q = p->next;
my_queue.head_page = q;
if( !is_valid_page(q) ) {
my_queue.tail_page = NULL;
}
}
itt_store_word_with_release(my_queue.head_counter, my_ticket);
if( is_valid_page(p) ) {
allocator.deallocate_page( p );
}
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif 

template<typename T> class concurrent_queue_iterator_rep ;
template<typename T> class concurrent_queue_iterator_base_v3;


template<typename T>
struct concurrent_queue_rep : public concurrent_queue_rep_base {
micro_queue<T> array[n_queue];

static size_t index( ticket k ) {
return k*phi%n_queue;
}

micro_queue<T>& choose( ticket k ) {
return array[index(k)];
}
};


template<typename T>
class concurrent_queue_base_v3: public concurrent_queue_page_allocator {
private:
concurrent_queue_rep<T>* my_rep;

friend struct concurrent_queue_rep<T>;
friend class micro_queue<T>;
friend class concurrent_queue_iterator_rep<T>;
friend class concurrent_queue_iterator_base_v3<T>;

protected:
typedef typename concurrent_queue_rep<T>::page page;

private:
typedef typename micro_queue<T>::padded_page padded_page;
typedef typename micro_queue<T>::item_constructor_t item_constructor_t;

virtual page *allocate_page() __TBB_override {
concurrent_queue_rep<T>& r = *my_rep;
size_t n = sizeof(padded_page) + (r.items_per_page-1)*sizeof(T);
return reinterpret_cast<page*>(allocate_block ( n ));
}

virtual void deallocate_page( concurrent_queue_rep_base::page *p ) __TBB_override {
concurrent_queue_rep<T>& r = *my_rep;
size_t n = sizeof(padded_page) + (r.items_per_page-1)*sizeof(T);
deallocate_block( reinterpret_cast<void*>(p), n );
}

virtual void *allocate_block( size_t n ) = 0;

virtual void deallocate_block( void *p, size_t n ) = 0;

protected:
concurrent_queue_base_v3();

virtual ~concurrent_queue_base_v3() {
#if TBB_USE_ASSERT
size_t nq = my_rep->n_queue;
for( size_t i=0; i<nq; i++ )
__TBB_ASSERT( my_rep->array[i].tail_page==NULL, "pages were not freed properly" );
#endif 
cache_aligned_allocator<concurrent_queue_rep<T> >().deallocate(my_rep,1);
}

void internal_push( const void* src, item_constructor_t construct_item ) {
concurrent_queue_rep<T>& r = *my_rep;
ticket k = r.tail_counter++;
r.choose(k).push( src, k, *this, construct_item );
}


bool internal_try_pop( void* dst ) ;

size_t internal_size() const ;

bool internal_empty() const ;


void internal_finish_clear() ;

void internal_throw_exception() const {
throw_exception( eid_bad_alloc );
}

void assign( const concurrent_queue_base_v3& src, item_constructor_t construct_item ) ;

#if __TBB_CPP11_RVALUE_REF_PRESENT
void internal_swap( concurrent_queue_base_v3& src ) {
std::swap( my_rep, src.my_rep );
}
#endif 
};

template<typename T>
concurrent_queue_base_v3<T>::concurrent_queue_base_v3() {
const size_t item_size = sizeof(T);
my_rep = cache_aligned_allocator<concurrent_queue_rep<T> >().allocate(1);
__TBB_ASSERT( (size_t)my_rep % NFS_GetLineSize()==0, "alignment error" );
__TBB_ASSERT( (size_t)&my_rep->head_counter % NFS_GetLineSize()==0, "alignment error" );
__TBB_ASSERT( (size_t)&my_rep->tail_counter % NFS_GetLineSize()==0, "alignment error" );
__TBB_ASSERT( (size_t)&my_rep->array % NFS_GetLineSize()==0, "alignment error" );
memset(static_cast<void*>(my_rep),0,sizeof(concurrent_queue_rep<T>));
my_rep->item_size = item_size;
my_rep->items_per_page = item_size<=  8 ? 32 :
item_size<= 16 ? 16 :
item_size<= 32 ?  8 :
item_size<= 64 ?  4 :
item_size<=128 ?  2 :
1;
}

template<typename T>
bool concurrent_queue_base_v3<T>::internal_try_pop( void* dst ) {
concurrent_queue_rep<T>& r = *my_rep;
ticket k;
do {
k = r.head_counter;
for(;;) {
if( (ptrdiff_t)(r.tail_counter-k)<=0 ) {
return false;
}
ticket tk=k;
#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (push)
#pragma warning (disable: 4267)
#endif
k = r.head_counter.compare_and_swap( tk+1, tk );
#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (pop)
#endif
if( k==tk )
break;
}
} while( !r.choose( k ).pop( dst, k, *this ) );
return true;
}

template<typename T>
size_t concurrent_queue_base_v3<T>::internal_size() const {
concurrent_queue_rep<T>& r = *my_rep;
__TBB_ASSERT( sizeof(ptrdiff_t)<=sizeof(size_t), NULL );
ticket hc = r.head_counter;
size_t nie = r.n_invalid_entries;
ticket tc = r.tail_counter;
__TBB_ASSERT( hc!=tc || !nie, NULL );
ptrdiff_t sz = tc-hc-nie;
return sz<0 ? 0 :  size_t(sz);
}

template<typename T>
bool concurrent_queue_base_v3<T>::internal_empty() const {
concurrent_queue_rep<T>& r = *my_rep;
ticket tc = r.tail_counter;
ticket hc = r.head_counter;
return tc==r.tail_counter && tc==hc+r.n_invalid_entries ;
}

template<typename T>
void concurrent_queue_base_v3<T>::internal_finish_clear() {
concurrent_queue_rep<T>& r = *my_rep;
size_t nq = r.n_queue;
for( size_t i=0; i<nq; ++i ) {
page* tp = r.array[i].tail_page;
if( is_valid_page(tp) ) {
__TBB_ASSERT( r.array[i].head_page==tp, "at most one page should remain" );
deallocate_page( tp );
r.array[i].tail_page = NULL;
} else
__TBB_ASSERT( !is_valid_page(r.array[i].head_page), "head page pointer corrupt?" );
}
}

template<typename T>
void concurrent_queue_base_v3<T>::assign( const concurrent_queue_base_v3& src,
item_constructor_t construct_item )
{
concurrent_queue_rep<T>& r = *my_rep;
r.items_per_page = src.my_rep->items_per_page;

r.head_counter = src.my_rep->head_counter;
r.tail_counter = src.my_rep->tail_counter;
r.n_invalid_entries = src.my_rep->n_invalid_entries;

for( size_t i = 0; i < r.n_queue; ++i )
r.array[i].assign( src.my_rep->array[i], *this, construct_item);

__TBB_ASSERT( r.head_counter==src.my_rep->head_counter && r.tail_counter==src.my_rep->tail_counter,
"the source concurrent queue should not be concurrently modified." );
}

template<typename Container, typename Value> class concurrent_queue_iterator;

template<typename T>
class concurrent_queue_iterator_rep: no_assign {
typedef typename micro_queue<T>::padded_page padded_page;
public:
ticket head_counter;
const concurrent_queue_base_v3<T>& my_queue;
typename concurrent_queue_base_v3<T>::page* array[concurrent_queue_rep<T>::n_queue];
concurrent_queue_iterator_rep( const concurrent_queue_base_v3<T>& queue ) :
head_counter(queue.my_rep->head_counter),
my_queue(queue)
{
for( size_t k=0; k<concurrent_queue_rep<T>::n_queue; ++k )
array[k] = queue.my_rep->array[k].head_page;
}

bool get_item( T*& item, size_t k ) ;
};

template<typename T>
bool concurrent_queue_iterator_rep<T>::get_item( T*& item, size_t k ) {
if( k==my_queue.my_rep->tail_counter ) {
item = NULL;
return true;
} else {
typename concurrent_queue_base_v3<T>::page* p = array[concurrent_queue_rep<T>::index(k)];
__TBB_ASSERT(p,NULL);
size_t i = modulo_power_of_two( k/concurrent_queue_rep<T>::n_queue, my_queue.my_rep->items_per_page );
item = &micro_queue<T>::get_ref(*p,i);
return (p->mask & uintptr_t(1)<<i)!=0;
}
}


template<typename Value>
class concurrent_queue_iterator_base_v3 {

concurrent_queue_iterator_rep<Value>* my_rep;

template<typename C, typename T, typename U>
friend bool operator==( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j );

template<typename C, typename T, typename U>
friend bool operator!=( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j );
protected:
Value* my_item;

concurrent_queue_iterator_base_v3() : my_rep(NULL), my_item(NULL) {
#if __TBB_GCC_OPTIMIZER_ORDERING_BROKEN
__TBB_compiler_fence();
#endif
}

concurrent_queue_iterator_base_v3( const concurrent_queue_iterator_base_v3& i )
: my_rep(NULL), my_item(NULL) {
assign(i);
}

concurrent_queue_iterator_base_v3& operator=( const concurrent_queue_iterator_base_v3& i ) {
assign(i);
return *this;
}

concurrent_queue_iterator_base_v3( const concurrent_queue_base_v3<Value>& queue ) ;

void assign( const concurrent_queue_iterator_base_v3<Value>& other ) ;

void advance() ;

~concurrent_queue_iterator_base_v3() {
cache_aligned_allocator<concurrent_queue_iterator_rep<Value> >().deallocate(my_rep, 1);
my_rep = NULL;
}
};

template<typename Value>
concurrent_queue_iterator_base_v3<Value>::concurrent_queue_iterator_base_v3( const concurrent_queue_base_v3<Value>& queue ) {
my_rep = cache_aligned_allocator<concurrent_queue_iterator_rep<Value> >().allocate(1);
new( my_rep ) concurrent_queue_iterator_rep<Value>(queue);
size_t k = my_rep->head_counter;
if( !my_rep->get_item(my_item, k) ) advance();
}

template<typename Value>
void concurrent_queue_iterator_base_v3<Value>::assign( const concurrent_queue_iterator_base_v3<Value>& other ) {
if( my_rep!=other.my_rep ) {
if( my_rep ) {
cache_aligned_allocator<concurrent_queue_iterator_rep<Value> >().deallocate(my_rep, 1);
my_rep = NULL;
}
if( other.my_rep ) {
my_rep = cache_aligned_allocator<concurrent_queue_iterator_rep<Value> >().allocate(1);
new( my_rep ) concurrent_queue_iterator_rep<Value>( *other.my_rep );
}
}
my_item = other.my_item;
}

template<typename Value>
void concurrent_queue_iterator_base_v3<Value>::advance() {
__TBB_ASSERT( my_item, "attempt to increment iterator past end of queue" );
size_t k = my_rep->head_counter;
const concurrent_queue_base_v3<Value>& queue = my_rep->my_queue;
#if TBB_USE_ASSERT
Value* tmp;
my_rep->get_item(tmp,k);
__TBB_ASSERT( my_item==tmp, NULL );
#endif 
size_t i = modulo_power_of_two( k/concurrent_queue_rep<Value>::n_queue, queue.my_rep->items_per_page );
if( i==queue.my_rep->items_per_page-1 ) {
typename concurrent_queue_base_v3<Value>::page*& root = my_rep->array[concurrent_queue_rep<Value>::index(k)];
root = root->next;
}
my_rep->head_counter = ++k;
if( !my_rep->get_item(my_item, k) ) advance();
}


template<typename T> struct tbb_remove_cv {typedef T type;};
template<typename T> struct tbb_remove_cv<const T> {typedef T type;};
template<typename T> struct tbb_remove_cv<volatile T> {typedef T type;};
template<typename T> struct tbb_remove_cv<const volatile T> {typedef T type;};


template<typename Container, typename Value>
class concurrent_queue_iterator: public concurrent_queue_iterator_base_v3<typename tbb_remove_cv<Value>::type>,
public std::iterator<std::forward_iterator_tag,Value> {
#if !__TBB_TEMPLATE_FRIENDS_BROKEN
template<typename T, class A>
friend class ::tbb::strict_ppl::concurrent_queue;
#else
public:
#endif
explicit concurrent_queue_iterator( const concurrent_queue_base_v3<typename tbb_remove_cv<Value>::type>& queue ) :
concurrent_queue_iterator_base_v3<typename tbb_remove_cv<Value>::type>(queue)
{
}

public:
concurrent_queue_iterator() {}


concurrent_queue_iterator( const concurrent_queue_iterator<Container,typename Container::value_type>& other ) :
concurrent_queue_iterator_base_v3<typename tbb_remove_cv<Value>::type>(other)
{}

concurrent_queue_iterator& operator=( const concurrent_queue_iterator<Container,typename Container::value_type>& other ) {
concurrent_queue_iterator_base_v3<typename tbb_remove_cv<Value>::type>::operator=(other);
return *this;
}

Value& operator*() const {
return *static_cast<Value*>(this->my_item);
}

Value* operator->() const {return &operator*();}

concurrent_queue_iterator& operator++() {
this->advance();
return *this;
}

Value* operator++(int) {
Value* result = &operator*();
operator++();
return result;
}
}; 


template<typename C, typename T, typename U>
bool operator==( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j ) {
return i.my_item==j.my_item;
}

template<typename C, typename T, typename U>
bool operator!=( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j ) {
return i.my_item!=j.my_item;
}

} 


} 

namespace internal {

class concurrent_queue_rep;
class concurrent_queue_iterator_rep;
class concurrent_queue_iterator_base_v3;
template<typename Container, typename Value> class concurrent_queue_iterator;


class concurrent_queue_base_v3: no_copy {
private:
concurrent_queue_rep* my_rep;

friend class concurrent_queue_rep;
friend struct micro_queue;
friend class micro_queue_pop_finalizer;
friend class concurrent_queue_iterator_rep;
friend class concurrent_queue_iterator_base_v3;
protected:
struct page {
page* next;
uintptr_t mask;
};

ptrdiff_t my_capacity;

size_t items_per_page;

size_t item_size;

enum copy_specifics { copy, move };

#if __TBB_PROTECTED_NESTED_CLASS_BROKEN
public:
#endif
template<typename T>
struct padded_page: page {
padded_page();
void operator=( const padded_page& );
T last;
};

private:
virtual void copy_item( page& dst, size_t index, const void* src ) = 0;
virtual void assign_and_destroy_item( void* dst, page& src, size_t index ) = 0;
protected:
__TBB_EXPORTED_METHOD concurrent_queue_base_v3( size_t item_size );
virtual __TBB_EXPORTED_METHOD ~concurrent_queue_base_v3();

void __TBB_EXPORTED_METHOD internal_push( const void* src );

void __TBB_EXPORTED_METHOD internal_pop( void* dst );

void __TBB_EXPORTED_METHOD internal_abort();

bool __TBB_EXPORTED_METHOD internal_push_if_not_full( const void* src );


bool __TBB_EXPORTED_METHOD internal_pop_if_present( void* dst );

ptrdiff_t __TBB_EXPORTED_METHOD internal_size() const;

bool __TBB_EXPORTED_METHOD internal_empty() const;

void __TBB_EXPORTED_METHOD internal_set_capacity( ptrdiff_t capacity, size_t element_size );

virtual page *allocate_page() = 0;

virtual void deallocate_page( page *p ) = 0;


void __TBB_EXPORTED_METHOD internal_finish_clear() ;

void __TBB_EXPORTED_METHOD internal_throw_exception() const;

void __TBB_EXPORTED_METHOD assign( const concurrent_queue_base_v3& src ) ;

#if __TBB_CPP11_RVALUE_REF_PRESENT
void internal_swap( concurrent_queue_base_v3& src ) {
std::swap( my_capacity, src.my_capacity );
std::swap( items_per_page, src.items_per_page );
std::swap( item_size, src.item_size );
std::swap( my_rep, src.my_rep );
}
#endif 

void internal_insert_item( const void* src, copy_specifics op_type );

bool internal_insert_if_not_full( const void* src, copy_specifics op_type );

void internal_assign( const concurrent_queue_base_v3& src, copy_specifics op_type );
private:
virtual void copy_page_item( page& dst, size_t dindex, const page& src, size_t sindex ) = 0;
};


class concurrent_queue_base_v8: public concurrent_queue_base_v3 {
protected:
concurrent_queue_base_v8( size_t item_sz ) : concurrent_queue_base_v3( item_sz ) {}

void __TBB_EXPORTED_METHOD move_content( concurrent_queue_base_v8& src ) ;

bool __TBB_EXPORTED_METHOD internal_push_move_if_not_full( const void* src );

void __TBB_EXPORTED_METHOD internal_push_move( const void* src );
private:
friend struct micro_queue;
virtual void move_page_item( page& dst, size_t dindex, const page& src, size_t sindex ) = 0;
virtual void move_item( page& dst, size_t index, const void* src ) = 0;
};


class concurrent_queue_iterator_base_v3 {

concurrent_queue_iterator_rep* my_rep;

template<typename C, typename T, typename U>
friend bool operator==( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j );

template<typename C, typename T, typename U>
friend bool operator!=( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j );

void initialize( const concurrent_queue_base_v3& queue, size_t offset_of_data );
protected:
void* my_item;

concurrent_queue_iterator_base_v3() : my_rep(NULL), my_item(NULL) {}

concurrent_queue_iterator_base_v3( const concurrent_queue_iterator_base_v3& i ) : my_rep(NULL), my_item(NULL) {
assign(i);
}

concurrent_queue_iterator_base_v3& operator=( const concurrent_queue_iterator_base_v3& i ) {
assign(i);
return *this;
}


__TBB_EXPORTED_METHOD concurrent_queue_iterator_base_v3( const concurrent_queue_base_v3& queue );

__TBB_EXPORTED_METHOD concurrent_queue_iterator_base_v3( const concurrent_queue_base_v3& queue, size_t offset_of_data );

void __TBB_EXPORTED_METHOD assign( const concurrent_queue_iterator_base_v3& i );

void __TBB_EXPORTED_METHOD advance();

__TBB_EXPORTED_METHOD ~concurrent_queue_iterator_base_v3();
};

typedef concurrent_queue_iterator_base_v3 concurrent_queue_iterator_base;


template<typename Container, typename Value>
class concurrent_queue_iterator: public concurrent_queue_iterator_base,
public std::iterator<std::forward_iterator_tag,Value> {

#if !__TBB_TEMPLATE_FRIENDS_BROKEN
template<typename T, class A>
friend class ::tbb::concurrent_bounded_queue;
#else
public:
#endif

explicit concurrent_queue_iterator( const concurrent_queue_base_v3& queue ) :
concurrent_queue_iterator_base_v3(queue,__TBB_offsetof(concurrent_queue_base_v3::padded_page<Value>,last))
{
}

public:
concurrent_queue_iterator() {}


concurrent_queue_iterator( const concurrent_queue_iterator<Container,typename Container::value_type>& other ) :
concurrent_queue_iterator_base_v3(other)
{}

concurrent_queue_iterator& operator=( const concurrent_queue_iterator<Container,typename Container::value_type>& other ) {
concurrent_queue_iterator_base_v3::operator=(other);
return *this;
}

Value& operator*() const {
return *static_cast<Value*>(my_item);
}

Value* operator->() const {return &operator*();}

concurrent_queue_iterator& operator++() {
advance();
return *this;
}

Value* operator++(int) {
Value* result = &operator*();
operator++();
return result;
}
}; 


template<typename C, typename T, typename U>
bool operator==( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j ) {
return i.my_item==j.my_item;
}

template<typename C, typename T, typename U>
bool operator!=( const concurrent_queue_iterator<C,T>& i, const concurrent_queue_iterator<C,U>& j ) {
return i.my_item!=j.my_item;
}

} 


} 

#endif 
