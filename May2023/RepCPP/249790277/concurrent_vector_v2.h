

#ifndef __TBB_concurrent_vector_H
#define __TBB_concurrent_vector_H

#include "tbb/tbb_stddef.h"
#include "tbb/atomic.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/blocked_range.h"
#include "tbb/tbb_machine.h"
#include <new>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <iterator>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

namespace tbb {

template<typename T>
class concurrent_vector;

namespace internal {


class concurrent_vector_base {
protected:

typedef unsigned long segment_index_t;
typedef size_t size_type;

static const int lg_min_segment_size = 4;

static const int min_segment_size = segment_index_t(1)<<lg_min_segment_size;

static segment_index_t segment_index_of( size_t index ) {
uintptr_t i = index|1<<(lg_min_segment_size-1);
uintptr_t j = __TBB_Log2(i);
return segment_index_t(j-(lg_min_segment_size-1));
}

static segment_index_t segment_base( segment_index_t k ) {
return min_segment_size>>1<<k & -min_segment_size;
}

static segment_index_t segment_size( segment_index_t k ) {
segment_index_t result = k==0 ? min_segment_size : min_segment_size/2<<k;
__TBB_ASSERT( result==segment_base(k+1)-segment_base(k), NULL );
return result;
}

void __TBB_EXPORTED_METHOD internal_reserve( size_type n, size_type element_size, size_type max_size );

size_type __TBB_EXPORTED_METHOD internal_capacity() const;

atomic<size_type> my_early_size;


struct segment_t {

void* volatile array;
#if TBB_USE_ASSERT
~segment_t() {
__TBB_ASSERT( !array, "should have been set to NULL by clear" );
}
#endif 
};


atomic<segment_t*> my_segment;

segment_t my_storage[2];


concurrent_vector_base() {
my_early_size = 0;
my_storage[0].array = NULL;
my_storage[1].array = NULL;
my_segment = my_storage;
}

typedef void(__TBB_EXPORTED_FUNC *internal_array_op1)(void* begin, size_type n );

typedef void(__TBB_EXPORTED_FUNC *internal_array_op2)(void* dst, const void* src, size_type n );

void __TBB_EXPORTED_METHOD internal_grow_to_at_least( size_type new_size, size_type element_size, internal_array_op1 init );
void internal_grow( size_type start, size_type finish, size_type element_size, internal_array_op1 init );
size_type __TBB_EXPORTED_METHOD internal_grow_by( size_type delta, size_type element_size, internal_array_op1 init );
void* __TBB_EXPORTED_METHOD internal_push_back( size_type element_size, size_type& index );
void __TBB_EXPORTED_METHOD internal_clear( internal_array_op1 destroy, bool reclaim_storage );
void __TBB_EXPORTED_METHOD internal_copy( const concurrent_vector_base& src, size_type element_size, internal_array_op2 copy );
void __TBB_EXPORTED_METHOD internal_assign( const concurrent_vector_base& src, size_type element_size,
internal_array_op1 destroy, internal_array_op2 assign, internal_array_op2 copy );
private:
class helper;
friend class helper;
};


template<typename Container, typename Value>
class vector_iterator
#if defined(_WIN64) && defined(_MSC_VER)
: public std::iterator<std::random_access_iterator_tag,Value>
#endif 
{
Container* my_vector;

size_t my_index;


mutable Value* my_item;

template<typename C, typename T, typename U>
friend bool operator==( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

template<typename C, typename T, typename U>
friend bool operator<( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

template<typename C, typename T, typename U>
friend ptrdiff_t operator-( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

template<typename C, typename U>
friend class internal::vector_iterator;

#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
template<typename T>
friend class tbb::concurrent_vector;
#else
public: 
#endif

vector_iterator( const Container& vector, size_t index ) :
my_vector(const_cast<Container*>(&vector)),
my_index(index),
my_item(NULL)
{}

public:
vector_iterator() : my_vector(NULL), my_index(~size_t(0)), my_item(NULL) {}

vector_iterator( const vector_iterator<Container,typename Container::value_type>& other ) :
my_vector(other.my_vector),
my_index(other.my_index),
my_item(other.my_item)
{}

vector_iterator operator+( ptrdiff_t offset ) const {
return vector_iterator( *my_vector, my_index+offset );
}
friend vector_iterator operator+( ptrdiff_t offset, const vector_iterator& v ) {
return vector_iterator( *v.my_vector, v.my_index+offset );
}
vector_iterator operator+=( ptrdiff_t offset ) {
my_index+=offset;
my_item = NULL;
return *this;
}
vector_iterator operator-( ptrdiff_t offset ) const {
return vector_iterator( *my_vector, my_index-offset );
}
vector_iterator operator-=( ptrdiff_t offset ) {
my_index-=offset;
my_item = NULL;
return *this;
}
Value& operator*() const {
Value* item = my_item;
if( !item ) {
item = my_item = &my_vector->internal_subscript(my_index);
}
__TBB_ASSERT( item==&my_vector->internal_subscript(my_index), "corrupt cache" );
return *item;
}
Value& operator[]( ptrdiff_t k ) const {
return my_vector->internal_subscript(my_index+k);
}
Value* operator->() const {return &operator*();}

vector_iterator& operator++() {
size_t k = ++my_index;
if( my_item ) {
if( (k& k-concurrent_vector<Container>::min_segment_size)==0 ) {
my_item= NULL;
} else {
++my_item;
}
}
return *this;
}

vector_iterator& operator--() {
__TBB_ASSERT( my_index>0, "operator--() applied to iterator already at beginning of concurrent_vector" );
size_t k = my_index--;
if( my_item ) {
if( (k& k-concurrent_vector<Container>::min_segment_size)==0 ) {
my_item= NULL;
} else {
--my_item;
}
}
return *this;
}

vector_iterator operator++(int) {
vector_iterator result = *this;
operator++();
return result;
}

vector_iterator operator--(int) {
vector_iterator result = *this;
operator--();
return result;
}


typedef ptrdiff_t difference_type;
typedef Value value_type;
typedef Value* pointer;
typedef Value& reference;
typedef std::random_access_iterator_tag iterator_category;
};

template<typename Container, typename T, typename U>
bool operator==( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return i.my_index==j.my_index;
}

template<typename Container, typename T, typename U>
bool operator!=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return !(i==j);
}

template<typename Container, typename T, typename U>
bool operator<( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return i.my_index<j.my_index;
}

template<typename Container, typename T, typename U>
bool operator>( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return j<i;
}

template<typename Container, typename T, typename U>
bool operator>=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return !(i<j);
}

template<typename Container, typename T, typename U>
bool operator<=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return !(j<i);
}

template<typename Container, typename T, typename U>
ptrdiff_t operator-( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
return ptrdiff_t(i.my_index)-ptrdiff_t(j.my_index);
}

} 


template<typename T>
class concurrent_vector: private internal::concurrent_vector_base {
public:
using internal::concurrent_vector_base::size_type;
private:
template<typename I>
class generic_range_type: public blocked_range<I> {
public:
typedef T value_type;
typedef T& reference;
typedef const T& const_reference;
typedef I iterator;
typedef ptrdiff_t difference_type;
generic_range_type( I begin_, I end_, size_t grainsize_ ) : blocked_range<I>(begin_,end_,grainsize_) {}
generic_range_type( generic_range_type& r, split ) : blocked_range<I>(r,split()) {}
};

template<typename C, typename U>
friend class internal::vector_iterator;
public:
typedef T& reference;
typedef const T& const_reference;
typedef T value_type;
typedef ptrdiff_t difference_type;

concurrent_vector() {}

concurrent_vector( const concurrent_vector& vector ) : internal::concurrent_vector_base()
{ internal_copy(vector,sizeof(T),&copy_array); }

concurrent_vector& operator=( const concurrent_vector& vector ) {
if( this!=&vector )
internal_assign(vector,sizeof(T),&destroy_array,&assign_array,&copy_array);
return *this;
}

~concurrent_vector() {internal_clear(&destroy_array,true);}


size_type grow_by( size_type delta ) {
return delta ? internal_grow_by( delta, sizeof(T), &initialize_array ) : my_early_size.load();
}

void grow_to_at_least( size_type n ) {
if( my_early_size<n )
internal_grow_to_at_least( n, sizeof(T), &initialize_array );
};

size_type push_back( const_reference item ) {
size_type k;
new( internal_push_back(sizeof(T),k) ) T(item);
return k;
}


reference operator[]( size_type index ) {
return internal_subscript(index);
}

const_reference operator[]( size_type index ) const {
return internal_subscript(index);
}

typedef internal::vector_iterator<concurrent_vector,T> iterator;
typedef internal::vector_iterator<concurrent_vector,const T> const_iterator;

#if !defined(_MSC_VER) || _CPPLIB_VER>=300
typedef std::reverse_iterator<iterator> reverse_iterator;
typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#else
typedef std::reverse_iterator<iterator,T,T&,T*> reverse_iterator;
typedef std::reverse_iterator<const_iterator,T,const T&,const T*> const_reverse_iterator;
#endif 

iterator begin() {return iterator(*this,0);}
iterator end() {return iterator(*this,size());}
const_iterator begin() const {return const_iterator(*this,0);}
const_iterator end() const {return const_iterator(*this,size());}

reverse_iterator rbegin() {return reverse_iterator(end());}
reverse_iterator rend() {return reverse_iterator(begin());}
const_reverse_iterator rbegin() const {return const_reverse_iterator(end());}
const_reverse_iterator rend() const {return const_reverse_iterator(begin());}

typedef generic_range_type<iterator> range_type;
typedef generic_range_type<const_iterator> const_range_type;

range_type range( size_t grainsize = 1 ) {
return range_type( begin(), end(), grainsize );
}

const_range_type range( size_t grainsize = 1 ) const {
return const_range_type( begin(), end(), grainsize );
}

size_type size() const {return my_early_size;}

bool empty() const {return !my_early_size;}

size_type capacity() const {return internal_capacity();}


void reserve( size_type n ) {
if( n )
internal_reserve(n, sizeof(T), max_size());
}

size_type max_size() const {return (~size_t(0))/sizeof(T);}


void clear() {internal_clear(&destroy_array,false);}
private:
T& internal_subscript( size_type index ) const;

static void __TBB_EXPORTED_FUNC initialize_array( void* begin, size_type n );

static void __TBB_EXPORTED_FUNC copy_array( void* dst, const void* src, size_type n );

static void __TBB_EXPORTED_FUNC assign_array( void* dst, const void* src, size_type n );

static void __TBB_EXPORTED_FUNC destroy_array( void* begin, size_type n );
};

template<typename T>
T& concurrent_vector<T>::internal_subscript( size_type index ) const {
__TBB_ASSERT( index<size(), "index out of bounds" );
segment_index_t k = segment_index_of( index );
size_type j = index-segment_base(k);
return static_cast<T*>(my_segment[k].array)[j];
}

template<typename T>
void concurrent_vector<T>::initialize_array( void* begin, size_type n ) {
T* array = static_cast<T*>(begin);
for( size_type j=0; j<n; ++j )
new( &array[j] ) T();
}

template<typename T>
void concurrent_vector<T>::copy_array( void* dst, const void* src, size_type n ) {
T* d = static_cast<T*>(dst);
const T* s = static_cast<const T*>(src);
for( size_type j=0; j<n; ++j )
new( &d[j] ) T(s[j]);
}

template<typename T>
void concurrent_vector<T>::assign_array( void* dst, const void* src, size_type n ) {
T* d = static_cast<T*>(dst);
const T* s = static_cast<const T*>(src);
for( size_type j=0; j<n; ++j )
d[j] = s[j];
}

template<typename T>
void concurrent_vector<T>::destroy_array( void* begin, size_type n ) {
T* array = static_cast<T*>(begin);
for( size_type j=n; j>0; --j )
array[j-1].~T();
}

} 

#endif 
