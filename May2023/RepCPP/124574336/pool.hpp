
#ifndef BOOST_POOL_HPP
#define BOOST_POOL_HPP

#include <boost/config.hpp>  

#include <functional>
#include <new>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <algorithm>

#include <boost/pool/poolfwd.hpp>

#include <boost/integer/common_factor_ct.hpp>
#include <boost/pool/simple_segregated_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/assert.hpp>

#ifdef BOOST_POOL_INSTRUMENT
#include <iostream>
#include<iomanip>
#endif
#ifdef BOOST_POOL_VALGRIND
#include <set>
#include <valgrind/memcheck.h>
#endif

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::malloc; using ::free; }
#endif






#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)  
#endif

namespace boost
{

struct default_user_allocator_new_delete
{
typedef std::size_t size_type; 
typedef std::ptrdiff_t difference_type; 

static char * malloc BOOST_PREVENT_MACRO_SUBSTITUTION(const size_type bytes)
{ 
return new (std::nothrow) char[bytes];
}
static void free BOOST_PREVENT_MACRO_SUBSTITUTION(char * const block)
{ 
delete [] block;
}
};

struct default_user_allocator_malloc_free
{
typedef std::size_t size_type; 
typedef std::ptrdiff_t difference_type; 

static char * malloc BOOST_PREVENT_MACRO_SUBSTITUTION(const size_type bytes)
{ return static_cast<char *>((std::malloc)(bytes)); }
static void free BOOST_PREVENT_MACRO_SUBSTITUTION(char * const block)
{ (std::free)(block); }
};

namespace details
{  

template <typename SizeType>
class PODptr
{ 


public:
typedef SizeType size_type;

private:
char * ptr;
size_type sz;

char * ptr_next_size() const
{
return (ptr + sz - sizeof(size_type));
}
char * ptr_next_ptr() const
{
return (ptr_next_size() -
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value);
}

public:
PODptr(char * const nptr, const size_type nsize)
:ptr(nptr), sz(nsize)
{
}
PODptr()
:  ptr(0), sz(0)
{ 
}

bool valid() const
{ 
return (begin() != 0);
}
void invalidate()
{ 
begin() = 0;
}
char * & begin()
{ 
return ptr;
}
char * begin() const
{ 
return ptr;
}
char * end() const
{ 
return ptr_next_ptr();
}
size_type total_size() const
{ 
return sz;
}
size_type element_size() const
{ 
return static_cast<size_type>(sz - sizeof(size_type) -
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value);
}

size_type & next_size() const
{ 
return *(static_cast<size_type *>(static_cast<void*>((ptr_next_size()))));
}
char * & next_ptr() const
{  
return *(static_cast<char **>(static_cast<void*>(ptr_next_ptr())));
}

PODptr next() const
{ 
return PODptr<size_type>(next_ptr(), next_size());
}
void next(const PODptr & arg) const
{ 
next_ptr() = arg.begin();
next_size() = arg.total_size();
}
}; 
} 

#ifndef BOOST_POOL_VALGRIND

template <typename UserAllocator>
class pool: protected simple_segregated_storage < typename UserAllocator::size_type >
{
public:
typedef UserAllocator user_allocator; 
typedef typename UserAllocator::size_type size_type;  
typedef typename UserAllocator::difference_type difference_type;  

private:
BOOST_STATIC_CONSTANT(size_type, min_alloc_size =
(::boost::integer::static_lcm<sizeof(void *), sizeof(size_type)>::value) );
BOOST_STATIC_CONSTANT(size_type, min_align =
(::boost::integer::static_lcm< ::boost::alignment_of<void *>::value, ::boost::alignment_of<size_type>::value>::value) );

void * malloc_need_resize(); 
void * ordered_malloc_need_resize();  

protected:
details::PODptr<size_type> list; 

simple_segregated_storage<size_type> & store()
{ 
return *this;
}
const simple_segregated_storage<size_type> & store() const
{ 
return *this;
}
const size_type requested_size;
size_type next_size;
size_type start_size;
size_type max_size;

details::PODptr<size_type> find_POD(void * const chunk) const;

static bool is_from(void * const chunk, char * const i,
const size_type sizeof_i)
{ 

std::less_equal<void *> lt_eq;
std::less<void *> lt;
return (lt_eq(i, chunk) && lt(chunk, i + sizeof_i));
}

size_type alloc_size() const
{ 
size_type s = (std::max)(requested_size, min_alloc_size);
size_type rem = s % min_align;
if(rem)
s += min_align - rem;
BOOST_ASSERT(s >= min_alloc_size);
BOOST_ASSERT(s % min_align == 0);
return s;
}

static void * & nextof(void * const ptr)
{ 
return *(static_cast<void **>(ptr));
}

public:
explicit pool(const size_type nrequested_size,
const size_type nnext_size = 32,
const size_type nmax_size = 0)
:
list(0, 0), requested_size(nrequested_size), next_size(nnext_size), start_size(nnext_size),max_size(nmax_size)
{ 
}

~pool()
{ 
purge_memory();
}

bool release_memory();

bool purge_memory();

size_type get_next_size() const
{ 
return next_size;
}
void set_next_size(const size_type nnext_size)
{ 
next_size = start_size = nnext_size;
}
size_type get_max_size() const
{ 
return max_size;
}
void set_max_size(const size_type nmax_size)
{ 
max_size = nmax_size;
}
size_type get_requested_size() const
{ 
return requested_size;
}

void * malloc BOOST_PREVENT_MACRO_SUBSTITUTION()
{ 
if (!store().empty())
return (store().malloc)();
return malloc_need_resize();
}

void * ordered_malloc()
{ 
if (!store().empty())
return (store().malloc)();
return ordered_malloc_need_resize();
}

void * ordered_malloc(size_type n);

void free BOOST_PREVENT_MACRO_SUBSTITUTION(void * const chunk)
{ 
(store().free)(chunk);
}

void ordered_free(void * const chunk)
{ 
store().ordered_free(chunk);
}

void free BOOST_PREVENT_MACRO_SUBSTITUTION(void * const chunks, const size_type n)
{ 
const size_type partition_size = alloc_size();
const size_type total_req_size = n * requested_size;
const size_type num_chunks = total_req_size / partition_size +
((total_req_size % partition_size) ? true : false);

store().free_n(chunks, num_chunks, partition_size);
}

void ordered_free(void * const chunks, const size_type n)
{ 

const size_type partition_size = alloc_size();
const size_type total_req_size = n * requested_size;
const size_type num_chunks = total_req_size / partition_size +
((total_req_size % partition_size) ? true : false);

store().ordered_free_n(chunks, num_chunks, partition_size);
}

bool is_from(void * const chunk) const
{ 
return (find_POD(chunk).valid());
}
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <typename UserAllocator>
typename pool<UserAllocator>::size_type const pool<UserAllocator>::min_alloc_size;
template <typename UserAllocator>
typename pool<UserAllocator>::size_type const pool<UserAllocator>::min_align;
#endif

template <typename UserAllocator>
bool pool<UserAllocator>::release_memory()
{ 

bool ret = false;

details::PODptr<size_type> ptr = list;
details::PODptr<size_type> prev;

void * free_p = this->first;
void * prev_free_p = 0;

const size_type partition_size = alloc_size();

while (ptr.valid())
{

if (free_p == 0)
break;

bool all_chunks_free = true;

void * saved_free = free_p;
for (char * i = ptr.begin(); i != ptr.end(); i += partition_size)
{
if (i != free_p)
{
all_chunks_free = false;

free_p = saved_free;
break;
}

free_p = nextof(free_p);
}


const details::PODptr<size_type> next = ptr.next();

if (!all_chunks_free)
{
if (is_from(free_p, ptr.begin(), ptr.element_size()))
{
std::less<void *> lt;
void * const end = ptr.end();
do
{
prev_free_p = free_p;
free_p = nextof(free_p);
} while (free_p && lt(free_p, end));
}

prev = ptr;
}
else
{

if (prev.valid())
prev.next(next);
else
list = next;

if (prev_free_p != 0)
nextof(prev_free_p) = free_p;
else
this->first = free_p;

(UserAllocator::free)(ptr.begin());
ret = true;
}

ptr = next;
}

next_size = start_size;
return ret;
}

template <typename UserAllocator>
bool pool<UserAllocator>::purge_memory()
{ 

details::PODptr<size_type> iter = list;

if (!iter.valid())
return false;

do
{
const details::PODptr<size_type> next = iter.next();

(UserAllocator::free)(iter.begin());

iter = next;
} while (iter.valid());

list.invalidate();
this->first = 0;
next_size = start_size;

return true;
}

template <typename UserAllocator>
void * pool<UserAllocator>::malloc_need_resize()
{ 
size_type partition_size = alloc_size();
size_type POD_size = static_cast<size_type>(next_size * partition_size +
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
char * ptr = (UserAllocator::malloc)(POD_size);
if (ptr == 0)
{
if(next_size > 4)
{
next_size >>= 1;
partition_size = alloc_size();
POD_size = static_cast<size_type>(next_size * partition_size +
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
ptr = (UserAllocator::malloc)(POD_size);
}
if(ptr == 0)
return 0;
}
const details::PODptr<size_type> node(ptr, POD_size);

BOOST_USING_STD_MIN();
if(!max_size)
next_size <<= 1;
else if( next_size*partition_size/requested_size < max_size)
next_size = min BOOST_PREVENT_MACRO_SUBSTITUTION(next_size << 1, max_size*requested_size/ partition_size);

store().add_block(node.begin(), node.element_size(), partition_size);

node.next(list);
list = node;

return (store().malloc)();
}

template <typename UserAllocator>
void * pool<UserAllocator>::ordered_malloc_need_resize()
{ 
size_type partition_size = alloc_size();
size_type POD_size = static_cast<size_type>(next_size * partition_size +
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
char * ptr = (UserAllocator::malloc)(POD_size);
if (ptr == 0)
{
if(next_size > 4)
{
next_size >>= 1;
partition_size = alloc_size();
POD_size = static_cast<size_type>(next_size * partition_size +
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
ptr = (UserAllocator::malloc)(POD_size);
}
if(ptr == 0)
return 0;
}
const details::PODptr<size_type> node(ptr, POD_size);

BOOST_USING_STD_MIN();
if(!max_size)
next_size <<= 1;
else if( next_size*partition_size/requested_size < max_size)
next_size = min BOOST_PREVENT_MACRO_SUBSTITUTION(next_size << 1, max_size*requested_size/ partition_size);

store().add_ordered_block(node.begin(), node.element_size(), partition_size);

if (!list.valid() || std::greater<void *>()(list.begin(), node.begin()))
{
node.next(list);
list = node;
}
else
{
details::PODptr<size_type> prev = list;

while (true)
{
if (prev.next_ptr() == 0
|| std::greater<void *>()(prev.next_ptr(), node.begin()))
break;

prev = prev.next();
}

node.next(prev.next());
prev.next(node);
}
return (store().malloc)();
}

template <typename UserAllocator>
void * pool<UserAllocator>::ordered_malloc(const size_type n)
{ 

const size_type partition_size = alloc_size();
const size_type total_req_size = n * requested_size;
const size_type num_chunks = total_req_size / partition_size +
((total_req_size % partition_size) ? true : false);

void * ret = store().malloc_n(num_chunks, partition_size);

#ifdef BOOST_POOL_INSTRUMENT
std::cout << "Allocating " << n << " chunks from pool of size " << partition_size << std::endl;
#endif
if ((ret != 0) || (n == 0))
return ret;

#ifdef BOOST_POOL_INSTRUMENT
std::cout << "Cache miss, allocating another chunk...\n";
#endif

BOOST_USING_STD_MAX();
next_size = max BOOST_PREVENT_MACRO_SUBSTITUTION(next_size, num_chunks);
size_type POD_size = static_cast<size_type>(next_size * partition_size +
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
char * ptr = (UserAllocator::malloc)(POD_size);
if (ptr == 0)
{
if(num_chunks < next_size)
{
next_size >>= 1;
next_size = max BOOST_PREVENT_MACRO_SUBSTITUTION(next_size, num_chunks);
POD_size = static_cast<size_type>(next_size * partition_size +
integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
ptr = (UserAllocator::malloc)(POD_size);
}
if(ptr == 0)
return 0;
}
const details::PODptr<size_type> node(ptr, POD_size);

if (next_size > num_chunks)
store().add_ordered_block(node.begin() + num_chunks * partition_size,
node.element_size() - num_chunks * partition_size, partition_size);

BOOST_USING_STD_MIN();
if(!max_size)
next_size <<= 1;
else if( next_size*partition_size/requested_size < max_size)
next_size = min BOOST_PREVENT_MACRO_SUBSTITUTION(next_size << 1, max_size*requested_size/ partition_size);

if (!list.valid() || std::greater<void *>()(list.begin(), node.begin()))
{
node.next(list);
list = node;
}
else
{
details::PODptr<size_type> prev = list;

while (true)
{
if (prev.next_ptr() == 0
|| std::greater<void *>()(prev.next_ptr(), node.begin()))
break;

prev = prev.next();
}

node.next(prev.next());
prev.next(node);
}

return node.begin();
}

template <typename UserAllocator>
details::PODptr<typename pool<UserAllocator>::size_type>
pool<UserAllocator>::find_POD(void * const chunk) const
{ 
details::PODptr<size_type> iter = list;
while (iter.valid())
{
if (is_from(chunk, iter.begin(), iter.element_size()))
return iter;
iter = iter.next();
}

return iter;
}

#else 

template<typename UserAllocator> 
class pool 
{
public:
typedef UserAllocator                  user_allocator;   
typedef typename UserAllocator::size_type       size_type;        
typedef typename UserAllocator::difference_type difference_type;  

explicit pool(const size_type s, const size_type = 32, const size_type m = 0) : chunk_size(s), max_alloc_size(m) {}
~pool()
{
purge_memory();
}

bool release_memory()
{
bool ret = free_list.empty() ? false : true;
for(std::set<void*>::iterator pos = free_list.begin(); pos != free_list.end(); ++pos)
{
(user_allocator::free)(static_cast<char*>(*pos));
}
free_list.clear();
return ret;
}
bool purge_memory()
{
bool ret = free_list.empty() && used_list.empty() ? false : true;
for(std::set<void*>::iterator pos = free_list.begin(); pos != free_list.end(); ++pos)
{
(user_allocator::free)(static_cast<char*>(*pos));
}
free_list.clear();
for(std::set<void*>::iterator pos = used_list.begin(); pos != used_list.end(); ++pos)
{
(user_allocator::free)(static_cast<char*>(*pos));
}
used_list.clear();
return ret;
}
size_type get_next_size() const
{
return 1;
}
void set_next_size(const size_type){}
size_type get_max_size() const
{
return max_alloc_size;
}
void set_max_size(const size_type s)
{
max_alloc_size = s;
}
size_type get_requested_size() const
{
return chunk_size;
}
void * malloc BOOST_PREVENT_MACRO_SUBSTITUTION()
{
void* ret;
if(free_list.empty())
{
ret = (user_allocator::malloc)(chunk_size);
VALGRIND_MAKE_MEM_UNDEFINED(ret, chunk_size);
}
else
{
ret = *free_list.begin();
free_list.erase(free_list.begin());
VALGRIND_MAKE_MEM_UNDEFINED(ret, chunk_size);
}
used_list.insert(ret);
return ret;
}
void * ordered_malloc()
{
return (this->malloc)();
}
void * ordered_malloc(size_type n)
{
if(max_alloc_size && (n > max_alloc_size))
return 0;
void* ret = (user_allocator::malloc)(chunk_size * n);
used_list.insert(ret);
return ret;
}
void free BOOST_PREVENT_MACRO_SUBSTITUTION(void *const chunk)
{
BOOST_ASSERT(used_list.count(chunk) == 1);
BOOST_ASSERT(free_list.count(chunk) == 0);
used_list.erase(chunk);
free_list.insert(chunk);
VALGRIND_MAKE_MEM_NOACCESS(chunk, chunk_size);
}
void ordered_free(void *const chunk)
{
return (this->free)(chunk);
}
void free BOOST_PREVENT_MACRO_SUBSTITUTION(void *const chunk, const size_type)
{
BOOST_ASSERT(used_list.count(chunk) == 1);
BOOST_ASSERT(free_list.count(chunk) == 0);
used_list.erase(chunk);
(user_allocator::free)(static_cast<char*>(chunk));
}
void ordered_free(void *const chunk, const size_type n)
{
(this->free)(chunk, n);
}
bool is_from(void *const chunk) const
{
return used_list.count(chunk) || free_list.count(chunk);
}

protected:
size_type chunk_size, max_alloc_size;
std::set<void*> free_list, used_list;
};

#endif

} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 

