
#ifndef BOOST_SIMPLE_SEGREGATED_STORAGE_HPP
#define BOOST_SIMPLE_SEGREGATED_STORAGE_HPP



#include <functional>

#include <boost/pool/poolfwd.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)  
#endif

#ifdef BOOST_POOL_VALIDATE
#  define BOOST_POOL_VALIDATE_INTERNALS validate();
#else
#  define BOOST_POOL_VALIDATE_INTERNALS
#endif

namespace boost {


template <typename SizeType>
class simple_segregated_storage
{
public:
typedef SizeType size_type;

private:
simple_segregated_storage(const simple_segregated_storage &);
void operator=(const simple_segregated_storage &);

static void * try_malloc_n(void * & start, size_type n,
size_type partition_size);

protected:
void * first; 

void * find_prev(void * ptr);

static void * & nextof(void * const ptr)
{ 
return *(static_cast<void **>(ptr));
}

public:
simple_segregated_storage()
:first(0)
{ 
}

static void * segregate(void * block,
size_type nsz, size_type npartition_sz,
void * end = 0);

void add_block(void * const block,
const size_type nsz, const size_type npartition_sz)
{ 
BOOST_POOL_VALIDATE_INTERNALS
first = segregate(block, nsz, npartition_sz, first);
BOOST_POOL_VALIDATE_INTERNALS
}

void add_ordered_block(void * const block,
const size_type nsz, const size_type npartition_sz)
{ 
BOOST_POOL_VALIDATE_INTERNALS
void * const loc = find_prev(block);

if (loc == 0)
add_block(block, nsz, npartition_sz);
else
nextof(loc) = segregate(block, nsz, npartition_sz, nextof(loc));
BOOST_POOL_VALIDATE_INTERNALS
}


bool empty() const
{ 
return (first == 0);
}

void * malloc BOOST_PREVENT_MACRO_SUBSTITUTION()
{ 
BOOST_POOL_VALIDATE_INTERNALS
void * const ret = first;

first = nextof(first);
BOOST_POOL_VALIDATE_INTERNALS
return ret;
}

void free BOOST_PREVENT_MACRO_SUBSTITUTION(void * const chunk)
{ 
BOOST_POOL_VALIDATE_INTERNALS
nextof(chunk) = first;
first = chunk;
BOOST_POOL_VALIDATE_INTERNALS
}

void ordered_free(void * const chunk)
{ 

BOOST_POOL_VALIDATE_INTERNALS
void * const loc = find_prev(chunk);

if (loc == 0)
(free)(chunk);
else
{
nextof(chunk) = nextof(loc);
nextof(loc) = chunk;
}
BOOST_POOL_VALIDATE_INTERNALS
}

void * malloc_n(size_type n, size_type partition_size);

void free_n(void * const chunks, const size_type n,
const size_type partition_size)
{ 
BOOST_POOL_VALIDATE_INTERNALS
if(n != 0)
add_block(chunks, n * partition_size, partition_size);
BOOST_POOL_VALIDATE_INTERNALS
}

void ordered_free_n(void * const chunks, const size_type n,
const size_type partition_size)
{ 

BOOST_POOL_VALIDATE_INTERNALS
if(n != 0)
add_ordered_block(chunks, n * partition_size, partition_size);
BOOST_POOL_VALIDATE_INTERNALS
}
#ifdef BOOST_POOL_VALIDATE
void validate()
{
int index = 0;
void* old = 0;
void* ptr = first;
while(ptr)
{
void* pt = nextof(ptr); 
++index;
old = ptr;
ptr = nextof(ptr);
}
}
#endif
};


template <typename SizeType>
void * simple_segregated_storage<SizeType>::find_prev(void * const ptr)
{ 
if (first == 0 || std::greater<void *>()(first, ptr))
return 0;

void * iter = first;
while (true)
{
if (nextof(iter) == 0 || std::greater<void *>()(nextof(iter), ptr))
return iter;

iter = nextof(iter);
}
}

template <typename SizeType>
void * simple_segregated_storage<SizeType>::segregate(
void * const block,
const size_type sz,
const size_type partition_sz,
void * const end)
{
char * old = static_cast<char *>(block)
+ ((sz - partition_sz) / partition_sz) * partition_sz;

nextof(old) = end;

if (old == block)
return block;

for (char * iter = old - partition_sz; iter != block;
old = iter, iter -= partition_sz)
nextof(iter) = old;

nextof(block) = old;

return block;
}

template <typename SizeType>
void * simple_segregated_storage<SizeType>::try_malloc_n(
void * & start, size_type n, const size_type partition_size)
{
void * iter = nextof(start);
while (--n != 0)
{
void * next = nextof(iter);
if (next != static_cast<char *>(iter) + partition_size)
{
start = iter;
return 0;
}
iter = next;
}
return iter;
}

template <typename SizeType>
void * simple_segregated_storage<SizeType>::malloc_n(const size_type n,
const size_type partition_size)
{
BOOST_POOL_VALIDATE_INTERNALS
if(n == 0)
return 0;
void * start = &first;
void * iter;
do
{
if (nextof(start) == 0)
return 0;
iter = try_malloc_n(start, n, partition_size);
} while (iter == 0);
void * const ret = nextof(start);
nextof(start) = nextof(iter);
BOOST_POOL_VALIDATE_INTERNALS
return ret;
}

} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
