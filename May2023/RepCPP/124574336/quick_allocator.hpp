#ifndef BOOST_SMART_PTR_DETAIL_QUICK_ALLOCATOR_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_QUICK_ALLOCATOR_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>

#include <boost/smart_ptr/detail/lightweight_mutex.hpp>
#include <boost/type_traits/type_with_alignment.hpp>
#include <boost/type_traits/alignment_of.hpp>

#include <new>              
#include <cstddef>          

namespace boost
{

namespace detail
{

template<unsigned size, unsigned align_> union freeblock
{
typedef typename boost::type_with_alignment<align_>::type aligner_type;
aligner_type aligner;
char bytes[size];
freeblock * next;
};

template<unsigned size, unsigned align_> struct allocator_impl
{
typedef freeblock<size, align_> block;


#if defined(BOOST_QA_PAGE_SIZE)

enum { items_per_page = BOOST_QA_PAGE_SIZE / size };

#else

enum { items_per_page = 512 / size }; 

#endif

#ifdef BOOST_HAS_THREADS

static lightweight_mutex & mutex()
{
static freeblock< sizeof( lightweight_mutex ), boost::alignment_of< lightweight_mutex >::value > fbm;
static lightweight_mutex * pm = new( &fbm ) lightweight_mutex;
return *pm;
}

static lightweight_mutex * mutex_init;

#endif

static block * free;
static block * page;
static unsigned last;

static inline void * alloc()
{
#ifdef BOOST_HAS_THREADS
lightweight_mutex::scoped_lock lock( mutex() );
#endif
if(block * x = free)
{
free = x->next;
return x;
}
else
{
if(last == items_per_page)
{
page = ::new block[items_per_page];
last = 0;
}

return &page[last++];
}
}

static inline void * alloc(std::size_t n)
{
if(n != size) 
{
return ::operator new(n);
}
else
{
#ifdef BOOST_HAS_THREADS
lightweight_mutex::scoped_lock lock( mutex() );
#endif
if(block * x = free)
{
free = x->next;
return x;
}
else
{
if(last == items_per_page)
{
page = ::new block[items_per_page];
last = 0;
}

return &page[last++];
}
}
}

static inline void dealloc(void * pv)
{
if(pv != 0) 
{
#ifdef BOOST_HAS_THREADS
lightweight_mutex::scoped_lock lock( mutex() );
#endif
block * pb = static_cast<block *>(pv);
pb->next = free;
free = pb;
}
}

static inline void dealloc(void * pv, std::size_t n)
{
if(n != size) 
{
::operator delete(pv);
}
else if(pv != 0) 
{
#ifdef BOOST_HAS_THREADS
lightweight_mutex::scoped_lock lock( mutex() );
#endif
block * pb = static_cast<block *>(pv);
pb->next = free;
free = pb;
}
}
};

#ifdef BOOST_HAS_THREADS

template<unsigned size, unsigned align_>
lightweight_mutex * allocator_impl<size, align_>::mutex_init = &allocator_impl<size, align_>::mutex();

#endif

template<unsigned size, unsigned align_>
freeblock<size, align_> * allocator_impl<size, align_>::free = 0;

template<unsigned size, unsigned align_>
freeblock<size, align_> * allocator_impl<size, align_>::page = 0;

template<unsigned size, unsigned align_>
unsigned allocator_impl<size, align_>::last = allocator_impl<size, align_>::items_per_page;

template<class T>
struct quick_allocator: public allocator_impl< sizeof(T), boost::alignment_of<T>::value >
{
};

} 

} 

#endif  
