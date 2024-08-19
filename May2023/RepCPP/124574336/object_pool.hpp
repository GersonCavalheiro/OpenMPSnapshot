
#ifndef BOOST_OBJECT_POOL_HPP
#define BOOST_OBJECT_POOL_HPP


#include <boost/pool/poolfwd.hpp>

#include <boost/pool/pool.hpp>

#if defined(BOOST_MSVC) || defined(__KCC)
# define BOOST_NO_TEMPLATE_CV_REF_OVERLOADS
#endif

#ifdef BOOST_BORLANDC
# pragma option push -w-inl
#endif


namespace boost {



template <typename T, typename UserAllocator>
class object_pool: protected pool<UserAllocator>
{ 
public:
typedef T element_type; 
typedef UserAllocator user_allocator; 
typedef typename pool<UserAllocator>::size_type size_type; 
typedef typename pool<UserAllocator>::difference_type difference_type; 

protected:
pool<UserAllocator> & store()
{ 
return *this;
}
const pool<UserAllocator> & store() const
{ 
return *this;
}

static void * & nextof(void * const ptr)
{ 
return *(static_cast<void **>(ptr));
}

public:
explicit object_pool(const size_type arg_next_size = 32, const size_type arg_max_size = 0)
:
pool<UserAllocator>(sizeof(T), arg_next_size, arg_max_size)
{ 
}

~object_pool();

element_type * malloc BOOST_PREVENT_MACRO_SUBSTITUTION()
{ 
return static_cast<element_type *>(store().ordered_malloc());
}
void free BOOST_PREVENT_MACRO_SUBSTITUTION(element_type * const chunk)
{ 
store().ordered_free(chunk);
}
bool is_from(element_type * const chunk) const
{ 
return store().is_from(chunk);
}

element_type * construct()
{ 
element_type * const ret = (malloc)();
if (ret == 0)
return ret;
try { new (ret) element_type(); }
catch (...) { (free)(ret); throw; }
return ret;
}


#if defined(BOOST_DOXYGEN)
template <class Arg1, ... class ArgN>
element_type * construct(Arg1&, ... ArgN&)
{
}
#else

#ifndef BOOST_NO_TEMPLATE_CV_REF_OVERLOADS
#   include <boost/pool/detail/pool_construct.ipp>
#else
#   include <boost/pool/detail/pool_construct_simple.ipp>
#endif
#endif
void destroy(element_type * const chunk)
{ 
chunk->~T();
(free)(chunk);
}

size_type get_next_size() const
{ 
return store().get_next_size();
}
void set_next_size(const size_type x)
{ 
store().set_next_size(x);
}
};

template <typename T, typename UserAllocator>
object_pool<T, UserAllocator>::~object_pool()
{
#ifndef BOOST_POOL_VALGRIND
if (!this->list.valid())
return;

details::PODptr<size_type> iter = this->list;
details::PODptr<size_type> next = iter;

void * freed_iter = this->first;

const size_type partition_size = this->alloc_size();

do
{
next = next.next();


for (char * i = iter.begin(); i != iter.end(); i += partition_size)
{
if (i == freed_iter)
{
freed_iter = nextof(freed_iter);

continue;
}

static_cast<T *>(static_cast<void *>(i))->~T();
}

(UserAllocator::free)(iter.begin());

iter = next;
} while (iter.valid());

this->list.invalidate();
#else
for(std::set<void*>::iterator pos = this->used_list.begin(); pos != this->used_list.end(); ++pos)
{
static_cast<T*>(*pos)->~T();
}
#endif
}

} 

#ifdef BOOST_BORLANDC
# pragma option pop
#endif

#endif
