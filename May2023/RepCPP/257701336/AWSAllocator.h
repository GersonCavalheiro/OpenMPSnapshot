

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/SDKConfig.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/MemorySystemInterface.h>

#include <memory>
#include <cstdlib>

namespace Aws
{
#ifdef USE_AWS_MEMORY_MANAGEMENT

template <typename T>
class Allocator : public std::allocator<T>
{
public:

typedef std::allocator<T> Base;

Allocator() throw() :
Base()
{}

Allocator(const Allocator<T>& a) throw() :
Base(a)
{}

template <class U>
Allocator(const Allocator<U>& a) throw() :
Base(a)
{}

~Allocator() throw() {}

typedef std::size_t size_type;

template<typename U>
struct rebind
{
typedef Allocator<U> other;
};

typename Base::pointer allocate(size_type n, const void *hint = nullptr)
{
AWS_UNREFERENCED_PARAM(hint);

return reinterpret_cast<typename Base::pointer>(Malloc("AWSSTL", n * sizeof(T)));
}

void deallocate(typename Base::pointer p, size_type n)
{
AWS_UNREFERENCED_PARAM(n);

Free(p);
}

};

#ifdef __ANDROID__
#if _GLIBCXX_FULLY_DYNAMIC_STRING == 0
template< typename T >
bool operator ==(const Allocator< T >& lhs, const Allocator< T >& rhs)
{
AWS_UNREFERENCED_PARAM(lhs);
AWS_UNREFERENCED_PARAM(rhs);

return false;
}
#endif 
#endif 

#else

template< typename T > using Allocator = std::allocator<T>;

#endif 

template<typename T, typename ...ArgTypes>
std::shared_ptr<T> MakeShared(const char* allocationTag, ArgTypes&&... args)
{
AWS_UNREFERENCED_PARAM(allocationTag);

return std::allocate_shared<T, Aws::Allocator<T>>(Aws::Allocator<T>(), std::forward<ArgTypes>(args)...);
}


} 
