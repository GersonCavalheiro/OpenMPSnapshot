

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSAllocator.h>
#include <functional>

#ifndef __GNUG__

namespace Aws
{

template<typename F>
std::function< F > BuildFunction(const F& f) 
{
return std::function< F >( std::allocator_arg_t(), Aws::Allocator<void>(), f );
}

template<typename F>
std::function< F > BuildFunction(const std::function< F >& f) 
{
return std::function< F >( std::allocator_arg_t(), Aws::Allocator<void>(), f );
}

template<typename F>
std::function< F > BuildFunction(std::function< F >&& f) 
{
return std::function< F >( std::allocator_arg_t(), Aws::Allocator<void>(), f );
}

} 

#else 

namespace Aws
{


template<typename F>
F BuildFunction(F f)
{
return f;
}

template<typename F>
std::function< F > BuildFunction(const std::function< F >& f)
{
return std::function< F >( f );
}

template<typename F>
std::function< F > BuildFunction(std::function< F >&& f)
{
return std::function< F >( std::move(f));
}

} 

#endif 

#define AWS_BUILD_FUNCTION(func) Aws::BuildFunction(func)
#define AWS_BUILD_TYPED_FUNCTION(func, type) Aws::BuildFunction<type>(func)
