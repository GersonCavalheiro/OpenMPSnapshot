

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/cpp/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_system_tag.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{


struct tag;

template<typename> struct execution_policy;

template<>
struct execution_policy<tag>
: hydra_thrust::system::cpp::detail::execution_policy<tag>
{};

struct tag : execution_policy<tag> {};

template<typename Derived>
struct execution_policy
: hydra_thrust::system::cpp::detail::execution_policy<Derived>
{
typedef tag tag_type; 
operator tag() const { return tag(); }
};

} 

using hydra_thrust::system::tbb::detail::execution_policy;
using hydra_thrust::system::tbb::detail::tag;

} 
} 

namespace tbb
{

using hydra_thrust::system::tbb::execution_policy;
using hydra_thrust::system::tbb::tag;

} 
} 

