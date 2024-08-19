


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/block/shared/dyn/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>

namespace alpaka
{
class BlockSharedMemDynGenericSycl
: public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
{
public:
using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

BlockSharedMemDynGenericSycl(
sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> accessor)
: m_accessor{accessor}
{
}

sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> m_accessor;
};
} 

namespace alpaka::trait
{
template<typename T>
struct GetDynSharedMem<T, BlockSharedMemDynGenericSycl>
{
static auto getMem(BlockSharedMemDynGenericSycl const& shared) -> T*
{
auto void_ptr = sycl::multi_ptr<void, sycl::access::address_space::local_space>{shared.m_accessor};
auto t_ptr = static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space>>(void_ptr);
return t_ptr.get();
}
};
} 

#endif
