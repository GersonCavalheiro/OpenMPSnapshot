


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/block/shared/st/Traits.hpp>
#    include <alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>

namespace alpaka
{
class BlockSharedMemStGenericSycl
: public alpaka::detail::BlockSharedMemStMemberImpl<>
, public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStGenericSycl>
{
public:
BlockSharedMemStGenericSycl(
sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> accessor)
: BlockSharedMemStMemberImpl(
reinterpret_cast<std::uint8_t*>(accessor.get_pointer().get()),
accessor.size())
, m_accessor{accessor}
{
}

private:
sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> m_accessor;
};
} 

namespace alpaka::trait
{
template<typename T, std::size_t TUniqueId>
struct DeclareSharedVar<T, TUniqueId, BlockSharedMemStGenericSycl>
{
static auto declareVar(BlockSharedMemStGenericSycl const& smem) -> T&
{
auto* data = smem.template getVarPtr<T>(TUniqueId);

if(!data)
{
smem.template alloc<T>(TUniqueId);
data = smem.template getLatestVarPtr<T>();
}
ALPAKA_ASSERT(data != nullptr);
return *data;
}
};

template<>
struct FreeSharedVars<BlockSharedMemStGenericSycl>
{
static auto freeVars(BlockSharedMemStGenericSycl const&) -> void
{
}
};
} 

#endif
