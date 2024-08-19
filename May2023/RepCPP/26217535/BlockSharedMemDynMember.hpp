

#pragma once

#include <alpaka/block/shared/dyn/BlockSharedDynMemberAllocKiB.hpp>
#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <array>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
namespace detail
{
template<std::size_t TStaticAllocKiB>
struct BlockSharedMemDynMemberStatic
{
static constexpr std::uint32_t staticAllocBytes = static_cast<std::uint32_t>(TStaticAllocKiB << 10u);
};
} 

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4324) 
#endif
template<std::size_t TStaticAllocKiB = BlockSharedDynMemberAllocKiB>
class alignas(core::vectorization::defaultAlignment) BlockSharedMemDynMember
: public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynMember<TStaticAllocKiB>>
{
public:
BlockSharedMemDynMember(std::size_t sizeBytes) : m_dynPitch(getPitch(sizeBytes))
{
ALPAKA_ASSERT_OFFLOAD(static_cast<std::uint32_t>(sizeBytes) <= staticAllocBytes());
}

auto dynMemBegin() const -> uint8_t*
{
return std::data(m_mem);
}


auto staticMemBegin() const -> uint8_t*
{
return std::data(m_mem) + m_dynPitch;
}


auto staticMemCapacity() const -> std::uint32_t
{
return staticAllocBytes() - m_dynPitch;
}

static constexpr auto staticAllocBytes() -> std::uint32_t
{
return detail::BlockSharedMemDynMemberStatic<TStaticAllocKiB>::staticAllocBytes;
}

private:
static auto getPitch(std::size_t sizeBytes) -> std::uint32_t
{
constexpr auto alignment = core::vectorization::defaultAlignment;
return static_cast<std::uint32_t>((sizeBytes / alignment + (sizeBytes % alignment > 0u)) * alignment);
}

mutable std::array<uint8_t, detail::BlockSharedMemDynMemberStatic<TStaticAllocKiB>::staticAllocBytes> m_mem;
std::uint32_t m_dynPitch;
};
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif

namespace trait
{
template<typename T, std::size_t TStaticAllocKiB>
struct GetDynSharedMem<T, BlockSharedMemDynMember<TStaticAllocKiB>>
{
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
"-Wcast-align" 
#endif
static auto getMem(BlockSharedMemDynMember<TStaticAllocKiB> const& mem) -> T*
{
static_assert(
core::vectorization::defaultAlignment >= alignof(T),
"Unable to get block shared dynamic memory for types with alignment higher than "
"defaultAlignment!");
return reinterpret_cast<T*>(mem.dynMemBegin());
}
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
};
} 
} 
