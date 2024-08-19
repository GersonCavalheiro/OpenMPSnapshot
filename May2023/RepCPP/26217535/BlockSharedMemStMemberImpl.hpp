

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

namespace alpaka::detail
{
template<std::size_t TMinDataAlignBytes = core::vectorization::defaultAlignment>
class BlockSharedMemStMemberImpl
{
struct MetaData
{
std::uint32_t id = std::numeric_limits<std::uint32_t>::max();
std::uint32_t offset = 0;
};

static constexpr std::uint32_t metaDataSize = sizeof(MetaData);

public:
#ifndef NDEBUG
BlockSharedMemStMemberImpl(std::uint8_t* mem, std::size_t capacity)
: m_mem(mem)
, m_capacity(static_cast<std::uint32_t>(capacity))
{
ALPAKA_ASSERT_OFFLOAD((m_mem == nullptr) == (m_capacity == 0u));
}
#else
BlockSharedMemStMemberImpl(std::uint8_t* mem, std::size_t) : m_mem(mem)
{
}
#endif

template<typename T>
void alloc(std::uint32_t id) const
{
m_allocdBytes = varChunkEnd<MetaData>(m_allocdBytes);
ALPAKA_ASSERT_OFFLOAD(m_allocdBytes <= m_capacity);
auto* meta = getLatestVarPtr<MetaData>();

m_allocdBytes = varChunkEnd<T>(m_allocdBytes);
ALPAKA_ASSERT_OFFLOAD(m_allocdBytes <= m_capacity);

meta->id = id;
meta->offset = m_allocdBytes;
}

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
"-Wcast-align" 
#endif

template<typename T>
auto getVarPtr(std::uint32_t id) const -> T*
{
std::uint32_t off = 0;

while(off < m_allocdBytes)
{
std::uint32_t const alignedMetaDataOffset
= varChunkEnd<MetaData>(off) - static_cast<std::uint32_t>(sizeof(MetaData));
ALPAKA_ASSERT_OFFLOAD(
(alignedMetaDataOffset + static_cast<std::uint32_t>(sizeof(MetaData))) <= m_allocdBytes);
auto* metaDataPtr = reinterpret_cast<MetaData*>(m_mem + alignedMetaDataOffset);
off = metaDataPtr->offset;

if(metaDataPtr->id == id)
return reinterpret_cast<T*>(&m_mem[off - sizeof(T)]);
}

return nullptr;
}

template<typename T>
auto getLatestVarPtr() const -> T*
{
return reinterpret_cast<T*>(&m_mem[m_allocdBytes - sizeof(T)]);
}

private:
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

template<typename T>
auto varChunkEnd(std::uint32_t byteOffset) const -> std::uint32_t
{
auto const ptr = reinterpret_cast<std::size_t>(m_mem + byteOffset);
constexpr size_t align = std::max(TMinDataAlignBytes, alignof(T));
std::size_t const newPtrAdress = ((ptr + align - 1u) / align) * align + sizeof(T);
return static_cast<uint32_t>(newPtrAdress - reinterpret_cast<std::size_t>(m_mem));
}

mutable std::uint32_t m_allocdBytes = 0u;

std::uint8_t* const m_mem;
#ifndef NDEBUG
const std::uint32_t m_capacity;
#endif
};
} 
