

#pragma once

#include <alpaka/core/Utility.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>
#include <alpaka/meta/TypeListOps.hpp>

#include <tuple>

namespace alpaka::experimental
{
struct ReadAccess
{
};

struct WriteAccess
{
};

struct ReadWriteAccess
{
};

template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim, typename TAccessModes>
struct Accessor;

namespace trait
{
template<typename TMemoryObject, typename SFINAE = void>
struct BuildAccessor
{
template<typename... TAccessModes, typename TMemoryObjectForwardRef>
ALPAKA_FN_HOST static auto buildAccessor(TMemoryObjectForwardRef&&)
{
static_assert(
meta::DependentFalseType<TMemoryObject>::value,
"BuildAccessor<TMemoryObject> is not specialized for your TMemoryObject.");
}
};
} 

namespace internal
{
template<typename AccessorOrBuffer>
struct MemoryHandle
{
};

template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim, typename TAccessModes>
struct MemoryHandle<Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TAccessModes>>
{
using type = TMemoryHandle;
};
} 

template<typename Accessor>
using MemoryHandle = typename internal::MemoryHandle<Accessor>::type;

namespace internal
{
template<typename T>
struct IsAccessor : std::false_type
{
};

template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t Dim, typename TAccessModes>
struct IsAccessor<Accessor<TMemoryHandle, TElem, TBufferIdx, Dim, TAccessModes>> : std::true_type
{
};
} 

template<
typename... TAccessModes,
typename TMemoryObject,
typename = std::enable_if_t<!internal::IsAccessor<std::decay_t<TMemoryObject>>::value>>
ALPAKA_FN_HOST auto accessWith(TMemoryObject&& memoryObject)
{
return trait::BuildAccessor<std::decay_t<TMemoryObject>>::template buildAccessor<TAccessModes...>(
memoryObject);
}

template<
typename TNewAccessMode,
typename TMemoryHandle,
typename TElem,
typename TBufferIdx,
std::size_t TDim,
typename... TPrevAccessModes>
ALPAKA_FN_HOST auto accessWith(
Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, std::tuple<TPrevAccessModes...>> const& acc)
{
static_assert(
meta::Contains<std::tuple<TPrevAccessModes...>, TNewAccessMode>::value,
"The accessed accessor must already contain the requested access mode");
return Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TNewAccessMode>{acc};
}

template<typename TNewAccessMode, typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim>
ALPAKA_FN_HOST auto accessWith(Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TNewAccessMode> const& acc)
{
return acc;
}

template<typename TMemoryObjectOrAccessor>
ALPAKA_FN_HOST auto access(TMemoryObjectOrAccessor&& viewOrAccessor)
{
return accessWith<ReadWriteAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
}

template<typename TMemoryObjectOrAccessor>
ALPAKA_FN_HOST auto readAccess(TMemoryObjectOrAccessor&& viewOrAccessor)
{
return accessWith<ReadAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
}

template<typename TMemoryObjectOrAccessor>
ALPAKA_FN_HOST auto writeAccess(TMemoryObjectOrAccessor&& viewOrAccessor)
{
return accessWith<WriteAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
}

template<
typename TAcc,
typename TElem,
std::size_t TDim,
typename TAccessModes = ReadWriteAccess,
typename TIdx = Idx<TAcc>>
using BufferAccessor = Accessor<
MemoryHandle<decltype(accessWith<TAccessModes>(core::declval<Buf<TAcc, TElem, DimInt<TDim>, TIdx>>()))>,
TElem,
TIdx,
TDim,
TAccessModes>;
} 
