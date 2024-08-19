

#pragma once

#include <type_traits>

namespace alpaka::concepts
{
template<typename TConcept, typename TBase>
struct Implements
{
};

template<typename TConcept, typename TDerived>
struct ImplementsConcept
{
template<typename TBase>
static auto implements(Implements<TConcept, TBase>&) -> std::true_type;
static auto implements(...) -> std::false_type;

static constexpr auto value = decltype(implements(std::declval<TDerived&>()))::value;
};

namespace detail
{
template<typename TConcept, typename TDerived, typename Sfinae = void>
struct ImplementationBaseType;

template<typename TConcept, typename TDerived>
struct ImplementationBaseType<
TConcept,
TDerived,
std::enable_if_t<!ImplementsConcept<TConcept, TDerived>::value>>
{
using type = TDerived;
};

template<typename TConcept, typename TDerived>
struct ImplementationBaseType<
TConcept,
TDerived,
std::enable_if_t<ImplementsConcept<TConcept, TDerived>::value>>
{
template<typename TBase>
static auto implementer(Implements<TConcept, TBase>&) -> TBase;

using type = decltype(implementer(std::declval<TDerived&>()));

static_assert(
std::is_base_of_v<type, TDerived>,
"The type implementing the concept has to be a publicly accessible base class!");
};
} 

template<typename TConcept, typename TDerived>
using ImplementationBase = typename detail::ImplementationBaseType<TConcept, TDerived>::type;
} 
