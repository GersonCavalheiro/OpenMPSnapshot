
#ifndef ABSL_TYPES_variant_internal_H_
#define ABSL_TYPES_variant_internal_H_

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "absl/base/config.h"
#include "absl/base/internal/identity.h"
#include "absl/base/internal/inline_variable.h"
#include "absl/base/internal/invoke.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
#include "absl/types/bad_variant_access.h"
#include "absl/utility/utility.h"

#if !defined(ABSL_HAVE_STD_VARIANT)

namespace absl {

template <class... Types>
class variant;

ABSL_INTERNAL_INLINE_CONSTEXPR(size_t, variant_npos, -1);

template <class T>
struct variant_size;

template <std::size_t I, class T>
struct variant_alternative;

namespace variant_internal {

template <std::size_t I, class T>
struct VariantAlternativeSfinae {};

template <std::size_t I, class T0, class... Tn>
struct VariantAlternativeSfinae<I, variant<T0, Tn...>>
: VariantAlternativeSfinae<I - 1, variant<Tn...>> {};

template <class T0, class... Ts>
struct VariantAlternativeSfinae<0, variant<T0, Ts...>> {
using type = T0;
};

template <std::size_t I, class T>
using VariantAlternativeSfinaeT = typename VariantAlternativeSfinae<I, T>::type;

template <class T, class U>
struct GiveQualsTo;

template <class T, class U>
struct GiveQualsTo<T&, U> {
using type = U&;
};

template <class T, class U>
struct GiveQualsTo<T&&, U> {
using type = U&&;
};

template <class T, class U>
struct GiveQualsTo<const T&, U> {
using type = const U&;
};

template <class T, class U>
struct GiveQualsTo<const T&&, U> {
using type = const U&&;
};

template <class T, class U>
struct GiveQualsTo<volatile T&, U> {
using type = volatile U&;
};

template <class T, class U>
struct GiveQualsTo<volatile T&&, U> {
using type = volatile U&&;
};

template <class T, class U>
struct GiveQualsTo<volatile const T&, U> {
using type = volatile const U&;
};

template <class T, class U>
struct GiveQualsTo<volatile const T&&, U> {
using type = volatile const U&&;
};

template <class T, class U>
using GiveQualsToT = typename GiveQualsTo<T, U>::type;

template <std::size_t I>
using SizeT = std::integral_constant<std::size_t, I>;

using NPos = SizeT<variant_npos>;

template <class Variant, class T, class = void>
struct IndexOfConstructedType {};

template <std::size_t I, class Variant>
struct VariantAccessResultImpl;

template <std::size_t I, template <class...> class Variantemplate, class... T>
struct VariantAccessResultImpl<I, Variantemplate<T...>&> {
using type = typename absl::variant_alternative<I, variant<T...>>::type&;
};

template <std::size_t I, template <class...> class Variantemplate, class... T>
struct VariantAccessResultImpl<I, const Variantemplate<T...>&> {
using type =
const typename absl::variant_alternative<I, variant<T...>>::type&;
};

template <std::size_t I, template <class...> class Variantemplate, class... T>
struct VariantAccessResultImpl<I, Variantemplate<T...>&&> {
using type = typename absl::variant_alternative<I, variant<T...>>::type&&;
};

template <std::size_t I, template <class...> class Variantemplate, class... T>
struct VariantAccessResultImpl<I, const Variantemplate<T...>&&> {
using type =
const typename absl::variant_alternative<I, variant<T...>>::type&&;
};

template <std::size_t I, class Variant>
using VariantAccessResult =
typename VariantAccessResultImpl<I, Variant&&>::type;

template <class T, std::size_t Size>
struct SimpleArray {
static_assert(Size != 0, "");
T value[Size];
};

template <class T>
struct AccessedType {
using type = T;
};

template <class T>
using AccessedTypeT = typename AccessedType<T>::type;

template <class T, std::size_t Size>
struct AccessedType<SimpleArray<T, Size>> {
using type = AccessedTypeT<T>;
};

template <class T>
constexpr T AccessSimpleArray(const T& value) {
return value;
}

template <class T, std::size_t Size, class... SizeT>
constexpr AccessedTypeT<T> AccessSimpleArray(const SimpleArray<T, Size>& table,
std::size_t head_index,
SizeT... tail_indices) {
return AccessSimpleArray(table.value[head_index], tail_indices...);
}

template <class T>
using AlwaysZero = SizeT<0>;

template <class Op, class... Vs>
struct VisitIndicesResultImpl {
using type = absl::result_of_t<Op(AlwaysZero<Vs>...)>;
};

template <class Op, class... Vs>
using VisitIndicesResultT = typename VisitIndicesResultImpl<Op, Vs...>::type;

template <class ReturnType, class FunctionObject, class EndIndices,
std::size_t... BoundIndices>
struct MakeVisitationMatrix;

template <class ReturnType, class FunctionObject, std::size_t... Indices>
constexpr ReturnType call_with_indices(FunctionObject&& function) {
static_assert(
std::is_same<ReturnType, decltype(std::declval<FunctionObject>()(
SizeT<Indices>()...))>::value,
"Not all visitation overloads have the same return type.");
return absl::forward<FunctionObject>(function)(SizeT<Indices>()...);
}

template <class ReturnType, class FunctionObject, std::size_t... BoundIndices>
struct MakeVisitationMatrix<ReturnType, FunctionObject, index_sequence<>,
BoundIndices...> {
using ResultType = ReturnType (*)(FunctionObject&&);
static constexpr ResultType Run() {
return &call_with_indices<ReturnType, FunctionObject,
(BoundIndices - 1)...>;
}
};

template <class ReturnType, class FunctionObject, class EndIndices,
class CurrIndices, std::size_t... BoundIndices>
struct MakeVisitationMatrixImpl;

template <class ReturnType, class FunctionObject, std::size_t... EndIndices,
std::size_t... CurrIndices, std::size_t... BoundIndices>
struct MakeVisitationMatrixImpl<
ReturnType, FunctionObject, index_sequence<EndIndices...>,
index_sequence<CurrIndices...>, BoundIndices...> {
using ResultType = SimpleArray<
typename MakeVisitationMatrix<ReturnType, FunctionObject,
index_sequence<EndIndices...>>::ResultType,
sizeof...(CurrIndices)>;

static constexpr ResultType Run() {
return {{MakeVisitationMatrix<ReturnType, FunctionObject,
index_sequence<EndIndices...>,
BoundIndices..., CurrIndices>::Run()...}};
}
};

template <class ReturnType, class FunctionObject, std::size_t HeadEndIndex,
std::size_t... TailEndIndices, std::size_t... BoundIndices>
struct MakeVisitationMatrix<ReturnType, FunctionObject,
index_sequence<HeadEndIndex, TailEndIndices...>,
BoundIndices...>
: MakeVisitationMatrixImpl<
ReturnType, FunctionObject, index_sequence<TailEndIndices...>,
absl::make_index_sequence<HeadEndIndex>, BoundIndices...> {};

struct UnreachableSwitchCase {
template <class Op>
[[noreturn]] static VisitIndicesResultT<Op, std::size_t> Run(
Op&& ) {
#if ABSL_HAVE_BUILTIN(__builtin_unreachable) || \
(defined(__GNUC__) && !defined(__clang__))
__builtin_unreachable();
#elif defined(_MSC_VER)
__assume(false);
#else
assert(false);  

return Run(absl::forward<Op>(op));
#endif  
}
};

template <class Op, std::size_t I>
struct ReachableSwitchCase {
static VisitIndicesResultT<Op, std::size_t> Run(Op&& op) {
return absl::base_internal::Invoke(absl::forward<Op>(op), SizeT<I>());
}
};

ABSL_INTERNAL_INLINE_CONSTEXPR(std::size_t, MaxUnrolledVisitCases, 33);

template <bool IsReachable>
struct PickCaseImpl {
template <class Op, std::size_t I>
using Apply = UnreachableSwitchCase;
};

template <>
struct PickCaseImpl<true> {
template <class Op, std::size_t I>
using Apply = ReachableSwitchCase<Op, I>;
};

template <class Op, std::size_t I, std::size_t EndIndex>
using PickCase = typename PickCaseImpl<(I < EndIndex)>::template Apply<Op, I>;

template <class ReturnType>
[[noreturn]] ReturnType TypedThrowBadVariantAccess() {
absl::variant_internal::ThrowBadVariantAccess();
}

template <std::size_t... NumAlternatives>
struct NumCasesOfSwitch;

template <std::size_t HeadNumAlternatives, std::size_t... TailNumAlternatives>
struct NumCasesOfSwitch<HeadNumAlternatives, TailNumAlternatives...> {
static constexpr std::size_t value =
(HeadNumAlternatives + 1) *
NumCasesOfSwitch<TailNumAlternatives...>::value;
};

template <>
struct NumCasesOfSwitch<> {
static constexpr std::size_t value = 1;
};

template <std::size_t EndIndex>
struct VisitIndicesSwitch {
static_assert(EndIndex <= MaxUnrolledVisitCases,
"Maximum unrolled switch size exceeded.");

template <class Op>
static VisitIndicesResultT<Op, std::size_t> Run(Op&& op, std::size_t i) {
switch (i) {
case 0:
return PickCase<Op, 0, EndIndex>::Run(absl::forward<Op>(op));
case 1:
return PickCase<Op, 1, EndIndex>::Run(absl::forward<Op>(op));
case 2:
return PickCase<Op, 2, EndIndex>::Run(absl::forward<Op>(op));
case 3:
return PickCase<Op, 3, EndIndex>::Run(absl::forward<Op>(op));
case 4:
return PickCase<Op, 4, EndIndex>::Run(absl::forward<Op>(op));
case 5:
return PickCase<Op, 5, EndIndex>::Run(absl::forward<Op>(op));
case 6:
return PickCase<Op, 6, EndIndex>::Run(absl::forward<Op>(op));
case 7:
return PickCase<Op, 7, EndIndex>::Run(absl::forward<Op>(op));
case 8:
return PickCase<Op, 8, EndIndex>::Run(absl::forward<Op>(op));
case 9:
return PickCase<Op, 9, EndIndex>::Run(absl::forward<Op>(op));
case 10:
return PickCase<Op, 10, EndIndex>::Run(absl::forward<Op>(op));
case 11:
return PickCase<Op, 11, EndIndex>::Run(absl::forward<Op>(op));
case 12:
return PickCase<Op, 12, EndIndex>::Run(absl::forward<Op>(op));
case 13:
return PickCase<Op, 13, EndIndex>::Run(absl::forward<Op>(op));
case 14:
return PickCase<Op, 14, EndIndex>::Run(absl::forward<Op>(op));
case 15:
return PickCase<Op, 15, EndIndex>::Run(absl::forward<Op>(op));
case 16:
return PickCase<Op, 16, EndIndex>::Run(absl::forward<Op>(op));
case 17:
return PickCase<Op, 17, EndIndex>::Run(absl::forward<Op>(op));
case 18:
return PickCase<Op, 18, EndIndex>::Run(absl::forward<Op>(op));
case 19:
return PickCase<Op, 19, EndIndex>::Run(absl::forward<Op>(op));
case 20:
return PickCase<Op, 20, EndIndex>::Run(absl::forward<Op>(op));
case 21:
return PickCase<Op, 21, EndIndex>::Run(absl::forward<Op>(op));
case 22:
return PickCase<Op, 22, EndIndex>::Run(absl::forward<Op>(op));
case 23:
return PickCase<Op, 23, EndIndex>::Run(absl::forward<Op>(op));
case 24:
return PickCase<Op, 24, EndIndex>::Run(absl::forward<Op>(op));
case 25:
return PickCase<Op, 25, EndIndex>::Run(absl::forward<Op>(op));
case 26:
return PickCase<Op, 26, EndIndex>::Run(absl::forward<Op>(op));
case 27:
return PickCase<Op, 27, EndIndex>::Run(absl::forward<Op>(op));
case 28:
return PickCase<Op, 28, EndIndex>::Run(absl::forward<Op>(op));
case 29:
return PickCase<Op, 29, EndIndex>::Run(absl::forward<Op>(op));
case 30:
return PickCase<Op, 30, EndIndex>::Run(absl::forward<Op>(op));
case 31:
return PickCase<Op, 31, EndIndex>::Run(absl::forward<Op>(op));
case 32:
return PickCase<Op, 32, EndIndex>::Run(absl::forward<Op>(op));
default:
ABSL_ASSERT(i == variant_npos);
return absl::base_internal::Invoke(absl::forward<Op>(op), NPos());
}
}
};

template <std::size_t... EndIndices>
struct VisitIndicesFallback {
template <class Op, class... SizeT>
static VisitIndicesResultT<Op, SizeT...> Run(Op&& op, SizeT... indices) {
return AccessSimpleArray(
MakeVisitationMatrix<VisitIndicesResultT<Op, SizeT...>, Op,
index_sequence<(EndIndices + 1)...>>::Run(),
(indices + 1)...)(absl::forward<Op>(op));
}
};

template <std::size_t...>
struct FlattenIndices;

template <std::size_t HeadSize, std::size_t... TailSize>
struct FlattenIndices<HeadSize, TailSize...> {
template<class... SizeType>
static constexpr std::size_t Run(std::size_t head, SizeType... tail) {
return head + HeadSize * FlattenIndices<TailSize...>::Run(tail...);
}
};

template <>
struct FlattenIndices<> {
static constexpr std::size_t Run() { return 0; }
};

template <std::size_t I, std::size_t IndexToGet, std::size_t HeadSize,
std::size_t... TailSize>
struct UnflattenIndex {
static constexpr std::size_t value =
UnflattenIndex<I / HeadSize, IndexToGet - 1, TailSize...>::value;
};

template <std::size_t I, std::size_t HeadSize, std::size_t... TailSize>
struct UnflattenIndex<I, 0, HeadSize, TailSize...> {
static constexpr std::size_t value = (I % HeadSize);
};

template <class IndexSequence, std::size_t... EndIndices>
struct VisitIndicesVariadicImpl;

template <std::size_t... N, std::size_t... EndIndices>
struct VisitIndicesVariadicImpl<absl::index_sequence<N...>, EndIndices...> {
template <class Op>
struct FlattenedOp {
template <std::size_t I>
VisitIndicesResultT<Op, decltype(EndIndices)...> operator()(
SizeT<I> ) && {
return base_internal::Invoke(
absl::forward<Op>(op),
SizeT<UnflattenIndex<I, N, (EndIndices + 1)...>::value -
std::size_t{1}>()...);
}

Op&& op;
};

template <class Op, class... SizeType>
static VisitIndicesResultT<Op, decltype(EndIndices)...> Run(
Op&& op, SizeType... i) {
return VisitIndicesSwitch<NumCasesOfSwitch<EndIndices...>::value>::Run(
FlattenedOp<Op>{absl::forward<Op>(op)},
FlattenIndices<(EndIndices + std::size_t{1})...>::Run(
(i + std::size_t{1})...));
}
};

template <std::size_t... EndIndices>
struct VisitIndicesVariadic
: VisitIndicesVariadicImpl<absl::make_index_sequence<sizeof...(EndIndices)>,
EndIndices...> {};

template <std::size_t... EndIndices>
struct VisitIndices
: absl::conditional_t<(NumCasesOfSwitch<EndIndices...>::value <=
MaxUnrolledVisitCases),
VisitIndicesVariadic<EndIndices...>,
VisitIndicesFallback<EndIndices...>> {};

template <std::size_t EndIndex>
struct VisitIndices<EndIndex>
: absl::conditional_t<(EndIndex <= MaxUnrolledVisitCases),
VisitIndicesSwitch<EndIndex>,
VisitIndicesFallback<EndIndex>> {};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4172)
#endif  

template <class Self, std::size_t I>
inline VariantAccessResult<I, Self> AccessUnion(Self&& self, SizeT<I> ) {
return reinterpret_cast<VariantAccessResult<I, Self>>(self);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif  

template <class T>
void DeducedDestroy(T& self) {  
self.~T();
}

struct VariantCoreAccess {
template <class VariantType>
static typename VariantType::Variant& Derived(VariantType& self) {  
return static_cast<typename VariantType::Variant&>(self);
}

template <class VariantType>
static const typename VariantType::Variant& Derived(
const VariantType& self) {  
return static_cast<const typename VariantType::Variant&>(self);
}

template <class VariantType>
static void Destroy(VariantType& self) {  
Derived(self).destroy();
self.index_ = absl::variant_npos;
}

template <class Variant>
static void SetIndex(Variant& self, std::size_t i) {  
self.index_ = i;
}

template <class Variant>
static void InitFrom(Variant& self, Variant&& other) {  
VisitIndices<absl::variant_size<Variant>::value>::Run(
InitFromVisitor<Variant, Variant&&>{&self,
std::forward<Variant>(other)},
other.index());
self.index_ = other.index();
}

template <std::size_t I, class Variant>
static VariantAccessResult<I, Variant> Access(Variant&& self) {
return static_cast<VariantAccessResult<I, Variant>>(
variant_internal::AccessUnion(self.state_, SizeT<I>()));
}

template <std::size_t I, class Variant>
static VariantAccessResult<I, Variant> CheckedAccess(Variant&& self) {
if (ABSL_PREDICT_FALSE(self.index_ != I)) {
TypedThrowBadVariantAccess<VariantAccessResult<I, Variant>>();
}

return Access<I>(absl::forward<Variant>(self));
}

template <class VType>
struct MoveAssignVisitor {
using DerivedType = typename VType::Variant;
template <std::size_t NewIndex>
void operator()(SizeT<NewIndex> ) const {
if (left->index_ == NewIndex) {
Access<NewIndex>(*left) = std::move(Access<NewIndex>(*right));
} else {
Derived(*left).template emplace<NewIndex>(
std::move(Access<NewIndex>(*right)));
}
}

void operator()(SizeT<absl::variant_npos> ) const {
Destroy(*left);
}

VType* left;
VType* right;
};

template <class VType>
static MoveAssignVisitor<VType> MakeMoveAssignVisitor(VType* left,
VType* other) {
return {left, other};
}

template <class VType>
struct CopyAssignVisitor {
using DerivedType = typename VType::Variant;
template <std::size_t NewIndex>
void operator()(SizeT<NewIndex> ) const {
using New =
typename absl::variant_alternative<NewIndex, DerivedType>::type;

if (left->index_ == NewIndex) {
Access<NewIndex>(*left) = Access<NewIndex>(*right);
} else if (std::is_nothrow_copy_constructible<New>::value ||
!std::is_nothrow_move_constructible<New>::value) {
Derived(*left).template emplace<NewIndex>(Access<NewIndex>(*right));
} else {
Derived(*left) = DerivedType(Derived(*right));
}
}

void operator()(SizeT<absl::variant_npos> ) const {
Destroy(*left);
}

VType* left;
const VType* right;
};

template <class VType>
static CopyAssignVisitor<VType> MakeCopyAssignVisitor(VType* left,
const VType& other) {
return {left, &other};
}

template <class Left, class QualifiedNew>
struct ConversionAssignVisitor {
using NewIndex =
variant_internal::IndexOfConstructedType<Left, QualifiedNew>;

void operator()(SizeT<NewIndex::value> 
) const {
Access<NewIndex::value>(*left) = absl::forward<QualifiedNew>(other);
}

template <std::size_t OldIndex>
void operator()(SizeT<OldIndex> 
) const {
using New =
typename absl::variant_alternative<NewIndex::value, Left>::type;
if (std::is_nothrow_constructible<New, QualifiedNew>::value ||
!std::is_nothrow_move_constructible<New>::value) {
left->template emplace<NewIndex::value>(
absl::forward<QualifiedNew>(other));
} else {
left->template emplace<NewIndex::value>(
New(absl::forward<QualifiedNew>(other)));
}
}

Left* left;
QualifiedNew&& other;
};

template <class Left, class QualifiedNew>
static ConversionAssignVisitor<Left, QualifiedNew>
MakeConversionAssignVisitor(Left* left, QualifiedNew&& qual) {
return {left, absl::forward<QualifiedNew>(qual)};
}

template <std::size_t NewIndex, class Self, class... Args>
static typename absl::variant_alternative<NewIndex, Self>::type& Replace(
Self* self, Args&&... args) {
Destroy(*self);
using New = typename absl::variant_alternative<NewIndex, Self>::type;
New* const result = ::new (static_cast<void*>(&self->state_))
New(absl::forward<Args>(args)...);
self->index_ = NewIndex;
return *result;
}

template <class LeftVariant, class QualifiedRightVariant>
struct InitFromVisitor {
template <std::size_t NewIndex>
void operator()(SizeT<NewIndex> ) const {
using Alternative =
typename variant_alternative<NewIndex, LeftVariant>::type;
::new (static_cast<void*>(&left->state_)) Alternative(
Access<NewIndex>(std::forward<QualifiedRightVariant>(right)));
}

void operator()(SizeT<absl::variant_npos> ) const {
}
LeftVariant* left;
QualifiedRightVariant&& right;
};
};

template <class Expected, class... T>
struct IndexOfImpl;

template <class Expected>
struct IndexOfImpl<Expected> {
using IndexFromEnd = SizeT<0>;
using MatchedIndexFromEnd = IndexFromEnd;
using MultipleMatches = std::false_type;
};

template <class Expected, class Head, class... Tail>
struct IndexOfImpl<Expected, Head, Tail...> : IndexOfImpl<Expected, Tail...> {
using IndexFromEnd =
SizeT<IndexOfImpl<Expected, Tail...>::IndexFromEnd::value + 1>;
};

template <class Expected, class... Tail>
struct IndexOfImpl<Expected, Expected, Tail...>
: IndexOfImpl<Expected, Tail...> {
using IndexFromEnd =
SizeT<IndexOfImpl<Expected, Tail...>::IndexFromEnd::value + 1>;
using MatchedIndexFromEnd = IndexFromEnd;
using MultipleMatches = std::integral_constant<
bool, IndexOfImpl<Expected, Tail...>::MatchedIndexFromEnd::value != 0>;
};

template <class Expected, class... Types>
struct IndexOfMeta {
using Results = IndexOfImpl<Expected, Types...>;
static_assert(!Results::MultipleMatches::value,
"Attempted to access a variant by specifying a type that "
"matches more than one alternative.");
static_assert(Results::MatchedIndexFromEnd::value != 0,
"Attempted to access a variant by specifying a type that does "
"not match any alternative.");
using type = SizeT<sizeof...(Types) - Results::MatchedIndexFromEnd::value>;
};

template <class Expected, class... Types>
using IndexOf = typename IndexOfMeta<Expected, Types...>::type;

template <class Variant, class T, std::size_t CurrIndex>
struct UnambiguousIndexOfImpl;

template <class T, std::size_t CurrIndex>
struct UnambiguousIndexOfImpl<variant<>, T, CurrIndex> : SizeT<CurrIndex> {};

template <class Head, class... Tail, class T, std::size_t CurrIndex>
struct UnambiguousIndexOfImpl<variant<Head, Tail...>, T, CurrIndex>
: UnambiguousIndexOfImpl<variant<Tail...>, T, CurrIndex + 1>::type {};

template <class Head, class... Tail, std::size_t CurrIndex>
struct UnambiguousIndexOfImpl<variant<Head, Tail...>, Head, CurrIndex>
: SizeT<UnambiguousIndexOfImpl<variant<Tail...>, Head, 0>::value ==
sizeof...(Tail)
? CurrIndex
: CurrIndex + sizeof...(Tail) + 1> {};

template <class Variant, class T>
struct UnambiguousIndexOf;

struct NoMatch {
struct type {};
};

template <class... Alts, class T>
struct UnambiguousIndexOf<variant<Alts...>, T>
: std::conditional<UnambiguousIndexOfImpl<variant<Alts...>, T, 0>::value !=
sizeof...(Alts),
UnambiguousIndexOfImpl<variant<Alts...>, T, 0>,
NoMatch>::type::type {};

template <class T, std::size_t >
using UnambiguousTypeOfImpl = T;

template <class Variant, class T>
using UnambiguousTypeOfT =
UnambiguousTypeOfImpl<T, UnambiguousIndexOf<Variant, T>::value>;

template <class H, class... T>
class VariantStateBase;

template <class Variant, std::size_t I = 0>
struct ImaginaryFun;

template <std::size_t I>
struct ImaginaryFun<variant<>, I> {
static void Run() = delete;
};

template <class H, class... T, std::size_t I>
struct ImaginaryFun<variant<H, T...>, I> : ImaginaryFun<variant<T...>, I + 1> {
using ImaginaryFun<variant<T...>, I + 1>::Run;

static SizeT<I> Run(const H&);
static SizeT<I> Run(H&&);
};

template <class Self, class T>
struct IsNeitherSelfNorInPlace : std::true_type {};

template <class Self>
struct IsNeitherSelfNorInPlace<Self, Self> : std::false_type {};

template <class Self, class T>
struct IsNeitherSelfNorInPlace<Self, in_place_type_t<T>> : std::false_type {};

template <class Self, std::size_t I>
struct IsNeitherSelfNorInPlace<Self, in_place_index_t<I>> : std::false_type {};

template <class Variant, class T, class = void>
struct ConversionIsPossibleImpl : std::false_type {};

template <class Variant, class T>
struct ConversionIsPossibleImpl<
Variant, T, void_t<decltype(ImaginaryFun<Variant>::Run(std::declval<T>()))>>
: std::true_type {};

template <class Variant, class T>
struct ConversionIsPossible : ConversionIsPossibleImpl<Variant, T>::type {};

template <class Variant, class T>
struct IndexOfConstructedType<
Variant, T, void_t<decltype(ImaginaryFun<Variant>::Run(std::declval<T>()))>>
: decltype(ImaginaryFun<Variant>::Run(std::declval<T>())) {};

template <std::size_t... Is>
struct ContainsVariantNPos
: absl::negation<std::is_same<  
absl::integer_sequence<bool, 0 <= Is...>,
absl::integer_sequence<bool, Is != absl::variant_npos...>>> {};

template <class Op, class... QualifiedVariants>
using RawVisitResult =
absl::result_of_t<Op(VariantAccessResult<0, QualifiedVariants>...)>;

template <class Op, class... QualifiedVariants>
struct VisitResultImpl {
using type =
absl::result_of_t<Op(VariantAccessResult<0, QualifiedVariants>...)>;
};

template <class Op, class... QualifiedVariants>
using VisitResult = typename VisitResultImpl<Op, QualifiedVariants...>::type;

template <class Op, class... QualifiedVariants>
struct PerformVisitation {
using ReturnType = VisitResult<Op, QualifiedVariants...>;

template <std::size_t... Is>
constexpr ReturnType operator()(SizeT<Is>... indices) const {
return Run(typename ContainsVariantNPos<Is...>::type{},
absl::index_sequence_for<QualifiedVariants...>(), indices...);
}

template <std::size_t... TupIs, std::size_t... Is>
constexpr ReturnType Run(std::false_type ,
index_sequence<TupIs...>, SizeT<Is>...) const {
static_assert(
std::is_same<ReturnType,
absl::result_of_t<Op(VariantAccessResult<
Is, QualifiedVariants>...)>>::value,
"All visitation overloads must have the same return type.");
return absl::base_internal::Invoke(
absl::forward<Op>(op),
VariantCoreAccess::Access<Is>(
absl::forward<QualifiedVariants>(std::get<TupIs>(variant_tup)))...);
}

template <std::size_t... TupIs, std::size_t... Is>
[[noreturn]] ReturnType Run(std::true_type ,
index_sequence<TupIs...>, SizeT<Is>...) const {
absl::variant_internal::ThrowBadVariantAccess();
}

std::tuple<QualifiedVariants&&...> variant_tup;
Op&& op;
};

template <class... T>
union Union;

struct NoopConstructorTag {};

template <std::size_t I>
struct EmplaceTag {};

template <>
union Union<> {
constexpr explicit Union(NoopConstructorTag) noexcept {}
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4624)
#endif  

template <class Head, class... Tail>
union Union<Head, Tail...> {
using TailUnion = Union<Tail...>;

explicit constexpr Union(NoopConstructorTag ) noexcept
: tail(NoopConstructorTag()) {}

template <class... P>
explicit constexpr Union(EmplaceTag<0>, P&&... args)
: head(absl::forward<P>(args)...) {}

template <std::size_t I, class... P>
explicit constexpr Union(EmplaceTag<I>, P&&... args)
: tail(EmplaceTag<I - 1>{}, absl::forward<P>(args)...) {}

Head head;
TailUnion tail;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif  

template <class... T>
union DestructibleUnionImpl;

template <>
union DestructibleUnionImpl<> {
constexpr explicit DestructibleUnionImpl(NoopConstructorTag) noexcept {}
};

template <class Head, class... Tail>
union DestructibleUnionImpl<Head, Tail...> {
using TailUnion = DestructibleUnionImpl<Tail...>;

explicit constexpr DestructibleUnionImpl(NoopConstructorTag ) noexcept
: tail(NoopConstructorTag()) {}

template <class... P>
explicit constexpr DestructibleUnionImpl(EmplaceTag<0>, P&&... args)
: head(absl::forward<P>(args)...) {}

template <std::size_t I, class... P>
explicit constexpr DestructibleUnionImpl(EmplaceTag<I>, P&&... args)
: tail(EmplaceTag<I - 1>{}, absl::forward<P>(args)...) {}

~DestructibleUnionImpl() {}

Head head;
TailUnion tail;
};

template <class... T>
using DestructibleUnion =
absl::conditional_t<std::is_destructible<Union<T...>>::value, Union<T...>,
DestructibleUnionImpl<T...>>;

template <class H, class... T>
class VariantStateBase {
protected:
using Variant = variant<H, T...>;

template <class LazyH = H,
class ConstructibleH = absl::enable_if_t<
std::is_default_constructible<LazyH>::value, LazyH>>
constexpr VariantStateBase() noexcept(
std::is_nothrow_default_constructible<ConstructibleH>::value)
: state_(EmplaceTag<0>()), index_(0) {}

template <std::size_t I, class... P>
explicit constexpr VariantStateBase(EmplaceTag<I> tag, P&&... args)
: state_(tag, absl::forward<P>(args)...), index_(I) {}

explicit constexpr VariantStateBase(NoopConstructorTag)
: state_(NoopConstructorTag()), index_(variant_npos) {}

void destroy() {}  

DestructibleUnion<H, T...> state_;
std::size_t index_;
};

using absl::internal::identity;

template <typename... Ts>
struct OverloadSet;

template <typename T, typename... Ts>
struct OverloadSet<T, Ts...> : OverloadSet<Ts...> {
using Base = OverloadSet<Ts...>;
static identity<T> Overload(const T&);
using Base::Overload;
};

template <>
struct OverloadSet<> {
static void Overload(...);
};

template <class T>
using LessThanResult = decltype(std::declval<T>() < std::declval<T>());

template <class T>
using GreaterThanResult = decltype(std::declval<T>() > std::declval<T>());

template <class T>
using LessThanOrEqualResult = decltype(std::declval<T>() <= std::declval<T>());

template <class T>
using GreaterThanOrEqualResult =
decltype(std::declval<T>() >= std::declval<T>());

template <class T>
using EqualResult = decltype(std::declval<T>() == std::declval<T>());

template <class T>
using NotEqualResult = decltype(std::declval<T>() != std::declval<T>());

using type_traits_internal::is_detected_convertible;

template <class... T>
using RequireAllHaveEqualT = absl::enable_if_t<
absl::conjunction<is_detected_convertible<bool, EqualResult, T>...>::value,
bool>;

template <class... T>
using RequireAllHaveNotEqualT =
absl::enable_if_t<absl::conjunction<is_detected_convertible<
bool, NotEqualResult, T>...>::value,
bool>;

template <class... T>
using RequireAllHaveLessThanT =
absl::enable_if_t<absl::conjunction<is_detected_convertible<
bool, LessThanResult, T>...>::value,
bool>;

template <class... T>
using RequireAllHaveLessThanOrEqualT =
absl::enable_if_t<absl::conjunction<is_detected_convertible<
bool, LessThanOrEqualResult, T>...>::value,
bool>;

template <class... T>
using RequireAllHaveGreaterThanOrEqualT =
absl::enable_if_t<absl::conjunction<is_detected_convertible<
bool, GreaterThanOrEqualResult, T>...>::value,
bool>;

template <class... T>
using RequireAllHaveGreaterThanT =
absl::enable_if_t<absl::conjunction<is_detected_convertible<
bool, GreaterThanResult, T>...>::value,
bool>;

template <typename T>
struct VariantHelper;

template <typename... Ts>
struct VariantHelper<variant<Ts...>> {
template <typename U>
using BestMatch = decltype(
variant_internal::OverloadSet<Ts...>::Overload(std::declval<U>()));

template <typename U>
struct CanAccept :
std::integral_constant<bool, !std::is_void<BestMatch<U>>::value> {};

template <typename Other>
struct CanConvertFrom;

template <typename... Us>
struct CanConvertFrom<variant<Us...>>
: public absl::conjunction<CanAccept<Us>...> {};
};

struct TrivialMoveOnly {
TrivialMoveOnly(TrivialMoveOnly&&) = default;
};

template <typename T>
struct IsTriviallyMoveConstructible:
std::is_move_constructible<Union<T, TrivialMoveOnly>> {};


template <class... T>
class VariantStateBaseDestructorNontrivial;

template <class... T>
class VariantMoveBaseNontrivial;

template <class... T>
class VariantCopyBaseNontrivial;

template <class... T>
class VariantMoveAssignBaseNontrivial;

template <class... T>
class VariantCopyAssignBaseNontrivial;

template <class... T>
using VariantStateBaseDestructor =
absl::conditional_t<std::is_destructible<Union<T...>>::value,
VariantStateBase<T...>,
VariantStateBaseDestructorNontrivial<T...>>;

template <class... T>
using VariantMoveBase = absl::conditional_t<
absl::disjunction<
absl::negation<absl::conjunction<std::is_move_constructible<T>...>>,
absl::conjunction<IsTriviallyMoveConstructible<T>...>>::value,
VariantStateBaseDestructor<T...>, VariantMoveBaseNontrivial<T...>>;

template <class... T>
using VariantCopyBase = absl::conditional_t<
absl::disjunction<
absl::negation<absl::conjunction<std::is_copy_constructible<T>...>>,
std::is_copy_constructible<Union<T...>>>::value,
VariantMoveBase<T...>, VariantCopyBaseNontrivial<T...>>;

template <class... T>
using VariantMoveAssignBase = absl::conditional_t<
absl::disjunction<
absl::conjunction<absl::is_move_assignable<Union<T...>>,
std::is_move_constructible<Union<T...>>,
std::is_destructible<Union<T...>>>,
absl::negation<absl::conjunction<std::is_move_constructible<T>...,
is_move_assignable<T>...>>>::value,
VariantCopyBase<T...>, VariantMoveAssignBaseNontrivial<T...>>;

template <class... T>
using VariantCopyAssignBase = absl::conditional_t<
absl::disjunction<
absl::conjunction<absl::is_copy_assignable<Union<T...>>,
std::is_copy_constructible<Union<T...>>,
std::is_destructible<Union<T...>>>,
absl::negation<absl::conjunction<std::is_copy_constructible<T>...,
is_copy_assignable<T>...>>>::value,
VariantMoveAssignBase<T...>, VariantCopyAssignBaseNontrivial<T...>>;

template <class... T>
using VariantBase = VariantCopyAssignBase<T...>;

template <class... T>
class VariantStateBaseDestructorNontrivial : protected VariantStateBase<T...> {
private:
using Base = VariantStateBase<T...>;

protected:
using Base::Base;

VariantStateBaseDestructorNontrivial() = default;
VariantStateBaseDestructorNontrivial(VariantStateBaseDestructorNontrivial&&) =
default;
VariantStateBaseDestructorNontrivial(
const VariantStateBaseDestructorNontrivial&) = default;
VariantStateBaseDestructorNontrivial& operator=(
VariantStateBaseDestructorNontrivial&&) = default;
VariantStateBaseDestructorNontrivial& operator=(
const VariantStateBaseDestructorNontrivial&) = default;

struct Destroyer {
template <std::size_t I>
void operator()(SizeT<I> i) const {
using Alternative =
typename absl::variant_alternative<I, variant<T...>>::type;
variant_internal::AccessUnion(self->state_, i).~Alternative();
}

void operator()(SizeT<absl::variant_npos> ) const {
}

VariantStateBaseDestructorNontrivial* self;
};

void destroy() { VisitIndices<sizeof...(T)>::Run(Destroyer{this}, index_); }

~VariantStateBaseDestructorNontrivial() { destroy(); }

protected:
using Base::index_;
using Base::state_;
};

template <class... T>
class VariantMoveBaseNontrivial : protected VariantStateBaseDestructor<T...> {
private:
using Base = VariantStateBaseDestructor<T...>;

protected:
using Base::Base;

struct Construct {
template <std::size_t I>
void operator()(SizeT<I> i) const {
using Alternative =
typename absl::variant_alternative<I, variant<T...>>::type;
::new (static_cast<void*>(&self->state_)) Alternative(
variant_internal::AccessUnion(absl::move(other->state_), i));
}

void operator()(SizeT<absl::variant_npos> ) const {}

VariantMoveBaseNontrivial* self;
VariantMoveBaseNontrivial* other;
};

VariantMoveBaseNontrivial() = default;
VariantMoveBaseNontrivial(VariantMoveBaseNontrivial&& other) noexcept(
absl::conjunction<std::is_nothrow_move_constructible<T>...>::value)
: Base(NoopConstructorTag()) {
VisitIndices<sizeof...(T)>::Run(Construct{this, &other}, other.index_);
index_ = other.index_;
}

VariantMoveBaseNontrivial(VariantMoveBaseNontrivial const&) = default;

VariantMoveBaseNontrivial& operator=(VariantMoveBaseNontrivial&&) = default;
VariantMoveBaseNontrivial& operator=(VariantMoveBaseNontrivial const&) =
default;

protected:
using Base::index_;
using Base::state_;
};

template <class... T>
class VariantCopyBaseNontrivial : protected VariantMoveBase<T...> {
private:
using Base = VariantMoveBase<T...>;

protected:
using Base::Base;

VariantCopyBaseNontrivial() = default;
VariantCopyBaseNontrivial(VariantCopyBaseNontrivial&&) = default;

struct Construct {
template <std::size_t I>
void operator()(SizeT<I> i) const {
using Alternative =
typename absl::variant_alternative<I, variant<T...>>::type;
::new (static_cast<void*>(&self->state_))
Alternative(variant_internal::AccessUnion(other->state_, i));
}

void operator()(SizeT<absl::variant_npos> ) const {}

VariantCopyBaseNontrivial* self;
const VariantCopyBaseNontrivial* other;
};

VariantCopyBaseNontrivial(VariantCopyBaseNontrivial const& other)
: Base(NoopConstructorTag()) {
VisitIndices<sizeof...(T)>::Run(Construct{this, &other}, other.index_);
index_ = other.index_;
}

VariantCopyBaseNontrivial& operator=(VariantCopyBaseNontrivial&&) = default;
VariantCopyBaseNontrivial& operator=(VariantCopyBaseNontrivial const&) =
default;

protected:
using Base::index_;
using Base::state_;
};

template <class... T>
class VariantMoveAssignBaseNontrivial : protected VariantCopyBase<T...> {
friend struct VariantCoreAccess;

private:
using Base = VariantCopyBase<T...>;

protected:
using Base::Base;

VariantMoveAssignBaseNontrivial() = default;
VariantMoveAssignBaseNontrivial(VariantMoveAssignBaseNontrivial&&) = default;
VariantMoveAssignBaseNontrivial(const VariantMoveAssignBaseNontrivial&) =
default;
VariantMoveAssignBaseNontrivial& operator=(
VariantMoveAssignBaseNontrivial const&) = default;

VariantMoveAssignBaseNontrivial&
operator=(VariantMoveAssignBaseNontrivial&& other) noexcept(
absl::conjunction<std::is_nothrow_move_constructible<T>...,
std::is_nothrow_move_assignable<T>...>::value) {
VisitIndices<sizeof...(T)>::Run(
VariantCoreAccess::MakeMoveAssignVisitor(this, &other), other.index_);
return *this;
}

protected:
using Base::index_;
using Base::state_;
};

template <class... T>
class VariantCopyAssignBaseNontrivial : protected VariantMoveAssignBase<T...> {
friend struct VariantCoreAccess;

private:
using Base = VariantMoveAssignBase<T...>;

protected:
using Base::Base;

VariantCopyAssignBaseNontrivial() = default;
VariantCopyAssignBaseNontrivial(VariantCopyAssignBaseNontrivial&&) = default;
VariantCopyAssignBaseNontrivial(const VariantCopyAssignBaseNontrivial&) =
default;
VariantCopyAssignBaseNontrivial& operator=(
VariantCopyAssignBaseNontrivial&&) = default;

VariantCopyAssignBaseNontrivial& operator=(
const VariantCopyAssignBaseNontrivial& other) {
VisitIndices<sizeof...(T)>::Run(
VariantCoreAccess::MakeCopyAssignVisitor(this, other), other.index_);
return *this;
}

protected:
using Base::index_;
using Base::state_;
};


template <class... Types>
struct EqualsOp {
const variant<Types...>* v;
const variant<Types...>* w;

constexpr bool operator()(SizeT<absl::variant_npos> ) const {
return true;
}

template <std::size_t I>
constexpr bool operator()(SizeT<I> ) const {
return VariantCoreAccess::Access<I>(*v) == VariantCoreAccess::Access<I>(*w);
}
};

template <class... Types>
struct NotEqualsOp {
const variant<Types...>* v;
const variant<Types...>* w;

constexpr bool operator()(SizeT<absl::variant_npos> ) const {
return false;
}

template <std::size_t I>
constexpr bool operator()(SizeT<I> ) const {
return VariantCoreAccess::Access<I>(*v) != VariantCoreAccess::Access<I>(*w);
}
};

template <class... Types>
struct LessThanOp {
const variant<Types...>* v;
const variant<Types...>* w;

constexpr bool operator()(SizeT<absl::variant_npos> ) const {
return false;
}

template <std::size_t I>
constexpr bool operator()(SizeT<I> ) const {
return VariantCoreAccess::Access<I>(*v) < VariantCoreAccess::Access<I>(*w);
}
};

template <class... Types>
struct GreaterThanOp {
const variant<Types...>* v;
const variant<Types...>* w;

constexpr bool operator()(SizeT<absl::variant_npos> ) const {
return false;
}

template <std::size_t I>
constexpr bool operator()(SizeT<I> ) const {
return VariantCoreAccess::Access<I>(*v) > VariantCoreAccess::Access<I>(*w);
}
};

template <class... Types>
struct LessThanOrEqualsOp {
const variant<Types...>* v;
const variant<Types...>* w;

constexpr bool operator()(SizeT<absl::variant_npos> ) const {
return true;
}

template <std::size_t I>
constexpr bool operator()(SizeT<I> ) const {
return VariantCoreAccess::Access<I>(*v) <= VariantCoreAccess::Access<I>(*w);
}
};

template <class... Types>
struct GreaterThanOrEqualsOp {
const variant<Types...>* v;
const variant<Types...>* w;

constexpr bool operator()(SizeT<absl::variant_npos> ) const {
return true;
}

template <std::size_t I>
constexpr bool operator()(SizeT<I> ) const {
return VariantCoreAccess::Access<I>(*v) >= VariantCoreAccess::Access<I>(*w);
}
};

template <class... Types>
struct SwapSameIndex {
variant<Types...>* v;
variant<Types...>* w;
template <std::size_t I>
void operator()(SizeT<I>) const {
using std::swap;
swap(VariantCoreAccess::Access<I>(*v), VariantCoreAccess::Access<I>(*w));
}

void operator()(SizeT<variant_npos>) const {}
};

template <class... Types>
struct Swap {
variant<Types...>* v;
variant<Types...>* w;

void generic_swap() const {
variant<Types...> tmp(std::move(*w));
VariantCoreAccess::Destroy(*w);
VariantCoreAccess::InitFrom(*w, std::move(*v));
VariantCoreAccess::Destroy(*v);
VariantCoreAccess::InitFrom(*v, std::move(tmp));
}

void operator()(SizeT<absl::variant_npos> ) const {
if (!v->valueless_by_exception()) {
generic_swap();
}
}

template <std::size_t Wi>
void operator()(SizeT<Wi> ) {
if (v->index() == Wi) {
VisitIndices<sizeof...(Types)>::Run(SwapSameIndex<Types...>{v, w}, Wi);
} else {
generic_swap();
}
}
};

template <typename Variant, typename = void, typename... Ts>
struct VariantHashBase {
VariantHashBase() = delete;
VariantHashBase(const VariantHashBase&) = delete;
VariantHashBase(VariantHashBase&&) = delete;
VariantHashBase& operator=(const VariantHashBase&) = delete;
VariantHashBase& operator=(VariantHashBase&&) = delete;
};

struct VariantHashVisitor {
template <typename T>
size_t operator()(const T& t) {
return std::hash<T>{}(t);
}
};

template <typename Variant, typename... Ts>
struct VariantHashBase<Variant,
absl::enable_if_t<absl::conjunction<
type_traits_internal::IsHashEnabled<Ts>...>::value>,
Ts...> {
using argument_type = Variant;
using result_type = size_t;
size_t operator()(const Variant& var) const {
if (var.valueless_by_exception()) {
return 239799884;
}
size_t result = VisitIndices<variant_size<Variant>::value>::Run(
PerformVisitation<VariantHashVisitor, const Variant&>{
std::forward_as_tuple(var), VariantHashVisitor{}},
var.index());
return result ^ var.index();
}
};

}  
}  

#endif  
#endif  
