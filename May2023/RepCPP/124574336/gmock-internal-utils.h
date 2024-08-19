



#ifndef GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_INTERNAL_UTILS_H_
#define GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_INTERNAL_UTILS_H_

#include <stdio.h>
#include <ostream>  
#include <string>
#include <type_traits>
#include "gmock/internal/gmock-port.h"
#include "gtest/gtest.h"

namespace testing {

template <typename>
class Matcher;

namespace internal {

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
# pragma warning(disable:4805)
#endif

GTEST_API_ std::string JoinAsTuple(const Strings& fields);

GTEST_API_ std::string ConvertIdentifierNameToWords(const char* id_name);

template <typename Pointer>
struct PointeeOf {
typedef typename Pointer::element_type type;
};
template <typename T>
struct PointeeOf<T*> { typedef T type; };  

template <typename Pointer>
inline const typename Pointer::element_type* GetRawPointer(const Pointer& p) {
return p.get();
}
template <typename Element>
inline Element* GetRawPointer(Element* p) { return p; }

#if defined(_MSC_VER) && !defined(_NATIVE_WCHAR_T_DEFINED)
#else
# define GMOCK_WCHAR_T_IS_NATIVE_ 1
#endif

enum TypeKind {
kBool, kInteger, kFloatingPoint, kOther
};

template <typename T> struct KindOf {
enum { value = kOther };  
};

#define GMOCK_DECLARE_KIND_(type, kind) \
template <> struct KindOf<type> { enum { value = kind }; }

GMOCK_DECLARE_KIND_(bool, kBool);

GMOCK_DECLARE_KIND_(char, kInteger);
GMOCK_DECLARE_KIND_(signed char, kInteger);
GMOCK_DECLARE_KIND_(unsigned char, kInteger);
GMOCK_DECLARE_KIND_(short, kInteger);  
GMOCK_DECLARE_KIND_(unsigned short, kInteger);  
GMOCK_DECLARE_KIND_(int, kInteger);
GMOCK_DECLARE_KIND_(unsigned int, kInteger);
GMOCK_DECLARE_KIND_(long, kInteger);  
GMOCK_DECLARE_KIND_(unsigned long, kInteger);  
GMOCK_DECLARE_KIND_(long long, kInteger);  
GMOCK_DECLARE_KIND_(unsigned long long, kInteger);  

#if GMOCK_WCHAR_T_IS_NATIVE_
GMOCK_DECLARE_KIND_(wchar_t, kInteger);
#endif

GMOCK_DECLARE_KIND_(float, kFloatingPoint);
GMOCK_DECLARE_KIND_(double, kFloatingPoint);
GMOCK_DECLARE_KIND_(long double, kFloatingPoint);

#undef GMOCK_DECLARE_KIND_

#define GMOCK_KIND_OF_(type) \
static_cast< ::testing::internal::TypeKind>( \
::testing::internal::KindOf<type>::value)

template <TypeKind kFromKind, typename From, TypeKind kToKind, typename To>
using LosslessArithmeticConvertibleImpl = std::integral_constant<
bool,
(kFromKind == kBool) ? true
: (kFromKind != kToKind) ? false
: (kFromKind == kInteger &&
(((sizeof(From) < sizeof(To)) &&
!(std::is_signed<From>::value && !std::is_signed<To>::value)) ||
((sizeof(From) == sizeof(To)) &&
(std::is_signed<From>::value == std::is_signed<To>::value)))
) ? true
: (kFromKind == kFloatingPoint && (sizeof(From) <= sizeof(To))) ? true
: false
>;

template <typename From, typename To>
using LosslessArithmeticConvertible =
LosslessArithmeticConvertibleImpl<GMOCK_KIND_OF_(From), From,
GMOCK_KIND_OF_(To), To>;

class FailureReporterInterface {
public:
enum FailureType {
kNonfatal, kFatal
};

virtual ~FailureReporterInterface() {}

virtual void ReportFailure(FailureType type, const char* file, int line,
const std::string& message) = 0;
};

GTEST_API_ FailureReporterInterface* GetFailureReporter();

inline void Assert(bool condition, const char* file, int line,
const std::string& msg) {
if (!condition) {
GetFailureReporter()->ReportFailure(FailureReporterInterface::kFatal,
file, line, msg);
}
}
inline void Assert(bool condition, const char* file, int line) {
Assert(condition, file, line, "Assertion failed.");
}

inline void Expect(bool condition, const char* file, int line,
const std::string& msg) {
if (!condition) {
GetFailureReporter()->ReportFailure(FailureReporterInterface::kNonfatal,
file, line, msg);
}
}
inline void Expect(bool condition, const char* file, int line) {
Expect(condition, file, line, "Expectation failed.");
}

enum LogSeverity {
kInfo = 0,
kWarning = 1
};


const char kInfoVerbosity[] = "info";
const char kWarningVerbosity[] = "warning";
const char kErrorVerbosity[] = "error";

GTEST_API_ bool LogIsVisible(LogSeverity severity);

GTEST_API_ void Log(LogSeverity severity, const std::string& message,
int stack_frames_to_skip);

class WithoutMatchers {
private:
WithoutMatchers() {}
friend GTEST_API_ WithoutMatchers GetWithoutMatchers();
};

GTEST_API_ WithoutMatchers GetWithoutMatchers();

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4717)
#endif

template <typename T>
inline T Invalid() {
Assert(false, "", -1, "Internal error: attempt to return invalid value");
return Invalid<T>();
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

template <class RawContainer>
class StlContainerView {
public:
typedef RawContainer type;
typedef const type& const_reference;

static const_reference ConstReference(const RawContainer& container) {
static_assert(!std::is_const<RawContainer>::value,
"RawContainer type must not be const");
return container;
}
static type Copy(const RawContainer& container) { return container; }
};

template <typename Element, size_t N>
class StlContainerView<Element[N]> {
public:
typedef typename std::remove_const<Element>::type RawElement;
typedef internal::NativeArray<RawElement> type;
typedef const type const_reference;

static const_reference ConstReference(const Element (&array)[N]) {
static_assert(std::is_same<Element, RawElement>::value,
"Element type must not be const");
return type(array, N, RelationToSourceReference());
}
static type Copy(const Element (&array)[N]) {
return type(array, N, RelationToSourceCopy());
}
};

template <typename ElementPointer, typename Size>
class StlContainerView< ::std::tuple<ElementPointer, Size> > {
public:
typedef typename std::remove_const<
typename internal::PointeeOf<ElementPointer>::type>::type RawElement;
typedef internal::NativeArray<RawElement> type;
typedef const type const_reference;

static const_reference ConstReference(
const ::std::tuple<ElementPointer, Size>& array) {
return type(std::get<0>(array), std::get<1>(array),
RelationToSourceReference());
}
static type Copy(const ::std::tuple<ElementPointer, Size>& array) {
return type(std::get<0>(array), std::get<1>(array), RelationToSourceCopy());
}
};

template <typename T> class StlContainerView<T&>;

template <typename T>
struct RemoveConstFromKey {
typedef T type;
};

template <typename K, typename V>
struct RemoveConstFromKey<std::pair<const K, V> > {
typedef std::pair<K, V> type;
};

GTEST_API_ void IllegalDoDefault(const char* file, int line);

template <typename F, typename Tuple, size_t... Idx>
auto ApplyImpl(F&& f, Tuple&& args, IndexSequence<Idx...>) -> decltype(
std::forward<F>(f)(std::get<Idx>(std::forward<Tuple>(args))...)) {
return std::forward<F>(f)(std::get<Idx>(std::forward<Tuple>(args))...);
}

template <typename F, typename Tuple>
auto Apply(F&& f, Tuple&& args) -> decltype(
ApplyImpl(std::forward<F>(f), std::forward<Tuple>(args),
MakeIndexSequence<std::tuple_size<
typename std::remove_reference<Tuple>::type>::value>())) {
return ApplyImpl(std::forward<F>(f), std::forward<Tuple>(args),
MakeIndexSequence<std::tuple_size<
typename std::remove_reference<Tuple>::type>::value>());
}

template <typename T>
struct Function;

template <typename R, typename... Args>
struct Function<R(Args...)> {
using Result = R;
static constexpr size_t ArgumentCount = sizeof...(Args);
template <size_t I>
using Arg = ElemFromList<I, Args...>;
using ArgumentTuple = std::tuple<Args...>;
using ArgumentMatcherTuple = std::tuple<Matcher<Args>...>;
using MakeResultVoid = void(Args...);
using MakeResultIgnoredValue = IgnoredValue(Args...);
};

template <typename R, typename... Args>
constexpr size_t Function<R(Args...)>::ArgumentCount;

#ifdef _MSC_VER
# pragma warning(pop)
#endif

}  
}  

#endif  
