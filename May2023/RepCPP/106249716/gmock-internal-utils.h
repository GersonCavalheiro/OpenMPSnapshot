

#ifndef GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_INTERNAL_UTILS_H_
#define GMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_INTERNAL_UTILS_H_

#include <stdio.h>
#include <ostream>  
#include <string>

#include "gmock/internal/gmock-generated-internal-utils.h"
#include "gmock/internal/gmock-port.h"
#include "gtest/gtest.h"

namespace testing {
namespace internal {

GTEST_API_ string ConvertIdentifierNameToWords(const char* id_name);

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

template <typename T>
struct LinkedPtrLessThan {
bool operator()(const ::testing::internal::linked_ptr<T>& lhs,
const ::testing::internal::linked_ptr<T>& rhs) const {
return lhs.get() < rhs.get();
}
};

#if (GTEST_OS_SYMBIAN && defined(_STLP_NO_WCHAR_T)) || \
(defined(_MSC_VER) && !defined(_NATIVE_WCHAR_T_DEFINED))
#else
# define GMOCK_WCHAR_T_IS_NATIVE_ 1
#endif

#ifdef __GNUC__
# define GMOCK_HAS_SIGNED_WCHAR_T_ 1
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

#if GMOCK_WCHAR_T_IS_NATIVE_
GMOCK_DECLARE_KIND_(wchar_t, kInteger);
#endif

GMOCK_DECLARE_KIND_(Int64, kInteger);
GMOCK_DECLARE_KIND_(UInt64, kInteger);

GMOCK_DECLARE_KIND_(float, kFloatingPoint);
GMOCK_DECLARE_KIND_(double, kFloatingPoint);
GMOCK_DECLARE_KIND_(long double, kFloatingPoint);

#undef GMOCK_DECLARE_KIND_

#define GMOCK_KIND_OF_(type) \
static_cast< ::testing::internal::TypeKind>( \
::testing::internal::KindOf<type>::value)

#define GMOCK_IS_SIGNED_(T) (static_cast<T>(-1) < 0)

template <TypeKind kFromKind, typename From, TypeKind kToKind, typename To>
struct LosslessArithmeticConvertibleImpl : public false_type {};

template <>
struct LosslessArithmeticConvertibleImpl<kBool, bool, kBool, bool>
: public true_type {};  

template <typename To>
struct LosslessArithmeticConvertibleImpl<kBool, bool, kInteger, To>
: public true_type {};  

template <typename To>
struct LosslessArithmeticConvertibleImpl<kBool, bool, kFloatingPoint, To>
: public true_type {};  

template <typename From>
struct LosslessArithmeticConvertibleImpl<kInteger, From, kBool, bool>
: public false_type {};  

template <typename From, typename To>
struct LosslessArithmeticConvertibleImpl<kInteger, From, kInteger, To>
: public bool_constant<
((sizeof(From) < sizeof(To)) &&
(!GMOCK_IS_SIGNED_(From) || GMOCK_IS_SIGNED_(To))) ||
((sizeof(From) == sizeof(To)) &&
(GMOCK_IS_SIGNED_(From) == GMOCK_IS_SIGNED_(To)))> {};  

#undef GMOCK_IS_SIGNED_

template <typename From, typename To>
struct LosslessArithmeticConvertibleImpl<kInteger, From, kFloatingPoint, To>
: public false_type {};  

template <typename From>
struct LosslessArithmeticConvertibleImpl<kFloatingPoint, From, kBool, bool>
: public false_type {};  

template <typename From, typename To>
struct LosslessArithmeticConvertibleImpl<kFloatingPoint, From, kInteger, To>
: public false_type {};  

template <typename From, typename To>
struct LosslessArithmeticConvertibleImpl<
kFloatingPoint, From, kFloatingPoint, To>
: public bool_constant<sizeof(From) <= sizeof(To)> {};  

template <typename From, typename To>
struct LosslessArithmeticConvertible
: public LosslessArithmeticConvertibleImpl<
GMOCK_KIND_OF_(From), From, GMOCK_KIND_OF_(To), To> {};  

class FailureReporterInterface {
public:
enum FailureType {
kNonfatal, kFatal
};

virtual ~FailureReporterInterface() {}

virtual void ReportFailure(FailureType type, const char* file, int line,
const string& message) = 0;
};

GTEST_API_ FailureReporterInterface* GetFailureReporter();

inline void Assert(bool condition, const char* file, int line,
const string& msg) {
if (!condition) {
GetFailureReporter()->ReportFailure(FailureReporterInterface::kFatal,
file, line, msg);
}
}
inline void Assert(bool condition, const char* file, int line) {
Assert(condition, file, line, "Assertion failed.");
}

inline void Expect(bool condition, const char* file, int line,
const string& msg) {
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

GTEST_API_ void Log(LogSeverity severity,
const string& message,
int stack_frames_to_skip);



template <typename T> struct is_reference : public false_type {};
template <typename T> struct is_reference<T&> : public true_type {};

template <typename T1, typename T2> struct type_equals : public false_type {};
template <typename T> struct type_equals<T, T> : public true_type {};

template <typename T> struct remove_reference { typedef T type; };  
template <typename T> struct remove_reference<T&> { typedef T type; }; 

template <typename T> struct DecayArray { typedef T type; };  
template <typename T, size_t N> struct DecayArray<T[N]> {
typedef const T* type;
};
template <typename T> struct DecayArray<T[]> {
typedef const T* type;
};

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
testing::StaticAssertTypeEq<RawContainer,
GTEST_REMOVE_CONST_(RawContainer)>();
return container;
}
static type Copy(const RawContainer& container) { return container; }
};

template <typename Element, size_t N>
class StlContainerView<Element[N]> {
public:
typedef GTEST_REMOVE_CONST_(Element) RawElement;
typedef internal::NativeArray<RawElement> type;
typedef const type const_reference;

static const_reference ConstReference(const Element (&array)[N]) {
testing::StaticAssertTypeEq<Element, RawElement>();
#if GTEST_OS_SYMBIAN
return type(const_cast<Element*>(&array[0]), N,
RelationToSourceReference());
#else
return type(array, N, RelationToSourceReference());
#endif  
}
static type Copy(const Element (&array)[N]) {
#if GTEST_OS_SYMBIAN
return type(const_cast<Element*>(&array[0]), N, RelationToSourceCopy());
#else
return type(array, N, RelationToSourceCopy());
#endif  
}
};

template <typename ElementPointer, typename Size>
class StlContainerView< ::testing::tuple<ElementPointer, Size> > {
public:
typedef GTEST_REMOVE_CONST_(
typename internal::PointeeOf<ElementPointer>::type) RawElement;
typedef internal::NativeArray<RawElement> type;
typedef const type const_reference;

static const_reference ConstReference(
const ::testing::tuple<ElementPointer, Size>& array) {
return type(get<0>(array), get<1>(array), RelationToSourceReference());
}
static type Copy(const ::testing::tuple<ElementPointer, Size>& array) {
return type(get<0>(array), get<1>(array), RelationToSourceCopy());
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

template <bool kValue>
struct BooleanConstant {};

}  
}  

#endif  

