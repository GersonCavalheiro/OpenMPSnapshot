


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4244)
# pragma warning(disable:4100)
#endif

#include "gmock/gmock-matchers.h"

#include <string.h>
#include <time.h>

#include <array>
#include <cstdint>
#include <deque>
#include <forward_list>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gmock/gmock-more-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace testing {
namespace gmock_matchers_test {
namespace {

using std::greater;
using std::less;
using std::list;
using std::make_pair;
using std::map;
using std::multimap;
using std::multiset;
using std::ostream;
using std::pair;
using std::set;
using std::stringstream;
using std::vector;
using testing::internal::DummyMatchResultListener;
using testing::internal::ElementMatcherPair;
using testing::internal::ElementMatcherPairs;
using testing::internal::ElementsAreArrayMatcher;
using testing::internal::ExplainMatchFailureTupleTo;
using testing::internal::FloatingEqMatcher;
using testing::internal::FormatMatcherDescription;
using testing::internal::IsReadableTypeName;
using testing::internal::MatchMatrix;
using testing::internal::PredicateFormatterFromMatcher;
using testing::internal::RE;
using testing::internal::StreamMatchResultListener;
using testing::internal::Strings;

struct ContainerHelper {
MOCK_METHOD1(Call, void(std::vector<std::unique_ptr<int>>));
};

std::vector<std::unique_ptr<int>> MakeUniquePtrs(const std::vector<int>& ints) {
std::vector<std::unique_ptr<int>> pointers;
for (int i : ints) pointers.emplace_back(new int(i));
return pointers;
}

class GreaterThanMatcher : public MatcherInterface<int> {
public:
explicit GreaterThanMatcher(int rhs) : rhs_(rhs) {}

void DescribeTo(ostream* os) const override { *os << "is > " << rhs_; }

bool MatchAndExplain(int lhs, MatchResultListener* listener) const override {
const int diff = lhs - rhs_;
if (diff > 0) {
*listener << "which is " << diff << " more than " << rhs_;
} else if (diff == 0) {
*listener << "which is the same as " << rhs_;
} else {
*listener << "which is " << -diff << " less than " << rhs_;
}

return lhs > rhs_;
}

private:
int rhs_;
};

Matcher<int> GreaterThan(int n) {
return MakeMatcher(new GreaterThanMatcher(n));
}

std::string OfType(const std::string& type_name) {
#if GTEST_HAS_RTTI
return IsReadableTypeName(type_name) ? " (of type " + type_name + ")" : "";
#else
return "";
#endif
}

template <typename T>
std::string Describe(const Matcher<T>& m) {
return DescribeMatcher<T>(m);
}

template <typename T>
std::string DescribeNegation(const Matcher<T>& m) {
return DescribeMatcher<T>(m, true);
}

template <typename MatcherType, typename Value>
std::string Explain(const MatcherType& m, const Value& x) {
StringMatchResultListener listener;
ExplainMatchResult(m, x, &listener);
return listener.str();
}

TEST(MonotonicMatcherTest, IsPrintable) {
stringstream ss;
ss << GreaterThan(5);
EXPECT_EQ("is > 5", ss.str());
}

TEST(MatchResultListenerTest, StreamingWorks) {
StringMatchResultListener listener;
listener << "hi" << 5;
EXPECT_EQ("hi5", listener.str());

listener.Clear();
EXPECT_EQ("", listener.str());

listener << 42;
EXPECT_EQ("42", listener.str());

DummyMatchResultListener dummy;
dummy << "hi" << 5;
}

TEST(MatchResultListenerTest, CanAccessUnderlyingStream) {
EXPECT_TRUE(DummyMatchResultListener().stream() == nullptr);
EXPECT_TRUE(StreamMatchResultListener(nullptr).stream() == nullptr);

EXPECT_EQ(&std::cout, StreamMatchResultListener(&std::cout).stream());
}

TEST(MatchResultListenerTest, IsInterestedWorks) {
EXPECT_TRUE(StringMatchResultListener().IsInterested());
EXPECT_TRUE(StreamMatchResultListener(&std::cout).IsInterested());

EXPECT_FALSE(DummyMatchResultListener().IsInterested());
EXPECT_FALSE(StreamMatchResultListener(nullptr).IsInterested());
}

class EvenMatcherImpl : public MatcherInterface<int> {
public:
bool MatchAndExplain(int x,
MatchResultListener* ) const override {
return x % 2 == 0;
}

void DescribeTo(ostream* os) const override { *os << "is an even number"; }

};

TEST(MatcherInterfaceTest, CanBeImplementedUsingPublishedAPI) {
EvenMatcherImpl m;
}


class NewEvenMatcherImpl : public MatcherInterface<int> {
public:
bool MatchAndExplain(int x, MatchResultListener* listener) const override {
const bool match = x % 2 == 0;
*listener << "value % " << 2;
if (listener->stream() != nullptr) {
*listener->stream() << " == " << (x % 2);
}
return match;
}

void DescribeTo(ostream* os) const override { *os << "is an even number"; }
};

TEST(MatcherInterfaceTest, CanBeImplementedUsingNewAPI) {
Matcher<int> m = MakeMatcher(new NewEvenMatcherImpl);
EXPECT_TRUE(m.Matches(2));
EXPECT_FALSE(m.Matches(3));
EXPECT_EQ("value % 2 == 0", Explain(m, 2));
EXPECT_EQ("value % 2 == 1", Explain(m, 3));
}

TEST(MatcherTest, CanBeDefaultConstructed) {
Matcher<double> m;
}

TEST(MatcherTest, CanBeConstructedFromMatcherInterface) {
const MatcherInterface<int>* impl = new EvenMatcherImpl;
Matcher<int> m(impl);
EXPECT_TRUE(m.Matches(4));
EXPECT_FALSE(m.Matches(5));
}

TEST(MatcherTest, CanBeImplicitlyConstructedFromValue) {
Matcher<int> m1 = 5;
EXPECT_TRUE(m1.Matches(5));
EXPECT_FALSE(m1.Matches(6));
}

TEST(MatcherTest, CanBeImplicitlyConstructedFromNULL) {
Matcher<int*> m1 = nullptr;
EXPECT_TRUE(m1.Matches(nullptr));
int n = 0;
EXPECT_FALSE(m1.Matches(&n));
}

struct Undefined {
virtual ~Undefined() = 0;
static const int kInt = 1;
};

TEST(MatcherTest, CanBeConstructedFromUndefinedVariable) {
Matcher<int> m1 = Undefined::kInt;
EXPECT_TRUE(m1.Matches(1));
EXPECT_FALSE(m1.Matches(2));
}

TEST(MatcherTest, CanAcceptAbstractClass) { Matcher<const Undefined&> m = _; }

TEST(MatcherTest, IsCopyable) {
Matcher<bool> m1 = Eq(false);
EXPECT_TRUE(m1.Matches(false));
EXPECT_FALSE(m1.Matches(true));

m1 = Eq(true);
EXPECT_TRUE(m1.Matches(true));
EXPECT_FALSE(m1.Matches(false));
}

TEST(MatcherTest, CanDescribeItself) {
EXPECT_EQ("is an even number",
Describe(Matcher<int>(new EvenMatcherImpl)));
}

TEST(MatcherTest, MatchAndExplain) {
Matcher<int> m = GreaterThan(0);
StringMatchResultListener listener1;
EXPECT_TRUE(m.MatchAndExplain(42, &listener1));
EXPECT_EQ("which is 42 more than 0", listener1.str());

StringMatchResultListener listener2;
EXPECT_FALSE(m.MatchAndExplain(-9, &listener2));
EXPECT_EQ("which is 9 less than 0", listener2.str());
}

TEST(StringMatcherTest, CanBeImplicitlyConstructedFromCStringLiteral) {
Matcher<std::string> m1 = "hi";
EXPECT_TRUE(m1.Matches("hi"));
EXPECT_FALSE(m1.Matches("hello"));

Matcher<const std::string&> m2 = "hi";
EXPECT_TRUE(m2.Matches("hi"));
EXPECT_FALSE(m2.Matches("hello"));
}

TEST(StringMatcherTest, CanBeImplicitlyConstructedFromString) {
Matcher<std::string> m1 = std::string("hi");
EXPECT_TRUE(m1.Matches("hi"));
EXPECT_FALSE(m1.Matches("hello"));

Matcher<const std::string&> m2 = std::string("hi");
EXPECT_TRUE(m2.Matches("hi"));
EXPECT_FALSE(m2.Matches("hello"));
}

#if GTEST_INTERNAL_HAS_STRING_VIEW
TEST(StringViewMatcherTest, CanBeImplicitlyConstructedFromCStringLiteral) {
Matcher<internal::StringView> m1 = "cats";
EXPECT_TRUE(m1.Matches("cats"));
EXPECT_FALSE(m1.Matches("dogs"));

Matcher<const internal::StringView&> m2 = "cats";
EXPECT_TRUE(m2.Matches("cats"));
EXPECT_FALSE(m2.Matches("dogs"));
}

TEST(StringViewMatcherTest, CanBeImplicitlyConstructedFromString) {
Matcher<internal::StringView> m1 = std::string("cats");
EXPECT_TRUE(m1.Matches("cats"));
EXPECT_FALSE(m1.Matches("dogs"));

Matcher<const internal::StringView&> m2 = std::string("cats");
EXPECT_TRUE(m2.Matches("cats"));
EXPECT_FALSE(m2.Matches("dogs"));
}

TEST(StringViewMatcherTest, CanBeImplicitlyConstructedFromStringView) {
Matcher<internal::StringView> m1 = internal::StringView("cats");
EXPECT_TRUE(m1.Matches("cats"));
EXPECT_FALSE(m1.Matches("dogs"));

Matcher<const internal::StringView&> m2 = internal::StringView("cats");
EXPECT_TRUE(m2.Matches("cats"));
EXPECT_FALSE(m2.Matches("dogs"));
}
#endif  

TEST(StringMatcherTest,
CanBeImplicitlyConstructedFromEqReferenceWrapperString) {
std::string value = "cats";
Matcher<std::string> m1 = Eq(std::ref(value));
EXPECT_TRUE(m1.Matches("cats"));
EXPECT_FALSE(m1.Matches("dogs"));

Matcher<const std::string&> m2 = Eq(std::ref(value));
EXPECT_TRUE(m2.Matches("cats"));
EXPECT_FALSE(m2.Matches("dogs"));
}

TEST(MakeMatcherTest, ConstructsMatcherFromMatcherInterface) {
const MatcherInterface<int>* dummy_impl = nullptr;
Matcher<int> m = MakeMatcher(dummy_impl);
}

const int g_bar = 1;
class ReferencesBarOrIsZeroImpl {
public:
template <typename T>
bool MatchAndExplain(const T& x,
MatchResultListener* ) const {
const void* p = &x;
return p == &g_bar || x == 0;
}

void DescribeTo(ostream* os) const { *os << "g_bar or zero"; }

void DescribeNegationTo(ostream* os) const {
*os << "doesn't reference g_bar and is not zero";
}
};

PolymorphicMatcher<ReferencesBarOrIsZeroImpl> ReferencesBarOrIsZero() {
return MakePolymorphicMatcher(ReferencesBarOrIsZeroImpl());
}

TEST(MakePolymorphicMatcherTest, ConstructsMatcherUsingOldAPI) {
Matcher<const int&> m1 = ReferencesBarOrIsZero();
EXPECT_TRUE(m1.Matches(0));
EXPECT_TRUE(m1.Matches(g_bar));
EXPECT_FALSE(m1.Matches(1));
EXPECT_EQ("g_bar or zero", Describe(m1));

Matcher<double> m2 = ReferencesBarOrIsZero();
EXPECT_TRUE(m2.Matches(0.0));
EXPECT_FALSE(m2.Matches(0.1));
EXPECT_EQ("g_bar or zero", Describe(m2));
}


class PolymorphicIsEvenImpl {
public:
void DescribeTo(ostream* os) const { *os << "is even"; }

void DescribeNegationTo(ostream* os) const {
*os << "is odd";
}

template <typename T>
bool MatchAndExplain(const T& x, MatchResultListener* listener) const {
*listener << "% " << 2;
if (listener->stream() != nullptr) {
*listener->stream() << " == " << (x % 2);
}
return (x % 2) == 0;
}
};

PolymorphicMatcher<PolymorphicIsEvenImpl> PolymorphicIsEven() {
return MakePolymorphicMatcher(PolymorphicIsEvenImpl());
}

TEST(MakePolymorphicMatcherTest, ConstructsMatcherUsingNewAPI) {
const Matcher<int> m1 = PolymorphicIsEven();
EXPECT_TRUE(m1.Matches(42));
EXPECT_FALSE(m1.Matches(43));
EXPECT_EQ("is even", Describe(m1));

const Matcher<int> not_m1 = Not(m1);
EXPECT_EQ("is odd", Describe(not_m1));

EXPECT_EQ("% 2 == 0", Explain(m1, 42));

const Matcher<char> m2 = PolymorphicIsEven();
EXPECT_TRUE(m2.Matches('\x42'));
EXPECT_FALSE(m2.Matches('\x43'));
EXPECT_EQ("is even", Describe(m2));

const Matcher<char> not_m2 = Not(m2);
EXPECT_EQ("is odd", Describe(not_m2));

EXPECT_EQ("% 2 == 0", Explain(m2, '\x42'));
}

TEST(MatcherCastTest, FromPolymorphicMatcher) {
Matcher<int> m = MatcherCast<int>(Eq(5));
EXPECT_TRUE(m.Matches(5));
EXPECT_FALSE(m.Matches(6));
}

class IntValue {
public:
explicit IntValue(int a_value) : value_(a_value) {}

int value() const { return value_; }
private:
int value_;
};

bool IsPositiveIntValue(const IntValue& foo) {
return foo.value() > 0;
}

TEST(MatcherCastTest, FromCompatibleType) {
Matcher<double> m1 = Eq(2.0);
Matcher<int> m2 = MatcherCast<int>(m1);
EXPECT_TRUE(m2.Matches(2));
EXPECT_FALSE(m2.Matches(3));

Matcher<IntValue> m3 = Truly(IsPositiveIntValue);
Matcher<int> m4 = MatcherCast<int>(m3);
EXPECT_TRUE(m4.Matches(1));
EXPECT_FALSE(m4.Matches(0));
}

TEST(MatcherCastTest, FromConstReferenceToNonReference) {
Matcher<const int&> m1 = Eq(0);
Matcher<int> m2 = MatcherCast<int>(m1);
EXPECT_TRUE(m2.Matches(0));
EXPECT_FALSE(m2.Matches(1));
}

TEST(MatcherCastTest, FromReferenceToNonReference) {
Matcher<int&> m1 = Eq(0);
Matcher<int> m2 = MatcherCast<int>(m1);
EXPECT_TRUE(m2.Matches(0));
EXPECT_FALSE(m2.Matches(1));
}

TEST(MatcherCastTest, FromNonReferenceToConstReference) {
Matcher<int> m1 = Eq(0);
Matcher<const int&> m2 = MatcherCast<const int&>(m1);
EXPECT_TRUE(m2.Matches(0));
EXPECT_FALSE(m2.Matches(1));
}

TEST(MatcherCastTest, FromNonReferenceToReference) {
Matcher<int> m1 = Eq(0);
Matcher<int&> m2 = MatcherCast<int&>(m1);
int n = 0;
EXPECT_TRUE(m2.Matches(n));
n = 1;
EXPECT_FALSE(m2.Matches(n));
}

TEST(MatcherCastTest, FromSameType) {
Matcher<int> m1 = Eq(0);
Matcher<int> m2 = MatcherCast<int>(m1);
EXPECT_TRUE(m2.Matches(0));
EXPECT_FALSE(m2.Matches(1));
}

TEST(MatcherCastTest, FromAValue) {
Matcher<int> m = MatcherCast<int>(42);
EXPECT_TRUE(m.Matches(42));
EXPECT_FALSE(m.Matches(239));
}

TEST(MatcherCastTest, FromAnImplicitlyConvertibleValue) {
const int kExpected = 'c';
Matcher<int> m = MatcherCast<int>('c');
EXPECT_TRUE(m.Matches(kExpected));
EXPECT_FALSE(m.Matches(kExpected + 1));
}

struct NonImplicitlyConstructibleTypeWithOperatorEq {
friend bool operator==(
const NonImplicitlyConstructibleTypeWithOperatorEq& ,
int rhs) {
return 42 == rhs;
}
friend bool operator==(
int lhs,
const NonImplicitlyConstructibleTypeWithOperatorEq& ) {
return lhs == 42;
}
};

TEST(MatcherCastTest, NonImplicitlyConstructibleTypeWithOperatorEq) {
Matcher<NonImplicitlyConstructibleTypeWithOperatorEq> m1 =
MatcherCast<NonImplicitlyConstructibleTypeWithOperatorEq>(42);
EXPECT_TRUE(m1.Matches(NonImplicitlyConstructibleTypeWithOperatorEq()));

Matcher<NonImplicitlyConstructibleTypeWithOperatorEq> m2 =
MatcherCast<NonImplicitlyConstructibleTypeWithOperatorEq>(239);
EXPECT_FALSE(m2.Matches(NonImplicitlyConstructibleTypeWithOperatorEq()));

Matcher<int> m3 =
MatcherCast<int>(NonImplicitlyConstructibleTypeWithOperatorEq());
EXPECT_TRUE(m3.Matches(42));
EXPECT_FALSE(m3.Matches(239));
}


#if !defined _MSC_VER

namespace convertible_from_any {
struct ConvertibleFromAny {
ConvertibleFromAny(int a_value) : value(a_value) {}
template <typename T>
ConvertibleFromAny(const T& ) : value(-1) {
ADD_FAILURE() << "Conversion constructor called";
}
int value;
};

bool operator==(const ConvertibleFromAny& a, const ConvertibleFromAny& b) {
return a.value == b.value;
}

ostream& operator<<(ostream& os, const ConvertibleFromAny& a) {
return os << a.value;
}

TEST(MatcherCastTest, ConversionConstructorIsUsed) {
Matcher<ConvertibleFromAny> m = MatcherCast<ConvertibleFromAny>(1);
EXPECT_TRUE(m.Matches(ConvertibleFromAny(1)));
EXPECT_FALSE(m.Matches(ConvertibleFromAny(2)));
}

TEST(MatcherCastTest, FromConvertibleFromAny) {
Matcher<ConvertibleFromAny> m =
MatcherCast<ConvertibleFromAny>(Eq(ConvertibleFromAny(1)));
EXPECT_TRUE(m.Matches(ConvertibleFromAny(1)));
EXPECT_FALSE(m.Matches(ConvertibleFromAny(2)));
}
}  

#endif  

struct IntReferenceWrapper {
IntReferenceWrapper(const int& a_value) : value(&a_value) {}
const int* value;
};

bool operator==(const IntReferenceWrapper& a, const IntReferenceWrapper& b) {
return a.value == b.value;
}

TEST(MatcherCastTest, ValueIsNotCopied) {
int n = 42;
Matcher<IntReferenceWrapper> m = MatcherCast<IntReferenceWrapper>(n);
EXPECT_TRUE(m.Matches(n));
}

class Base {
public:
virtual ~Base() {}
Base() {}
private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(Base);
};

class Derived : public Base {
public:
Derived() : Base() {}
int i;
};

class OtherDerived : public Base {};

TEST(SafeMatcherCastTest, FromPolymorphicMatcher) {
Matcher<char> m2 = SafeMatcherCast<char>(Eq(32));
EXPECT_TRUE(m2.Matches(' '));
EXPECT_FALSE(m2.Matches('\n'));
}

TEST(SafeMatcherCastTest, FromLosslesslyConvertibleArithmeticType) {
Matcher<double> m1 = DoubleEq(1.0);
Matcher<float> m2 = SafeMatcherCast<float>(m1);
EXPECT_TRUE(m2.Matches(1.0f));
EXPECT_FALSE(m2.Matches(2.0f));

Matcher<char> m3 = SafeMatcherCast<char>(TypedEq<int>('a'));
EXPECT_TRUE(m3.Matches('a'));
EXPECT_FALSE(m3.Matches('b'));
}

TEST(SafeMatcherCastTest, FromBaseClass) {
Derived d, d2;
Matcher<Base*> m1 = Eq(&d);
Matcher<Derived*> m2 = SafeMatcherCast<Derived*>(m1);
EXPECT_TRUE(m2.Matches(&d));
EXPECT_FALSE(m2.Matches(&d2));

Matcher<Base&> m3 = Ref(d);
Matcher<Derived&> m4 = SafeMatcherCast<Derived&>(m3);
EXPECT_TRUE(m4.Matches(d));
EXPECT_FALSE(m4.Matches(d2));
}

TEST(SafeMatcherCastTest, FromConstReferenceToReference) {
int n = 0;
Matcher<const int&> m1 = Ref(n);
Matcher<int&> m2 = SafeMatcherCast<int&>(m1);
int n1 = 0;
EXPECT_TRUE(m2.Matches(n));
EXPECT_FALSE(m2.Matches(n1));
}

TEST(SafeMatcherCastTest, FromNonReferenceToConstReference) {
Matcher<std::unique_ptr<int>> m1 = IsNull();
Matcher<const std::unique_ptr<int>&> m2 =
SafeMatcherCast<const std::unique_ptr<int>&>(m1);
EXPECT_TRUE(m2.Matches(std::unique_ptr<int>()));
EXPECT_FALSE(m2.Matches(std::unique_ptr<int>(new int)));
}

TEST(SafeMatcherCastTest, FromNonReferenceToReference) {
Matcher<int> m1 = Eq(0);
Matcher<int&> m2 = SafeMatcherCast<int&>(m1);
int n = 0;
EXPECT_TRUE(m2.Matches(n));
n = 1;
EXPECT_FALSE(m2.Matches(n));
}

TEST(SafeMatcherCastTest, FromSameType) {
Matcher<int> m1 = Eq(0);
Matcher<int> m2 = SafeMatcherCast<int>(m1);
EXPECT_TRUE(m2.Matches(0));
EXPECT_FALSE(m2.Matches(1));
}

#if !defined _MSC_VER

namespace convertible_from_any {
TEST(SafeMatcherCastTest, ConversionConstructorIsUsed) {
Matcher<ConvertibleFromAny> m = SafeMatcherCast<ConvertibleFromAny>(1);
EXPECT_TRUE(m.Matches(ConvertibleFromAny(1)));
EXPECT_FALSE(m.Matches(ConvertibleFromAny(2)));
}

TEST(SafeMatcherCastTest, FromConvertibleFromAny) {
Matcher<ConvertibleFromAny> m =
SafeMatcherCast<ConvertibleFromAny>(Eq(ConvertibleFromAny(1)));
EXPECT_TRUE(m.Matches(ConvertibleFromAny(1)));
EXPECT_FALSE(m.Matches(ConvertibleFromAny(2)));
}
}  

#endif  

TEST(SafeMatcherCastTest, ValueIsNotCopied) {
int n = 42;
Matcher<IntReferenceWrapper> m = SafeMatcherCast<IntReferenceWrapper>(n);
EXPECT_TRUE(m.Matches(n));
}

TEST(ExpectThat, TakesLiterals) {
EXPECT_THAT(1, 1);
EXPECT_THAT(1.0, 1.0);
EXPECT_THAT(std::string(), "");
}

TEST(ExpectThat, TakesFunctions) {
struct Helper {
static void Func() {}
};
void (*func)() = Helper::Func;
EXPECT_THAT(func, Helper::Func);
EXPECT_THAT(func, &Helper::Func);
}

TEST(ATest, MatchesAnyValue) {
Matcher<double> m1 = A<double>();
EXPECT_TRUE(m1.Matches(91.43));
EXPECT_TRUE(m1.Matches(-15.32));

int a = 2;
int b = -6;
Matcher<int&> m2 = A<int&>();
EXPECT_TRUE(m2.Matches(a));
EXPECT_TRUE(m2.Matches(b));
}

TEST(ATest, WorksForDerivedClass) {
Base base;
Derived derived;
EXPECT_THAT(&base, A<Base*>());
EXPECT_THAT(&derived, A<Base*>());
EXPECT_THAT(&derived, A<Derived*>());
}

TEST(ATest, CanDescribeSelf) {
EXPECT_EQ("is anything", Describe(A<bool>()));
}

TEST(AnTest, MatchesAnyValue) {
Matcher<int> m1 = An<int>();
EXPECT_TRUE(m1.Matches(9143));
EXPECT_TRUE(m1.Matches(-1532));

int a = 2;
int b = -6;
Matcher<int&> m2 = An<int&>();
EXPECT_TRUE(m2.Matches(a));
EXPECT_TRUE(m2.Matches(b));
}

TEST(AnTest, CanDescribeSelf) {
EXPECT_EQ("is anything", Describe(An<int>()));
}

TEST(UnderscoreTest, MatchesAnyValue) {
Matcher<int> m1 = _;
EXPECT_TRUE(m1.Matches(123));
EXPECT_TRUE(m1.Matches(-242));

bool a = false;
const bool b = true;
Matcher<const bool&> m2 = _;
EXPECT_TRUE(m2.Matches(a));
EXPECT_TRUE(m2.Matches(b));
}

TEST(UnderscoreTest, CanDescribeSelf) {
Matcher<int> m = _;
EXPECT_EQ("is anything", Describe(m));
}

TEST(EqTest, MatchesEqualValue) {
const char a1[] = "hi";
const char a2[] = "hi";

Matcher<const char*> m1 = Eq(a1);
EXPECT_TRUE(m1.Matches(a1));
EXPECT_FALSE(m1.Matches(a2));
}


class Unprintable {
public:
Unprintable() : c_('a') {}

bool operator==(const Unprintable& ) const { return true; }
char dummy_c() { return c_; }
private:
char c_;
};

TEST(EqTest, CanDescribeSelf) {
Matcher<Unprintable> m = Eq(Unprintable());
EXPECT_EQ("is equal to 1-byte object <61>", Describe(m));
}

TEST(EqTest, IsPolymorphic) {
Matcher<int> m1 = Eq(1);
EXPECT_TRUE(m1.Matches(1));
EXPECT_FALSE(m1.Matches(2));

Matcher<char> m2 = Eq(1);
EXPECT_TRUE(m2.Matches('\1'));
EXPECT_FALSE(m2.Matches('a'));
}

TEST(TypedEqTest, ChecksEqualityForGivenType) {
Matcher<char> m1 = TypedEq<char>('a');
EXPECT_TRUE(m1.Matches('a'));
EXPECT_FALSE(m1.Matches('b'));

Matcher<int> m2 = TypedEq<int>(6);
EXPECT_TRUE(m2.Matches(6));
EXPECT_FALSE(m2.Matches(7));
}

TEST(TypedEqTest, CanDescribeSelf) {
EXPECT_EQ("is equal to 2", Describe(TypedEq<int>(2)));
}


template <typename T>
struct Type {
static bool IsTypeOf(const T& ) { return true; }

template <typename T2>
static void IsTypeOf(T2 v);
};

TEST(TypedEqTest, HasSpecifiedType) {
Type<Matcher<int> >::IsTypeOf(TypedEq<int>(5));
Type<Matcher<double> >::IsTypeOf(TypedEq<double>(5));
}

TEST(GeTest, ImplementsGreaterThanOrEqual) {
Matcher<int> m1 = Ge(0);
EXPECT_TRUE(m1.Matches(1));
EXPECT_TRUE(m1.Matches(0));
EXPECT_FALSE(m1.Matches(-1));
}

TEST(GeTest, CanDescribeSelf) {
Matcher<int> m = Ge(5);
EXPECT_EQ("is >= 5", Describe(m));
}

TEST(GtTest, ImplementsGreaterThan) {
Matcher<double> m1 = Gt(0);
EXPECT_TRUE(m1.Matches(1.0));
EXPECT_FALSE(m1.Matches(0.0));
EXPECT_FALSE(m1.Matches(-1.0));
}

TEST(GtTest, CanDescribeSelf) {
Matcher<int> m = Gt(5);
EXPECT_EQ("is > 5", Describe(m));
}

TEST(LeTest, ImplementsLessThanOrEqual) {
Matcher<char> m1 = Le('b');
EXPECT_TRUE(m1.Matches('a'));
EXPECT_TRUE(m1.Matches('b'));
EXPECT_FALSE(m1.Matches('c'));
}

TEST(LeTest, CanDescribeSelf) {
Matcher<int> m = Le(5);
EXPECT_EQ("is <= 5", Describe(m));
}

TEST(LtTest, ImplementsLessThan) {
Matcher<const std::string&> m1 = Lt("Hello");
EXPECT_TRUE(m1.Matches("Abc"));
EXPECT_FALSE(m1.Matches("Hello"));
EXPECT_FALSE(m1.Matches("Hello, world!"));
}

TEST(LtTest, CanDescribeSelf) {
Matcher<int> m = Lt(5);
EXPECT_EQ("is < 5", Describe(m));
}

TEST(NeTest, ImplementsNotEqual) {
Matcher<int> m1 = Ne(0);
EXPECT_TRUE(m1.Matches(1));
EXPECT_TRUE(m1.Matches(-1));
EXPECT_FALSE(m1.Matches(0));
}

TEST(NeTest, CanDescribeSelf) {
Matcher<int> m = Ne(5);
EXPECT_EQ("isn't equal to 5", Describe(m));
}

class MoveOnly {
public:
explicit MoveOnly(int i) : i_(i) {}
MoveOnly(const MoveOnly&) = delete;
MoveOnly(MoveOnly&&) = default;
MoveOnly& operator=(const MoveOnly&) = delete;
MoveOnly& operator=(MoveOnly&&) = default;

bool operator==(const MoveOnly& other) const { return i_ == other.i_; }
bool operator!=(const MoveOnly& other) const { return i_ != other.i_; }
bool operator<(const MoveOnly& other) const { return i_ < other.i_; }
bool operator<=(const MoveOnly& other) const { return i_ <= other.i_; }
bool operator>(const MoveOnly& other) const { return i_ > other.i_; }
bool operator>=(const MoveOnly& other) const { return i_ >= other.i_; }

private:
int i_;
};

struct MoveHelper {
MOCK_METHOD1(Call, void(MoveOnly));
};

TEST(ComparisonBaseTest, WorksWithMoveOnly) {
MoveOnly m{0};
MoveHelper helper;

EXPECT_CALL(helper, Call(Eq(ByRef(m))));
helper.Call(MoveOnly(0));
EXPECT_CALL(helper, Call(Ne(ByRef(m))));
helper.Call(MoveOnly(1));
EXPECT_CALL(helper, Call(Le(ByRef(m))));
helper.Call(MoveOnly(0));
EXPECT_CALL(helper, Call(Lt(ByRef(m))));
helper.Call(MoveOnly(-1));
EXPECT_CALL(helper, Call(Ge(ByRef(m))));
helper.Call(MoveOnly(0));
EXPECT_CALL(helper, Call(Gt(ByRef(m))));
helper.Call(MoveOnly(1));
}

TEST(IsNullTest, MatchesNullPointer) {
Matcher<int*> m1 = IsNull();
int* p1 = nullptr;
int n = 0;
EXPECT_TRUE(m1.Matches(p1));
EXPECT_FALSE(m1.Matches(&n));

Matcher<const char*> m2 = IsNull();
const char* p2 = nullptr;
EXPECT_TRUE(m2.Matches(p2));
EXPECT_FALSE(m2.Matches("hi"));

Matcher<void*> m3 = IsNull();
void* p3 = nullptr;
EXPECT_TRUE(m3.Matches(p3));
EXPECT_FALSE(m3.Matches(reinterpret_cast<void*>(0xbeef)));
}

TEST(IsNullTest, StdFunction) {
const Matcher<std::function<void()>> m = IsNull();

EXPECT_TRUE(m.Matches(std::function<void()>()));
EXPECT_FALSE(m.Matches([]{}));
}

TEST(IsNullTest, CanDescribeSelf) {
Matcher<int*> m = IsNull();
EXPECT_EQ("is NULL", Describe(m));
EXPECT_EQ("isn't NULL", DescribeNegation(m));
}

TEST(NotNullTest, MatchesNonNullPointer) {
Matcher<int*> m1 = NotNull();
int* p1 = nullptr;
int n = 0;
EXPECT_FALSE(m1.Matches(p1));
EXPECT_TRUE(m1.Matches(&n));

Matcher<const char*> m2 = NotNull();
const char* p2 = nullptr;
EXPECT_FALSE(m2.Matches(p2));
EXPECT_TRUE(m2.Matches("hi"));
}

TEST(NotNullTest, LinkedPtr) {
const Matcher<std::shared_ptr<int>> m = NotNull();
const std::shared_ptr<int> null_p;
const std::shared_ptr<int> non_null_p(new int);

EXPECT_FALSE(m.Matches(null_p));
EXPECT_TRUE(m.Matches(non_null_p));
}

TEST(NotNullTest, ReferenceToConstLinkedPtr) {
const Matcher<const std::shared_ptr<double>&> m = NotNull();
const std::shared_ptr<double> null_p;
const std::shared_ptr<double> non_null_p(new double);

EXPECT_FALSE(m.Matches(null_p));
EXPECT_TRUE(m.Matches(non_null_p));
}

TEST(NotNullTest, StdFunction) {
const Matcher<std::function<void()>> m = NotNull();

EXPECT_TRUE(m.Matches([]{}));
EXPECT_FALSE(m.Matches(std::function<void()>()));
}

TEST(NotNullTest, CanDescribeSelf) {
Matcher<int*> m = NotNull();
EXPECT_EQ("isn't NULL", Describe(m));
}

TEST(RefTest, MatchesSameVariable) {
int a = 0;
int b = 0;
Matcher<int&> m = Ref(a);
EXPECT_TRUE(m.Matches(a));
EXPECT_FALSE(m.Matches(b));
}

TEST(RefTest, CanDescribeSelf) {
int n = 5;
Matcher<int&> m = Ref(n);
stringstream ss;
ss << "references the variable @" << &n << " 5";
EXPECT_EQ(ss.str(), Describe(m));
}

TEST(RefTest, CanBeUsedAsMatcherForConstReference) {
int a = 0;
int b = 0;
Matcher<const int&> m = Ref(a);
EXPECT_TRUE(m.Matches(a));
EXPECT_FALSE(m.Matches(b));
}


TEST(RefTest, IsCovariant) {
Base base, base2;
Derived derived;
Matcher<const Base&> m1 = Ref(base);
EXPECT_TRUE(m1.Matches(base));
EXPECT_FALSE(m1.Matches(base2));
EXPECT_FALSE(m1.Matches(derived));

m1 = Ref(derived);
EXPECT_TRUE(m1.Matches(derived));
EXPECT_FALSE(m1.Matches(base));
EXPECT_FALSE(m1.Matches(base2));
}

TEST(RefTest, ExplainsResult) {
int n = 0;
EXPECT_THAT(Explain(Matcher<const int&>(Ref(n)), n),
StartsWith("which is located @"));

int m = 0;
EXPECT_THAT(Explain(Matcher<const int&>(Ref(n)), m),
StartsWith("which is located @"));
}


template <typename T = std::string>
std::string FromStringLike(internal::StringLike<T> str) {
return std::string(str);
}

TEST(StringLike, TestConversions) {
EXPECT_EQ("foo", FromStringLike("foo"));
EXPECT_EQ("foo", FromStringLike(std::string("foo")));
#if GTEST_INTERNAL_HAS_STRING_VIEW
EXPECT_EQ("foo", FromStringLike(internal::StringView("foo")));
#endif  

EXPECT_EQ("", FromStringLike({}));
EXPECT_EQ("foo", FromStringLike({'f', 'o', 'o'}));
const char buf[] = "foo";
EXPECT_EQ("foo", FromStringLike({buf, buf + 3}));
}

TEST(StrEqTest, MatchesEqualString) {
Matcher<const char*> m = StrEq(std::string("Hello"));
EXPECT_TRUE(m.Matches("Hello"));
EXPECT_FALSE(m.Matches("hello"));
EXPECT_FALSE(m.Matches(nullptr));

Matcher<const std::string&> m2 = StrEq("Hello");
EXPECT_TRUE(m2.Matches("Hello"));
EXPECT_FALSE(m2.Matches("Hi"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
Matcher<const internal::StringView&> m3 =
StrEq(internal::StringView("Hello"));
EXPECT_TRUE(m3.Matches(internal::StringView("Hello")));
EXPECT_FALSE(m3.Matches(internal::StringView("hello")));
EXPECT_FALSE(m3.Matches(internal::StringView()));

Matcher<const internal::StringView&> m_empty = StrEq("");
EXPECT_TRUE(m_empty.Matches(internal::StringView("")));
EXPECT_TRUE(m_empty.Matches(internal::StringView()));
EXPECT_FALSE(m_empty.Matches(internal::StringView("hello")));
#endif  
}

TEST(StrEqTest, CanDescribeSelf) {
Matcher<std::string> m = StrEq("Hi-\'\"?\\\a\b\f\n\r\t\v\xD3");
EXPECT_EQ("is equal to \"Hi-\'\\\"?\\\\\\a\\b\\f\\n\\r\\t\\v\\xD3\"",
Describe(m));

std::string str("01204500800");
str[3] = '\0';
Matcher<std::string> m2 = StrEq(str);
EXPECT_EQ("is equal to \"012\\04500800\"", Describe(m2));
str[0] = str[6] = str[7] = str[9] = str[10] = '\0';
Matcher<std::string> m3 = StrEq(str);
EXPECT_EQ("is equal to \"\\012\\045\\0\\08\\0\\0\"", Describe(m3));
}

TEST(StrNeTest, MatchesUnequalString) {
Matcher<const char*> m = StrNe("Hello");
EXPECT_TRUE(m.Matches(""));
EXPECT_TRUE(m.Matches(nullptr));
EXPECT_FALSE(m.Matches("Hello"));

Matcher<std::string> m2 = StrNe(std::string("Hello"));
EXPECT_TRUE(m2.Matches("hello"));
EXPECT_FALSE(m2.Matches("Hello"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
Matcher<const internal::StringView> m3 = StrNe(internal::StringView("Hello"));
EXPECT_TRUE(m3.Matches(internal::StringView("")));
EXPECT_TRUE(m3.Matches(internal::StringView()));
EXPECT_FALSE(m3.Matches(internal::StringView("Hello")));
#endif  
}

TEST(StrNeTest, CanDescribeSelf) {
Matcher<const char*> m = StrNe("Hi");
EXPECT_EQ("isn't equal to \"Hi\"", Describe(m));
}

TEST(StrCaseEqTest, MatchesEqualStringIgnoringCase) {
Matcher<const char*> m = StrCaseEq(std::string("Hello"));
EXPECT_TRUE(m.Matches("Hello"));
EXPECT_TRUE(m.Matches("hello"));
EXPECT_FALSE(m.Matches("Hi"));
EXPECT_FALSE(m.Matches(nullptr));

Matcher<const std::string&> m2 = StrCaseEq("Hello");
EXPECT_TRUE(m2.Matches("hello"));
EXPECT_FALSE(m2.Matches("Hi"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
Matcher<const internal::StringView&> m3 =
StrCaseEq(internal::StringView("Hello"));
EXPECT_TRUE(m3.Matches(internal::StringView("Hello")));
EXPECT_TRUE(m3.Matches(internal::StringView("hello")));
EXPECT_FALSE(m3.Matches(internal::StringView("Hi")));
EXPECT_FALSE(m3.Matches(internal::StringView()));
#endif  
}

TEST(StrCaseEqTest, MatchesEqualStringWith0IgnoringCase) {
std::string str1("oabocdooeoo");
std::string str2("OABOCDOOEOO");
Matcher<const std::string&> m0 = StrCaseEq(str1);
EXPECT_FALSE(m0.Matches(str2 + std::string(1, '\0')));

str1[3] = str2[3] = '\0';
Matcher<const std::string&> m1 = StrCaseEq(str1);
EXPECT_TRUE(m1.Matches(str2));

str1[0] = str1[6] = str1[7] = str1[10] = '\0';
str2[0] = str2[6] = str2[7] = str2[10] = '\0';
Matcher<const std::string&> m2 = StrCaseEq(str1);
str1[9] = str2[9] = '\0';
EXPECT_FALSE(m2.Matches(str2));

Matcher<const std::string&> m3 = StrCaseEq(str1);
EXPECT_TRUE(m3.Matches(str2));

EXPECT_FALSE(m3.Matches(str2 + "x"));
str2.append(1, '\0');
EXPECT_FALSE(m3.Matches(str2));
EXPECT_FALSE(m3.Matches(std::string(str2, 0, 9)));
}

TEST(StrCaseEqTest, CanDescribeSelf) {
Matcher<std::string> m = StrCaseEq("Hi");
EXPECT_EQ("is equal to (ignoring case) \"Hi\"", Describe(m));
}

TEST(StrCaseNeTest, MatchesUnequalStringIgnoringCase) {
Matcher<const char*> m = StrCaseNe("Hello");
EXPECT_TRUE(m.Matches("Hi"));
EXPECT_TRUE(m.Matches(nullptr));
EXPECT_FALSE(m.Matches("Hello"));
EXPECT_FALSE(m.Matches("hello"));

Matcher<std::string> m2 = StrCaseNe(std::string("Hello"));
EXPECT_TRUE(m2.Matches(""));
EXPECT_FALSE(m2.Matches("Hello"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
Matcher<const internal::StringView> m3 =
StrCaseNe(internal::StringView("Hello"));
EXPECT_TRUE(m3.Matches(internal::StringView("Hi")));
EXPECT_TRUE(m3.Matches(internal::StringView()));
EXPECT_FALSE(m3.Matches(internal::StringView("Hello")));
EXPECT_FALSE(m3.Matches(internal::StringView("hello")));
#endif  
}

TEST(StrCaseNeTest, CanDescribeSelf) {
Matcher<const char*> m = StrCaseNe("Hi");
EXPECT_EQ("isn't equal to (ignoring case) \"Hi\"", Describe(m));
}

TEST(HasSubstrTest, WorksForStringClasses) {
const Matcher<std::string> m1 = HasSubstr("foo");
EXPECT_TRUE(m1.Matches(std::string("I love food.")));
EXPECT_FALSE(m1.Matches(std::string("tofo")));

const Matcher<const std::string&> m2 = HasSubstr("foo");
EXPECT_TRUE(m2.Matches(std::string("I love food.")));
EXPECT_FALSE(m2.Matches(std::string("tofo")));

const Matcher<std::string> m_empty = HasSubstr("");
EXPECT_TRUE(m_empty.Matches(std::string()));
EXPECT_TRUE(m_empty.Matches(std::string("not empty")));
}

TEST(HasSubstrTest, WorksForCStrings) {
const Matcher<char*> m1 = HasSubstr("foo");
EXPECT_TRUE(m1.Matches(const_cast<char*>("I love food.")));
EXPECT_FALSE(m1.Matches(const_cast<char*>("tofo")));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const char*> m2 = HasSubstr("foo");
EXPECT_TRUE(m2.Matches("I love food."));
EXPECT_FALSE(m2.Matches("tofo"));
EXPECT_FALSE(m2.Matches(nullptr));

const Matcher<const char*> m_empty = HasSubstr("");
EXPECT_TRUE(m_empty.Matches("not empty"));
EXPECT_TRUE(m_empty.Matches(""));
EXPECT_FALSE(m_empty.Matches(nullptr));
}

#if GTEST_INTERNAL_HAS_STRING_VIEW
TEST(HasSubstrTest, WorksForStringViewClasses) {
const Matcher<internal::StringView> m1 =
HasSubstr(internal::StringView("foo"));
EXPECT_TRUE(m1.Matches(internal::StringView("I love food.")));
EXPECT_FALSE(m1.Matches(internal::StringView("tofo")));
EXPECT_FALSE(m1.Matches(internal::StringView()));

const Matcher<const internal::StringView&> m2 = HasSubstr("foo");
EXPECT_TRUE(m2.Matches(internal::StringView("I love food.")));
EXPECT_FALSE(m2.Matches(internal::StringView("tofo")));
EXPECT_FALSE(m2.Matches(internal::StringView()));

const Matcher<const internal::StringView&> m3 = HasSubstr("");
EXPECT_TRUE(m3.Matches(internal::StringView("foo")));
EXPECT_TRUE(m3.Matches(internal::StringView("")));
EXPECT_TRUE(m3.Matches(internal::StringView()));
}
#endif  

TEST(HasSubstrTest, CanDescribeSelf) {
Matcher<std::string> m = HasSubstr("foo\n\"");
EXPECT_EQ("has substring \"foo\\n\\\"\"", Describe(m));
}

TEST(KeyTest, CanDescribeSelf) {
Matcher<const pair<std::string, int>&> m = Key("foo");
EXPECT_EQ("has a key that is equal to \"foo\"", Describe(m));
EXPECT_EQ("doesn't have a key that is equal to \"foo\"", DescribeNegation(m));
}

TEST(KeyTest, ExplainsResult) {
Matcher<pair<int, bool> > m = Key(GreaterThan(10));
EXPECT_EQ("whose first field is a value which is 5 less than 10",
Explain(m, make_pair(5, true)));
EXPECT_EQ("whose first field is a value which is 5 more than 10",
Explain(m, make_pair(15, true)));
}

TEST(KeyTest, MatchesCorrectly) {
pair<int, std::string> p(25, "foo");
EXPECT_THAT(p, Key(25));
EXPECT_THAT(p, Not(Key(42)));
EXPECT_THAT(p, Key(Ge(20)));
EXPECT_THAT(p, Not(Key(Lt(25))));
}

TEST(KeyTest, WorksWithMoveOnly) {
pair<std::unique_ptr<int>, std::unique_ptr<int>> p;
EXPECT_THAT(p, Key(Eq(nullptr)));
}

template <size_t I>
struct Tag {};

struct PairWithGet {
int member_1;
std::string member_2;
using first_type = int;
using second_type = std::string;

const int& GetImpl(Tag<0>) const { return member_1; }
const std::string& GetImpl(Tag<1>) const { return member_2; }
};
template <size_t I>
auto get(const PairWithGet& value) -> decltype(value.GetImpl(Tag<I>())) {
return value.GetImpl(Tag<I>());
}
TEST(PairTest, MatchesPairWithGetCorrectly) {
PairWithGet p{25, "foo"};
EXPECT_THAT(p, Key(25));
EXPECT_THAT(p, Not(Key(42)));
EXPECT_THAT(p, Key(Ge(20)));
EXPECT_THAT(p, Not(Key(Lt(25))));

std::vector<PairWithGet> v = {{11, "Foo"}, {29, "gMockIsBestMock"}};
EXPECT_THAT(v, Contains(Key(29)));
}

TEST(KeyTest, SafelyCastsInnerMatcher) {
Matcher<int> is_positive = Gt(0);
Matcher<int> is_negative = Lt(0);
pair<char, bool> p('a', true);
EXPECT_THAT(p, Key(is_positive));
EXPECT_THAT(p, Not(Key(is_negative)));
}

TEST(KeyTest, InsideContainsUsingMap) {
map<int, char> container;
container.insert(make_pair(1, 'a'));
container.insert(make_pair(2, 'b'));
container.insert(make_pair(4, 'c'));
EXPECT_THAT(container, Contains(Key(1)));
EXPECT_THAT(container, Not(Contains(Key(3))));
}

TEST(KeyTest, InsideContainsUsingMultimap) {
multimap<int, char> container;
container.insert(make_pair(1, 'a'));
container.insert(make_pair(2, 'b'));
container.insert(make_pair(4, 'c'));

EXPECT_THAT(container, Not(Contains(Key(25))));
container.insert(make_pair(25, 'd'));
EXPECT_THAT(container, Contains(Key(25)));
container.insert(make_pair(25, 'e'));
EXPECT_THAT(container, Contains(Key(25)));

EXPECT_THAT(container, Contains(Key(1)));
EXPECT_THAT(container, Not(Contains(Key(3))));
}

TEST(PairTest, Typing) {
Matcher<const pair<const char*, int>&> m1 = Pair("foo", 42);
Matcher<const pair<const char*, int> > m2 = Pair("foo", 42);
Matcher<pair<const char*, int> > m3 = Pair("foo", 42);

Matcher<pair<int, const std::string> > m4 = Pair(25, "42");
Matcher<pair<const std::string, int> > m5 = Pair("25", 42);
}

TEST(PairTest, CanDescribeSelf) {
Matcher<const pair<std::string, int>&> m1 = Pair("foo", 42);
EXPECT_EQ("has a first field that is equal to \"foo\""
", and has a second field that is equal to 42",
Describe(m1));
EXPECT_EQ("has a first field that isn't equal to \"foo\""
", or has a second field that isn't equal to 42",
DescribeNegation(m1));
Matcher<const pair<int, int>&> m2 = Not(Pair(Not(13), 42));
EXPECT_EQ("has a first field that isn't equal to 13"
", and has a second field that is equal to 42",
DescribeNegation(m2));
}

TEST(PairTest, CanExplainMatchResultTo) {
const Matcher<pair<int, int> > m = Pair(GreaterThan(0), GreaterThan(0));
EXPECT_EQ("whose first field does not match, which is 1 less than 0",
Explain(m, make_pair(-1, -2)));

EXPECT_EQ("whose second field does not match, which is 2 less than 0",
Explain(m, make_pair(1, -2)));

EXPECT_EQ("whose first field does not match, which is 1 less than 0",
Explain(m, make_pair(-1, 2)));

EXPECT_EQ("whose both fields match, where the first field is a value "
"which is 1 more than 0, and the second field is a value "
"which is 2 more than 0",
Explain(m, make_pair(1, 2)));

const Matcher<pair<int, int> > explain_first = Pair(GreaterThan(0), 0);
EXPECT_EQ("whose both fields match, where the first field is a value "
"which is 1 more than 0",
Explain(explain_first, make_pair(1, 0)));

const Matcher<pair<int, int> > explain_second = Pair(0, GreaterThan(0));
EXPECT_EQ("whose both fields match, where the second field is a value "
"which is 1 more than 0",
Explain(explain_second, make_pair(0, 1)));
}

TEST(PairTest, MatchesCorrectly) {
pair<int, std::string> p(25, "foo");

EXPECT_THAT(p, Pair(25, "foo"));
EXPECT_THAT(p, Pair(Ge(20), HasSubstr("o")));

EXPECT_THAT(p, Not(Pair(42, "foo")));
EXPECT_THAT(p, Not(Pair(Lt(25), "foo")));

EXPECT_THAT(p, Not(Pair(25, "bar")));
EXPECT_THAT(p, Not(Pair(25, Not("foo"))));

EXPECT_THAT(p, Not(Pair(13, "bar")));
EXPECT_THAT(p, Not(Pair(Lt(13), HasSubstr("a"))));
}

TEST(PairTest, WorksWithMoveOnly) {
pair<std::unique_ptr<int>, std::unique_ptr<int>> p;
p.second.reset(new int(7));
EXPECT_THAT(p, Pair(Eq(nullptr), Ne(nullptr)));
}

TEST(PairTest, SafelyCastsInnerMatchers) {
Matcher<int> is_positive = Gt(0);
Matcher<int> is_negative = Lt(0);
pair<char, bool> p('a', true);
EXPECT_THAT(p, Pair(is_positive, _));
EXPECT_THAT(p, Not(Pair(is_negative, _)));
EXPECT_THAT(p, Pair(_, is_positive));
EXPECT_THAT(p, Not(Pair(_, is_negative)));
}

TEST(PairTest, InsideContainsUsingMap) {
map<int, char> container;
container.insert(make_pair(1, 'a'));
container.insert(make_pair(2, 'b'));
container.insert(make_pair(4, 'c'));
EXPECT_THAT(container, Contains(Pair(1, 'a')));
EXPECT_THAT(container, Contains(Pair(1, _)));
EXPECT_THAT(container, Contains(Pair(_, 'a')));
EXPECT_THAT(container, Not(Contains(Pair(3, _))));
}

TEST(ContainsTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(Contains(Pointee(2))));
helper.Call(MakeUniquePtrs({1, 2}));
}

TEST(PairTest, UseGetInsteadOfMembers) {
PairWithGet pair{7, "ABC"};
EXPECT_THAT(pair, Pair(7, "ABC"));
EXPECT_THAT(pair, Pair(Ge(7), HasSubstr("AB")));
EXPECT_THAT(pair, Not(Pair(Lt(7), "ABC")));

std::vector<PairWithGet> v = {{11, "Foo"}, {29, "gMockIsBestMock"}};
EXPECT_THAT(v,
ElementsAre(Pair(11, std::string("Foo")), Pair(Ge(10), Not(""))));
}


TEST(StartsWithTest, MatchesStringWithGivenPrefix) {
const Matcher<const char*> m1 = StartsWith(std::string(""));
EXPECT_TRUE(m1.Matches("Hi"));
EXPECT_TRUE(m1.Matches(""));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const std::string&> m2 = StartsWith("Hi");
EXPECT_TRUE(m2.Matches("Hi"));
EXPECT_TRUE(m2.Matches("Hi Hi!"));
EXPECT_TRUE(m2.Matches("High"));
EXPECT_FALSE(m2.Matches("H"));
EXPECT_FALSE(m2.Matches(" Hi"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
const Matcher<internal::StringView> m_empty =
StartsWith(internal::StringView(""));
EXPECT_TRUE(m_empty.Matches(internal::StringView()));
EXPECT_TRUE(m_empty.Matches(internal::StringView("")));
EXPECT_TRUE(m_empty.Matches(internal::StringView("not empty")));
#endif  
}

TEST(StartsWithTest, CanDescribeSelf) {
Matcher<const std::string> m = StartsWith("Hi");
EXPECT_EQ("starts with \"Hi\"", Describe(m));
}


TEST(EndsWithTest, MatchesStringWithGivenSuffix) {
const Matcher<const char*> m1 = EndsWith("");
EXPECT_TRUE(m1.Matches("Hi"));
EXPECT_TRUE(m1.Matches(""));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const std::string&> m2 = EndsWith(std::string("Hi"));
EXPECT_TRUE(m2.Matches("Hi"));
EXPECT_TRUE(m2.Matches("Wow Hi Hi"));
EXPECT_TRUE(m2.Matches("Super Hi"));
EXPECT_FALSE(m2.Matches("i"));
EXPECT_FALSE(m2.Matches("Hi "));

#if GTEST_INTERNAL_HAS_STRING_VIEW
const Matcher<const internal::StringView&> m4 =
EndsWith(internal::StringView(""));
EXPECT_TRUE(m4.Matches("Hi"));
EXPECT_TRUE(m4.Matches(""));
EXPECT_TRUE(m4.Matches(internal::StringView()));
EXPECT_TRUE(m4.Matches(internal::StringView("")));
#endif  
}

TEST(EndsWithTest, CanDescribeSelf) {
Matcher<const std::string> m = EndsWith("Hi");
EXPECT_EQ("ends with \"Hi\"", Describe(m));
}


TEST(MatchesRegexTest, MatchesStringMatchingGivenRegex) {
const Matcher<const char*> m1 = MatchesRegex("a.*z");
EXPECT_TRUE(m1.Matches("az"));
EXPECT_TRUE(m1.Matches("abcz"));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const std::string&> m2 = MatchesRegex(new RE("a.*z"));
EXPECT_TRUE(m2.Matches("azbz"));
EXPECT_FALSE(m2.Matches("az1"));
EXPECT_FALSE(m2.Matches("1az"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
const Matcher<const internal::StringView&> m3 = MatchesRegex("a.*z");
EXPECT_TRUE(m3.Matches(internal::StringView("az")));
EXPECT_TRUE(m3.Matches(internal::StringView("abcz")));
EXPECT_FALSE(m3.Matches(internal::StringView("1az")));
EXPECT_FALSE(m3.Matches(internal::StringView()));
const Matcher<const internal::StringView&> m4 =
MatchesRegex(internal::StringView(""));
EXPECT_TRUE(m4.Matches(internal::StringView("")));
EXPECT_TRUE(m4.Matches(internal::StringView()));
#endif  
}

TEST(MatchesRegexTest, CanDescribeSelf) {
Matcher<const std::string> m1 = MatchesRegex(std::string("Hi.*"));
EXPECT_EQ("matches regular expression \"Hi.*\"", Describe(m1));

Matcher<const char*> m2 = MatchesRegex(new RE("a.*"));
EXPECT_EQ("matches regular expression \"a.*\"", Describe(m2));

#if GTEST_INTERNAL_HAS_STRING_VIEW
Matcher<const internal::StringView> m3 = MatchesRegex(new RE("0.*"));
EXPECT_EQ("matches regular expression \"0.*\"", Describe(m3));
#endif  
}


TEST(ContainsRegexTest, MatchesStringContainingGivenRegex) {
const Matcher<const char*> m1 = ContainsRegex(std::string("a.*z"));
EXPECT_TRUE(m1.Matches("az"));
EXPECT_TRUE(m1.Matches("0abcz1"));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const std::string&> m2 = ContainsRegex(new RE("a.*z"));
EXPECT_TRUE(m2.Matches("azbz"));
EXPECT_TRUE(m2.Matches("az1"));
EXPECT_FALSE(m2.Matches("1a"));

#if GTEST_INTERNAL_HAS_STRING_VIEW
const Matcher<const internal::StringView&> m3 =
ContainsRegex(new RE("a.*z"));
EXPECT_TRUE(m3.Matches(internal::StringView("azbz")));
EXPECT_TRUE(m3.Matches(internal::StringView("az1")));
EXPECT_FALSE(m3.Matches(internal::StringView("1a")));
EXPECT_FALSE(m3.Matches(internal::StringView()));
const Matcher<const internal::StringView&> m4 =
ContainsRegex(internal::StringView(""));
EXPECT_TRUE(m4.Matches(internal::StringView("")));
EXPECT_TRUE(m4.Matches(internal::StringView()));
#endif  
}

TEST(ContainsRegexTest, CanDescribeSelf) {
Matcher<const std::string> m1 = ContainsRegex("Hi.*");
EXPECT_EQ("contains regular expression \"Hi.*\"", Describe(m1));

Matcher<const char*> m2 = ContainsRegex(new RE("a.*"));
EXPECT_EQ("contains regular expression \"a.*\"", Describe(m2));

#if GTEST_INTERNAL_HAS_STRING_VIEW
Matcher<const internal::StringView> m3 = ContainsRegex(new RE("0.*"));
EXPECT_EQ("contains regular expression \"0.*\"", Describe(m3));
#endif  
}

#if GTEST_HAS_STD_WSTRING
TEST(StdWideStrEqTest, MatchesEqual) {
Matcher<const wchar_t*> m = StrEq(::std::wstring(L"Hello"));
EXPECT_TRUE(m.Matches(L"Hello"));
EXPECT_FALSE(m.Matches(L"hello"));
EXPECT_FALSE(m.Matches(nullptr));

Matcher<const ::std::wstring&> m2 = StrEq(L"Hello");
EXPECT_TRUE(m2.Matches(L"Hello"));
EXPECT_FALSE(m2.Matches(L"Hi"));

Matcher<const ::std::wstring&> m3 = StrEq(L"\xD3\x576\x8D3\xC74D");
EXPECT_TRUE(m3.Matches(L"\xD3\x576\x8D3\xC74D"));
EXPECT_FALSE(m3.Matches(L"\xD3\x576\x8D3\xC74E"));

::std::wstring str(L"01204500800");
str[3] = L'\0';
Matcher<const ::std::wstring&> m4 = StrEq(str);
EXPECT_TRUE(m4.Matches(str));
str[0] = str[6] = str[7] = str[9] = str[10] = L'\0';
Matcher<const ::std::wstring&> m5 = StrEq(str);
EXPECT_TRUE(m5.Matches(str));
}

TEST(StdWideStrEqTest, CanDescribeSelf) {
Matcher< ::std::wstring> m = StrEq(L"Hi-\'\"?\\\a\b\f\n\r\t\v");
EXPECT_EQ("is equal to L\"Hi-\'\\\"?\\\\\\a\\b\\f\\n\\r\\t\\v\"",
Describe(m));

Matcher< ::std::wstring> m2 = StrEq(L"\xD3\x576\x8D3\xC74D");
EXPECT_EQ("is equal to L\"\\xD3\\x576\\x8D3\\xC74D\"",
Describe(m2));

::std::wstring str(L"01204500800");
str[3] = L'\0';
Matcher<const ::std::wstring&> m4 = StrEq(str);
EXPECT_EQ("is equal to L\"012\\04500800\"", Describe(m4));
str[0] = str[6] = str[7] = str[9] = str[10] = L'\0';
Matcher<const ::std::wstring&> m5 = StrEq(str);
EXPECT_EQ("is equal to L\"\\012\\045\\0\\08\\0\\0\"", Describe(m5));
}

TEST(StdWideStrNeTest, MatchesUnequalString) {
Matcher<const wchar_t*> m = StrNe(L"Hello");
EXPECT_TRUE(m.Matches(L""));
EXPECT_TRUE(m.Matches(nullptr));
EXPECT_FALSE(m.Matches(L"Hello"));

Matcher< ::std::wstring> m2 = StrNe(::std::wstring(L"Hello"));
EXPECT_TRUE(m2.Matches(L"hello"));
EXPECT_FALSE(m2.Matches(L"Hello"));
}

TEST(StdWideStrNeTest, CanDescribeSelf) {
Matcher<const wchar_t*> m = StrNe(L"Hi");
EXPECT_EQ("isn't equal to L\"Hi\"", Describe(m));
}

TEST(StdWideStrCaseEqTest, MatchesEqualStringIgnoringCase) {
Matcher<const wchar_t*> m = StrCaseEq(::std::wstring(L"Hello"));
EXPECT_TRUE(m.Matches(L"Hello"));
EXPECT_TRUE(m.Matches(L"hello"));
EXPECT_FALSE(m.Matches(L"Hi"));
EXPECT_FALSE(m.Matches(nullptr));

Matcher<const ::std::wstring&> m2 = StrCaseEq(L"Hello");
EXPECT_TRUE(m2.Matches(L"hello"));
EXPECT_FALSE(m2.Matches(L"Hi"));
}

TEST(StdWideStrCaseEqTest, MatchesEqualStringWith0IgnoringCase) {
::std::wstring str1(L"oabocdooeoo");
::std::wstring str2(L"OABOCDOOEOO");
Matcher<const ::std::wstring&> m0 = StrCaseEq(str1);
EXPECT_FALSE(m0.Matches(str2 + ::std::wstring(1, L'\0')));

str1[3] = str2[3] = L'\0';
Matcher<const ::std::wstring&> m1 = StrCaseEq(str1);
EXPECT_TRUE(m1.Matches(str2));

str1[0] = str1[6] = str1[7] = str1[10] = L'\0';
str2[0] = str2[6] = str2[7] = str2[10] = L'\0';
Matcher<const ::std::wstring&> m2 = StrCaseEq(str1);
str1[9] = str2[9] = L'\0';
EXPECT_FALSE(m2.Matches(str2));

Matcher<const ::std::wstring&> m3 = StrCaseEq(str1);
EXPECT_TRUE(m3.Matches(str2));

EXPECT_FALSE(m3.Matches(str2 + L"x"));
str2.append(1, L'\0');
EXPECT_FALSE(m3.Matches(str2));
EXPECT_FALSE(m3.Matches(::std::wstring(str2, 0, 9)));
}

TEST(StdWideStrCaseEqTest, CanDescribeSelf) {
Matcher< ::std::wstring> m = StrCaseEq(L"Hi");
EXPECT_EQ("is equal to (ignoring case) L\"Hi\"", Describe(m));
}

TEST(StdWideStrCaseNeTest, MatchesUnequalStringIgnoringCase) {
Matcher<const wchar_t*> m = StrCaseNe(L"Hello");
EXPECT_TRUE(m.Matches(L"Hi"));
EXPECT_TRUE(m.Matches(nullptr));
EXPECT_FALSE(m.Matches(L"Hello"));
EXPECT_FALSE(m.Matches(L"hello"));

Matcher< ::std::wstring> m2 = StrCaseNe(::std::wstring(L"Hello"));
EXPECT_TRUE(m2.Matches(L""));
EXPECT_FALSE(m2.Matches(L"Hello"));
}

TEST(StdWideStrCaseNeTest, CanDescribeSelf) {
Matcher<const wchar_t*> m = StrCaseNe(L"Hi");
EXPECT_EQ("isn't equal to (ignoring case) L\"Hi\"", Describe(m));
}

TEST(StdWideHasSubstrTest, WorksForStringClasses) {
const Matcher< ::std::wstring> m1 = HasSubstr(L"foo");
EXPECT_TRUE(m1.Matches(::std::wstring(L"I love food.")));
EXPECT_FALSE(m1.Matches(::std::wstring(L"tofo")));

const Matcher<const ::std::wstring&> m2 = HasSubstr(L"foo");
EXPECT_TRUE(m2.Matches(::std::wstring(L"I love food.")));
EXPECT_FALSE(m2.Matches(::std::wstring(L"tofo")));
}

TEST(StdWideHasSubstrTest, WorksForCStrings) {
const Matcher<wchar_t*> m1 = HasSubstr(L"foo");
EXPECT_TRUE(m1.Matches(const_cast<wchar_t*>(L"I love food.")));
EXPECT_FALSE(m1.Matches(const_cast<wchar_t*>(L"tofo")));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const wchar_t*> m2 = HasSubstr(L"foo");
EXPECT_TRUE(m2.Matches(L"I love food."));
EXPECT_FALSE(m2.Matches(L"tofo"));
EXPECT_FALSE(m2.Matches(nullptr));
}

TEST(StdWideHasSubstrTest, CanDescribeSelf) {
Matcher< ::std::wstring> m = HasSubstr(L"foo\n\"");
EXPECT_EQ("has substring L\"foo\\n\\\"\"", Describe(m));
}


TEST(StdWideStartsWithTest, MatchesStringWithGivenPrefix) {
const Matcher<const wchar_t*> m1 = StartsWith(::std::wstring(L""));
EXPECT_TRUE(m1.Matches(L"Hi"));
EXPECT_TRUE(m1.Matches(L""));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const ::std::wstring&> m2 = StartsWith(L"Hi");
EXPECT_TRUE(m2.Matches(L"Hi"));
EXPECT_TRUE(m2.Matches(L"Hi Hi!"));
EXPECT_TRUE(m2.Matches(L"High"));
EXPECT_FALSE(m2.Matches(L"H"));
EXPECT_FALSE(m2.Matches(L" Hi"));
}

TEST(StdWideStartsWithTest, CanDescribeSelf) {
Matcher<const ::std::wstring> m = StartsWith(L"Hi");
EXPECT_EQ("starts with L\"Hi\"", Describe(m));
}


TEST(StdWideEndsWithTest, MatchesStringWithGivenSuffix) {
const Matcher<const wchar_t*> m1 = EndsWith(L"");
EXPECT_TRUE(m1.Matches(L"Hi"));
EXPECT_TRUE(m1.Matches(L""));
EXPECT_FALSE(m1.Matches(nullptr));

const Matcher<const ::std::wstring&> m2 = EndsWith(::std::wstring(L"Hi"));
EXPECT_TRUE(m2.Matches(L"Hi"));
EXPECT_TRUE(m2.Matches(L"Wow Hi Hi"));
EXPECT_TRUE(m2.Matches(L"Super Hi"));
EXPECT_FALSE(m2.Matches(L"i"));
EXPECT_FALSE(m2.Matches(L"Hi "));
}

TEST(StdWideEndsWithTest, CanDescribeSelf) {
Matcher<const ::std::wstring> m = EndsWith(L"Hi");
EXPECT_EQ("ends with L\"Hi\"", Describe(m));
}

#endif  

typedef ::std::tuple<long, int> Tuple2;  

TEST(Eq2Test, MatchesEqualArguments) {
Matcher<const Tuple2&> m = Eq();
EXPECT_TRUE(m.Matches(Tuple2(5L, 5)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 6)));
}

TEST(Eq2Test, CanDescribeSelf) {
Matcher<const Tuple2&> m = Eq();
EXPECT_EQ("are an equal pair", Describe(m));
}

TEST(Ge2Test, MatchesGreaterThanOrEqualArguments) {
Matcher<const Tuple2&> m = Ge();
EXPECT_TRUE(m.Matches(Tuple2(5L, 4)));
EXPECT_TRUE(m.Matches(Tuple2(5L, 5)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 6)));
}

TEST(Ge2Test, CanDescribeSelf) {
Matcher<const Tuple2&> m = Ge();
EXPECT_EQ("are a pair where the first >= the second", Describe(m));
}

TEST(Gt2Test, MatchesGreaterThanArguments) {
Matcher<const Tuple2&> m = Gt();
EXPECT_TRUE(m.Matches(Tuple2(5L, 4)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 5)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 6)));
}

TEST(Gt2Test, CanDescribeSelf) {
Matcher<const Tuple2&> m = Gt();
EXPECT_EQ("are a pair where the first > the second", Describe(m));
}

TEST(Le2Test, MatchesLessThanOrEqualArguments) {
Matcher<const Tuple2&> m = Le();
EXPECT_TRUE(m.Matches(Tuple2(5L, 6)));
EXPECT_TRUE(m.Matches(Tuple2(5L, 5)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 4)));
}

TEST(Le2Test, CanDescribeSelf) {
Matcher<const Tuple2&> m = Le();
EXPECT_EQ("are a pair where the first <= the second", Describe(m));
}

TEST(Lt2Test, MatchesLessThanArguments) {
Matcher<const Tuple2&> m = Lt();
EXPECT_TRUE(m.Matches(Tuple2(5L, 6)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 5)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 4)));
}

TEST(Lt2Test, CanDescribeSelf) {
Matcher<const Tuple2&> m = Lt();
EXPECT_EQ("are a pair where the first < the second", Describe(m));
}

TEST(Ne2Test, MatchesUnequalArguments) {
Matcher<const Tuple2&> m = Ne();
EXPECT_TRUE(m.Matches(Tuple2(5L, 6)));
EXPECT_TRUE(m.Matches(Tuple2(5L, 4)));
EXPECT_FALSE(m.Matches(Tuple2(5L, 5)));
}

TEST(Ne2Test, CanDescribeSelf) {
Matcher<const Tuple2&> m = Ne();
EXPECT_EQ("are an unequal pair", Describe(m));
}

TEST(PairMatchBaseTest, WorksWithMoveOnly) {
using Pointers = std::tuple<std::unique_ptr<int>, std::unique_ptr<int>>;
Matcher<Pointers> matcher = Eq();
Pointers pointers;
EXPECT_TRUE(matcher.Matches(pointers));
}

TEST(IsNan, FloatMatchesNan) {
float quiet_nan = std::numeric_limits<float>::quiet_NaN();
float other_nan = std::nanf("1");
float real_value = 1.0f;

Matcher<float> m = IsNan();
EXPECT_TRUE(m.Matches(quiet_nan));
EXPECT_TRUE(m.Matches(other_nan));
EXPECT_FALSE(m.Matches(real_value));

Matcher<float&> m_ref = IsNan();
EXPECT_TRUE(m_ref.Matches(quiet_nan));
EXPECT_TRUE(m_ref.Matches(other_nan));
EXPECT_FALSE(m_ref.Matches(real_value));

Matcher<const float&> m_cref = IsNan();
EXPECT_TRUE(m_cref.Matches(quiet_nan));
EXPECT_TRUE(m_cref.Matches(other_nan));
EXPECT_FALSE(m_cref.Matches(real_value));
}

TEST(IsNan, DoubleMatchesNan) {
double quiet_nan = std::numeric_limits<double>::quiet_NaN();
double other_nan = std::nan("1");
double real_value = 1.0;

Matcher<double> m = IsNan();
EXPECT_TRUE(m.Matches(quiet_nan));
EXPECT_TRUE(m.Matches(other_nan));
EXPECT_FALSE(m.Matches(real_value));

Matcher<double&> m_ref = IsNan();
EXPECT_TRUE(m_ref.Matches(quiet_nan));
EXPECT_TRUE(m_ref.Matches(other_nan));
EXPECT_FALSE(m_ref.Matches(real_value));

Matcher<const double&> m_cref = IsNan();
EXPECT_TRUE(m_cref.Matches(quiet_nan));
EXPECT_TRUE(m_cref.Matches(other_nan));
EXPECT_FALSE(m_cref.Matches(real_value));
}

TEST(IsNan, LongDoubleMatchesNan) {
long double quiet_nan = std::numeric_limits<long double>::quiet_NaN();
long double other_nan = std::nan("1");
long double real_value = 1.0;

Matcher<long double> m = IsNan();
EXPECT_TRUE(m.Matches(quiet_nan));
EXPECT_TRUE(m.Matches(other_nan));
EXPECT_FALSE(m.Matches(real_value));

Matcher<long double&> m_ref = IsNan();
EXPECT_TRUE(m_ref.Matches(quiet_nan));
EXPECT_TRUE(m_ref.Matches(other_nan));
EXPECT_FALSE(m_ref.Matches(real_value));

Matcher<const long double&> m_cref = IsNan();
EXPECT_TRUE(m_cref.Matches(quiet_nan));
EXPECT_TRUE(m_cref.Matches(other_nan));
EXPECT_FALSE(m_cref.Matches(real_value));
}

TEST(IsNan, NotMatchesNan) {
Matcher<float> mf = Not(IsNan());
EXPECT_FALSE(mf.Matches(std::numeric_limits<float>::quiet_NaN()));
EXPECT_FALSE(mf.Matches(std::nanf("1")));
EXPECT_TRUE(mf.Matches(1.0));

Matcher<double> md = Not(IsNan());
EXPECT_FALSE(md.Matches(std::numeric_limits<double>::quiet_NaN()));
EXPECT_FALSE(md.Matches(std::nan("1")));
EXPECT_TRUE(md.Matches(1.0));

Matcher<long double> mld = Not(IsNan());
EXPECT_FALSE(mld.Matches(std::numeric_limits<long double>::quiet_NaN()));
EXPECT_FALSE(mld.Matches(std::nanl("1")));
EXPECT_TRUE(mld.Matches(1.0));
}

TEST(IsNan, CanDescribeSelf) {
Matcher<float> mf = IsNan();
EXPECT_EQ("is NaN", Describe(mf));

Matcher<double> md = IsNan();
EXPECT_EQ("is NaN", Describe(md));

Matcher<long double> mld = IsNan();
EXPECT_EQ("is NaN", Describe(mld));
}

TEST(IsNan, CanDescribeSelfWithNot) {
Matcher<float> mf = Not(IsNan());
EXPECT_EQ("isn't NaN", Describe(mf));

Matcher<double> md = Not(IsNan());
EXPECT_EQ("isn't NaN", Describe(md));

Matcher<long double> mld = Not(IsNan());
EXPECT_EQ("isn't NaN", Describe(mld));
}

TEST(FloatEq2Test, MatchesEqualArguments) {
typedef ::std::tuple<float, float> Tpl;
Matcher<const Tpl&> m = FloatEq();
EXPECT_TRUE(m.Matches(Tpl(1.0f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(0.3f, 0.1f + 0.1f + 0.1f)));
EXPECT_FALSE(m.Matches(Tpl(1.1f, 1.0f)));
}

TEST(FloatEq2Test, CanDescribeSelf) {
Matcher<const ::std::tuple<float, float>&> m = FloatEq();
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(NanSensitiveFloatEqTest, MatchesEqualArgumentsWithNaN) {
typedef ::std::tuple<float, float> Tpl;
Matcher<const Tpl&> m = NanSensitiveFloatEq();
EXPECT_TRUE(m.Matches(Tpl(1.0f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(std::numeric_limits<float>::quiet_NaN(),
std::numeric_limits<float>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(1.1f, 1.0f)));
EXPECT_FALSE(m.Matches(Tpl(1.0f, std::numeric_limits<float>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(std::numeric_limits<float>::quiet_NaN(), 1.0f)));
}

TEST(NanSensitiveFloatEqTest, CanDescribeSelfWithNaNs) {
Matcher<const ::std::tuple<float, float>&> m = NanSensitiveFloatEq();
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(DoubleEq2Test, MatchesEqualArguments) {
typedef ::std::tuple<double, double> Tpl;
Matcher<const Tpl&> m = DoubleEq();
EXPECT_TRUE(m.Matches(Tpl(1.0, 1.0)));
EXPECT_TRUE(m.Matches(Tpl(0.3, 0.1 + 0.1 + 0.1)));
EXPECT_FALSE(m.Matches(Tpl(1.1, 1.0)));
}

TEST(DoubleEq2Test, CanDescribeSelf) {
Matcher<const ::std::tuple<double, double>&> m = DoubleEq();
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(NanSensitiveDoubleEqTest, MatchesEqualArgumentsWithNaN) {
typedef ::std::tuple<double, double> Tpl;
Matcher<const Tpl&> m = NanSensitiveDoubleEq();
EXPECT_TRUE(m.Matches(Tpl(1.0f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(std::numeric_limits<double>::quiet_NaN(),
std::numeric_limits<double>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(1.1f, 1.0f)));
EXPECT_FALSE(m.Matches(Tpl(1.0f, std::numeric_limits<double>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(std::numeric_limits<double>::quiet_NaN(), 1.0f)));
}

TEST(NanSensitiveDoubleEqTest, CanDescribeSelfWithNaNs) {
Matcher<const ::std::tuple<double, double>&> m = NanSensitiveDoubleEq();
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(FloatNear2Test, MatchesEqualArguments) {
typedef ::std::tuple<float, float> Tpl;
Matcher<const Tpl&> m = FloatNear(0.5f);
EXPECT_TRUE(m.Matches(Tpl(1.0f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(1.3f, 1.0f)));
EXPECT_FALSE(m.Matches(Tpl(1.8f, 1.0f)));
}

TEST(FloatNear2Test, CanDescribeSelf) {
Matcher<const ::std::tuple<float, float>&> m = FloatNear(0.5f);
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(NanSensitiveFloatNearTest, MatchesNearbyArgumentsWithNaN) {
typedef ::std::tuple<float, float> Tpl;
Matcher<const Tpl&> m = NanSensitiveFloatNear(0.5f);
EXPECT_TRUE(m.Matches(Tpl(1.0f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(1.1f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(std::numeric_limits<float>::quiet_NaN(),
std::numeric_limits<float>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(1.6f, 1.0f)));
EXPECT_FALSE(m.Matches(Tpl(1.0f, std::numeric_limits<float>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(std::numeric_limits<float>::quiet_NaN(), 1.0f)));
}

TEST(NanSensitiveFloatNearTest, CanDescribeSelfWithNaNs) {
Matcher<const ::std::tuple<float, float>&> m = NanSensitiveFloatNear(0.5f);
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(DoubleNear2Test, MatchesEqualArguments) {
typedef ::std::tuple<double, double> Tpl;
Matcher<const Tpl&> m = DoubleNear(0.5);
EXPECT_TRUE(m.Matches(Tpl(1.0, 1.0)));
EXPECT_TRUE(m.Matches(Tpl(1.3, 1.0)));
EXPECT_FALSE(m.Matches(Tpl(1.8, 1.0)));
}

TEST(DoubleNear2Test, CanDescribeSelf) {
Matcher<const ::std::tuple<double, double>&> m = DoubleNear(0.5);
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(NanSensitiveDoubleNearTest, MatchesNearbyArgumentsWithNaN) {
typedef ::std::tuple<double, double> Tpl;
Matcher<const Tpl&> m = NanSensitiveDoubleNear(0.5f);
EXPECT_TRUE(m.Matches(Tpl(1.0f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(1.1f, 1.0f)));
EXPECT_TRUE(m.Matches(Tpl(std::numeric_limits<double>::quiet_NaN(),
std::numeric_limits<double>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(1.6f, 1.0f)));
EXPECT_FALSE(m.Matches(Tpl(1.0f, std::numeric_limits<double>::quiet_NaN())));
EXPECT_FALSE(m.Matches(Tpl(std::numeric_limits<double>::quiet_NaN(), 1.0f)));
}

TEST(NanSensitiveDoubleNearTest, CanDescribeSelfWithNaNs) {
Matcher<const ::std::tuple<double, double>&> m = NanSensitiveDoubleNear(0.5f);
EXPECT_EQ("are an almost-equal pair", Describe(m));
}

TEST(NotTest, NegatesMatcher) {
Matcher<int> m;
m = Not(Eq(2));
EXPECT_TRUE(m.Matches(3));
EXPECT_FALSE(m.Matches(2));
}

TEST(NotTest, CanDescribeSelf) {
Matcher<int> m = Not(Eq(5));
EXPECT_EQ("isn't equal to 5", Describe(m));
}

TEST(NotTest, NotMatcherSafelyCastsMonomorphicMatchers) {
Matcher<int> greater_than_5 = Gt(5);

Matcher<const int&> m = Not(greater_than_5);
Matcher<int&> m2 = Not(greater_than_5);
Matcher<int&> m3 = Not(m);
}

void AllOfMatches(int num, const Matcher<int>& m) {
SCOPED_TRACE(Describe(m));
EXPECT_TRUE(m.Matches(0));
for (int i = 1; i <= num; ++i) {
EXPECT_FALSE(m.Matches(i));
}
EXPECT_TRUE(m.Matches(num + 1));
}

TEST(AllOfTest, MatchesWhenAllMatch) {
Matcher<int> m;
m = AllOf(Le(2), Ge(1));
EXPECT_TRUE(m.Matches(1));
EXPECT_TRUE(m.Matches(2));
EXPECT_FALSE(m.Matches(0));
EXPECT_FALSE(m.Matches(3));

m = AllOf(Gt(0), Ne(1), Ne(2));
EXPECT_TRUE(m.Matches(3));
EXPECT_FALSE(m.Matches(2));
EXPECT_FALSE(m.Matches(1));
EXPECT_FALSE(m.Matches(0));

m = AllOf(Gt(0), Ne(1), Ne(2), Ne(3));
EXPECT_TRUE(m.Matches(4));
EXPECT_FALSE(m.Matches(3));
EXPECT_FALSE(m.Matches(2));
EXPECT_FALSE(m.Matches(1));
EXPECT_FALSE(m.Matches(0));

m = AllOf(Ge(0), Lt(10), Ne(3), Ne(5), Ne(7));
EXPECT_TRUE(m.Matches(0));
EXPECT_TRUE(m.Matches(1));
EXPECT_FALSE(m.Matches(3));

AllOfMatches(2, AllOf(Ne(1), Ne(2)));
AllOfMatches(3, AllOf(Ne(1), Ne(2), Ne(3)));
AllOfMatches(4, AllOf(Ne(1), Ne(2), Ne(3), Ne(4)));
AllOfMatches(5, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5)));
AllOfMatches(6, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6)));
AllOfMatches(7, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6), Ne(7)));
AllOfMatches(8, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6), Ne(7),
Ne(8)));
AllOfMatches(9, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6), Ne(7),
Ne(8), Ne(9)));
AllOfMatches(10, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6), Ne(7), Ne(8),
Ne(9), Ne(10)));
AllOfMatches(
50, AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6), Ne(7), Ne(8), Ne(9),
Ne(10), Ne(11), Ne(12), Ne(13), Ne(14), Ne(15), Ne(16), Ne(17),
Ne(18), Ne(19), Ne(20), Ne(21), Ne(22), Ne(23), Ne(24), Ne(25),
Ne(26), Ne(27), Ne(28), Ne(29), Ne(30), Ne(31), Ne(32), Ne(33),
Ne(34), Ne(35), Ne(36), Ne(37), Ne(38), Ne(39), Ne(40), Ne(41),
Ne(42), Ne(43), Ne(44), Ne(45), Ne(46), Ne(47), Ne(48), Ne(49),
Ne(50)));
}


TEST(AllOfTest, CanDescribeSelf) {
Matcher<int> m;
m = AllOf(Le(2), Ge(1));
EXPECT_EQ("(is <= 2) and (is >= 1)", Describe(m));

m = AllOf(Gt(0), Ne(1), Ne(2));
std::string expected_descr1 =
"(is > 0) and (isn't equal to 1) and (isn't equal to 2)";
EXPECT_EQ(expected_descr1, Describe(m));

m = AllOf(Gt(0), Ne(1), Ne(2), Ne(3));
std::string expected_descr2 =
"(is > 0) and (isn't equal to 1) and (isn't equal to 2) and (isn't equal "
"to 3)";
EXPECT_EQ(expected_descr2, Describe(m));

m = AllOf(Ge(0), Lt(10), Ne(3), Ne(5), Ne(7));
std::string expected_descr3 =
"(is >= 0) and (is < 10) and (isn't equal to 3) and (isn't equal to 5) "
"and (isn't equal to 7)";
EXPECT_EQ(expected_descr3, Describe(m));
}

TEST(AllOfTest, CanDescribeNegation) {
Matcher<int> m;
m = AllOf(Le(2), Ge(1));
std::string expected_descr4 = "(isn't <= 2) or (isn't >= 1)";
EXPECT_EQ(expected_descr4, DescribeNegation(m));

m = AllOf(Gt(0), Ne(1), Ne(2));
std::string expected_descr5 =
"(isn't > 0) or (is equal to 1) or (is equal to 2)";
EXPECT_EQ(expected_descr5, DescribeNegation(m));

m = AllOf(Gt(0), Ne(1), Ne(2), Ne(3));
std::string expected_descr6 =
"(isn't > 0) or (is equal to 1) or (is equal to 2) or (is equal to 3)";
EXPECT_EQ(expected_descr6, DescribeNegation(m));

m = AllOf(Ge(0), Lt(10), Ne(3), Ne(5), Ne(7));
std::string expected_desr7 =
"(isn't >= 0) or (isn't < 10) or (is equal to 3) or (is equal to 5) or "
"(is equal to 7)";
EXPECT_EQ(expected_desr7, DescribeNegation(m));

m = AllOf(Ne(1), Ne(2), Ne(3), Ne(4), Ne(5), Ne(6), Ne(7), Ne(8), Ne(9),
Ne(10), Ne(11));
AllOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
EXPECT_THAT(Describe(m), EndsWith("and (isn't equal to 11)"));
AllOfMatches(11, m);
}

TEST(AllOfTest, AllOfMatcherSafelyCastsMonomorphicMatchers) {
Matcher<int> greater_than_5 = Gt(5);
Matcher<int> less_than_10 = Lt(10);

Matcher<const int&> m = AllOf(greater_than_5, less_than_10);
Matcher<int&> m2 = AllOf(greater_than_5, less_than_10);
Matcher<int&> m3 = AllOf(greater_than_5, m2);

Matcher<const int&> m4 = AllOf(greater_than_5, less_than_10, less_than_10);
Matcher<int&> m5 = AllOf(greater_than_5, less_than_10, less_than_10);
}

TEST(AllOfTest, ExplainsResult) {
Matcher<int> m;

m = AllOf(GreaterThan(10), Lt(30));
EXPECT_EQ("which is 15 more than 10", Explain(m, 25));

m = AllOf(GreaterThan(10), GreaterThan(20));
EXPECT_EQ("which is 20 more than 10, and which is 10 more than 20",
Explain(m, 30));

m = AllOf(GreaterThan(10), Lt(30), GreaterThan(20));
EXPECT_EQ("which is 15 more than 10, and which is 5 more than 20",
Explain(m, 25));

m = AllOf(GreaterThan(10), GreaterThan(20), GreaterThan(30));
EXPECT_EQ("which is 30 more than 10, and which is 20 more than 20, "
"and which is 10 more than 30",
Explain(m, 40));

m = AllOf(GreaterThan(10), GreaterThan(20));
EXPECT_EQ("which is 5 less than 10", Explain(m, 5));

m = AllOf(GreaterThan(10), Lt(30));
EXPECT_EQ("", Explain(m, 40));

m = AllOf(GreaterThan(10), GreaterThan(20));
EXPECT_EQ("which is 5 less than 20", Explain(m, 15));
}

static void AnyOfMatches(int num, const Matcher<int>& m) {
SCOPED_TRACE(Describe(m));
EXPECT_FALSE(m.Matches(0));
for (int i = 1; i <= num; ++i) {
EXPECT_TRUE(m.Matches(i));
}
EXPECT_FALSE(m.Matches(num + 1));
}

static void AnyOfStringMatches(int num, const Matcher<std::string>& m) {
SCOPED_TRACE(Describe(m));
EXPECT_FALSE(m.Matches(std::to_string(0)));

for (int i = 1; i <= num; ++i) {
EXPECT_TRUE(m.Matches(std::to_string(i)));
}
EXPECT_FALSE(m.Matches(std::to_string(num + 1)));
}

TEST(AnyOfTest, MatchesWhenAnyMatches) {
Matcher<int> m;
m = AnyOf(Le(1), Ge(3));
EXPECT_TRUE(m.Matches(1));
EXPECT_TRUE(m.Matches(4));
EXPECT_FALSE(m.Matches(2));

m = AnyOf(Lt(0), Eq(1), Eq(2));
EXPECT_TRUE(m.Matches(-1));
EXPECT_TRUE(m.Matches(1));
EXPECT_TRUE(m.Matches(2));
EXPECT_FALSE(m.Matches(0));

m = AnyOf(Lt(0), Eq(1), Eq(2), Eq(3));
EXPECT_TRUE(m.Matches(-1));
EXPECT_TRUE(m.Matches(1));
EXPECT_TRUE(m.Matches(2));
EXPECT_TRUE(m.Matches(3));
EXPECT_FALSE(m.Matches(0));

m = AnyOf(Le(0), Gt(10), 3, 5, 7);
EXPECT_TRUE(m.Matches(0));
EXPECT_TRUE(m.Matches(11));
EXPECT_TRUE(m.Matches(3));
EXPECT_FALSE(m.Matches(2));

AnyOfMatches(2, AnyOf(1, 2));
AnyOfMatches(3, AnyOf(1, 2, 3));
AnyOfMatches(4, AnyOf(1, 2, 3, 4));
AnyOfMatches(5, AnyOf(1, 2, 3, 4, 5));
AnyOfMatches(6, AnyOf(1, 2, 3, 4, 5, 6));
AnyOfMatches(7, AnyOf(1, 2, 3, 4, 5, 6, 7));
AnyOfMatches(8, AnyOf(1, 2, 3, 4, 5, 6, 7, 8));
AnyOfMatches(9, AnyOf(1, 2, 3, 4, 5, 6, 7, 8, 9));
AnyOfMatches(10, AnyOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(AnyOfTest, VariadicMatchesWhenAnyMatches) {
Matcher<int> m = ::testing::AnyOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

EXPECT_THAT(Describe(m), EndsWith("or (is equal to 11)"));
AnyOfMatches(11, m);
AnyOfMatches(50, AnyOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 46, 47, 48, 49, 50));
AnyOfStringMatches(
50, AnyOf("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
"13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
"23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
"33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
"43", "44", "45", "46", "47", "48", "49", "50"));
}

TEST(ElementsAreTest, HugeMatcher) {
vector<int> test_vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

EXPECT_THAT(test_vector,
ElementsAre(Eq(1), Eq(2), Lt(13), Eq(4), Eq(5), Eq(6), Eq(7),
Eq(8), Eq(9), Eq(10), Gt(1), Eq(12)));
}

TEST(ElementsAreTest, HugeMatcherStr) {
vector<std::string> test_vector{
"literal_string", "", "", "", "", "", "", "", "", "", "", ""};

EXPECT_THAT(test_vector, UnorderedElementsAre("literal_string", _, _, _, _, _,
_, _, _, _, _, _));
}

TEST(ElementsAreTest, HugeMatcherUnordered) {
vector<int> test_vector{2, 1, 8, 5, 4, 6, 7, 3, 9, 12, 11, 10};

EXPECT_THAT(test_vector, UnorderedElementsAre(
Eq(2), Eq(1), Gt(7), Eq(5), Eq(4), Eq(6), Eq(7),
Eq(3), Eq(9), Eq(12), Eq(11), Ne(122)));
}


TEST(AnyOfTest, CanDescribeSelf) {
Matcher<int> m;
m = AnyOf(Le(1), Ge(3));

EXPECT_EQ("(is <= 1) or (is >= 3)",
Describe(m));

m = AnyOf(Lt(0), Eq(1), Eq(2));
EXPECT_EQ("(is < 0) or (is equal to 1) or (is equal to 2)", Describe(m));

m = AnyOf(Lt(0), Eq(1), Eq(2), Eq(3));
EXPECT_EQ("(is < 0) or (is equal to 1) or (is equal to 2) or (is equal to 3)",
Describe(m));

m = AnyOf(Le(0), Gt(10), 3, 5, 7);
EXPECT_EQ(
"(is <= 0) or (is > 10) or (is equal to 3) or (is equal to 5) or (is "
"equal to 7)",
Describe(m));
}

TEST(AnyOfTest, CanDescribeNegation) {
Matcher<int> m;
m = AnyOf(Le(1), Ge(3));
EXPECT_EQ("(isn't <= 1) and (isn't >= 3)",
DescribeNegation(m));

m = AnyOf(Lt(0), Eq(1), Eq(2));
EXPECT_EQ("(isn't < 0) and (isn't equal to 1) and (isn't equal to 2)",
DescribeNegation(m));

m = AnyOf(Lt(0), Eq(1), Eq(2), Eq(3));
EXPECT_EQ(
"(isn't < 0) and (isn't equal to 1) and (isn't equal to 2) and (isn't "
"equal to 3)",
DescribeNegation(m));

m = AnyOf(Le(0), Gt(10), 3, 5, 7);
EXPECT_EQ(
"(isn't <= 0) and (isn't > 10) and (isn't equal to 3) and (isn't equal "
"to 5) and (isn't equal to 7)",
DescribeNegation(m));
}

TEST(AnyOfTest, AnyOfMatcherSafelyCastsMonomorphicMatchers) {
Matcher<int> greater_than_5 = Gt(5);
Matcher<int> less_than_10 = Lt(10);

Matcher<const int&> m = AnyOf(greater_than_5, less_than_10);
Matcher<int&> m2 = AnyOf(greater_than_5, less_than_10);
Matcher<int&> m3 = AnyOf(greater_than_5, m2);

Matcher<const int&> m4 = AnyOf(greater_than_5, less_than_10, less_than_10);
Matcher<int&> m5 = AnyOf(greater_than_5, less_than_10, less_than_10);
}

TEST(AnyOfTest, ExplainsResult) {
Matcher<int> m;

m = AnyOf(GreaterThan(10), Lt(0));
EXPECT_EQ("which is 5 less than 10", Explain(m, 5));

m = AnyOf(GreaterThan(10), GreaterThan(20));
EXPECT_EQ("which is 5 less than 10, and which is 15 less than 20",
Explain(m, 5));

m = AnyOf(GreaterThan(10), Gt(20), GreaterThan(30));
EXPECT_EQ("which is 5 less than 10, and which is 25 less than 30",
Explain(m, 5));

m = AnyOf(GreaterThan(10), GreaterThan(20), GreaterThan(30));
EXPECT_EQ("which is 5 less than 10, and which is 15 less than 20, "
"and which is 25 less than 30",
Explain(m, 5));

m = AnyOf(GreaterThan(10), GreaterThan(20));
EXPECT_EQ("which is 5 more than 10", Explain(m, 15));

m = AnyOf(GreaterThan(10), Lt(30));
EXPECT_EQ("", Explain(m, 0));

m = AnyOf(GreaterThan(30), GreaterThan(20));
EXPECT_EQ("which is 5 more than 20", Explain(m, 25));
}


int IsPositive(double x) {
return x > 0 ? 1 : 0;
}

class IsGreaterThan {
public:
explicit IsGreaterThan(int threshold) : threshold_(threshold) {}

bool operator()(int n) const { return n > threshold_; }

private:
int threshold_;
};

const int foo = 0;

bool ReferencesFooAndIsZero(const int& n) {
return (&n == &foo) && (n == 0);
}

TEST(TrulyTest, MatchesWhatSatisfiesThePredicate) {
Matcher<double> m = Truly(IsPositive);
EXPECT_TRUE(m.Matches(2.0));
EXPECT_FALSE(m.Matches(-1.5));
}

TEST(TrulyTest, CanBeUsedWithFunctor) {
Matcher<int> m = Truly(IsGreaterThan(5));
EXPECT_TRUE(m.Matches(6));
EXPECT_FALSE(m.Matches(4));
}

class ConvertibleToBool {
public:
explicit ConvertibleToBool(int number) : number_(number) {}
operator bool() const { return number_ != 0; }

private:
int number_;
};

ConvertibleToBool IsNotZero(int number) {
return ConvertibleToBool(number);
}

TEST(TrulyTest, PredicateCanReturnAClassConvertibleToBool) {
Matcher<int> m = Truly(IsNotZero);
EXPECT_TRUE(m.Matches(1));
EXPECT_FALSE(m.Matches(0));
}

TEST(TrulyTest, CanDescribeSelf) {
Matcher<double> m = Truly(IsPositive);
EXPECT_EQ("satisfies the given predicate",
Describe(m));
}

TEST(TrulyTest, WorksForByRefArguments) {
Matcher<const int&> m = Truly(ReferencesFooAndIsZero);
EXPECT_TRUE(m.Matches(foo));
int n = 0;
EXPECT_FALSE(m.Matches(n));
}

TEST(MatchesTest, IsSatisfiedByWhatMatchesTheMatcher) {
EXPECT_TRUE(Matches(Ge(0))(1));
EXPECT_FALSE(Matches(Eq('a'))('b'));
}

TEST(MatchesTest, WorksOnByRefArguments) {
int m = 0, n = 0;
EXPECT_TRUE(Matches(AllOf(Ref(n), Eq(0)))(n));
EXPECT_FALSE(Matches(Ref(m))(n));
}

TEST(MatchesTest, WorksWithMatcherOnNonRefType) {
Matcher<int> eq5 = Eq(5);
EXPECT_TRUE(Matches(eq5)(5));
EXPECT_FALSE(Matches(eq5)(2));
}

TEST(ValueTest, WorksWithPolymorphicMatcher) {
EXPECT_TRUE(Value("hi", StartsWith("h")));
EXPECT_FALSE(Value(5, Gt(10)));
}

TEST(ValueTest, WorksWithMonomorphicMatcher) {
const Matcher<int> is_zero = Eq(0);
EXPECT_TRUE(Value(0, is_zero));
EXPECT_FALSE(Value('a', is_zero));

int n = 0;
const Matcher<const int&> ref_n = Ref(n);
EXPECT_TRUE(Value(n, ref_n));
EXPECT_FALSE(Value(1, ref_n));
}

TEST(ExplainMatchResultTest, WorksWithPolymorphicMatcher) {
StringMatchResultListener listener1;
EXPECT_TRUE(ExplainMatchResult(PolymorphicIsEven(), 42, &listener1));
EXPECT_EQ("% 2 == 0", listener1.str());

StringMatchResultListener listener2;
EXPECT_FALSE(ExplainMatchResult(Ge(42), 1.5, &listener2));
EXPECT_EQ("", listener2.str());
}

TEST(ExplainMatchResultTest, WorksWithMonomorphicMatcher) {
const Matcher<int> is_even = PolymorphicIsEven();
StringMatchResultListener listener1;
EXPECT_TRUE(ExplainMatchResult(is_even, 42, &listener1));
EXPECT_EQ("% 2 == 0", listener1.str());

const Matcher<const double&> is_zero = Eq(0);
StringMatchResultListener listener2;
EXPECT_FALSE(ExplainMatchResult(is_zero, 1.5, &listener2));
EXPECT_EQ("", listener2.str());
}

MATCHER(ConstructNoArg, "") { return true; }
MATCHER_P(Construct1Arg, arg1, "") { return true; }
MATCHER_P2(Construct2Args, arg1, arg2, "") { return true; }

TEST(MatcherConstruct, ExplicitVsImplicit) {
{
ConstructNoArgMatcher m = {};
(void)m;
ConstructNoArgMatcher m2;
(void)m2;
}
{
using M = Construct1ArgMatcherP<int>;
EXPECT_TRUE((std::is_constructible<M, int>::value));
EXPECT_FALSE((std::is_convertible<int, M>::value));
}
{
Construct2ArgsMatcherP2<int, double> m = {1, 2.2};
(void)m;
}
}

MATCHER_P(Really, inner_matcher, "") {
return ExplainMatchResult(inner_matcher, arg, result_listener);
}

TEST(ExplainMatchResultTest, WorksInsideMATCHER) {
EXPECT_THAT(0, Really(Eq(0)));
}

TEST(DescribeMatcherTest, WorksWithValue) {
EXPECT_EQ("is equal to 42", DescribeMatcher<int>(42));
EXPECT_EQ("isn't equal to 42", DescribeMatcher<int>(42, true));
}

TEST(DescribeMatcherTest, WorksWithMonomorphicMatcher) {
const Matcher<int> monomorphic = Le(0);
EXPECT_EQ("is <= 0", DescribeMatcher<int>(monomorphic));
EXPECT_EQ("isn't <= 0", DescribeMatcher<int>(monomorphic, true));
}

TEST(DescribeMatcherTest, WorksWithPolymorphicMatcher) {
EXPECT_EQ("is even", DescribeMatcher<int>(PolymorphicIsEven()));
EXPECT_EQ("is odd", DescribeMatcher<int>(PolymorphicIsEven(), true));
}

TEST(AllArgsTest, WorksForTuple) {
EXPECT_THAT(std::make_tuple(1, 2L), AllArgs(Lt()));
EXPECT_THAT(std::make_tuple(2L, 1), Not(AllArgs(Lt())));
}

TEST(AllArgsTest, WorksForNonTuple) {
EXPECT_THAT(42, AllArgs(Gt(0)));
EXPECT_THAT('a', Not(AllArgs(Eq('b'))));
}

class AllArgsHelper {
public:
AllArgsHelper() {}

MOCK_METHOD2(Helper, int(char x, int y));

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(AllArgsHelper);
};

TEST(AllArgsTest, WorksInWithClause) {
AllArgsHelper helper;
ON_CALL(helper, Helper(_, _))
.With(AllArgs(Lt()))
.WillByDefault(Return(1));
EXPECT_CALL(helper, Helper(_, _));
EXPECT_CALL(helper, Helper(_, _))
.With(AllArgs(Gt()))
.WillOnce(Return(2));

EXPECT_EQ(1, helper.Helper('\1', 2));
EXPECT_EQ(2, helper.Helper('a', 1));
}

class OptionalMatchersHelper {
public:
OptionalMatchersHelper() {}

MOCK_METHOD0(NoArgs, int());

MOCK_METHOD1(OneArg, int(int y));

MOCK_METHOD2(TwoArgs, int(char x, int y));

MOCK_METHOD1(Overloaded, int(char x));
MOCK_METHOD2(Overloaded, int(char x, int y));

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(OptionalMatchersHelper);
};

TEST(AllArgsTest, WorksWithoutMatchers) {
OptionalMatchersHelper helper;

ON_CALL(helper, NoArgs).WillByDefault(Return(10));
ON_CALL(helper, OneArg).WillByDefault(Return(20));
ON_CALL(helper, TwoArgs).WillByDefault(Return(30));

EXPECT_EQ(10, helper.NoArgs());
EXPECT_EQ(20, helper.OneArg(1));
EXPECT_EQ(30, helper.TwoArgs('\1', 2));

EXPECT_CALL(helper, NoArgs).Times(1);
EXPECT_CALL(helper, OneArg).WillOnce(Return(100));
EXPECT_CALL(helper, OneArg(17)).WillOnce(Return(200));
EXPECT_CALL(helper, TwoArgs).Times(0);

EXPECT_EQ(10, helper.NoArgs());
EXPECT_EQ(100, helper.OneArg(1));
EXPECT_EQ(200, helper.OneArg(17));
}

TEST(MatcherAssertionTest, WorksWhenMatcherIsSatisfied) {
ASSERT_THAT(5, Ge(2)) << "This should succeed.";
ASSERT_THAT("Foo", EndsWith("oo"));
EXPECT_THAT(2, AllOf(Le(7), Ge(0))) << "This should succeed too.";
EXPECT_THAT("Hello", StartsWith("Hell"));
}

TEST(MatcherAssertionTest, WorksWhenMatcherIsNotSatisfied) {
static unsigned short n;  
n = 5;

EXPECT_FATAL_FAILURE(ASSERT_THAT(n, Gt(10)),
"Value of: n\n"
"Expected: is > 10\n"
"  Actual: 5" + OfType("unsigned short"));
n = 0;
EXPECT_NONFATAL_FAILURE(
EXPECT_THAT(n, AllOf(Le(7), Ge(5))),
"Value of: n\n"
"Expected: (is <= 7) and (is >= 5)\n"
"  Actual: 0" + OfType("unsigned short"));
}

TEST(MatcherAssertionTest, WorksForByRefArguments) {
static int n;
n = 0;
EXPECT_THAT(n, AllOf(Le(7), Ref(n)));
EXPECT_FATAL_FAILURE(ASSERT_THAT(n, Not(Ref(n))),
"Value of: n\n"
"Expected: does not reference the variable @");
EXPECT_FATAL_FAILURE(ASSERT_THAT(n, Not(Ref(n))),
"Actual: 0" + OfType("int") + ", which is located @");
}

TEST(MatcherAssertionTest, WorksForMonomorphicMatcher) {
Matcher<const char*> starts_with_he = StartsWith("he");
ASSERT_THAT("hello", starts_with_he);

Matcher<const std::string&> ends_with_ok = EndsWith("ok");
ASSERT_THAT("book", ends_with_ok);
const std::string bad = "bad";
EXPECT_NONFATAL_FAILURE(EXPECT_THAT(bad, ends_with_ok),
"Value of: bad\n"
"Expected: ends with \"ok\"\n"
"  Actual: \"bad\"");
Matcher<int> is_greater_than_5 = Gt(5);
EXPECT_NONFATAL_FAILURE(EXPECT_THAT(5, is_greater_than_5),
"Value of: 5\n"
"Expected: is > 5\n"
"  Actual: 5" + OfType("int"));
}

template <typename RawType>
class FloatingPointTest : public testing::Test {
protected:
typedef testing::internal::FloatingPoint<RawType> Floating;
typedef typename Floating::Bits Bits;

FloatingPointTest()
: max_ulps_(Floating::kMaxUlps),
zero_bits_(Floating(0).bits()),
one_bits_(Floating(1).bits()),
infinity_bits_(Floating(Floating::Infinity()).bits()),
close_to_positive_zero_(
Floating::ReinterpretBits(zero_bits_ + max_ulps_/2)),
close_to_negative_zero_(
-Floating::ReinterpretBits(zero_bits_ + max_ulps_ - max_ulps_/2)),
further_from_negative_zero_(-Floating::ReinterpretBits(
zero_bits_ + max_ulps_ + 1 - max_ulps_/2)),
close_to_one_(Floating::ReinterpretBits(one_bits_ + max_ulps_)),
further_from_one_(Floating::ReinterpretBits(one_bits_ + max_ulps_ + 1)),
infinity_(Floating::Infinity()),
close_to_infinity_(
Floating::ReinterpretBits(infinity_bits_ - max_ulps_)),
further_from_infinity_(
Floating::ReinterpretBits(infinity_bits_ - max_ulps_ - 1)),
max_(Floating::Max()),
nan1_(Floating::ReinterpretBits(Floating::kExponentBitMask | 1)),
nan2_(Floating::ReinterpretBits(Floating::kExponentBitMask | 200)) {
}

void TestSize() {
EXPECT_EQ(sizeof(RawType), sizeof(Bits));
}

void TestMatches(
testing::internal::FloatingEqMatcher<RawType> (*matcher_maker)(RawType)) {
Matcher<RawType> m1 = matcher_maker(0.0);
EXPECT_TRUE(m1.Matches(-0.0));
EXPECT_TRUE(m1.Matches(close_to_positive_zero_));
EXPECT_TRUE(m1.Matches(close_to_negative_zero_));
EXPECT_FALSE(m1.Matches(1.0));

Matcher<RawType> m2 = matcher_maker(close_to_positive_zero_);
EXPECT_FALSE(m2.Matches(further_from_negative_zero_));

Matcher<RawType> m3 = matcher_maker(1.0);
EXPECT_TRUE(m3.Matches(close_to_one_));
EXPECT_FALSE(m3.Matches(further_from_one_));

EXPECT_FALSE(m3.Matches(0.0));

Matcher<RawType> m4 = matcher_maker(-infinity_);
EXPECT_TRUE(m4.Matches(-close_to_infinity_));

Matcher<RawType> m5 = matcher_maker(infinity_);
EXPECT_TRUE(m5.Matches(close_to_infinity_));

EXPECT_FALSE(m5.Matches(nan1_));

Matcher<const RawType&> m6 = matcher_maker(0.0);
EXPECT_TRUE(m6.Matches(-0.0));
EXPECT_TRUE(m6.Matches(close_to_positive_zero_));
EXPECT_FALSE(m6.Matches(1.0));

Matcher<RawType&> m7 = matcher_maker(0.0);
RawType x = 0.0;
EXPECT_TRUE(m7.Matches(x));
x = 0.01f;
EXPECT_FALSE(m7.Matches(x));
}


const Bits max_ulps_;

const Bits zero_bits_;  
const Bits one_bits_;  
const Bits infinity_bits_;  

const RawType close_to_positive_zero_;
const RawType close_to_negative_zero_;
const RawType further_from_negative_zero_;

const RawType close_to_one_;
const RawType further_from_one_;

const RawType infinity_;
const RawType close_to_infinity_;
const RawType further_from_infinity_;

const RawType max_;

const RawType nan1_;
const RawType nan2_;
};

template <typename RawType>
class FloatingPointNearTest : public FloatingPointTest<RawType> {
protected:
typedef FloatingPointTest<RawType> ParentType;

void TestNearMatches(
testing::internal::FloatingEqMatcher<RawType>
(*matcher_maker)(RawType, RawType)) {
Matcher<RawType> m1 = matcher_maker(0.0, 0.0);
EXPECT_TRUE(m1.Matches(0.0));
EXPECT_TRUE(m1.Matches(-0.0));
EXPECT_FALSE(m1.Matches(ParentType::close_to_positive_zero_));
EXPECT_FALSE(m1.Matches(ParentType::close_to_negative_zero_));
EXPECT_FALSE(m1.Matches(1.0));

Matcher<RawType> m2 = matcher_maker(0.0, 1.0);
EXPECT_TRUE(m2.Matches(0.0));
EXPECT_TRUE(m2.Matches(-0.0));
EXPECT_TRUE(m2.Matches(1.0));
EXPECT_TRUE(m2.Matches(-1.0));
EXPECT_FALSE(m2.Matches(ParentType::close_to_one_));
EXPECT_FALSE(m2.Matches(-ParentType::close_to_one_));

Matcher<RawType> m3 = matcher_maker(ParentType::infinity_, 0.0);
EXPECT_TRUE(m3.Matches(ParentType::infinity_));
EXPECT_FALSE(m3.Matches(ParentType::close_to_infinity_));
EXPECT_FALSE(m3.Matches(-ParentType::infinity_));

Matcher<RawType> m4 = matcher_maker(-ParentType::infinity_, 0.0);
EXPECT_TRUE(m4.Matches(-ParentType::infinity_));
EXPECT_FALSE(m4.Matches(-ParentType::close_to_infinity_));
EXPECT_FALSE(m4.Matches(ParentType::infinity_));

Matcher<RawType> m5 = matcher_maker(ParentType::max_, ParentType::max_);
EXPECT_TRUE(m5.Matches(ParentType::max_));
EXPECT_FALSE(m5.Matches(-ParentType::max_));

Matcher<RawType> m6 = matcher_maker(-ParentType::max_, ParentType::max_);
EXPECT_FALSE(m6.Matches(ParentType::max_));
EXPECT_TRUE(m6.Matches(-ParentType::max_));

Matcher<RawType> m7 = matcher_maker(ParentType::max_, 0);
EXPECT_TRUE(m7.Matches(ParentType::max_));
EXPECT_FALSE(m7.Matches(-ParentType::max_));

Matcher<RawType> m8 = matcher_maker(-ParentType::max_, 0);
EXPECT_FALSE(m8.Matches(ParentType::max_));
EXPECT_TRUE(m8.Matches(-ParentType::max_));

Matcher<RawType> m9 = matcher_maker(
ParentType::max_, ParentType::infinity_);
EXPECT_TRUE(m8.Matches(-ParentType::max_));

Matcher<const RawType&> m10 = matcher_maker(0.0, 1.0);
EXPECT_TRUE(m10.Matches(-0.0));
EXPECT_TRUE(m10.Matches(ParentType::close_to_positive_zero_));
EXPECT_FALSE(m10.Matches(ParentType::close_to_one_));

Matcher<RawType&> m11 = matcher_maker(0.0, 1.0);
RawType x = 0.0;
EXPECT_TRUE(m11.Matches(x));
x = 1.0f;
EXPECT_TRUE(m11.Matches(x));
x = -1.0f;
EXPECT_TRUE(m11.Matches(x));
x = 1.1f;
EXPECT_FALSE(m11.Matches(x));
x = -1.1f;
EXPECT_FALSE(m11.Matches(x));
}
};

typedef FloatingPointTest<float> FloatTest;

TEST_F(FloatTest, FloatEqApproximatelyMatchesFloats) {
TestMatches(&FloatEq);
}

TEST_F(FloatTest, NanSensitiveFloatEqApproximatelyMatchesFloats) {
TestMatches(&NanSensitiveFloatEq);
}

TEST_F(FloatTest, FloatEqCannotMatchNaN) {
Matcher<float> m = FloatEq(nan1_);
EXPECT_FALSE(m.Matches(nan1_));
EXPECT_FALSE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST_F(FloatTest, NanSensitiveFloatEqCanMatchNaN) {
Matcher<float> m = NanSensitiveFloatEq(nan1_);
EXPECT_TRUE(m.Matches(nan1_));
EXPECT_TRUE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST_F(FloatTest, FloatEqCanDescribeSelf) {
Matcher<float> m1 = FloatEq(2.0f);
EXPECT_EQ("is approximately 2", Describe(m1));
EXPECT_EQ("isn't approximately 2", DescribeNegation(m1));

Matcher<float> m2 = FloatEq(0.5f);
EXPECT_EQ("is approximately 0.5", Describe(m2));
EXPECT_EQ("isn't approximately 0.5", DescribeNegation(m2));

Matcher<float> m3 = FloatEq(nan1_);
EXPECT_EQ("never matches", Describe(m3));
EXPECT_EQ("is anything", DescribeNegation(m3));
}

TEST_F(FloatTest, NanSensitiveFloatEqCanDescribeSelf) {
Matcher<float> m1 = NanSensitiveFloatEq(2.0f);
EXPECT_EQ("is approximately 2", Describe(m1));
EXPECT_EQ("isn't approximately 2", DescribeNegation(m1));

Matcher<float> m2 = NanSensitiveFloatEq(0.5f);
EXPECT_EQ("is approximately 0.5", Describe(m2));
EXPECT_EQ("isn't approximately 0.5", DescribeNegation(m2));

Matcher<float> m3 = NanSensitiveFloatEq(nan1_);
EXPECT_EQ("is NaN", Describe(m3));
EXPECT_EQ("isn't NaN", DescribeNegation(m3));
}

typedef FloatingPointNearTest<float> FloatNearTest;

TEST_F(FloatNearTest, FloatNearMatches) {
TestNearMatches(&FloatNear);
}

TEST_F(FloatNearTest, NanSensitiveFloatNearApproximatelyMatchesFloats) {
TestNearMatches(&NanSensitiveFloatNear);
}

TEST_F(FloatNearTest, FloatNearCanDescribeSelf) {
Matcher<float> m1 = FloatNear(2.0f, 0.5f);
EXPECT_EQ("is approximately 2 (absolute error <= 0.5)", Describe(m1));
EXPECT_EQ(
"isn't approximately 2 (absolute error > 0.5)", DescribeNegation(m1));

Matcher<float> m2 = FloatNear(0.5f, 0.5f);
EXPECT_EQ("is approximately 0.5 (absolute error <= 0.5)", Describe(m2));
EXPECT_EQ(
"isn't approximately 0.5 (absolute error > 0.5)", DescribeNegation(m2));

Matcher<float> m3 = FloatNear(nan1_, 0.0);
EXPECT_EQ("never matches", Describe(m3));
EXPECT_EQ("is anything", DescribeNegation(m3));
}

TEST_F(FloatNearTest, NanSensitiveFloatNearCanDescribeSelf) {
Matcher<float> m1 = NanSensitiveFloatNear(2.0f, 0.5f);
EXPECT_EQ("is approximately 2 (absolute error <= 0.5)", Describe(m1));
EXPECT_EQ(
"isn't approximately 2 (absolute error > 0.5)", DescribeNegation(m1));

Matcher<float> m2 = NanSensitiveFloatNear(0.5f, 0.5f);
EXPECT_EQ("is approximately 0.5 (absolute error <= 0.5)", Describe(m2));
EXPECT_EQ(
"isn't approximately 0.5 (absolute error > 0.5)", DescribeNegation(m2));

Matcher<float> m3 = NanSensitiveFloatNear(nan1_, 0.1f);
EXPECT_EQ("is NaN", Describe(m3));
EXPECT_EQ("isn't NaN", DescribeNegation(m3));
}

TEST_F(FloatNearTest, FloatNearCannotMatchNaN) {
Matcher<float> m = FloatNear(ParentType::nan1_, 0.1f);
EXPECT_FALSE(m.Matches(nan1_));
EXPECT_FALSE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST_F(FloatNearTest, NanSensitiveFloatNearCanMatchNaN) {
Matcher<float> m = NanSensitiveFloatNear(nan1_, 0.1f);
EXPECT_TRUE(m.Matches(nan1_));
EXPECT_TRUE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

typedef FloatingPointTest<double> DoubleTest;

TEST_F(DoubleTest, DoubleEqApproximatelyMatchesDoubles) {
TestMatches(&DoubleEq);
}

TEST_F(DoubleTest, NanSensitiveDoubleEqApproximatelyMatchesDoubles) {
TestMatches(&NanSensitiveDoubleEq);
}

TEST_F(DoubleTest, DoubleEqCannotMatchNaN) {
Matcher<double> m = DoubleEq(nan1_);
EXPECT_FALSE(m.Matches(nan1_));
EXPECT_FALSE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST_F(DoubleTest, NanSensitiveDoubleEqCanMatchNaN) {
Matcher<double> m = NanSensitiveDoubleEq(nan1_);
EXPECT_TRUE(m.Matches(nan1_));
EXPECT_TRUE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST_F(DoubleTest, DoubleEqCanDescribeSelf) {
Matcher<double> m1 = DoubleEq(2.0);
EXPECT_EQ("is approximately 2", Describe(m1));
EXPECT_EQ("isn't approximately 2", DescribeNegation(m1));

Matcher<double> m2 = DoubleEq(0.5);
EXPECT_EQ("is approximately 0.5", Describe(m2));
EXPECT_EQ("isn't approximately 0.5", DescribeNegation(m2));

Matcher<double> m3 = DoubleEq(nan1_);
EXPECT_EQ("never matches", Describe(m3));
EXPECT_EQ("is anything", DescribeNegation(m3));
}

TEST_F(DoubleTest, NanSensitiveDoubleEqCanDescribeSelf) {
Matcher<double> m1 = NanSensitiveDoubleEq(2.0);
EXPECT_EQ("is approximately 2", Describe(m1));
EXPECT_EQ("isn't approximately 2", DescribeNegation(m1));

Matcher<double> m2 = NanSensitiveDoubleEq(0.5);
EXPECT_EQ("is approximately 0.5", Describe(m2));
EXPECT_EQ("isn't approximately 0.5", DescribeNegation(m2));

Matcher<double> m3 = NanSensitiveDoubleEq(nan1_);
EXPECT_EQ("is NaN", Describe(m3));
EXPECT_EQ("isn't NaN", DescribeNegation(m3));
}

typedef FloatingPointNearTest<double> DoubleNearTest;

TEST_F(DoubleNearTest, DoubleNearMatches) {
TestNearMatches(&DoubleNear);
}

TEST_F(DoubleNearTest, NanSensitiveDoubleNearApproximatelyMatchesDoubles) {
TestNearMatches(&NanSensitiveDoubleNear);
}

TEST_F(DoubleNearTest, DoubleNearCanDescribeSelf) {
Matcher<double> m1 = DoubleNear(2.0, 0.5);
EXPECT_EQ("is approximately 2 (absolute error <= 0.5)", Describe(m1));
EXPECT_EQ(
"isn't approximately 2 (absolute error > 0.5)", DescribeNegation(m1));

Matcher<double> m2 = DoubleNear(0.5, 0.5);
EXPECT_EQ("is approximately 0.5 (absolute error <= 0.5)", Describe(m2));
EXPECT_EQ(
"isn't approximately 0.5 (absolute error > 0.5)", DescribeNegation(m2));

Matcher<double> m3 = DoubleNear(nan1_, 0.0);
EXPECT_EQ("never matches", Describe(m3));
EXPECT_EQ("is anything", DescribeNegation(m3));
}

TEST_F(DoubleNearTest, ExplainsResultWhenMatchFails) {
EXPECT_EQ("", Explain(DoubleNear(2.0, 0.1), 2.05));
EXPECT_EQ("which is 0.2 from 2", Explain(DoubleNear(2.0, 0.1), 2.2));
EXPECT_EQ("which is -0.3 from 2", Explain(DoubleNear(2.0, 0.1), 1.7));

const std::string explanation =
Explain(DoubleNear(2.1, 1e-10), 2.1 + 1.2e-10);
EXPECT_TRUE(explanation == "which is 1.2e-10 from 2.1" ||  
explanation == "which is 1.2e-010 from 2.1")   
<< " where explanation is \"" << explanation << "\".";
}

TEST_F(DoubleNearTest, NanSensitiveDoubleNearCanDescribeSelf) {
Matcher<double> m1 = NanSensitiveDoubleNear(2.0, 0.5);
EXPECT_EQ("is approximately 2 (absolute error <= 0.5)", Describe(m1));
EXPECT_EQ(
"isn't approximately 2 (absolute error > 0.5)", DescribeNegation(m1));

Matcher<double> m2 = NanSensitiveDoubleNear(0.5, 0.5);
EXPECT_EQ("is approximately 0.5 (absolute error <= 0.5)", Describe(m2));
EXPECT_EQ(
"isn't approximately 0.5 (absolute error > 0.5)", DescribeNegation(m2));

Matcher<double> m3 = NanSensitiveDoubleNear(nan1_, 0.1);
EXPECT_EQ("is NaN", Describe(m3));
EXPECT_EQ("isn't NaN", DescribeNegation(m3));
}

TEST_F(DoubleNearTest, DoubleNearCannotMatchNaN) {
Matcher<double> m = DoubleNear(ParentType::nan1_, 0.1);
EXPECT_FALSE(m.Matches(nan1_));
EXPECT_FALSE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST_F(DoubleNearTest, NanSensitiveDoubleNearCanMatchNaN) {
Matcher<double> m = NanSensitiveDoubleNear(nan1_, 0.1);
EXPECT_TRUE(m.Matches(nan1_));
EXPECT_TRUE(m.Matches(nan2_));
EXPECT_FALSE(m.Matches(1.0));
}

TEST(PointeeTest, RawPointer) {
const Matcher<int*> m = Pointee(Ge(0));

int n = 1;
EXPECT_TRUE(m.Matches(&n));
n = -1;
EXPECT_FALSE(m.Matches(&n));
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(PointeeTest, RawPointerToConst) {
const Matcher<const double*> m = Pointee(Ge(0));

double x = 1;
EXPECT_TRUE(m.Matches(&x));
x = -1;
EXPECT_FALSE(m.Matches(&x));
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(PointeeTest, ReferenceToConstRawPointer) {
const Matcher<int* const &> m = Pointee(Ge(0));

int n = 1;
EXPECT_TRUE(m.Matches(&n));
n = -1;
EXPECT_FALSE(m.Matches(&n));
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(PointeeTest, ReferenceToNonConstRawPointer) {
const Matcher<double* &> m = Pointee(Ge(0));

double x = 1.0;
double* p = &x;
EXPECT_TRUE(m.Matches(p));
x = -1;
EXPECT_FALSE(m.Matches(p));
p = nullptr;
EXPECT_FALSE(m.Matches(p));
}

MATCHER_P(FieldIIs, inner_matcher, "") {
return ExplainMatchResult(inner_matcher, arg.i, result_listener);
}

#if GTEST_HAS_RTTI
TEST(WhenDynamicCastToTest, SameType) {
Derived derived;
derived.i = 4;

Base* as_base_ptr = &derived;
EXPECT_THAT(as_base_ptr, WhenDynamicCastTo<Derived*>(Not(IsNull())));
EXPECT_THAT(as_base_ptr, WhenDynamicCastTo<Derived*>(Pointee(FieldIIs(4))));
EXPECT_THAT(as_base_ptr,
Not(WhenDynamicCastTo<Derived*>(Pointee(FieldIIs(5)))));
}

TEST(WhenDynamicCastToTest, WrongTypes) {
Base base;
Derived derived;
OtherDerived other_derived;

EXPECT_THAT(&base, Not(WhenDynamicCastTo<Derived*>(Pointee(_))));
EXPECT_THAT(&base, WhenDynamicCastTo<Derived*>(IsNull()));
Base* as_base_ptr = &derived;
EXPECT_THAT(as_base_ptr, Not(WhenDynamicCastTo<OtherDerived*>(Pointee(_))));
EXPECT_THAT(as_base_ptr, WhenDynamicCastTo<OtherDerived*>(IsNull()));
as_base_ptr = &other_derived;
EXPECT_THAT(as_base_ptr, Not(WhenDynamicCastTo<Derived*>(Pointee(_))));
EXPECT_THAT(as_base_ptr, WhenDynamicCastTo<Derived*>(IsNull()));
}

TEST(WhenDynamicCastToTest, AlreadyNull) {
Base* as_base_ptr = nullptr;
EXPECT_THAT(as_base_ptr, WhenDynamicCastTo<Derived*>(IsNull()));
}

struct AmbiguousCastTypes {
class VirtualDerived : public virtual Base {};
class DerivedSub1 : public VirtualDerived {};
class DerivedSub2 : public VirtualDerived {};
class ManyDerivedInHierarchy : public DerivedSub1, public DerivedSub2 {};
};

TEST(WhenDynamicCastToTest, AmbiguousCast) {
AmbiguousCastTypes::DerivedSub1 sub1;
AmbiguousCastTypes::ManyDerivedInHierarchy many_derived;
Base* as_base_ptr =
static_cast<AmbiguousCastTypes::DerivedSub1*>(&many_derived);
EXPECT_THAT(as_base_ptr,
WhenDynamicCastTo<AmbiguousCastTypes::VirtualDerived*>(IsNull()));
as_base_ptr = &sub1;
EXPECT_THAT(
as_base_ptr,
WhenDynamicCastTo<AmbiguousCastTypes::VirtualDerived*>(Not(IsNull())));
}

TEST(WhenDynamicCastToTest, Describe) {
Matcher<Base*> matcher = WhenDynamicCastTo<Derived*>(Pointee(_));
const std::string prefix =
"when dynamic_cast to " + internal::GetTypeName<Derived*>() + ", ";
EXPECT_EQ(prefix + "points to a value that is anything", Describe(matcher));
EXPECT_EQ(prefix + "does not point to a value that is anything",
DescribeNegation(matcher));
}

TEST(WhenDynamicCastToTest, Explain) {
Matcher<Base*> matcher = WhenDynamicCastTo<Derived*>(Pointee(_));
Base* null = nullptr;
EXPECT_THAT(Explain(matcher, null), HasSubstr("NULL"));
Derived derived;
EXPECT_TRUE(matcher.Matches(&derived));
EXPECT_THAT(Explain(matcher, &derived), HasSubstr("which points to "));

Matcher<const Base&> ref_matcher = WhenDynamicCastTo<const OtherDerived&>(_);
EXPECT_THAT(Explain(ref_matcher, derived),
HasSubstr("which cannot be dynamic_cast"));
}

TEST(WhenDynamicCastToTest, GoodReference) {
Derived derived;
derived.i = 4;
Base& as_base_ref = derived;
EXPECT_THAT(as_base_ref, WhenDynamicCastTo<const Derived&>(FieldIIs(4)));
EXPECT_THAT(as_base_ref, WhenDynamicCastTo<const Derived&>(Not(FieldIIs(5))));
}

TEST(WhenDynamicCastToTest, BadReference) {
Derived derived;
Base& as_base_ref = derived;
EXPECT_THAT(as_base_ref, Not(WhenDynamicCastTo<const OtherDerived&>(_)));
}
#endif  

template <typename T>
class ConstPropagatingPtr {
public:
typedef T element_type;

ConstPropagatingPtr() : val_() {}
explicit ConstPropagatingPtr(T* t) : val_(t) {}
ConstPropagatingPtr(const ConstPropagatingPtr& other) : val_(other.val_) {}

T* get() { return val_; }
T& operator*() { return *val_; }
const T* get() const { return val_; }
const T& operator*() const { return *val_; }

private:
T* val_;
};

TEST(PointeeTest, WorksWithConstPropagatingPointers) {
const Matcher< ConstPropagatingPtr<int> > m = Pointee(Lt(5));
int three = 3;
const ConstPropagatingPtr<int> co(&three);
ConstPropagatingPtr<int> o(&three);
EXPECT_TRUE(m.Matches(o));
EXPECT_TRUE(m.Matches(co));
*o = 6;
EXPECT_FALSE(m.Matches(o));
EXPECT_FALSE(m.Matches(ConstPropagatingPtr<int>()));
}

TEST(PointeeTest, NeverMatchesNull) {
const Matcher<const char*> m = Pointee(_);
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(PointeeTest, MatchesAgainstAValue) {
const Matcher<int*> m = Pointee(5);

int n = 5;
EXPECT_TRUE(m.Matches(&n));
n = -1;
EXPECT_FALSE(m.Matches(&n));
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(PointeeTest, CanDescribeSelf) {
const Matcher<int*> m = Pointee(Gt(3));
EXPECT_EQ("points to a value that is > 3", Describe(m));
EXPECT_EQ("does not point to a value that is > 3",
DescribeNegation(m));
}

TEST(PointeeTest, CanExplainMatchResult) {
const Matcher<const std::string*> m = Pointee(StartsWith("Hi"));

EXPECT_EQ("", Explain(m, static_cast<const std::string*>(nullptr)));

const Matcher<long*> m2 = Pointee(GreaterThan(1));  
long n = 3;  
EXPECT_EQ("which points to 3" + OfType("long") + ", which is 2 more than 1",
Explain(m2, &n));
}

TEST(PointeeTest, AlwaysExplainsPointee) {
const Matcher<int*> m = Pointee(0);
int n = 42;
EXPECT_EQ("which points to 42" + OfType("int"), Explain(m, &n));
}

class Uncopyable {
public:
Uncopyable() : value_(-1) {}
explicit Uncopyable(int a_value) : value_(a_value) {}

int value() const { return value_; }
void set_value(int i) { value_ = i; }

private:
int value_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(Uncopyable);
};

bool ValueIsPositive(const Uncopyable& x) { return x.value() > 0; }

MATCHER_P(UncopyableIs, inner_matcher, "") {
return ExplainMatchResult(inner_matcher, arg.value(), result_listener);
}

struct AStruct {
AStruct() : x(0), y(1.0), z(5), p(nullptr) {}
AStruct(const AStruct& rhs)
: x(rhs.x), y(rhs.y), z(rhs.z.value()), p(rhs.p) {}

int x;           
const double y;  
Uncopyable z;    
const char* p;   
};

struct DerivedStruct : public AStruct {
char ch;
};

TEST(FieldTest, WorksForNonConstField) {
Matcher<AStruct> m = Field(&AStruct::x, Ge(0));
Matcher<AStruct> m_with_name = Field("x", &AStruct::x, Ge(0));

AStruct a;
EXPECT_TRUE(m.Matches(a));
EXPECT_TRUE(m_with_name.Matches(a));
a.x = -1;
EXPECT_FALSE(m.Matches(a));
EXPECT_FALSE(m_with_name.Matches(a));
}

TEST(FieldTest, WorksForConstField) {
AStruct a;

Matcher<AStruct> m = Field(&AStruct::y, Ge(0.0));
Matcher<AStruct> m_with_name = Field("y", &AStruct::y, Ge(0.0));
EXPECT_TRUE(m.Matches(a));
EXPECT_TRUE(m_with_name.Matches(a));
m = Field(&AStruct::y, Le(0.0));
m_with_name = Field("y", &AStruct::y, Le(0.0));
EXPECT_FALSE(m.Matches(a));
EXPECT_FALSE(m_with_name.Matches(a));
}

TEST(FieldTest, WorksForUncopyableField) {
AStruct a;

Matcher<AStruct> m = Field(&AStruct::z, Truly(ValueIsPositive));
EXPECT_TRUE(m.Matches(a));
m = Field(&AStruct::z, Not(Truly(ValueIsPositive)));
EXPECT_FALSE(m.Matches(a));
}

TEST(FieldTest, WorksForPointerField) {
Matcher<AStruct> m = Field(&AStruct::p, static_cast<const char*>(nullptr));
AStruct a;
EXPECT_TRUE(m.Matches(a));
a.p = "hi";
EXPECT_FALSE(m.Matches(a));

m = Field(&AStruct::p, StartsWith("hi"));
a.p = "hill";
EXPECT_TRUE(m.Matches(a));
a.p = "hole";
EXPECT_FALSE(m.Matches(a));
}

TEST(FieldTest, WorksForByRefArgument) {
Matcher<const AStruct&> m = Field(&AStruct::x, Ge(0));

AStruct a;
EXPECT_TRUE(m.Matches(a));
a.x = -1;
EXPECT_FALSE(m.Matches(a));
}

TEST(FieldTest, WorksForArgumentOfSubType) {
Matcher<const DerivedStruct&> m = Field(&AStruct::x, Ge(0));

DerivedStruct d;
EXPECT_TRUE(m.Matches(d));
d.x = -1;
EXPECT_FALSE(m.Matches(d));
}

TEST(FieldTest, WorksForCompatibleMatcherType) {
Matcher<const AStruct&> m = Field(&AStruct::x,
Matcher<signed char>(Ge(0)));

AStruct a;
EXPECT_TRUE(m.Matches(a));
a.x = -1;
EXPECT_FALSE(m.Matches(a));
}

TEST(FieldTest, CanDescribeSelf) {
Matcher<const AStruct&> m = Field(&AStruct::x, Ge(0));

EXPECT_EQ("is an object whose given field is >= 0", Describe(m));
EXPECT_EQ("is an object whose given field isn't >= 0", DescribeNegation(m));
}

TEST(FieldTest, CanDescribeSelfWithFieldName) {
Matcher<const AStruct&> m = Field("field_name", &AStruct::x, Ge(0));

EXPECT_EQ("is an object whose field `field_name` is >= 0", Describe(m));
EXPECT_EQ("is an object whose field `field_name` isn't >= 0",
DescribeNegation(m));
}

TEST(FieldTest, CanExplainMatchResult) {
Matcher<const AStruct&> m = Field(&AStruct::x, Ge(0));

AStruct a;
a.x = 1;
EXPECT_EQ("whose given field is 1" + OfType("int"), Explain(m, a));

m = Field(&AStruct::x, GreaterThan(0));
EXPECT_EQ(
"whose given field is 1" + OfType("int") + ", which is 1 more than 0",
Explain(m, a));
}

TEST(FieldTest, CanExplainMatchResultWithFieldName) {
Matcher<const AStruct&> m = Field("field_name", &AStruct::x, Ge(0));

AStruct a;
a.x = 1;
EXPECT_EQ("whose field `field_name` is 1" + OfType("int"), Explain(m, a));

m = Field("field_name", &AStruct::x, GreaterThan(0));
EXPECT_EQ("whose field `field_name` is 1" + OfType("int") +
", which is 1 more than 0",
Explain(m, a));
}

TEST(FieldForPointerTest, WorksForPointerToConst) {
Matcher<const AStruct*> m = Field(&AStruct::x, Ge(0));

AStruct a;
EXPECT_TRUE(m.Matches(&a));
a.x = -1;
EXPECT_FALSE(m.Matches(&a));
}

TEST(FieldForPointerTest, WorksForPointerToNonConst) {
Matcher<AStruct*> m = Field(&AStruct::x, Ge(0));

AStruct a;
EXPECT_TRUE(m.Matches(&a));
a.x = -1;
EXPECT_FALSE(m.Matches(&a));
}

TEST(FieldForPointerTest, WorksForReferenceToConstPointer) {
Matcher<AStruct* const&> m = Field(&AStruct::x, Ge(0));

AStruct a;
EXPECT_TRUE(m.Matches(&a));
a.x = -1;
EXPECT_FALSE(m.Matches(&a));
}

TEST(FieldForPointerTest, DoesNotMatchNull) {
Matcher<const AStruct*> m = Field(&AStruct::x, _);
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(FieldForPointerTest, WorksForArgumentOfSubType) {
Matcher<DerivedStruct*> m = Field(&AStruct::x, Ge(0));

DerivedStruct d;
EXPECT_TRUE(m.Matches(&d));
d.x = -1;
EXPECT_FALSE(m.Matches(&d));
}

TEST(FieldForPointerTest, CanDescribeSelf) {
Matcher<const AStruct*> m = Field(&AStruct::x, Ge(0));

EXPECT_EQ("is an object whose given field is >= 0", Describe(m));
EXPECT_EQ("is an object whose given field isn't >= 0", DescribeNegation(m));
}

TEST(FieldForPointerTest, CanDescribeSelfWithFieldName) {
Matcher<const AStruct*> m = Field("field_name", &AStruct::x, Ge(0));

EXPECT_EQ("is an object whose field `field_name` is >= 0", Describe(m));
EXPECT_EQ("is an object whose field `field_name` isn't >= 0",
DescribeNegation(m));
}

TEST(FieldForPointerTest, CanExplainMatchResult) {
Matcher<const AStruct*> m = Field(&AStruct::x, Ge(0));

AStruct a;
a.x = 1;
EXPECT_EQ("", Explain(m, static_cast<const AStruct*>(nullptr)));
EXPECT_EQ("which points to an object whose given field is 1" + OfType("int"),
Explain(m, &a));

m = Field(&AStruct::x, GreaterThan(0));
EXPECT_EQ("which points to an object whose given field is 1" + OfType("int") +
", which is 1 more than 0", Explain(m, &a));
}

TEST(FieldForPointerTest, CanExplainMatchResultWithFieldName) {
Matcher<const AStruct*> m = Field("field_name", &AStruct::x, Ge(0));

AStruct a;
a.x = 1;
EXPECT_EQ("", Explain(m, static_cast<const AStruct*>(nullptr)));
EXPECT_EQ(
"which points to an object whose field `field_name` is 1" + OfType("int"),
Explain(m, &a));

m = Field("field_name", &AStruct::x, GreaterThan(0));
EXPECT_EQ("which points to an object whose field `field_name` is 1" +
OfType("int") + ", which is 1 more than 0",
Explain(m, &a));
}

class AClass {
public:
AClass() : n_(0) {}

int n() const { return n_; }

void set_n(int new_n) { n_ = new_n; }

const std::string& s() const { return s_; }

const std::string& s_ref() const & { return s_; }

void set_s(const std::string& new_s) { s_ = new_s; }

double& x() const { return x_; }

private:
int n_;
std::string s_;

static double x_;
};

double AClass::x_ = 0.0;

class DerivedClass : public AClass {
public:
int k() const { return k_; }
private:
int k_;
};

TEST(PropertyTest, WorksForNonReferenceProperty) {
Matcher<const AClass&> m = Property(&AClass::n, Ge(0));
Matcher<const AClass&> m_with_name = Property("n", &AClass::n, Ge(0));

AClass a;
a.set_n(1);
EXPECT_TRUE(m.Matches(a));
EXPECT_TRUE(m_with_name.Matches(a));

a.set_n(-1);
EXPECT_FALSE(m.Matches(a));
EXPECT_FALSE(m_with_name.Matches(a));
}

TEST(PropertyTest, WorksForReferenceToConstProperty) {
Matcher<const AClass&> m = Property(&AClass::s, StartsWith("hi"));
Matcher<const AClass&> m_with_name =
Property("s", &AClass::s, StartsWith("hi"));

AClass a;
a.set_s("hill");
EXPECT_TRUE(m.Matches(a));
EXPECT_TRUE(m_with_name.Matches(a));

a.set_s("hole");
EXPECT_FALSE(m.Matches(a));
EXPECT_FALSE(m_with_name.Matches(a));
}

TEST(PropertyTest, WorksForRefQualifiedProperty) {
Matcher<const AClass&> m = Property(&AClass::s_ref, StartsWith("hi"));
Matcher<const AClass&> m_with_name =
Property("s", &AClass::s_ref, StartsWith("hi"));

AClass a;
a.set_s("hill");
EXPECT_TRUE(m.Matches(a));
EXPECT_TRUE(m_with_name.Matches(a));

a.set_s("hole");
EXPECT_FALSE(m.Matches(a));
EXPECT_FALSE(m_with_name.Matches(a));
}

TEST(PropertyTest, WorksForReferenceToNonConstProperty) {
double x = 0.0;
AClass a;

Matcher<const AClass&> m = Property(&AClass::x, Ref(x));
EXPECT_FALSE(m.Matches(a));

m = Property(&AClass::x, Not(Ref(x)));
EXPECT_TRUE(m.Matches(a));
}

TEST(PropertyTest, WorksForByValueArgument) {
Matcher<AClass> m = Property(&AClass::s, StartsWith("hi"));

AClass a;
a.set_s("hill");
EXPECT_TRUE(m.Matches(a));

a.set_s("hole");
EXPECT_FALSE(m.Matches(a));
}

TEST(PropertyTest, WorksForArgumentOfSubType) {
Matcher<const DerivedClass&> m = Property(&AClass::n, Ge(0));

DerivedClass d;
d.set_n(1);
EXPECT_TRUE(m.Matches(d));

d.set_n(-1);
EXPECT_FALSE(m.Matches(d));
}

TEST(PropertyTest, WorksForCompatibleMatcherType) {
Matcher<const AClass&> m = Property(&AClass::n,
Matcher<signed char>(Ge(0)));

Matcher<const AClass&> m_with_name =
Property("n", &AClass::n, Matcher<signed char>(Ge(0)));

AClass a;
EXPECT_TRUE(m.Matches(a));
EXPECT_TRUE(m_with_name.Matches(a));
a.set_n(-1);
EXPECT_FALSE(m.Matches(a));
EXPECT_FALSE(m_with_name.Matches(a));
}

TEST(PropertyTest, CanDescribeSelf) {
Matcher<const AClass&> m = Property(&AClass::n, Ge(0));

EXPECT_EQ("is an object whose given property is >= 0", Describe(m));
EXPECT_EQ("is an object whose given property isn't >= 0",
DescribeNegation(m));
}

TEST(PropertyTest, CanDescribeSelfWithPropertyName) {
Matcher<const AClass&> m = Property("fancy_name", &AClass::n, Ge(0));

EXPECT_EQ("is an object whose property `fancy_name` is >= 0", Describe(m));
EXPECT_EQ("is an object whose property `fancy_name` isn't >= 0",
DescribeNegation(m));
}

TEST(PropertyTest, CanExplainMatchResult) {
Matcher<const AClass&> m = Property(&AClass::n, Ge(0));

AClass a;
a.set_n(1);
EXPECT_EQ("whose given property is 1" + OfType("int"), Explain(m, a));

m = Property(&AClass::n, GreaterThan(0));
EXPECT_EQ(
"whose given property is 1" + OfType("int") + ", which is 1 more than 0",
Explain(m, a));
}

TEST(PropertyTest, CanExplainMatchResultWithPropertyName) {
Matcher<const AClass&> m = Property("fancy_name", &AClass::n, Ge(0));

AClass a;
a.set_n(1);
EXPECT_EQ("whose property `fancy_name` is 1" + OfType("int"), Explain(m, a));

m = Property("fancy_name", &AClass::n, GreaterThan(0));
EXPECT_EQ("whose property `fancy_name` is 1" + OfType("int") +
", which is 1 more than 0",
Explain(m, a));
}

TEST(PropertyForPointerTest, WorksForPointerToConst) {
Matcher<const AClass*> m = Property(&AClass::n, Ge(0));

AClass a;
a.set_n(1);
EXPECT_TRUE(m.Matches(&a));

a.set_n(-1);
EXPECT_FALSE(m.Matches(&a));
}

TEST(PropertyForPointerTest, WorksForPointerToNonConst) {
Matcher<AClass*> m = Property(&AClass::s, StartsWith("hi"));

AClass a;
a.set_s("hill");
EXPECT_TRUE(m.Matches(&a));

a.set_s("hole");
EXPECT_FALSE(m.Matches(&a));
}

TEST(PropertyForPointerTest, WorksForReferenceToConstPointer) {
Matcher<AClass* const&> m = Property(&AClass::s, StartsWith("hi"));

AClass a;
a.set_s("hill");
EXPECT_TRUE(m.Matches(&a));

a.set_s("hole");
EXPECT_FALSE(m.Matches(&a));
}

TEST(PropertyForPointerTest, WorksForReferenceToNonConstProperty) {
Matcher<const AClass*> m = Property(&AClass::x, _);
EXPECT_FALSE(m.Matches(nullptr));
}

TEST(PropertyForPointerTest, WorksForArgumentOfSubType) {
Matcher<const DerivedClass*> m = Property(&AClass::n, Ge(0));

DerivedClass d;
d.set_n(1);
EXPECT_TRUE(m.Matches(&d));

d.set_n(-1);
EXPECT_FALSE(m.Matches(&d));
}

TEST(PropertyForPointerTest, CanDescribeSelf) {
Matcher<const AClass*> m = Property(&AClass::n, Ge(0));

EXPECT_EQ("is an object whose given property is >= 0", Describe(m));
EXPECT_EQ("is an object whose given property isn't >= 0",
DescribeNegation(m));
}

TEST(PropertyForPointerTest, CanDescribeSelfWithPropertyDescription) {
Matcher<const AClass*> m = Property("fancy_name", &AClass::n, Ge(0));

EXPECT_EQ("is an object whose property `fancy_name` is >= 0", Describe(m));
EXPECT_EQ("is an object whose property `fancy_name` isn't >= 0",
DescribeNegation(m));
}

TEST(PropertyForPointerTest, CanExplainMatchResult) {
Matcher<const AClass*> m = Property(&AClass::n, Ge(0));

AClass a;
a.set_n(1);
EXPECT_EQ("", Explain(m, static_cast<const AClass*>(nullptr)));
EXPECT_EQ(
"which points to an object whose given property is 1" + OfType("int"),
Explain(m, &a));

m = Property(&AClass::n, GreaterThan(0));
EXPECT_EQ("which points to an object whose given property is 1" +
OfType("int") + ", which is 1 more than 0",
Explain(m, &a));
}

TEST(PropertyForPointerTest, CanExplainMatchResultWithPropertyName) {
Matcher<const AClass*> m = Property("fancy_name", &AClass::n, Ge(0));

AClass a;
a.set_n(1);
EXPECT_EQ("", Explain(m, static_cast<const AClass*>(nullptr)));
EXPECT_EQ("which points to an object whose property `fancy_name` is 1" +
OfType("int"),
Explain(m, &a));

m = Property("fancy_name", &AClass::n, GreaterThan(0));
EXPECT_EQ("which points to an object whose property `fancy_name` is 1" +
OfType("int") + ", which is 1 more than 0",
Explain(m, &a));
}


std::string IntToStringFunction(int input) {
return input == 1 ? "foo" : "bar";
}

TEST(ResultOfTest, WorksForFunctionPointers) {
Matcher<int> matcher = ResultOf(&IntToStringFunction, Eq(std::string("foo")));

EXPECT_TRUE(matcher.Matches(1));
EXPECT_FALSE(matcher.Matches(2));
}

TEST(ResultOfTest, CanDescribeItself) {
Matcher<int> matcher = ResultOf(&IntToStringFunction, StrEq("foo"));

EXPECT_EQ("is mapped by the given callable to a value that "
"is equal to \"foo\"", Describe(matcher));
EXPECT_EQ("is mapped by the given callable to a value that "
"isn't equal to \"foo\"", DescribeNegation(matcher));
}

int IntFunction(int input) { return input == 42 ? 80 : 90; }

TEST(ResultOfTest, CanExplainMatchResult) {
Matcher<int> matcher = ResultOf(&IntFunction, Ge(85));
EXPECT_EQ("which is mapped by the given callable to 90" + OfType("int"),
Explain(matcher, 36));

matcher = ResultOf(&IntFunction, GreaterThan(85));
EXPECT_EQ("which is mapped by the given callable to 90" + OfType("int") +
", which is 5 more than 85", Explain(matcher, 36));
}

TEST(ResultOfTest, WorksForNonReferenceResults) {
Matcher<int> matcher = ResultOf(&IntFunction, Eq(80));

EXPECT_TRUE(matcher.Matches(42));
EXPECT_FALSE(matcher.Matches(36));
}

double& DoubleFunction(double& input) { return input; }  

Uncopyable& RefUncopyableFunction(Uncopyable& obj) {  
return obj;
}

TEST(ResultOfTest, WorksForReferenceToNonConstResults) {
double x = 3.14;
double x2 = x;
Matcher<double&> matcher = ResultOf(&DoubleFunction, Ref(x));

EXPECT_TRUE(matcher.Matches(x));
EXPECT_FALSE(matcher.Matches(x2));

Uncopyable obj(0);
Uncopyable obj2(0);
Matcher<Uncopyable&> matcher2 =
ResultOf(&RefUncopyableFunction, Ref(obj));

EXPECT_TRUE(matcher2.Matches(obj));
EXPECT_FALSE(matcher2.Matches(obj2));
}

const std::string& StringFunction(const std::string& input) { return input; }

TEST(ResultOfTest, WorksForReferenceToConstResults) {
std::string s = "foo";
std::string s2 = s;
Matcher<const std::string&> matcher = ResultOf(&StringFunction, Ref(s));

EXPECT_TRUE(matcher.Matches(s));
EXPECT_FALSE(matcher.Matches(s2));
}

TEST(ResultOfTest, WorksForCompatibleMatcherTypes) {
Matcher<int> matcher = ResultOf(IntFunction, Matcher<signed char>(Ge(85)));

EXPECT_TRUE(matcher.Matches(36));
EXPECT_FALSE(matcher.Matches(42));
}

TEST(ResultOfDeathTest, DiesOnNullFunctionPointers) {
EXPECT_DEATH_IF_SUPPORTED(
ResultOf(static_cast<std::string (*)(int dummy)>(nullptr),
Eq(std::string("foo"))),
"NULL function pointer is passed into ResultOf\\(\\)\\.");
}

TEST(ResultOfTest, WorksForFunctionReferences) {
Matcher<int> matcher = ResultOf(IntToStringFunction, StrEq("foo"));
EXPECT_TRUE(matcher.Matches(1));
EXPECT_FALSE(matcher.Matches(2));
}

struct Functor {
std::string operator()(int input) const {
return IntToStringFunction(input);
}
};

TEST(ResultOfTest, WorksForFunctors) {
Matcher<int> matcher = ResultOf(Functor(), Eq(std::string("foo")));

EXPECT_TRUE(matcher.Matches(1));
EXPECT_FALSE(matcher.Matches(2));
}

struct PolymorphicFunctor {
typedef int result_type;
int operator()(int n) { return n; }
int operator()(const char* s) { return static_cast<int>(strlen(s)); }
std::string operator()(int *p) { return p ? "good ptr" : "null"; }
};

TEST(ResultOfTest, WorksForPolymorphicFunctors) {
Matcher<int> matcher_int = ResultOf(PolymorphicFunctor(), Ge(5));

EXPECT_TRUE(matcher_int.Matches(10));
EXPECT_FALSE(matcher_int.Matches(2));

Matcher<const char*> matcher_string = ResultOf(PolymorphicFunctor(), Ge(5));

EXPECT_TRUE(matcher_string.Matches("long string"));
EXPECT_FALSE(matcher_string.Matches("shrt"));
}

TEST(ResultOfTest, WorksForPolymorphicFunctorsIgnoringResultType) {
Matcher<int*> matcher = ResultOf(PolymorphicFunctor(), "good ptr");

int n = 0;
EXPECT_TRUE(matcher.Matches(&n));
EXPECT_FALSE(matcher.Matches(nullptr));
}

TEST(ResultOfTest, WorksForLambdas) {
Matcher<int> matcher = ResultOf(
[](int str_len) {
return std::string(static_cast<size_t>(str_len), 'x');
},
"xxx");
EXPECT_TRUE(matcher.Matches(3));
EXPECT_FALSE(matcher.Matches(1));
}

TEST(ResultOfTest, WorksForNonCopyableArguments) {
Matcher<std::unique_ptr<int>> matcher = ResultOf(
[](const std::unique_ptr<int>& str_len) {
return std::string(static_cast<size_t>(*str_len), 'x');
},
"xxx");
EXPECT_TRUE(matcher.Matches(std::unique_ptr<int>(new int(3))));
EXPECT_FALSE(matcher.Matches(std::unique_ptr<int>(new int(1))));
}

const int* ReferencingFunction(const int& n) { return &n; }

struct ReferencingFunctor {
typedef const int* result_type;
result_type operator()(const int& n) { return &n; }
};

TEST(ResultOfTest, WorksForReferencingCallables) {
const int n = 1;
const int n2 = 1;
Matcher<const int&> matcher2 = ResultOf(ReferencingFunction, Eq(&n));
EXPECT_TRUE(matcher2.Matches(n));
EXPECT_FALSE(matcher2.Matches(n2));

Matcher<const int&> matcher3 = ResultOf(ReferencingFunctor(), Eq(&n));
EXPECT_TRUE(matcher3.Matches(n));
EXPECT_FALSE(matcher3.Matches(n2));
}

class DivisibleByImpl {
public:
explicit DivisibleByImpl(int a_divider) : divider_(a_divider) {}

template <typename T>
bool MatchAndExplain(const T& n, MatchResultListener* listener) const {
*listener << "which is " << (n % divider_) << " modulo "
<< divider_;
return (n % divider_) == 0;
}

void DescribeTo(ostream* os) const {
*os << "is divisible by " << divider_;
}

void DescribeNegationTo(ostream* os) const {
*os << "is not divisible by " << divider_;
}

void set_divider(int a_divider) { divider_ = a_divider; }
int divider() const { return divider_; }

private:
int divider_;
};

PolymorphicMatcher<DivisibleByImpl> DivisibleBy(int n) {
return MakePolymorphicMatcher(DivisibleByImpl(n));
}

TEST(ExplainMatchResultTest, AllOf_False_False) {
const Matcher<int> m = AllOf(DivisibleBy(4), DivisibleBy(3));
EXPECT_EQ("which is 1 modulo 4", Explain(m, 5));
}

TEST(ExplainMatchResultTest, AllOf_False_True) {
const Matcher<int> m = AllOf(DivisibleBy(4), DivisibleBy(3));
EXPECT_EQ("which is 2 modulo 4", Explain(m, 6));
}

TEST(ExplainMatchResultTest, AllOf_True_False) {
const Matcher<int> m = AllOf(Ge(1), DivisibleBy(3));
EXPECT_EQ("which is 2 modulo 3", Explain(m, 5));
}

TEST(ExplainMatchResultTest, AllOf_True_True) {
const Matcher<int> m = AllOf(DivisibleBy(2), DivisibleBy(3));
EXPECT_EQ("which is 0 modulo 2, and which is 0 modulo 3", Explain(m, 6));
}

TEST(ExplainMatchResultTest, AllOf_True_True_2) {
const Matcher<int> m = AllOf(Ge(2), Le(3));
EXPECT_EQ("", Explain(m, 2));
}

TEST(ExplainmatcherResultTest, MonomorphicMatcher) {
const Matcher<int> m = GreaterThan(5);
EXPECT_EQ("which is 1 more than 5", Explain(m, 6));
}


class NotCopyable {
public:
explicit NotCopyable(int a_value) : value_(a_value) {}

int value() const { return value_; }

bool operator==(const NotCopyable& rhs) const {
return value() == rhs.value();
}

bool operator>=(const NotCopyable& rhs) const {
return value() >= rhs.value();
}
private:
int value_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(NotCopyable);
};

TEST(ByRefTest, AllowsNotCopyableConstValueInMatchers) {
const NotCopyable const_value1(1);
const Matcher<const NotCopyable&> m = Eq(ByRef(const_value1));

const NotCopyable n1(1), n2(2);
EXPECT_TRUE(m.Matches(n1));
EXPECT_FALSE(m.Matches(n2));
}

TEST(ByRefTest, AllowsNotCopyableValueInMatchers) {
NotCopyable value2(2);
const Matcher<NotCopyable&> m = Ge(ByRef(value2));

NotCopyable n1(1), n2(2);
EXPECT_FALSE(m.Matches(n1));
EXPECT_TRUE(m.Matches(n2));
}

TEST(IsEmptyTest, ImplementsIsEmpty) {
vector<int> container;
EXPECT_THAT(container, IsEmpty());
container.push_back(0);
EXPECT_THAT(container, Not(IsEmpty()));
container.push_back(1);
EXPECT_THAT(container, Not(IsEmpty()));
}

TEST(IsEmptyTest, WorksWithString) {
std::string text;
EXPECT_THAT(text, IsEmpty());
text = "foo";
EXPECT_THAT(text, Not(IsEmpty()));
text = std::string("\0", 1);
EXPECT_THAT(text, Not(IsEmpty()));
}

TEST(IsEmptyTest, CanDescribeSelf) {
Matcher<vector<int> > m = IsEmpty();
EXPECT_EQ("is empty", Describe(m));
EXPECT_EQ("isn't empty", DescribeNegation(m));
}

TEST(IsEmptyTest, ExplainsResult) {
Matcher<vector<int> > m = IsEmpty();
vector<int> container;
EXPECT_EQ("", Explain(m, container));
container.push_back(0);
EXPECT_EQ("whose size is 1", Explain(m, container));
}

TEST(IsEmptyTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(IsEmpty()));
helper.Call({});
}

TEST(IsTrueTest, IsTrueIsFalse) {
EXPECT_THAT(true, IsTrue());
EXPECT_THAT(false, IsFalse());
EXPECT_THAT(true, Not(IsFalse()));
EXPECT_THAT(false, Not(IsTrue()));
EXPECT_THAT(0, Not(IsTrue()));
EXPECT_THAT(0, IsFalse());
EXPECT_THAT(nullptr, Not(IsTrue()));
EXPECT_THAT(nullptr, IsFalse());
EXPECT_THAT(-1, IsTrue());
EXPECT_THAT(-1, Not(IsFalse()));
EXPECT_THAT(1, IsTrue());
EXPECT_THAT(1, Not(IsFalse()));
EXPECT_THAT(2, IsTrue());
EXPECT_THAT(2, Not(IsFalse()));
int a = 42;
EXPECT_THAT(a, IsTrue());
EXPECT_THAT(a, Not(IsFalse()));
EXPECT_THAT(&a, IsTrue());
EXPECT_THAT(&a, Not(IsFalse()));
EXPECT_THAT(false, Not(IsTrue()));
EXPECT_THAT(true, Not(IsFalse()));
EXPECT_THAT(std::true_type(), IsTrue());
EXPECT_THAT(std::true_type(), Not(IsFalse()));
EXPECT_THAT(std::false_type(), IsFalse());
EXPECT_THAT(std::false_type(), Not(IsTrue()));
EXPECT_THAT(nullptr, Not(IsTrue()));
EXPECT_THAT(nullptr, IsFalse());
std::unique_ptr<int> null_unique;
std::unique_ptr<int> nonnull_unique(new int(0));
EXPECT_THAT(null_unique, Not(IsTrue()));
EXPECT_THAT(null_unique, IsFalse());
EXPECT_THAT(nonnull_unique, IsTrue());
EXPECT_THAT(nonnull_unique, Not(IsFalse()));
}

TEST(SizeIsTest, ImplementsSizeIs) {
vector<int> container;
EXPECT_THAT(container, SizeIs(0));
EXPECT_THAT(container, Not(SizeIs(1)));
container.push_back(0);
EXPECT_THAT(container, Not(SizeIs(0)));
EXPECT_THAT(container, SizeIs(1));
container.push_back(0);
EXPECT_THAT(container, Not(SizeIs(0)));
EXPECT_THAT(container, SizeIs(2));
}

TEST(SizeIsTest, WorksWithMap) {
map<std::string, int> container;
EXPECT_THAT(container, SizeIs(0));
EXPECT_THAT(container, Not(SizeIs(1)));
container.insert(make_pair("foo", 1));
EXPECT_THAT(container, Not(SizeIs(0)));
EXPECT_THAT(container, SizeIs(1));
container.insert(make_pair("bar", 2));
EXPECT_THAT(container, Not(SizeIs(0)));
EXPECT_THAT(container, SizeIs(2));
}

TEST(SizeIsTest, WorksWithReferences) {
vector<int> container;
Matcher<const vector<int>&> m = SizeIs(1);
EXPECT_THAT(container, Not(m));
container.push_back(0);
EXPECT_THAT(container, m);
}

TEST(SizeIsTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(SizeIs(3)));
helper.Call(MakeUniquePtrs({1, 2, 3}));
}

struct MinimalistCustomType {
int size() const { return 1; }
};
TEST(SizeIsTest, WorksWithMinimalistCustomType) {
MinimalistCustomType container;
EXPECT_THAT(container, SizeIs(1));
EXPECT_THAT(container, Not(SizeIs(0)));
}

TEST(SizeIsTest, CanDescribeSelf) {
Matcher<vector<int> > m = SizeIs(2);
EXPECT_EQ("size is equal to 2", Describe(m));
EXPECT_EQ("size isn't equal to 2", DescribeNegation(m));
}

TEST(SizeIsTest, ExplainsResult) {
Matcher<vector<int> > m1 = SizeIs(2);
Matcher<vector<int> > m2 = SizeIs(Lt(2u));
Matcher<vector<int> > m3 = SizeIs(AnyOf(0, 3));
Matcher<vector<int> > m4 = SizeIs(Gt(1u));
vector<int> container;
EXPECT_EQ("whose size 0 doesn't match", Explain(m1, container));
EXPECT_EQ("whose size 0 matches", Explain(m2, container));
EXPECT_EQ("whose size 0 matches", Explain(m3, container));
EXPECT_EQ("whose size 0 doesn't match", Explain(m4, container));
container.push_back(0);
container.push_back(0);
EXPECT_EQ("whose size 2 matches", Explain(m1, container));
EXPECT_EQ("whose size 2 doesn't match", Explain(m2, container));
EXPECT_EQ("whose size 2 doesn't match", Explain(m3, container));
EXPECT_EQ("whose size 2 matches", Explain(m4, container));
}

#if GTEST_HAS_TYPED_TEST

template <typename T>
class ContainerEqTest : public testing::Test {};

typedef testing::Types<
set<int>,
vector<size_t>,
multiset<size_t>,
list<int> >
ContainerEqTestTypes;

TYPED_TEST_SUITE(ContainerEqTest, ContainerEqTestTypes);

TYPED_TEST(ContainerEqTest, EqualsSelf) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
TypeParam my_set(vals, vals + 6);
const Matcher<TypeParam> m = ContainerEq(my_set);
EXPECT_TRUE(m.Matches(my_set));
EXPECT_EQ("", Explain(m, my_set));
}

TYPED_TEST(ContainerEqTest, ValueMissing) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {2, 1, 8, 5};
TypeParam my_set(vals, vals + 6);
TypeParam test_set(test_vals, test_vals + 4);
const Matcher<TypeParam> m = ContainerEq(my_set);
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("which doesn't have these expected elements: 3",
Explain(m, test_set));
}

TYPED_TEST(ContainerEqTest, ValueAdded) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {1, 2, 3, 5, 8, 46};
TypeParam my_set(vals, vals + 6);
TypeParam test_set(test_vals, test_vals + 6);
const Matcher<const TypeParam&> m = ContainerEq(my_set);
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("which has these unexpected elements: 46", Explain(m, test_set));
}

TYPED_TEST(ContainerEqTest, ValueAddedAndRemoved) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {1, 2, 3, 8, 46};
TypeParam my_set(vals, vals + 6);
TypeParam test_set(test_vals, test_vals + 5);
const Matcher<TypeParam> m = ContainerEq(my_set);
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("which has these unexpected elements: 46,\n"
"and doesn't have these expected elements: 5",
Explain(m, test_set));
}

TYPED_TEST(ContainerEqTest, DuplicateDifference) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {1, 2, 3, 5, 8};
TypeParam my_set(vals, vals + 6);
TypeParam test_set(test_vals, test_vals + 5);
const Matcher<const TypeParam&> m = ContainerEq(my_set);
EXPECT_EQ("", Explain(m, test_set));
}
#endif  

TEST(ContainerEqExtraTest, MultipleValuesMissing) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {2, 1, 5};
vector<int> my_set(vals, vals + 6);
vector<int> test_set(test_vals, test_vals + 3);
const Matcher<vector<int> > m = ContainerEq(my_set);
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("which doesn't have these expected elements: 3, 8",
Explain(m, test_set));
}

TEST(ContainerEqExtraTest, MultipleValuesAdded) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {1, 2, 92, 3, 5, 8, 46};
list<size_t> my_set(vals, vals + 6);
list<size_t> test_set(test_vals, test_vals + 7);
const Matcher<const list<size_t>&> m = ContainerEq(my_set);
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("which has these unexpected elements: 92, 46",
Explain(m, test_set));
}

TEST(ContainerEqExtraTest, MultipleValuesAddedAndRemoved) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {1, 2, 3, 92, 46};
list<size_t> my_set(vals, vals + 6);
list<size_t> test_set(test_vals, test_vals + 5);
const Matcher<const list<size_t> > m = ContainerEq(my_set);
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("which has these unexpected elements: 92, 46,\n"
"and doesn't have these expected elements: 5, 8",
Explain(m, test_set));
}

TEST(ContainerEqExtraTest, MultiSetOfIntDuplicateDifference) {
static const int vals[] = {1, 1, 2, 3, 5, 8};
static const int test_vals[] = {1, 2, 3, 5, 8};
vector<int> my_set(vals, vals + 6);
vector<int> test_set(test_vals, test_vals + 5);
const Matcher<vector<int> > m = ContainerEq(my_set);
EXPECT_TRUE(m.Matches(my_set));
EXPECT_FALSE(m.Matches(test_set));
EXPECT_EQ("", Explain(m, test_set));
}

TEST(ContainerEqExtraTest, WorksForMaps) {
map<int, std::string> my_map;
my_map[0] = "a";
my_map[1] = "b";

map<int, std::string> test_map;
test_map[0] = "aa";
test_map[1] = "b";

const Matcher<const map<int, std::string>&> m = ContainerEq(my_map);
EXPECT_TRUE(m.Matches(my_map));
EXPECT_FALSE(m.Matches(test_map));

EXPECT_EQ("which has these unexpected elements: (0, \"aa\"),\n"
"and doesn't have these expected elements: (0, \"a\")",
Explain(m, test_map));
}

TEST(ContainerEqExtraTest, WorksForNativeArray) {
int a1[] = {1, 2, 3};
int a2[] = {1, 2, 3};
int b[] = {1, 2, 4};

EXPECT_THAT(a1, ContainerEq(a2));
EXPECT_THAT(a1, Not(ContainerEq(b)));
}

TEST(ContainerEqExtraTest, WorksForTwoDimensionalNativeArray) {
const char a1[][3] = {"hi", "lo"};
const char a2[][3] = {"hi", "lo"};
const char b[][3] = {"lo", "hi"};

EXPECT_THAT(a1, ContainerEq(a2));
EXPECT_THAT(a1, Not(ContainerEq(b)));

EXPECT_THAT(a1, ElementsAre(ContainerEq(a2[0]), ContainerEq(a2[1])));
EXPECT_THAT(a1, ElementsAre(Not(ContainerEq(b[0])), ContainerEq(a2[1])));
}

TEST(ContainerEqExtraTest, WorksForNativeArrayAsTuple) {
const int a1[] = {1, 2, 3};
const int a2[] = {1, 2, 3};
const int b[] = {1, 2, 3, 4};

const int* const p1 = a1;
EXPECT_THAT(std::make_tuple(p1, 3), ContainerEq(a2));
EXPECT_THAT(std::make_tuple(p1, 3), Not(ContainerEq(b)));

const int c[] = {1, 3, 2};
EXPECT_THAT(std::make_tuple(p1, 3), Not(ContainerEq(c)));
}

TEST(ContainerEqExtraTest, CopiesNativeArrayParameter) {
std::string a1[][3] = {
{"hi", "hello", "ciao"},
{"bye", "see you", "ciao"}
};

std::string a2[][3] = {
{"hi", "hello", "ciao"},
{"bye", "see you", "ciao"}
};

const Matcher<const std::string(&)[2][3]> m = ContainerEq(a2);
EXPECT_THAT(a1, m);

a2[0][0] = "ha";
EXPECT_THAT(a1, m);
}

TEST(WhenSortedByTest, WorksForEmptyContainer) {
const vector<int> numbers;
EXPECT_THAT(numbers, WhenSortedBy(less<int>(), ElementsAre()));
EXPECT_THAT(numbers, Not(WhenSortedBy(less<int>(), ElementsAre(1))));
}

TEST(WhenSortedByTest, WorksForNonEmptyContainer) {
vector<unsigned> numbers;
numbers.push_back(3);
numbers.push_back(1);
numbers.push_back(2);
numbers.push_back(2);
EXPECT_THAT(numbers, WhenSortedBy(greater<unsigned>(),
ElementsAre(3, 2, 2, 1)));
EXPECT_THAT(numbers, Not(WhenSortedBy(greater<unsigned>(),
ElementsAre(1, 2, 2, 3))));
}

TEST(WhenSortedByTest, WorksForNonVectorContainer) {
list<std::string> words;
words.push_back("say");
words.push_back("hello");
words.push_back("world");
EXPECT_THAT(words, WhenSortedBy(less<std::string>(),
ElementsAre("hello", "say", "world")));
EXPECT_THAT(words, Not(WhenSortedBy(less<std::string>(),
ElementsAre("say", "hello", "world"))));
}

TEST(WhenSortedByTest, WorksForNativeArray) {
const int numbers[] = {1, 3, 2, 4};
const int sorted_numbers[] = {1, 2, 3, 4};
EXPECT_THAT(numbers, WhenSortedBy(less<int>(), ElementsAre(1, 2, 3, 4)));
EXPECT_THAT(numbers, WhenSortedBy(less<int>(),
ElementsAreArray(sorted_numbers)));
EXPECT_THAT(numbers, Not(WhenSortedBy(less<int>(), ElementsAre(1, 3, 2, 4))));
}

TEST(WhenSortedByTest, CanDescribeSelf) {
const Matcher<vector<int> > m = WhenSortedBy(less<int>(), ElementsAre(1, 2));
EXPECT_EQ("(when sorted) has 2 elements where\n"
"element #0 is equal to 1,\n"
"element #1 is equal to 2",
Describe(m));
EXPECT_EQ("(when sorted) doesn't have 2 elements, or\n"
"element #0 isn't equal to 1, or\n"
"element #1 isn't equal to 2",
DescribeNegation(m));
}

TEST(WhenSortedByTest, ExplainsMatchResult) {
const int a[] = {2, 1};
EXPECT_EQ("which is { 1, 2 } when sorted, whose element #0 doesn't match",
Explain(WhenSortedBy(less<int>(), ElementsAre(2, 3)), a));
EXPECT_EQ("which is { 1, 2 } when sorted",
Explain(WhenSortedBy(less<int>(), ElementsAre(1, 2)), a));
}


TEST(WhenSortedTest, WorksForEmptyContainer) {
const vector<int> numbers;
EXPECT_THAT(numbers, WhenSorted(ElementsAre()));
EXPECT_THAT(numbers, Not(WhenSorted(ElementsAre(1))));
}

TEST(WhenSortedTest, WorksForNonEmptyContainer) {
list<std::string> words;
words.push_back("3");
words.push_back("1");
words.push_back("2");
words.push_back("2");
EXPECT_THAT(words, WhenSorted(ElementsAre("1", "2", "2", "3")));
EXPECT_THAT(words, Not(WhenSorted(ElementsAre("3", "1", "2", "2"))));
}

TEST(WhenSortedTest, WorksForMapTypes) {
map<std::string, int> word_counts;
word_counts["and"] = 1;
word_counts["the"] = 1;
word_counts["buffalo"] = 2;
EXPECT_THAT(word_counts,
WhenSorted(ElementsAre(Pair("and", 1), Pair("buffalo", 2),
Pair("the", 1))));
EXPECT_THAT(word_counts,
Not(WhenSorted(ElementsAre(Pair("and", 1), Pair("the", 1),
Pair("buffalo", 2)))));
}

TEST(WhenSortedTest, WorksForMultiMapTypes) {
multimap<int, int> ifib;
ifib.insert(make_pair(8, 6));
ifib.insert(make_pair(2, 3));
ifib.insert(make_pair(1, 1));
ifib.insert(make_pair(3, 4));
ifib.insert(make_pair(1, 2));
ifib.insert(make_pair(5, 5));
EXPECT_THAT(ifib, WhenSorted(ElementsAre(Pair(1, 1),
Pair(1, 2),
Pair(2, 3),
Pair(3, 4),
Pair(5, 5),
Pair(8, 6))));
EXPECT_THAT(ifib, Not(WhenSorted(ElementsAre(Pair(8, 6),
Pair(2, 3),
Pair(1, 1),
Pair(3, 4),
Pair(1, 2),
Pair(5, 5)))));
}

TEST(WhenSortedTest, WorksForPolymorphicMatcher) {
std::deque<int> d;
d.push_back(2);
d.push_back(1);
EXPECT_THAT(d, WhenSorted(ElementsAre(1, 2)));
EXPECT_THAT(d, Not(WhenSorted(ElementsAre(2, 1))));
}

TEST(WhenSortedTest, WorksForVectorConstRefMatcher) {
std::deque<int> d;
d.push_back(2);
d.push_back(1);
Matcher<const std::vector<int>&> vector_match = ElementsAre(1, 2);
EXPECT_THAT(d, WhenSorted(vector_match));
Matcher<const std::vector<int>&> not_vector_match = ElementsAre(2, 1);
EXPECT_THAT(d, Not(WhenSorted(not_vector_match)));
}

template <typename T>
class Streamlike {
private:
class ConstIter;
public:
typedef ConstIter const_iterator;
typedef T value_type;

template <typename InIter>
Streamlike(InIter first, InIter last) : remainder_(first, last) {}

const_iterator begin() const {
return const_iterator(this, remainder_.begin());
}
const_iterator end() const {
return const_iterator(this, remainder_.end());
}

private:
class ConstIter : public std::iterator<std::input_iterator_tag,
value_type,
ptrdiff_t,
const value_type*,
const value_type&> {
public:
ConstIter(const Streamlike* s,
typename std::list<value_type>::iterator pos)
: s_(s), pos_(pos) {}

const value_type& operator*() const { return *pos_; }
const value_type* operator->() const { return &*pos_; }
ConstIter& operator++() {
s_->remainder_.erase(pos_++);
return *this;
}

class PostIncrProxy {
public:
explicit PostIncrProxy(const value_type& value) : value_(value) {}
value_type operator*() const { return value_; }
private:
value_type value_;
};
PostIncrProxy operator++(int) {
PostIncrProxy proxy(**this);
++(*this);
return proxy;
}

friend bool operator==(const ConstIter& a, const ConstIter& b) {
return a.s_ == b.s_ && a.pos_ == b.pos_;
}
friend bool operator!=(const ConstIter& a, const ConstIter& b) {
return !(a == b);
}

private:
const Streamlike* s_;
typename std::list<value_type>::iterator pos_;
};

friend std::ostream& operator<<(std::ostream& os, const Streamlike& s) {
os << "[";
typedef typename std::list<value_type>::const_iterator Iter;
const char* sep = "";
for (Iter it = s.remainder_.begin(); it != s.remainder_.end(); ++it) {
os << sep << *it;
sep = ",";
}
os << "]";
return os;
}

mutable std::list<value_type> remainder_;  
};

TEST(StreamlikeTest, Iteration) {
const int a[5] = {2, 1, 4, 5, 3};
Streamlike<int> s(a, a + 5);
Streamlike<int>::const_iterator it = s.begin();
const int* ip = a;
while (it != s.end()) {
SCOPED_TRACE(ip - a);
EXPECT_EQ(*ip++, *it++);
}
}

TEST(BeginEndDistanceIsTest, WorksWithForwardList) {
std::forward_list<int> container;
EXPECT_THAT(container, BeginEndDistanceIs(0));
EXPECT_THAT(container, Not(BeginEndDistanceIs(1)));
container.push_front(0);
EXPECT_THAT(container, Not(BeginEndDistanceIs(0)));
EXPECT_THAT(container, BeginEndDistanceIs(1));
container.push_front(0);
EXPECT_THAT(container, Not(BeginEndDistanceIs(0)));
EXPECT_THAT(container, BeginEndDistanceIs(2));
}

TEST(BeginEndDistanceIsTest, WorksWithNonStdList) {
const int a[5] = {1, 2, 3, 4, 5};
Streamlike<int> s(a, a + 5);
EXPECT_THAT(s, BeginEndDistanceIs(5));
}

TEST(BeginEndDistanceIsTest, CanDescribeSelf) {
Matcher<vector<int> > m = BeginEndDistanceIs(2);
EXPECT_EQ("distance between begin() and end() is equal to 2", Describe(m));
EXPECT_EQ("distance between begin() and end() isn't equal to 2",
DescribeNegation(m));
}

TEST(BeginEndDistanceIsTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(BeginEndDistanceIs(2)));
helper.Call(MakeUniquePtrs({1, 2}));
}

TEST(BeginEndDistanceIsTest, ExplainsResult) {
Matcher<vector<int> > m1 = BeginEndDistanceIs(2);
Matcher<vector<int> > m2 = BeginEndDistanceIs(Lt(2));
Matcher<vector<int> > m3 = BeginEndDistanceIs(AnyOf(0, 3));
Matcher<vector<int> > m4 = BeginEndDistanceIs(GreaterThan(1));
vector<int> container;
EXPECT_EQ("whose distance between begin() and end() 0 doesn't match",
Explain(m1, container));
EXPECT_EQ("whose distance between begin() and end() 0 matches",
Explain(m2, container));
EXPECT_EQ("whose distance between begin() and end() 0 matches",
Explain(m3, container));
EXPECT_EQ(
"whose distance between begin() and end() 0 doesn't match, which is 1 "
"less than 1",
Explain(m4, container));
container.push_back(0);
container.push_back(0);
EXPECT_EQ("whose distance between begin() and end() 2 matches",
Explain(m1, container));
EXPECT_EQ("whose distance between begin() and end() 2 doesn't match",
Explain(m2, container));
EXPECT_EQ("whose distance between begin() and end() 2 doesn't match",
Explain(m3, container));
EXPECT_EQ(
"whose distance between begin() and end() 2 matches, which is 1 more "
"than 1",
Explain(m4, container));
}

TEST(WhenSortedTest, WorksForStreamlike) {
const int a[5] = {2, 1, 4, 5, 3};
Streamlike<int> s(std::begin(a), std::end(a));
EXPECT_THAT(s, WhenSorted(ElementsAre(1, 2, 3, 4, 5)));
EXPECT_THAT(s, Not(WhenSorted(ElementsAre(2, 1, 4, 5, 3))));
}

TEST(WhenSortedTest, WorksForVectorConstRefMatcherOnStreamlike) {
const int a[] = {2, 1, 4, 5, 3};
Streamlike<int> s(std::begin(a), std::end(a));
Matcher<const std::vector<int>&> vector_match = ElementsAre(1, 2, 3, 4, 5);
EXPECT_THAT(s, WhenSorted(vector_match));
EXPECT_THAT(s, Not(WhenSorted(ElementsAre(2, 1, 4, 5, 3))));
}

TEST(IsSupersetOfTest, WorksForNativeArray) {
const int subset[] = {1, 4};
const int superset[] = {1, 2, 4};
const int disjoint[] = {1, 0, 3};
EXPECT_THAT(subset, IsSupersetOf(subset));
EXPECT_THAT(subset, Not(IsSupersetOf(superset)));
EXPECT_THAT(superset, IsSupersetOf(subset));
EXPECT_THAT(subset, Not(IsSupersetOf(disjoint)));
EXPECT_THAT(disjoint, Not(IsSupersetOf(subset)));
}

TEST(IsSupersetOfTest, WorksWithDuplicates) {
const int not_enough[] = {1, 2};
const int enough[] = {1, 1, 2};
const int expected[] = {1, 1};
EXPECT_THAT(not_enough, Not(IsSupersetOf(expected)));
EXPECT_THAT(enough, IsSupersetOf(expected));
}

TEST(IsSupersetOfTest, WorksForEmpty) {
vector<int> numbers;
vector<int> expected;
EXPECT_THAT(numbers, IsSupersetOf(expected));
expected.push_back(1);
EXPECT_THAT(numbers, Not(IsSupersetOf(expected)));
expected.clear();
numbers.push_back(1);
numbers.push_back(2);
EXPECT_THAT(numbers, IsSupersetOf(expected));
expected.push_back(1);
EXPECT_THAT(numbers, IsSupersetOf(expected));
expected.push_back(2);
EXPECT_THAT(numbers, IsSupersetOf(expected));
expected.push_back(3);
EXPECT_THAT(numbers, Not(IsSupersetOf(expected)));
}

TEST(IsSupersetOfTest, WorksForStreamlike) {
const int a[5] = {1, 2, 3, 4, 5};
Streamlike<int> s(std::begin(a), std::end(a));

vector<int> expected;
expected.push_back(1);
expected.push_back(2);
expected.push_back(5);
EXPECT_THAT(s, IsSupersetOf(expected));

expected.push_back(0);
EXPECT_THAT(s, Not(IsSupersetOf(expected)));
}

TEST(IsSupersetOfTest, TakesStlContainer) {
const int actual[] = {3, 1, 2};

::std::list<int> expected;
expected.push_back(1);
expected.push_back(3);
EXPECT_THAT(actual, IsSupersetOf(expected));

expected.push_back(4);
EXPECT_THAT(actual, Not(IsSupersetOf(expected)));
}

TEST(IsSupersetOfTest, Describe) {
typedef std::vector<int> IntVec;
IntVec expected;
expected.push_back(111);
expected.push_back(222);
expected.push_back(333);
EXPECT_THAT(
Describe<IntVec>(IsSupersetOf(expected)),
Eq("a surjection from elements to requirements exists such that:\n"
" - an element is equal to 111\n"
" - an element is equal to 222\n"
" - an element is equal to 333"));
}

TEST(IsSupersetOfTest, DescribeNegation) {
typedef std::vector<int> IntVec;
IntVec expected;
expected.push_back(111);
expected.push_back(222);
expected.push_back(333);
EXPECT_THAT(
DescribeNegation<IntVec>(IsSupersetOf(expected)),
Eq("no surjection from elements to requirements exists such that:\n"
" - an element is equal to 111\n"
" - an element is equal to 222\n"
" - an element is equal to 333"));
}

TEST(IsSupersetOfTest, MatchAndExplain) {
std::vector<int> v;
v.push_back(2);
v.push_back(3);
std::vector<int> expected;
expected.push_back(1);
expected.push_back(2);
StringMatchResultListener listener;
ASSERT_FALSE(ExplainMatchResult(IsSupersetOf(expected), v, &listener))
<< listener.str();
EXPECT_THAT(listener.str(),
Eq("where the following matchers don't match any elements:\n"
"matcher #0: is equal to 1"));

v.push_back(1);
listener.Clear();
ASSERT_TRUE(ExplainMatchResult(IsSupersetOf(expected), v, &listener))
<< listener.str();
EXPECT_THAT(listener.str(), Eq("where:\n"
" - element #0 is matched by matcher #1,\n"
" - element #2 is matched by matcher #0"));
}

TEST(IsSupersetOfTest, WorksForRhsInitializerList) {
const int numbers[] = {1, 3, 6, 2, 4, 5};
EXPECT_THAT(numbers, IsSupersetOf({1, 2}));
EXPECT_THAT(numbers, Not(IsSupersetOf({3, 0})));
}

TEST(IsSupersetOfTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(IsSupersetOf({Pointee(1)})));
helper.Call(MakeUniquePtrs({1, 2}));
EXPECT_CALL(helper, Call(Not(IsSupersetOf({Pointee(1), Pointee(2)}))));
helper.Call(MakeUniquePtrs({2}));
}

TEST(IsSubsetOfTest, WorksForNativeArray) {
const int subset[] = {1, 4};
const int superset[] = {1, 2, 4};
const int disjoint[] = {1, 0, 3};
EXPECT_THAT(subset, IsSubsetOf(subset));
EXPECT_THAT(subset, IsSubsetOf(superset));
EXPECT_THAT(superset, Not(IsSubsetOf(subset)));
EXPECT_THAT(subset, Not(IsSubsetOf(disjoint)));
EXPECT_THAT(disjoint, Not(IsSubsetOf(subset)));
}

TEST(IsSubsetOfTest, WorksWithDuplicates) {
const int not_enough[] = {1, 2};
const int enough[] = {1, 1, 2};
const int actual[] = {1, 1};
EXPECT_THAT(actual, Not(IsSubsetOf(not_enough)));
EXPECT_THAT(actual, IsSubsetOf(enough));
}

TEST(IsSubsetOfTest, WorksForEmpty) {
vector<int> numbers;
vector<int> expected;
EXPECT_THAT(numbers, IsSubsetOf(expected));
expected.push_back(1);
EXPECT_THAT(numbers, IsSubsetOf(expected));
expected.clear();
numbers.push_back(1);
numbers.push_back(2);
EXPECT_THAT(numbers, Not(IsSubsetOf(expected)));
expected.push_back(1);
EXPECT_THAT(numbers, Not(IsSubsetOf(expected)));
expected.push_back(2);
EXPECT_THAT(numbers, IsSubsetOf(expected));
expected.push_back(3);
EXPECT_THAT(numbers, IsSubsetOf(expected));
}

TEST(IsSubsetOfTest, WorksForStreamlike) {
const int a[5] = {1, 2};
Streamlike<int> s(std::begin(a), std::end(a));

vector<int> expected;
expected.push_back(1);
EXPECT_THAT(s, Not(IsSubsetOf(expected)));
expected.push_back(2);
expected.push_back(5);
EXPECT_THAT(s, IsSubsetOf(expected));
}

TEST(IsSubsetOfTest, TakesStlContainer) {
const int actual[] = {3, 1, 2};

::std::list<int> expected;
expected.push_back(1);
expected.push_back(3);
EXPECT_THAT(actual, Not(IsSubsetOf(expected)));

expected.push_back(2);
expected.push_back(4);
EXPECT_THAT(actual, IsSubsetOf(expected));
}

TEST(IsSubsetOfTest, Describe) {
typedef std::vector<int> IntVec;
IntVec expected;
expected.push_back(111);
expected.push_back(222);
expected.push_back(333);

EXPECT_THAT(
Describe<IntVec>(IsSubsetOf(expected)),
Eq("an injection from elements to requirements exists such that:\n"
" - an element is equal to 111\n"
" - an element is equal to 222\n"
" - an element is equal to 333"));
}

TEST(IsSubsetOfTest, DescribeNegation) {
typedef std::vector<int> IntVec;
IntVec expected;
expected.push_back(111);
expected.push_back(222);
expected.push_back(333);
EXPECT_THAT(
DescribeNegation<IntVec>(IsSubsetOf(expected)),
Eq("no injection from elements to requirements exists such that:\n"
" - an element is equal to 111\n"
" - an element is equal to 222\n"
" - an element is equal to 333"));
}

TEST(IsSubsetOfTest, MatchAndExplain) {
std::vector<int> v;
v.push_back(2);
v.push_back(3);
std::vector<int> expected;
expected.push_back(1);
expected.push_back(2);
StringMatchResultListener listener;
ASSERT_FALSE(ExplainMatchResult(IsSubsetOf(expected), v, &listener))
<< listener.str();
EXPECT_THAT(listener.str(),
Eq("where the following elements don't match any matchers:\n"
"element #1: 3"));

expected.push_back(3);
listener.Clear();
ASSERT_TRUE(ExplainMatchResult(IsSubsetOf(expected), v, &listener))
<< listener.str();
EXPECT_THAT(listener.str(), Eq("where:\n"
" - element #0 is matched by matcher #1,\n"
" - element #1 is matched by matcher #2"));
}

TEST(IsSubsetOfTest, WorksForRhsInitializerList) {
const int numbers[] = {1, 2, 3};
EXPECT_THAT(numbers, IsSubsetOf({1, 2, 3, 4}));
EXPECT_THAT(numbers, Not(IsSubsetOf({1, 2})));
}

TEST(IsSubsetOfTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(IsSubsetOf({Pointee(1), Pointee(2)})));
helper.Call(MakeUniquePtrs({1}));
EXPECT_CALL(helper, Call(Not(IsSubsetOf({Pointee(1)}))));
helper.Call(MakeUniquePtrs({2}));
}


TEST(ElemensAreStreamTest, WorksForStreamlike) {
const int a[5] = {1, 2, 3, 4, 5};
Streamlike<int> s(std::begin(a), std::end(a));
EXPECT_THAT(s, ElementsAre(1, 2, 3, 4, 5));
EXPECT_THAT(s, Not(ElementsAre(2, 1, 4, 5, 3)));
}

TEST(ElemensAreArrayStreamTest, WorksForStreamlike) {
const int a[5] = {1, 2, 3, 4, 5};
Streamlike<int> s(std::begin(a), std::end(a));

vector<int> expected;
expected.push_back(1);
expected.push_back(2);
expected.push_back(3);
expected.push_back(4);
expected.push_back(5);
EXPECT_THAT(s, ElementsAreArray(expected));

expected[3] = 0;
EXPECT_THAT(s, Not(ElementsAreArray(expected)));
}

TEST(ElementsAreTest, WorksWithUncopyable) {
Uncopyable objs[2];
objs[0].set_value(-3);
objs[1].set_value(1);
EXPECT_THAT(objs, ElementsAre(UncopyableIs(-3), Truly(ValueIsPositive)));
}

TEST(ElementsAreTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(ElementsAre(Pointee(1), Pointee(2))));
helper.Call(MakeUniquePtrs({1, 2}));

EXPECT_CALL(helper, Call(ElementsAreArray({Pointee(3), Pointee(4)})));
helper.Call(MakeUniquePtrs({3, 4}));
}

TEST(ElementsAreTest, TakesStlContainer) {
const int actual[] = {3, 1, 2};

::std::list<int> expected;
expected.push_back(3);
expected.push_back(1);
expected.push_back(2);
EXPECT_THAT(actual, ElementsAreArray(expected));

expected.push_back(4);
EXPECT_THAT(actual, Not(ElementsAreArray(expected)));
}


TEST(UnorderedElementsAreArrayTest, SucceedsWhenExpected) {
const int a[] = {0, 1, 2, 3, 4};
std::vector<int> s(std::begin(a), std::end(a));
do {
StringMatchResultListener listener;
EXPECT_TRUE(ExplainMatchResult(UnorderedElementsAreArray(a),
s, &listener)) << listener.str();
} while (std::next_permutation(s.begin(), s.end()));
}

TEST(UnorderedElementsAreArrayTest, VectorBool) {
const bool a[] = {0, 1, 0, 1, 1};
const bool b[] = {1, 0, 1, 1, 0};
std::vector<bool> expected(std::begin(a), std::end(a));
std::vector<bool> actual(std::begin(b), std::end(b));
StringMatchResultListener listener;
EXPECT_TRUE(ExplainMatchResult(UnorderedElementsAreArray(expected),
actual, &listener)) << listener.str();
}

TEST(UnorderedElementsAreArrayTest, WorksForStreamlike) {
const int a[5] = {2, 1, 4, 5, 3};
Streamlike<int> s(std::begin(a), std::end(a));

::std::vector<int> expected;
expected.push_back(1);
expected.push_back(2);
expected.push_back(3);
expected.push_back(4);
expected.push_back(5);
EXPECT_THAT(s, UnorderedElementsAreArray(expected));

expected.push_back(6);
EXPECT_THAT(s, Not(UnorderedElementsAreArray(expected)));
}

TEST(UnorderedElementsAreArrayTest, TakesStlContainer) {
const int actual[] = {3, 1, 2};

::std::list<int> expected;
expected.push_back(1);
expected.push_back(2);
expected.push_back(3);
EXPECT_THAT(actual, UnorderedElementsAreArray(expected));

expected.push_back(4);
EXPECT_THAT(actual, Not(UnorderedElementsAreArray(expected)));
}


TEST(UnorderedElementsAreArrayTest, TakesInitializerList) {
const int a[5] = {2, 1, 4, 5, 3};
EXPECT_THAT(a, UnorderedElementsAreArray({1, 2, 3, 4, 5}));
EXPECT_THAT(a, Not(UnorderedElementsAreArray({1, 2, 3, 4, 6})));
}

TEST(UnorderedElementsAreArrayTest, TakesInitializerListOfCStrings) {
const std::string a[5] = {"a", "b", "c", "d", "e"};
EXPECT_THAT(a, UnorderedElementsAreArray({"a", "b", "c", "d", "e"}));
EXPECT_THAT(a, Not(UnorderedElementsAreArray({"a", "b", "c", "d", "ef"})));
}

TEST(UnorderedElementsAreArrayTest, TakesInitializerListOfSameTypedMatchers) {
const int a[5] = {2, 1, 4, 5, 3};
EXPECT_THAT(a, UnorderedElementsAreArray(
{Eq(1), Eq(2), Eq(3), Eq(4), Eq(5)}));
EXPECT_THAT(a, Not(UnorderedElementsAreArray(
{Eq(1), Eq(2), Eq(3), Eq(4), Eq(6)})));
}

TEST(UnorderedElementsAreArrayTest,
TakesInitializerListOfDifferentTypedMatchers) {
const int a[5] = {2, 1, 4, 5, 3};
EXPECT_THAT(a, UnorderedElementsAreArray<Matcher<int> >(
{Eq(1), Ne(-2), Ge(3), Le(4), Eq(5)}));
EXPECT_THAT(a, Not(UnorderedElementsAreArray<Matcher<int> >(
{Eq(1), Ne(-2), Ge(3), Le(4), Eq(6)})));
}


TEST(UnorderedElementsAreArrayTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper,
Call(UnorderedElementsAreArray({Pointee(1), Pointee(2)})));
helper.Call(MakeUniquePtrs({2, 1}));
}

class UnorderedElementsAreTest : public testing::Test {
protected:
typedef std::vector<int> IntVec;
};

TEST_F(UnorderedElementsAreTest, WorksWithUncopyable) {
Uncopyable objs[2];
objs[0].set_value(-3);
objs[1].set_value(1);
EXPECT_THAT(objs,
UnorderedElementsAre(Truly(ValueIsPositive), UncopyableIs(-3)));
}

TEST_F(UnorderedElementsAreTest, SucceedsWhenExpected) {
const int a[] = {1, 2, 3};
std::vector<int> s(std::begin(a), std::end(a));
do {
StringMatchResultListener listener;
EXPECT_TRUE(ExplainMatchResult(UnorderedElementsAre(1, 2, 3),
s, &listener)) << listener.str();
} while (std::next_permutation(s.begin(), s.end()));
}

TEST_F(UnorderedElementsAreTest, FailsWhenAnElementMatchesNoMatcher) {
const int a[] = {1, 2, 3};
std::vector<int> s(std::begin(a), std::end(a));
std::vector<Matcher<int> > mv;
mv.push_back(1);
mv.push_back(2);
mv.push_back(2);
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(UnorderedElementsAreArray(mv),
s, &listener)) << listener.str();
}

TEST_F(UnorderedElementsAreTest, WorksForStreamlike) {
const int a[5] = {2, 1, 4, 5, 3};
Streamlike<int> s(std::begin(a), std::end(a));

EXPECT_THAT(s, UnorderedElementsAre(1, 2, 3, 4, 5));
EXPECT_THAT(s, Not(UnorderedElementsAre(2, 2, 3, 4, 5)));
}

TEST_F(UnorderedElementsAreTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(UnorderedElementsAre(Pointee(1), Pointee(2))));
helper.Call(MakeUniquePtrs({2, 1}));
}

TEST_F(UnorderedElementsAreTest, Performance) {
std::vector<int> s;
std::vector<Matcher<int> > mv;
for (int i = 0; i < 100; ++i) {
s.push_back(i);
mv.push_back(_);
}
mv[50] = Eq(0);
StringMatchResultListener listener;
EXPECT_TRUE(ExplainMatchResult(UnorderedElementsAreArray(mv),
s, &listener)) << listener.str();
}

TEST_F(UnorderedElementsAreTest, PerformanceHalfStrict) {
std::vector<int> s;
std::vector<Matcher<int> > mv;
for (int i = 0; i < 100; ++i) {
s.push_back(i);
if (i & 1) {
mv.push_back(_);
} else {
mv.push_back(i);
}
}
StringMatchResultListener listener;
EXPECT_TRUE(ExplainMatchResult(UnorderedElementsAreArray(mv),
s, &listener)) << listener.str();
}

TEST_F(UnorderedElementsAreTest, FailMessageCountWrong) {
std::vector<int> v;
v.push_back(4);
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(UnorderedElementsAre(1, 2, 3),
v, &listener)) << listener.str();
EXPECT_THAT(listener.str(), Eq("which has 1 element"));
}

TEST_F(UnorderedElementsAreTest, FailMessageCountWrongZero) {
std::vector<int> v;
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(UnorderedElementsAre(1, 2, 3),
v, &listener)) << listener.str();
EXPECT_THAT(listener.str(), Eq(""));
}

TEST_F(UnorderedElementsAreTest, FailMessageUnmatchedMatchers) {
std::vector<int> v;
v.push_back(1);
v.push_back(1);
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(UnorderedElementsAre(1, 2),
v, &listener)) << listener.str();
EXPECT_THAT(
listener.str(),
Eq("where the following matchers don't match any elements:\n"
"matcher #1: is equal to 2"));
}

TEST_F(UnorderedElementsAreTest, FailMessageUnmatchedElements) {
std::vector<int> v;
v.push_back(1);
v.push_back(2);
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(UnorderedElementsAre(1, 1),
v, &listener)) << listener.str();
EXPECT_THAT(
listener.str(),
Eq("where the following elements don't match any matchers:\n"
"element #1: 2"));
}

TEST_F(UnorderedElementsAreTest, FailMessageUnmatchedMatcherAndElement) {
std::vector<int> v;
v.push_back(2);
v.push_back(3);
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(UnorderedElementsAre(1, 2),
v, &listener)) << listener.str();
EXPECT_THAT(
listener.str(),
Eq("where"
" the following matchers don't match any elements:\n"
"matcher #0: is equal to 1\n"
"and"
" where"
" the following elements don't match any matchers:\n"
"element #1: 3"));
}

static std::string EMString(int element, int matcher) {
stringstream ss;
ss << "(element #" << element << ", matcher #" << matcher << ")";
return ss.str();
}

TEST_F(UnorderedElementsAreTest, FailMessageImperfectMatchOnly) {
std::vector<std::string> v;
v.push_back("a");
v.push_back("b");
v.push_back("c");
StringMatchResultListener listener;
EXPECT_FALSE(ExplainMatchResult(
UnorderedElementsAre("a", "a", AnyOf("b", "c")), v, &listener))
<< listener.str();

std::string prefix =
"where no permutation of the elements can satisfy all matchers, "
"and the closest match is 2 of 3 matchers with the "
"pairings:\n";

EXPECT_THAT(
listener.str(),
AnyOf(prefix + "{\n  " + EMString(0, 0) +
",\n  " + EMString(1, 2) + "\n}",
prefix + "{\n  " + EMString(0, 1) +
",\n  " + EMString(1, 2) + "\n}",
prefix + "{\n  " + EMString(0, 0) +
",\n  " + EMString(2, 2) + "\n}",
prefix + "{\n  " + EMString(0, 1) +
",\n  " + EMString(2, 2) + "\n}"));
}

TEST_F(UnorderedElementsAreTest, Describe) {
EXPECT_THAT(Describe<IntVec>(UnorderedElementsAre()),
Eq("is empty"));
EXPECT_THAT(
Describe<IntVec>(UnorderedElementsAre(345)),
Eq("has 1 element and that element is equal to 345"));
EXPECT_THAT(
Describe<IntVec>(UnorderedElementsAre(111, 222, 333)),
Eq("has 3 elements and there exists some permutation "
"of elements such that:\n"
" - element #0 is equal to 111, and\n"
" - element #1 is equal to 222, and\n"
" - element #2 is equal to 333"));
}

TEST_F(UnorderedElementsAreTest, DescribeNegation) {
EXPECT_THAT(DescribeNegation<IntVec>(UnorderedElementsAre()),
Eq("isn't empty"));
EXPECT_THAT(
DescribeNegation<IntVec>(UnorderedElementsAre(345)),
Eq("doesn't have 1 element, or has 1 element that isn't equal to 345"));
EXPECT_THAT(
DescribeNegation<IntVec>(UnorderedElementsAre(123, 234, 345)),
Eq("doesn't have 3 elements, or there exists no permutation "
"of elements such that:\n"
" - element #0 is equal to 123, and\n"
" - element #1 is equal to 234, and\n"
" - element #2 is equal to 345"));
}

namespace {

template <typename Graph>
class BacktrackingMaxBPMState {
public:
explicit BacktrackingMaxBPMState(const Graph* g) : graph_(g) { }

ElementMatcherPairs Compute() {
if (graph_->LhsSize() == 0 || graph_->RhsSize() == 0) {
return best_so_far_;
}
lhs_used_.assign(graph_->LhsSize(), kUnused);
rhs_used_.assign(graph_->RhsSize(), kUnused);
for (size_t irhs = 0; irhs < graph_->RhsSize(); ++irhs) {
matches_.clear();
RecurseInto(irhs);
if (best_so_far_.size() == graph_->RhsSize())
break;
}
return best_so_far_;
}

private:
static const size_t kUnused = static_cast<size_t>(-1);

void PushMatch(size_t lhs, size_t rhs) {
matches_.push_back(ElementMatcherPair(lhs, rhs));
lhs_used_[lhs] = rhs;
rhs_used_[rhs] = lhs;
if (matches_.size() > best_so_far_.size()) {
best_so_far_ = matches_;
}
}

void PopMatch() {
const ElementMatcherPair& back = matches_.back();
lhs_used_[back.first] = kUnused;
rhs_used_[back.second] = kUnused;
matches_.pop_back();
}

bool RecurseInto(size_t irhs) {
if (rhs_used_[irhs] != kUnused) {
return true;
}
for (size_t ilhs = 0; ilhs < graph_->LhsSize(); ++ilhs) {
if (lhs_used_[ilhs] != kUnused) {
continue;
}
if (!graph_->HasEdge(ilhs, irhs)) {
continue;
}
PushMatch(ilhs, irhs);
if (best_so_far_.size() == graph_->RhsSize()) {
return false;
}
for (size_t mi = irhs + 1; mi < graph_->RhsSize(); ++mi) {
if (!RecurseInto(mi)) return false;
}
PopMatch();
}
return true;
}

const Graph* graph_;  
std::vector<size_t> lhs_used_;
std::vector<size_t> rhs_used_;
ElementMatcherPairs matches_;
ElementMatcherPairs best_so_far_;
};

template <typename Graph>
const size_t BacktrackingMaxBPMState<Graph>::kUnused;

}  

template <typename Graph>
ElementMatcherPairs
FindBacktrackingMaxBPM(const Graph& g) {
return BacktrackingMaxBPMState<Graph>(&g).Compute();
}

class BacktrackingBPMTest : public ::testing::Test { };

class BipartiteTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BipartiteTest, Exhaustive) {
size_t nodes = GetParam();
MatchMatrix graph(nodes, nodes);
do {
ElementMatcherPairs matches =
internal::FindMaxBipartiteMatching(graph);
EXPECT_EQ(FindBacktrackingMaxBPM(graph).size(), matches.size())
<< "graph: " << graph.DebugString();
std::vector<bool> seen_element(graph.LhsSize());
std::vector<bool> seen_matcher(graph.RhsSize());
SCOPED_TRACE(PrintToString(matches));
for (size_t i = 0; i < matches.size(); ++i) {
size_t ilhs = matches[i].first;
size_t irhs = matches[i].second;
EXPECT_TRUE(graph.HasEdge(ilhs, irhs));
EXPECT_FALSE(seen_element[ilhs]);
EXPECT_FALSE(seen_matcher[irhs]);
seen_element[ilhs] = true;
seen_matcher[irhs] = true;
}
} while (graph.NextGraph());
}

INSTANTIATE_TEST_SUITE_P(AllGraphs, BipartiteTest,
::testing::Range(size_t{0}, size_t{5}));

class BipartiteNonSquareTest
: public ::testing::TestWithParam<std::pair<size_t, size_t> > {
};

TEST_F(BipartiteNonSquareTest, SimpleBacktracking) {
MatchMatrix g(4, 3);
constexpr std::array<std::array<size_t, 2>, 4> kEdges = {
{{{0, 2}}, {{1, 1}}, {{2, 1}}, {{3, 0}}}};
for (size_t i = 0; i < kEdges.size(); ++i) {
g.SetEdge(kEdges[i][0], kEdges[i][1], true);
}
EXPECT_THAT(FindBacktrackingMaxBPM(g),
ElementsAre(Pair(3, 0),
Pair(AnyOf(1, 2), 1),
Pair(0, 2))) << g.DebugString();
}

TEST_P(BipartiteNonSquareTest, Exhaustive) {
size_t nlhs = GetParam().first;
size_t nrhs = GetParam().second;
MatchMatrix graph(nlhs, nrhs);
do {
EXPECT_EQ(FindBacktrackingMaxBPM(graph).size(),
internal::FindMaxBipartiteMatching(graph).size())
<< "graph: " << graph.DebugString()
<< "\nbacktracking: "
<< PrintToString(FindBacktrackingMaxBPM(graph))
<< "\nmax flow: "
<< PrintToString(internal::FindMaxBipartiteMatching(graph));
} while (graph.NextGraph());
}

INSTANTIATE_TEST_SUITE_P(AllGraphs, BipartiteNonSquareTest,
testing::Values(
std::make_pair(1, 2),
std::make_pair(2, 1),
std::make_pair(3, 2),
std::make_pair(2, 3),
std::make_pair(4, 1),
std::make_pair(1, 4),
std::make_pair(4, 3),
std::make_pair(3, 4)));

class BipartiteRandomTest
: public ::testing::TestWithParam<std::pair<int, int> > {
};

TEST_P(BipartiteRandomTest, LargerNets) {
int nodes = GetParam().first;
int iters = GetParam().second;
MatchMatrix graph(static_cast<size_t>(nodes), static_cast<size_t>(nodes));

auto seed = static_cast<uint32_t>(GTEST_FLAG(random_seed));
if (seed == 0) {
seed = static_cast<uint32_t>(time(nullptr));
}

for (; iters > 0; --iters, ++seed) {
srand(static_cast<unsigned int>(seed));
graph.Randomize();
EXPECT_EQ(FindBacktrackingMaxBPM(graph).size(),
internal::FindMaxBipartiteMatching(graph).size())
<< " graph: " << graph.DebugString()
<< "\nTo reproduce the failure, rerun the test with the flag"
" --" << GTEST_FLAG_PREFIX_ << "random_seed=" << seed;
}
}

INSTANTIATE_TEST_SUITE_P(Samples, BipartiteRandomTest,
testing::Values(
std::make_pair(5, 10000),
std::make_pair(6, 5000),
std::make_pair(7, 2000),
std::make_pair(8, 500),
std::make_pair(9, 100)));


TEST(IsReadableTypeNameTest, ReturnsTrueForShortNames) {
EXPECT_TRUE(IsReadableTypeName("int"));
EXPECT_TRUE(IsReadableTypeName("const unsigned char*"));
EXPECT_TRUE(IsReadableTypeName("MyMap<int, void*>"));
EXPECT_TRUE(IsReadableTypeName("void (*)(int, bool)"));
}

TEST(IsReadableTypeNameTest, ReturnsTrueForLongNonTemplateNonFunctionNames) {
EXPECT_TRUE(IsReadableTypeName("my_long_namespace::MyClassName"));
EXPECT_TRUE(IsReadableTypeName("int [5][6][7][8][9][10][11]"));
EXPECT_TRUE(IsReadableTypeName("my_namespace::MyOuterClass::MyInnerClass"));
}

TEST(IsReadableTypeNameTest, ReturnsFalseForLongTemplateNames) {
EXPECT_FALSE(
IsReadableTypeName("basic_string<char, std::char_traits<char> >"));
EXPECT_FALSE(IsReadableTypeName("std::vector<int, std::alloc_traits<int> >"));
}

TEST(IsReadableTypeNameTest, ReturnsFalseForLongFunctionTypeNames) {
EXPECT_FALSE(IsReadableTypeName("void (&)(int, bool, char, float)"));
}


TEST(FormatMatcherDescriptionTest, WorksForEmptyDescription) {
EXPECT_EQ("is even",
FormatMatcherDescription(false, "IsEven", Strings()));
EXPECT_EQ("not (is even)",
FormatMatcherDescription(true, "IsEven", Strings()));

const char* params[] = {"5"};
EXPECT_EQ("equals 5",
FormatMatcherDescription(false, "Equals",
Strings(params, params + 1)));

const char* params2[] = {"5", "8"};
EXPECT_EQ("is in range (5, 8)",
FormatMatcherDescription(false, "IsInRange",
Strings(params2, params2 + 2)));
}

TEST(PolymorphicMatcherTest, CanAccessMutableImpl) {
PolymorphicMatcher<DivisibleByImpl> m(DivisibleByImpl(42));
DivisibleByImpl& impl = m.mutable_impl();
EXPECT_EQ(42, impl.divider());

impl.set_divider(0);
EXPECT_EQ(0, m.mutable_impl().divider());
}

TEST(PolymorphicMatcherTest, CanAccessImpl) {
const PolymorphicMatcher<DivisibleByImpl> m(DivisibleByImpl(42));
const DivisibleByImpl& impl = m.impl();
EXPECT_EQ(42, impl.divider());
}

TEST(MatcherTupleTest, ExplainsMatchFailure) {
stringstream ss1;
ExplainMatchFailureTupleTo(
std::make_tuple(Matcher<char>(Eq('a')), GreaterThan(5)),
std::make_tuple('a', 10), &ss1);
EXPECT_EQ("", ss1.str());  

stringstream ss2;
ExplainMatchFailureTupleTo(
std::make_tuple(GreaterThan(5), Matcher<char>(Eq('a'))),
std::make_tuple(2, 'b'), &ss2);
EXPECT_EQ("  Expected arg #0: is > 5\n"
"           Actual: 2, which is 3 less than 5\n"
"  Expected arg #1: is equal to 'a' (97, 0x61)\n"
"           Actual: 'b' (98, 0x62)\n",
ss2.str());  

stringstream ss3;
ExplainMatchFailureTupleTo(
std::make_tuple(GreaterThan(5), Matcher<char>(Eq('a'))),
std::make_tuple(2, 'a'), &ss3);
EXPECT_EQ("  Expected arg #0: is > 5\n"
"           Actual: 2, which is 3 less than 5\n",
ss3.str());  
}


TEST(EachTest, ExplainsMatchResultCorrectly) {
set<int> a;  

Matcher<set<int> > m = Each(2);
EXPECT_EQ("", Explain(m, a));

Matcher<const int(&)[1]> n = Each(1);  

const int b[1] = {1};
EXPECT_EQ("", Explain(n, b));

n = Each(3);
EXPECT_EQ("whose element #0 doesn't match", Explain(n, b));

a.insert(1);
a.insert(2);
a.insert(3);
m = Each(GreaterThan(0));
EXPECT_EQ("", Explain(m, a));

m = Each(GreaterThan(10));
EXPECT_EQ("whose element #0 doesn't match, which is 9 less than 10",
Explain(m, a));
}

TEST(EachTest, DescribesItselfCorrectly) {
Matcher<vector<int> > m = Each(1);
EXPECT_EQ("only contains elements that is equal to 1", Describe(m));

Matcher<vector<int> > m2 = Not(m);
EXPECT_EQ("contains some element that isn't equal to 1", Describe(m2));
}

TEST(EachTest, MatchesVectorWhenAllElementsMatch) {
vector<int> some_vector;
EXPECT_THAT(some_vector, Each(1));
some_vector.push_back(3);
EXPECT_THAT(some_vector, Not(Each(1)));
EXPECT_THAT(some_vector, Each(3));
some_vector.push_back(1);
some_vector.push_back(2);
EXPECT_THAT(some_vector, Not(Each(3)));
EXPECT_THAT(some_vector, Each(Lt(3.5)));

vector<std::string> another_vector;
another_vector.push_back("fee");
EXPECT_THAT(another_vector, Each(std::string("fee")));
another_vector.push_back("fie");
another_vector.push_back("foe");
another_vector.push_back("fum");
EXPECT_THAT(another_vector, Not(Each(std::string("fee"))));
}

TEST(EachTest, MatchesMapWhenAllElementsMatch) {
map<const char*, int> my_map;
const char* bar = "a string";
my_map[bar] = 2;
EXPECT_THAT(my_map, Each(make_pair(bar, 2)));

map<std::string, int> another_map;
EXPECT_THAT(another_map, Each(make_pair(std::string("fee"), 1)));
another_map["fee"] = 1;
EXPECT_THAT(another_map, Each(make_pair(std::string("fee"), 1)));
another_map["fie"] = 2;
another_map["foe"] = 3;
another_map["fum"] = 4;
EXPECT_THAT(another_map, Not(Each(make_pair(std::string("fee"), 1))));
EXPECT_THAT(another_map, Not(Each(make_pair(std::string("fum"), 1))));
EXPECT_THAT(another_map, Each(Pair(_, Gt(0))));
}

TEST(EachTest, AcceptsMatcher) {
const int a[] = {1, 2, 3};
EXPECT_THAT(a, Each(Gt(0)));
EXPECT_THAT(a, Not(Each(Gt(1))));
}

TEST(EachTest, WorksForNativeArrayAsTuple) {
const int a[] = {1, 2};
const int* const pointer = a;
EXPECT_THAT(std::make_tuple(pointer, 2), Each(Gt(0)));
EXPECT_THAT(std::make_tuple(pointer, 2), Not(Each(Gt(1))));
}

TEST(EachTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(Each(Pointee(Gt(0)))));
helper.Call(MakeUniquePtrs({1, 2}));
}

class IsHalfOfMatcher {
public:
template <typename T1, typename T2>
bool MatchAndExplain(const std::tuple<T1, T2>& a_pair,
MatchResultListener* listener) const {
if (std::get<0>(a_pair) == std::get<1>(a_pair) / 2) {
*listener << "where the second is " << std::get<1>(a_pair);
return true;
} else {
*listener << "where the second/2 is " << std::get<1>(a_pair) / 2;
return false;
}
}

void DescribeTo(ostream* os) const {
*os << "are a pair where the first is half of the second";
}

void DescribeNegationTo(ostream* os) const {
*os << "are a pair where the first isn't half of the second";
}
};

PolymorphicMatcher<IsHalfOfMatcher> IsHalfOf() {
return MakePolymorphicMatcher(IsHalfOfMatcher());
}

TEST(PointwiseTest, DescribesSelf) {
vector<int> rhs;
rhs.push_back(1);
rhs.push_back(2);
rhs.push_back(3);
const Matcher<const vector<int>&> m = Pointwise(IsHalfOf(), rhs);
EXPECT_EQ("contains 3 values, where each value and its corresponding value "
"in { 1, 2, 3 } are a pair where the first is half of the second",
Describe(m));
EXPECT_EQ("doesn't contain exactly 3 values, or contains a value x at some "
"index i where x and the i-th value of { 1, 2, 3 } are a pair "
"where the first isn't half of the second",
DescribeNegation(m));
}

TEST(PointwiseTest, MakesCopyOfRhs) {
list<signed char> rhs;
rhs.push_back(2);
rhs.push_back(4);

int lhs[] = {1, 2};
const Matcher<const int (&)[2]> m = Pointwise(IsHalfOf(), rhs);
EXPECT_THAT(lhs, m);

rhs.push_back(6);
EXPECT_THAT(lhs, m);
}

TEST(PointwiseTest, WorksForLhsNativeArray) {
const int lhs[] = {1, 2, 3};
vector<int> rhs;
rhs.push_back(2);
rhs.push_back(4);
rhs.push_back(6);
EXPECT_THAT(lhs, Pointwise(Lt(), rhs));
EXPECT_THAT(lhs, Not(Pointwise(Gt(), rhs)));
}

TEST(PointwiseTest, WorksForRhsNativeArray) {
const int rhs[] = {1, 2, 3};
vector<int> lhs;
lhs.push_back(2);
lhs.push_back(4);
lhs.push_back(6);
EXPECT_THAT(lhs, Pointwise(Gt(), rhs));
EXPECT_THAT(lhs, Not(Pointwise(Lt(), rhs)));
}

TEST(PointwiseTest, WorksForVectorOfBool) {
vector<bool> rhs(3, false);
rhs[1] = true;
vector<bool> lhs = rhs;
EXPECT_THAT(lhs, Pointwise(Eq(), rhs));
rhs[0] = true;
EXPECT_THAT(lhs, Not(Pointwise(Eq(), rhs)));
}


TEST(PointwiseTest, WorksForRhsInitializerList) {
const vector<int> lhs{2, 4, 6};
EXPECT_THAT(lhs, Pointwise(Gt(), {1, 2, 3}));
EXPECT_THAT(lhs, Not(Pointwise(Lt(), {3, 3, 7})));
}


TEST(PointwiseTest, RejectsWrongSize) {
const double lhs[2] = {1, 2};
const int rhs[1] = {0};
EXPECT_THAT(lhs, Not(Pointwise(Gt(), rhs)));
EXPECT_EQ("which contains 2 values",
Explain(Pointwise(Gt(), rhs), lhs));

const int rhs2[3] = {0, 1, 2};
EXPECT_THAT(lhs, Not(Pointwise(Gt(), rhs2)));
}

TEST(PointwiseTest, RejectsWrongContent) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {2, 6, 4};
EXPECT_THAT(lhs, Not(Pointwise(IsHalfOf(), rhs)));
EXPECT_EQ("where the value pair (2, 6) at index #1 don't match, "
"where the second/2 is 3",
Explain(Pointwise(IsHalfOf(), rhs), lhs));
}

TEST(PointwiseTest, AcceptsCorrectContent) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {2, 4, 6};
EXPECT_THAT(lhs, Pointwise(IsHalfOf(), rhs));
EXPECT_EQ("", Explain(Pointwise(IsHalfOf(), rhs), lhs));
}

TEST(PointwiseTest, AllowsMonomorphicInnerMatcher) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {2, 4, 6};
const Matcher<std::tuple<const double&, const int&>> m1 = IsHalfOf();
EXPECT_THAT(lhs, Pointwise(m1, rhs));
EXPECT_EQ("", Explain(Pointwise(m1, rhs), lhs));

const Matcher<std::tuple<double, int>> m2 = IsHalfOf();
EXPECT_THAT(lhs, Pointwise(m2, rhs));
EXPECT_EQ("", Explain(Pointwise(m2, rhs), lhs));
}

MATCHER(PointeeEquals, "Points to an equal value") {
return ExplainMatchResult(::testing::Pointee(::testing::get<1>(arg)),
::testing::get<0>(arg), result_listener);
}

TEST(PointwiseTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(Pointwise(PointeeEquals(), std::vector<int>{1, 2})));
helper.Call(MakeUniquePtrs({1, 2}));
}

TEST(UnorderedPointwiseTest, DescribesSelf) {
vector<int> rhs;
rhs.push_back(1);
rhs.push_back(2);
rhs.push_back(3);
const Matcher<const vector<int>&> m = UnorderedPointwise(IsHalfOf(), rhs);
EXPECT_EQ(
"has 3 elements and there exists some permutation of elements such "
"that:\n"
" - element #0 and 1 are a pair where the first is half of the second, "
"and\n"
" - element #1 and 2 are a pair where the first is half of the second, "
"and\n"
" - element #2 and 3 are a pair where the first is half of the second",
Describe(m));
EXPECT_EQ(
"doesn't have 3 elements, or there exists no permutation of elements "
"such that:\n"
" - element #0 and 1 are a pair where the first is half of the second, "
"and\n"
" - element #1 and 2 are a pair where the first is half of the second, "
"and\n"
" - element #2 and 3 are a pair where the first is half of the second",
DescribeNegation(m));
}

TEST(UnorderedPointwiseTest, MakesCopyOfRhs) {
list<signed char> rhs;
rhs.push_back(2);
rhs.push_back(4);

int lhs[] = {2, 1};
const Matcher<const int (&)[2]> m = UnorderedPointwise(IsHalfOf(), rhs);
EXPECT_THAT(lhs, m);

rhs.push_back(6);
EXPECT_THAT(lhs, m);
}

TEST(UnorderedPointwiseTest, WorksForLhsNativeArray) {
const int lhs[] = {1, 2, 3};
vector<int> rhs;
rhs.push_back(4);
rhs.push_back(6);
rhs.push_back(2);
EXPECT_THAT(lhs, UnorderedPointwise(Lt(), rhs));
EXPECT_THAT(lhs, Not(UnorderedPointwise(Gt(), rhs)));
}

TEST(UnorderedPointwiseTest, WorksForRhsNativeArray) {
const int rhs[] = {1, 2, 3};
vector<int> lhs;
lhs.push_back(4);
lhs.push_back(2);
lhs.push_back(6);
EXPECT_THAT(lhs, UnorderedPointwise(Gt(), rhs));
EXPECT_THAT(lhs, Not(UnorderedPointwise(Lt(), rhs)));
}


TEST(UnorderedPointwiseTest, WorksForRhsInitializerList) {
const vector<int> lhs{2, 4, 6};
EXPECT_THAT(lhs, UnorderedPointwise(Gt(), {5, 1, 3}));
EXPECT_THAT(lhs, Not(UnorderedPointwise(Lt(), {1, 1, 7})));
}


TEST(UnorderedPointwiseTest, RejectsWrongSize) {
const double lhs[2] = {1, 2};
const int rhs[1] = {0};
EXPECT_THAT(lhs, Not(UnorderedPointwise(Gt(), rhs)));
EXPECT_EQ("which has 2 elements",
Explain(UnorderedPointwise(Gt(), rhs), lhs));

const int rhs2[3] = {0, 1, 2};
EXPECT_THAT(lhs, Not(UnorderedPointwise(Gt(), rhs2)));
}

TEST(UnorderedPointwiseTest, RejectsWrongContent) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {2, 6, 6};
EXPECT_THAT(lhs, Not(UnorderedPointwise(IsHalfOf(), rhs)));
EXPECT_EQ("where the following elements don't match any matchers:\n"
"element #1: 2",
Explain(UnorderedPointwise(IsHalfOf(), rhs), lhs));
}

TEST(UnorderedPointwiseTest, AcceptsCorrectContentInSameOrder) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {2, 4, 6};
EXPECT_THAT(lhs, UnorderedPointwise(IsHalfOf(), rhs));
}

TEST(UnorderedPointwiseTest, AcceptsCorrectContentInDifferentOrder) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {6, 4, 2};
EXPECT_THAT(lhs, UnorderedPointwise(IsHalfOf(), rhs));
}

TEST(UnorderedPointwiseTest, AllowsMonomorphicInnerMatcher) {
const double lhs[3] = {1, 2, 3};
const int rhs[3] = {4, 6, 2};
const Matcher<std::tuple<const double&, const int&>> m1 = IsHalfOf();
EXPECT_THAT(lhs, UnorderedPointwise(m1, rhs));

const Matcher<std::tuple<double, int>> m2 = IsHalfOf();
EXPECT_THAT(lhs, UnorderedPointwise(m2, rhs));
}

TEST(UnorderedPointwiseTest, WorksWithMoveOnly) {
ContainerHelper helper;
EXPECT_CALL(helper, Call(UnorderedPointwise(PointeeEquals(),
std::vector<int>{1, 2})));
helper.Call(MakeUniquePtrs({2, 1}));
}

template <typename T>
class SampleOptional {
public:
using value_type = T;
explicit SampleOptional(T value)
: value_(std::move(value)), has_value_(true) {}
SampleOptional() : value_(), has_value_(false) {}
operator bool() const { return has_value_; }
const T& operator*() const { return value_; }

private:
T value_;
bool has_value_;
};

TEST(OptionalTest, DescribesSelf) {
const Matcher<SampleOptional<int>> m = Optional(Eq(1));
EXPECT_EQ("value is equal to 1", Describe(m));
}

TEST(OptionalTest, ExplainsSelf) {
const Matcher<SampleOptional<int>> m = Optional(Eq(1));
EXPECT_EQ("whose value 1 matches", Explain(m, SampleOptional<int>(1)));
EXPECT_EQ("whose value 2 doesn't match", Explain(m, SampleOptional<int>(2)));
}

TEST(OptionalTest, MatchesNonEmptyOptional) {
const Matcher<SampleOptional<int>> m1 = Optional(1);
const Matcher<SampleOptional<int>> m2 = Optional(Eq(2));
const Matcher<SampleOptional<int>> m3 = Optional(Lt(3));
SampleOptional<int> opt(1);
EXPECT_TRUE(m1.Matches(opt));
EXPECT_FALSE(m2.Matches(opt));
EXPECT_TRUE(m3.Matches(opt));
}

TEST(OptionalTest, DoesNotMatchNullopt) {
const Matcher<SampleOptional<int>> m = Optional(1);
SampleOptional<int> empty;
EXPECT_FALSE(m.Matches(empty));
}

TEST(OptionalTest, WorksWithMoveOnly) {
Matcher<SampleOptional<std::unique_ptr<int>>> m = Optional(Eq(nullptr));
EXPECT_TRUE(m.Matches(SampleOptional<std::unique_ptr<int>>(nullptr)));
}

class SampleVariantIntString {
public:
SampleVariantIntString(int i) : i_(i), has_int_(true) {}
SampleVariantIntString(const std::string& s) : s_(s), has_int_(false) {}

template <typename T>
friend bool holds_alternative(const SampleVariantIntString& value) {
return value.has_int_ == std::is_same<T, int>::value;
}

template <typename T>
friend const T& get(const SampleVariantIntString& value) {
return value.get_impl(static_cast<T*>(nullptr));
}

private:
const int& get_impl(int*) const { return i_; }
const std::string& get_impl(std::string*) const { return s_; }

int i_;
std::string s_;
bool has_int_;
};

TEST(VariantTest, DescribesSelf) {
const Matcher<SampleVariantIntString> m = VariantWith<int>(Eq(1));
EXPECT_THAT(Describe(m), ContainsRegex("is a variant<> with value of type "
"'.*' and the value is equal to 1"));
}

TEST(VariantTest, ExplainsSelf) {
const Matcher<SampleVariantIntString> m = VariantWith<int>(Eq(1));
EXPECT_THAT(Explain(m, SampleVariantIntString(1)),
ContainsRegex("whose value 1"));
EXPECT_THAT(Explain(m, SampleVariantIntString("A")),
HasSubstr("whose value is not of type '"));
EXPECT_THAT(Explain(m, SampleVariantIntString(2)),
"whose value 2 doesn't match");
}

TEST(VariantTest, FullMatch) {
Matcher<SampleVariantIntString> m = VariantWith<int>(Eq(1));
EXPECT_TRUE(m.Matches(SampleVariantIntString(1)));

m = VariantWith<std::string>(Eq("1"));
EXPECT_TRUE(m.Matches(SampleVariantIntString("1")));
}

TEST(VariantTest, TypeDoesNotMatch) {
Matcher<SampleVariantIntString> m = VariantWith<int>(Eq(1));
EXPECT_FALSE(m.Matches(SampleVariantIntString("1")));

m = VariantWith<std::string>(Eq("1"));
EXPECT_FALSE(m.Matches(SampleVariantIntString(1)));
}

TEST(VariantTest, InnerDoesNotMatch) {
Matcher<SampleVariantIntString> m = VariantWith<int>(Eq(1));
EXPECT_FALSE(m.Matches(SampleVariantIntString(2)));

m = VariantWith<std::string>(Eq("1"));
EXPECT_FALSE(m.Matches(SampleVariantIntString("2")));
}

class SampleAnyType {
public:
explicit SampleAnyType(int i) : index_(0), i_(i) {}
explicit SampleAnyType(const std::string& s) : index_(1), s_(s) {}

template <typename T>
friend const T* any_cast(const SampleAnyType* any) {
return any->get_impl(static_cast<T*>(nullptr));
}

private:
int index_;
int i_;
std::string s_;

const int* get_impl(int*) const { return index_ == 0 ? &i_ : nullptr; }
const std::string* get_impl(std::string*) const {
return index_ == 1 ? &s_ : nullptr;
}
};

TEST(AnyWithTest, FullMatch) {
Matcher<SampleAnyType> m = AnyWith<int>(Eq(1));
EXPECT_TRUE(m.Matches(SampleAnyType(1)));
}

TEST(AnyWithTest, TestBadCastType) {
Matcher<SampleAnyType> m = AnyWith<std::string>(Eq("fail"));
EXPECT_FALSE(m.Matches(SampleAnyType(1)));
}

TEST(AnyWithTest, TestUseInContainers) {
std::vector<SampleAnyType> a;
a.emplace_back(1);
a.emplace_back(2);
a.emplace_back(3);
EXPECT_THAT(
a, ElementsAreArray({AnyWith<int>(1), AnyWith<int>(2), AnyWith<int>(3)}));

std::vector<SampleAnyType> b;
b.emplace_back("hello");
b.emplace_back("merhaba");
b.emplace_back("salut");
EXPECT_THAT(b, ElementsAreArray({AnyWith<std::string>("hello"),
AnyWith<std::string>("merhaba"),
AnyWith<std::string>("salut")}));
}
TEST(AnyWithTest, TestCompare) {
EXPECT_THAT(SampleAnyType(1), AnyWith<int>(Gt(0)));
}

TEST(AnyWithTest, DescribesSelf) {
const Matcher<const SampleAnyType&> m = AnyWith<int>(Eq(1));
EXPECT_THAT(Describe(m), ContainsRegex("is an 'any' type with value of type "
"'.*' and the value is equal to 1"));
}

TEST(AnyWithTest, ExplainsSelf) {
const Matcher<const SampleAnyType&> m = AnyWith<int>(Eq(1));

EXPECT_THAT(Explain(m, SampleAnyType(1)), ContainsRegex("whose value 1"));
EXPECT_THAT(Explain(m, SampleAnyType("A")),
HasSubstr("whose value is not of type '"));
EXPECT_THAT(Explain(m, SampleAnyType(2)), "whose value 2 doesn't match");
}

TEST(PointeeTest, WorksOnMoveOnlyType) {
std::unique_ptr<int> p(new int(3));
EXPECT_THAT(p, Pointee(Eq(3)));
EXPECT_THAT(p, Not(Pointee(Eq(2))));
}

TEST(NotTest, WorksOnMoveOnlyType) {
std::unique_ptr<int> p(new int(3));
EXPECT_THAT(p, Pointee(Eq(3)));
EXPECT_THAT(p, Not(Pointee(Eq(2))));
}


TEST(ArgsTest, AcceptsZeroTemplateArg) {
const std::tuple<int, bool> t(5, true);
EXPECT_THAT(t, Args<>(Eq(std::tuple<>())));
EXPECT_THAT(t, Not(Args<>(Ne(std::tuple<>()))));
}

TEST(ArgsTest, AcceptsOneTemplateArg) {
const std::tuple<int, bool> t(5, true);
EXPECT_THAT(t, Args<0>(Eq(std::make_tuple(5))));
EXPECT_THAT(t, Args<1>(Eq(std::make_tuple(true))));
EXPECT_THAT(t, Not(Args<1>(Eq(std::make_tuple(false)))));
}

TEST(ArgsTest, AcceptsTwoTemplateArgs) {
const std::tuple<short, int, long> t(4, 5, 6L);  

EXPECT_THAT(t, (Args<0, 1>(Lt())));
EXPECT_THAT(t, (Args<1, 2>(Lt())));
EXPECT_THAT(t, Not(Args<0, 2>(Gt())));
}

TEST(ArgsTest, AcceptsRepeatedTemplateArgs) {
const std::tuple<short, int, long> t(4, 5, 6L);  
EXPECT_THAT(t, (Args<0, 0>(Eq())));
EXPECT_THAT(t, Not(Args<1, 1>(Ne())));
}

TEST(ArgsTest, AcceptsDecreasingTemplateArgs) {
const std::tuple<short, int, long> t(4, 5, 6L);  
EXPECT_THAT(t, (Args<2, 0>(Gt())));
EXPECT_THAT(t, Not(Args<2, 1>(Lt())));
}

MATCHER(SumIsZero, "") {
return std::get<0>(arg) + std::get<1>(arg) + std::get<2>(arg) == 0;
}

TEST(ArgsTest, AcceptsMoreTemplateArgsThanArityOfOriginalTuple) {
EXPECT_THAT(std::make_tuple(-1, 2), (Args<0, 0, 1>(SumIsZero())));
EXPECT_THAT(std::make_tuple(1, 2), Not(Args<0, 0, 1>(SumIsZero())));
}

TEST(ArgsTest, CanBeNested) {
const std::tuple<short, int, long, int> t(4, 5, 6L, 6);  
EXPECT_THAT(t, (Args<1, 2, 3>(Args<1, 2>(Eq()))));
EXPECT_THAT(t, (Args<0, 1, 3>(Args<0, 2>(Lt()))));
}

TEST(ArgsTest, CanMatchTupleByValue) {
typedef std::tuple<char, int, int> Tuple3;
const Matcher<Tuple3> m = Args<1, 2>(Lt());
EXPECT_TRUE(m.Matches(Tuple3('a', 1, 2)));
EXPECT_FALSE(m.Matches(Tuple3('b', 2, 2)));
}

TEST(ArgsTest, CanMatchTupleByReference) {
typedef std::tuple<char, char, int> Tuple3;
const Matcher<const Tuple3&> m = Args<0, 1>(Lt());
EXPECT_TRUE(m.Matches(Tuple3('a', 'b', 2)));
EXPECT_FALSE(m.Matches(Tuple3('b', 'b', 2)));
}

MATCHER_P(PrintsAs, str, "") {
return testing::PrintToString(arg) == str;
}

TEST(ArgsTest, AcceptsTenTemplateArgs) {
EXPECT_THAT(std::make_tuple(0, 1L, 2, 3L, 4, 5, 6, 7, 8, 9),
(Args<9, 8, 7, 6, 5, 4, 3, 2, 1, 0>(
PrintsAs("(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)"))));
EXPECT_THAT(std::make_tuple(0, 1L, 2, 3L, 4, 5, 6, 7, 8, 9),
Not(Args<9, 8, 7, 6, 5, 4, 3, 2, 1, 0>(
PrintsAs("(0, 8, 7, 6, 5, 4, 3, 2, 1, 0)"))));
}

TEST(ArgsTest, DescirbesSelfCorrectly) {
const Matcher<std::tuple<int, bool, char> > m = Args<2, 0>(Lt());
EXPECT_EQ("are a tuple whose fields (#2, #0) are a pair where "
"the first < the second",
Describe(m));
}

TEST(ArgsTest, DescirbesNestedArgsCorrectly) {
const Matcher<const std::tuple<int, bool, char, int>&> m =
Args<0, 2, 3>(Args<2, 0>(Lt()));
EXPECT_EQ("are a tuple whose fields (#0, #2, #3) are a tuple "
"whose fields (#2, #0) are a pair where the first < the second",
Describe(m));
}

TEST(ArgsTest, DescribesNegationCorrectly) {
const Matcher<std::tuple<int, char> > m = Args<1, 0>(Gt());
EXPECT_EQ("are a tuple whose fields (#1, #0) aren't a pair "
"where the first > the second",
DescribeNegation(m));
}

TEST(ArgsTest, ExplainsMatchResultWithoutInnerExplanation) {
const Matcher<std::tuple<bool, int, int> > m = Args<1, 2>(Eq());
EXPECT_EQ("whose fields (#1, #2) are (42, 42)",
Explain(m, std::make_tuple(false, 42, 42)));
EXPECT_EQ("whose fields (#1, #2) are (42, 43)",
Explain(m, std::make_tuple(false, 42, 43)));
}

class LessThanMatcher : public MatcherInterface<std::tuple<char, int> > {
public:
void DescribeTo(::std::ostream* ) const override {}

bool MatchAndExplain(std::tuple<char, int> value,
MatchResultListener* listener) const override {
const int diff = std::get<0>(value) - std::get<1>(value);
if (diff > 0) {
*listener << "where the first value is " << diff
<< " more than the second";
}
return diff < 0;
}
};

Matcher<std::tuple<char, int> > LessThan() {
return MakeMatcher(new LessThanMatcher);
}

TEST(ArgsTest, ExplainsMatchResultWithInnerExplanation) {
const Matcher<std::tuple<char, int, int> > m = Args<0, 2>(LessThan());
EXPECT_EQ(
"whose fields (#0, #2) are ('a' (97, 0x61), 42), "
"where the first value is 55 more than the second",
Explain(m, std::make_tuple('a', 42, 42)));
EXPECT_EQ("whose fields (#0, #2) are ('\\0', 43)",
Explain(m, std::make_tuple('\0', 42, 43)));
}

class PredicateFormatterFromMatcherTest : public ::testing::Test {
protected:
enum Behavior { kInitialSuccess, kAlwaysFail, kFlaky };

class MockMatcher : public MatcherInterface<Behavior> {
public:
bool MatchAndExplain(Behavior behavior,
MatchResultListener* listener) const override {
*listener << "[MatchAndExplain]";
switch (behavior) {
case kInitialSuccess:
return !listener->IsInterested();

case kAlwaysFail:
return false;

case kFlaky:
return listener->IsInterested();
}

GTEST_LOG_(FATAL) << "This should never be reached";
return false;
}

void DescribeTo(ostream* os) const override { *os << "[DescribeTo]"; }

void DescribeNegationTo(ostream* os) const override {
*os << "[DescribeNegationTo]";
}
};

AssertionResult RunPredicateFormatter(Behavior behavior) {
auto matcher = MakeMatcher(new MockMatcher);
PredicateFormatterFromMatcher<Matcher<Behavior>> predicate_formatter(
matcher);
return predicate_formatter("dummy-name", behavior);
}
};

TEST_F(PredicateFormatterFromMatcherTest, ShortCircuitOnSuccess) {
AssertionResult result = RunPredicateFormatter(kInitialSuccess);
EXPECT_TRUE(result);  
std::string expect;
EXPECT_EQ(expect, result.message());
}

TEST_F(PredicateFormatterFromMatcherTest, NoShortCircuitOnFailure) {
AssertionResult result = RunPredicateFormatter(kAlwaysFail);
EXPECT_FALSE(result);  
std::string expect =
"Value of: dummy-name\nExpected: [DescribeTo]\n"
"  Actual: 1" +
OfType(internal::GetTypeName<Behavior>()) + ", [MatchAndExplain]";
EXPECT_EQ(expect, result.message());
}

TEST_F(PredicateFormatterFromMatcherTest, DetectsFlakyShortCircuit) {
AssertionResult result = RunPredicateFormatter(kFlaky);
EXPECT_FALSE(result);  
std::string expect =
"Value of: dummy-name\nExpected: [DescribeTo]\n"
"  The matcher failed on the initial attempt; but passed when rerun to "
"generate the explanation.\n"
"  Actual: 2" +
OfType(internal::GetTypeName<Behavior>()) + ", [MatchAndExplain]";
EXPECT_EQ(expect, result.message());
}


TEST(ElementsAreTest, CanDescribeExpectingNoElement) {
Matcher<const vector<int>&> m = ElementsAre();
EXPECT_EQ("is empty", Describe(m));
}

TEST(ElementsAreTest, CanDescribeExpectingOneElement) {
Matcher<vector<int>> m = ElementsAre(Gt(5));
EXPECT_EQ("has 1 element that is > 5", Describe(m));
}

TEST(ElementsAreTest, CanDescribeExpectingManyElements) {
Matcher<list<std::string>> m = ElementsAre(StrEq("one"), "two");
EXPECT_EQ(
"has 2 elements where\n"
"element #0 is equal to \"one\",\n"
"element #1 is equal to \"two\"",
Describe(m));
}

TEST(ElementsAreTest, CanDescribeNegationOfExpectingNoElement) {
Matcher<vector<int>> m = ElementsAre();
EXPECT_EQ("isn't empty", DescribeNegation(m));
}

TEST(ElementsAreTest, CanDescribeNegationOfExpectingOneElment) {
Matcher<const list<int>&> m = ElementsAre(Gt(5));
EXPECT_EQ(
"doesn't have 1 element, or\n"
"element #0 isn't > 5",
DescribeNegation(m));
}

TEST(ElementsAreTest, CanDescribeNegationOfExpectingManyElements) {
Matcher<const list<std::string>&> m = ElementsAre("one", "two");
EXPECT_EQ(
"doesn't have 2 elements, or\n"
"element #0 isn't equal to \"one\", or\n"
"element #1 isn't equal to \"two\"",
DescribeNegation(m));
}

TEST(ElementsAreTest, DoesNotExplainTrivialMatch) {
Matcher<const list<int>&> m = ElementsAre(1, Ne(2));

list<int> test_list;
test_list.push_back(1);
test_list.push_back(3);
EXPECT_EQ("", Explain(m, test_list));  
}

TEST(ElementsAreTest, ExplainsNonTrivialMatch) {
Matcher<const vector<int>&> m =
ElementsAre(GreaterThan(1), 0, GreaterThan(2));

const int a[] = {10, 0, 100};
vector<int> test_vector(std::begin(a), std::end(a));
EXPECT_EQ(
"whose element #0 matches, which is 9 more than 1,\n"
"and whose element #2 matches, which is 98 more than 2",
Explain(m, test_vector));
}

TEST(ElementsAreTest, CanExplainMismatchWrongSize) {
Matcher<const list<int>&> m = ElementsAre(1, 3);

list<int> test_list;
EXPECT_EQ("", Explain(m, test_list));

test_list.push_back(1);
EXPECT_EQ("which has 1 element", Explain(m, test_list));
}

TEST(ElementsAreTest, CanExplainMismatchRightSize) {
Matcher<const vector<int>&> m = ElementsAre(1, GreaterThan(5));

vector<int> v;
v.push_back(2);
v.push_back(1);
EXPECT_EQ("whose element #0 doesn't match", Explain(m, v));

v[0] = 1;
EXPECT_EQ("whose element #1 doesn't match, which is 4 less than 5",
Explain(m, v));
}

TEST(ElementsAreTest, MatchesOneElementVector) {
vector<std::string> test_vector;
test_vector.push_back("test string");

EXPECT_THAT(test_vector, ElementsAre(StrEq("test string")));
}

TEST(ElementsAreTest, MatchesOneElementList) {
list<std::string> test_list;
test_list.push_back("test string");

EXPECT_THAT(test_list, ElementsAre("test string"));
}

TEST(ElementsAreTest, MatchesThreeElementVector) {
vector<std::string> test_vector;
test_vector.push_back("one");
test_vector.push_back("two");
test_vector.push_back("three");

EXPECT_THAT(test_vector, ElementsAre("one", StrEq("two"), _));
}

TEST(ElementsAreTest, MatchesOneElementEqMatcher) {
vector<int> test_vector;
test_vector.push_back(4);

EXPECT_THAT(test_vector, ElementsAre(Eq(4)));
}

TEST(ElementsAreTest, MatchesOneElementAnyMatcher) {
vector<int> test_vector;
test_vector.push_back(4);

EXPECT_THAT(test_vector, ElementsAre(_));
}

TEST(ElementsAreTest, MatchesOneElementValue) {
vector<int> test_vector;
test_vector.push_back(4);

EXPECT_THAT(test_vector, ElementsAre(4));
}

TEST(ElementsAreTest, MatchesThreeElementsMixedMatchers) {
vector<int> test_vector;
test_vector.push_back(1);
test_vector.push_back(2);
test_vector.push_back(3);

EXPECT_THAT(test_vector, ElementsAre(1, Eq(2), _));
}

TEST(ElementsAreTest, MatchesTenElementVector) {
const int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
vector<int> test_vector(std::begin(a), std::end(a));

EXPECT_THAT(test_vector,
ElementsAre(0, Ge(0), _, 3, 4, Ne(2), Eq(6), 7, 8, _));
}

TEST(ElementsAreTest, DoesNotMatchWrongSize) {
vector<std::string> test_vector;
test_vector.push_back("test string");
test_vector.push_back("test string");

Matcher<vector<std::string>> m = ElementsAre(StrEq("test string"));
EXPECT_FALSE(m.Matches(test_vector));
}

TEST(ElementsAreTest, DoesNotMatchWrongValue) {
vector<std::string> test_vector;
test_vector.push_back("other string");

Matcher<vector<std::string>> m = ElementsAre(StrEq("test string"));
EXPECT_FALSE(m.Matches(test_vector));
}

TEST(ElementsAreTest, DoesNotMatchWrongOrder) {
vector<std::string> test_vector;
test_vector.push_back("one");
test_vector.push_back("three");
test_vector.push_back("two");

Matcher<vector<std::string>> m =
ElementsAre(StrEq("one"), StrEq("two"), StrEq("three"));
EXPECT_FALSE(m.Matches(test_vector));
}

TEST(ElementsAreTest, WorksForNestedContainer) {
constexpr std::array<const char*, 2> strings = {{"Hi", "world"}};

vector<list<char>> nested;
for (const auto& s : strings) {
nested.emplace_back(s, s + strlen(s));
}

EXPECT_THAT(nested, ElementsAre(ElementsAre('H', Ne('e')),
ElementsAre('w', 'o', _, _, 'd')));
EXPECT_THAT(nested, Not(ElementsAre(ElementsAre('H', 'e'),
ElementsAre('w', 'o', _, _, 'd'))));
}

TEST(ElementsAreTest, WorksWithByRefElementMatchers) {
int a[] = {0, 1, 2};
vector<int> v(std::begin(a), std::end(a));

EXPECT_THAT(v, ElementsAre(Ref(v[0]), Ref(v[1]), Ref(v[2])));
EXPECT_THAT(v, Not(ElementsAre(Ref(v[0]), Ref(v[1]), Ref(a[2]))));
}

TEST(ElementsAreTest, WorksWithContainerPointerUsingPointee) {
int a[] = {0, 1, 2};
vector<int> v(std::begin(a), std::end(a));

EXPECT_THAT(&v, Pointee(ElementsAre(0, 1, _)));
EXPECT_THAT(&v, Not(Pointee(ElementsAre(0, _, 3))));
}

TEST(ElementsAreTest, WorksWithNativeArrayPassedByReference) {
int array[] = {0, 1, 2};
EXPECT_THAT(array, ElementsAre(0, 1, _));
EXPECT_THAT(array, Not(ElementsAre(1, _, _)));
EXPECT_THAT(array, Not(ElementsAre(0, _)));
}

class NativeArrayPassedAsPointerAndSize {
public:
NativeArrayPassedAsPointerAndSize() {}

MOCK_METHOD(void, Helper, (int* array, int size));

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(NativeArrayPassedAsPointerAndSize);
};

TEST(ElementsAreTest, WorksWithNativeArrayPassedAsPointerAndSize) {
int array[] = {0, 1};
::std::tuple<int*, size_t> array_as_tuple(array, 2);
EXPECT_THAT(array_as_tuple, ElementsAre(0, 1));
EXPECT_THAT(array_as_tuple, Not(ElementsAre(0)));

NativeArrayPassedAsPointerAndSize helper;
EXPECT_CALL(helper, Helper(_, _)).With(ElementsAre(0, 1));
helper.Helper(array, 2);
}

TEST(ElementsAreTest, WorksWithTwoDimensionalNativeArray) {
const char a2[][3] = {"hi", "lo"};
EXPECT_THAT(a2, ElementsAre(ElementsAre('h', 'i', '\0'),
ElementsAre('l', 'o', '\0')));
EXPECT_THAT(a2, ElementsAre(StrEq("hi"), StrEq("lo")));
EXPECT_THAT(a2, ElementsAre(Not(ElementsAre('h', 'o', '\0')),
ElementsAre('l', 'o', '\0')));
}

TEST(ElementsAreTest, AcceptsStringLiteral) {
std::string array[] = {"hi", "one", "two"};
EXPECT_THAT(array, ElementsAre("hi", "one", "two"));
EXPECT_THAT(array, Not(ElementsAre("hi", "one", "too")));
}

extern const char kHi[];

TEST(ElementsAreTest, AcceptsArrayWithUnknownSize) {

std::string array1[] = {"hi"};
EXPECT_THAT(array1, ElementsAre(kHi));

std::string array2[] = {"ho"};
EXPECT_THAT(array2, Not(ElementsAre(kHi)));
}

const char kHi[] = "hi";

TEST(ElementsAreTest, MakesCopyOfArguments) {
int x = 1;
int y = 2;
::testing::internal::ElementsAreMatcher<std::tuple<int, int>>
polymorphic_matcher = ElementsAre(x, y);
x = y = 0;
const int array1[] = {1, 2};
EXPECT_THAT(array1, polymorphic_matcher);
const int array2[] = {0, 0};
EXPECT_THAT(array2, Not(polymorphic_matcher));
}


TEST(ElementsAreArrayTest, CanBeCreatedWithValueArray) {
const int a[] = {1, 2, 3};

vector<int> test_vector(std::begin(a), std::end(a));
EXPECT_THAT(test_vector, ElementsAreArray(a));

test_vector[2] = 0;
EXPECT_THAT(test_vector, Not(ElementsAreArray(a)));
}

TEST(ElementsAreArrayTest, CanBeCreatedWithArraySize) {
std::array<const char*, 3> a = {{"one", "two", "three"}};

vector<std::string> test_vector(std::begin(a), std::end(a));
EXPECT_THAT(test_vector, ElementsAreArray(a.data(), a.size()));

const char** p = a.data();
test_vector[0] = "1";
EXPECT_THAT(test_vector, Not(ElementsAreArray(p, a.size())));
}

TEST(ElementsAreArrayTest, CanBeCreatedWithoutArraySize) {
const char* a[] = {"one", "two", "three"};

vector<std::string> test_vector(std::begin(a), std::end(a));
EXPECT_THAT(test_vector, ElementsAreArray(a));

test_vector[0] = "1";
EXPECT_THAT(test_vector, Not(ElementsAreArray(a)));
}

TEST(ElementsAreArrayTest, CanBeCreatedWithMatcherArray) {
const Matcher<std::string> kMatcherArray[] = {StrEq("one"), StrEq("two"),
StrEq("three")};

vector<std::string> test_vector;
test_vector.push_back("one");
test_vector.push_back("two");
test_vector.push_back("three");
EXPECT_THAT(test_vector, ElementsAreArray(kMatcherArray));

test_vector.push_back("three");
EXPECT_THAT(test_vector, Not(ElementsAreArray(kMatcherArray)));
}

TEST(ElementsAreArrayTest, CanBeCreatedWithVector) {
const int a[] = {1, 2, 3};
vector<int> test_vector(std::begin(a), std::end(a));
const vector<int> expected(std::begin(a), std::end(a));
EXPECT_THAT(test_vector, ElementsAreArray(expected));
test_vector.push_back(4);
EXPECT_THAT(test_vector, Not(ElementsAreArray(expected)));
}

TEST(ElementsAreArrayTest, TakesInitializerList) {
const int a[5] = {1, 2, 3, 4, 5};
EXPECT_THAT(a, ElementsAreArray({1, 2, 3, 4, 5}));
EXPECT_THAT(a, Not(ElementsAreArray({1, 2, 3, 5, 4})));
EXPECT_THAT(a, Not(ElementsAreArray({1, 2, 3, 4, 6})));
}

TEST(ElementsAreArrayTest, TakesInitializerListOfCStrings) {
const std::string a[5] = {"a", "b", "c", "d", "e"};
EXPECT_THAT(a, ElementsAreArray({"a", "b", "c", "d", "e"}));
EXPECT_THAT(a, Not(ElementsAreArray({"a", "b", "c", "e", "d"})));
EXPECT_THAT(a, Not(ElementsAreArray({"a", "b", "c", "d", "ef"})));
}

TEST(ElementsAreArrayTest, TakesInitializerListOfSameTypedMatchers) {
const int a[5] = {1, 2, 3, 4, 5};
EXPECT_THAT(a, ElementsAreArray({Eq(1), Eq(2), Eq(3), Eq(4), Eq(5)}));
EXPECT_THAT(a, Not(ElementsAreArray({Eq(1), Eq(2), Eq(3), Eq(4), Eq(6)})));
}

TEST(ElementsAreArrayTest, TakesInitializerListOfDifferentTypedMatchers) {
const int a[5] = {1, 2, 3, 4, 5};
EXPECT_THAT(
a, ElementsAreArray<Matcher<int>>({Eq(1), Ne(-2), Ge(3), Le(4), Eq(5)}));
EXPECT_THAT(a, Not(ElementsAreArray<Matcher<int>>(
{Eq(1), Ne(-2), Ge(3), Le(4), Eq(6)})));
}

TEST(ElementsAreArrayTest, CanBeCreatedWithMatcherVector) {
const int a[] = {1, 2, 3};
const Matcher<int> kMatchers[] = {Eq(1), Eq(2), Eq(3)};
vector<int> test_vector(std::begin(a), std::end(a));
const vector<Matcher<int>> expected(std::begin(kMatchers),
std::end(kMatchers));
EXPECT_THAT(test_vector, ElementsAreArray(expected));
test_vector.push_back(4);
EXPECT_THAT(test_vector, Not(ElementsAreArray(expected)));
}

TEST(ElementsAreArrayTest, CanBeCreatedWithIteratorRange) {
const int a[] = {1, 2, 3};
const vector<int> test_vector(std::begin(a), std::end(a));
const vector<int> expected(std::begin(a), std::end(a));
EXPECT_THAT(test_vector, ElementsAreArray(expected.begin(), expected.end()));
EXPECT_THAT(test_vector, ElementsAreArray(std::begin(a), std::end(a)));
int* const null_int = nullptr;
EXPECT_THAT(test_vector, Not(ElementsAreArray(null_int, null_int)));
EXPECT_THAT((vector<int>()), ElementsAreArray(null_int, null_int));
}

TEST(ElementsAreArrayTest, WorksWithNativeArray) {
::std::string a[] = {"hi", "ho"};
::std::string b[] = {"hi", "ho"};

EXPECT_THAT(a, ElementsAreArray(b));
EXPECT_THAT(a, ElementsAreArray(b, 2));
EXPECT_THAT(a, Not(ElementsAreArray(b, 1)));
}

TEST(ElementsAreArrayTest, SourceLifeSpan) {
const int a[] = {1, 2, 3};
vector<int> test_vector(std::begin(a), std::end(a));
vector<int> expect(std::begin(a), std::end(a));
ElementsAreArrayMatcher<int> matcher_maker =
ElementsAreArray(expect.begin(), expect.end());
EXPECT_THAT(test_vector, matcher_maker);
for (int& i : expect) {
i += 10;
}
EXPECT_THAT(test_vector, matcher_maker);
test_vector.push_back(3);
EXPECT_THAT(test_vector, Not(matcher_maker));
}



MATCHER(IsEven, "") { return (arg % 2) == 0; }

TEST(MatcherMacroTest, Works) {
const Matcher<int> m = IsEven();
EXPECT_TRUE(m.Matches(6));
EXPECT_FALSE(m.Matches(7));

EXPECT_EQ("is even", Describe(m));
EXPECT_EQ("not (is even)", DescribeNegation(m));
EXPECT_EQ("", Explain(m, 6));
EXPECT_EQ("", Explain(m, 7));
}

MATCHER(IsEven2, negation ? "is odd" : "is even") {
if ((arg % 2) == 0) {
*result_listener << "OK";
return true;
} else {
*result_listener << "% 2 == " << (arg % 2);
return false;
}
}

MATCHER_P2(EqSumOf, x, y,
std::string(negation ? "doesn't equal" : "equals") + " the sum of " +
PrintToString(x) + " and " + PrintToString(y)) {
if (arg == (x + y)) {
*result_listener << "OK";
return true;
} else {
if (result_listener->stream() != nullptr) {
*result_listener->stream() << "diff == " << (x + y - arg);
}
return false;
}
}

TEST(MatcherMacroTest, DescriptionCanReferenceNegationAndParameters) {
const Matcher<int> m1 = IsEven2();
EXPECT_EQ("is even", Describe(m1));
EXPECT_EQ("is odd", DescribeNegation(m1));

const Matcher<int> m2 = EqSumOf(5, 9);
EXPECT_EQ("equals the sum of 5 and 9", Describe(m2));
EXPECT_EQ("doesn't equal the sum of 5 and 9", DescribeNegation(m2));
}

TEST(MatcherMacroTest, CanExplainMatchResult) {
const Matcher<int> m1 = IsEven2();
EXPECT_EQ("OK", Explain(m1, 4));
EXPECT_EQ("% 2 == 1", Explain(m1, 5));

const Matcher<int> m2 = EqSumOf(1, 2);
EXPECT_EQ("OK", Explain(m2, 3));
EXPECT_EQ("diff == -1", Explain(m2, 4));
}


MATCHER(IsEmptyString, "") {
StaticAssertTypeEq<::std::string, arg_type>();
return arg.empty();
}

MATCHER(IsEmptyStringByRef, "") {
StaticAssertTypeEq<const ::std::string&, arg_type>();
return arg.empty();
}

TEST(MatcherMacroTest, CanReferenceArgType) {
const Matcher<::std::string> m1 = IsEmptyString();
EXPECT_TRUE(m1.Matches(""));

const Matcher<const ::std::string&> m2 = IsEmptyStringByRef();
EXPECT_TRUE(m2.Matches(""));
}


namespace matcher_test {
MATCHER(IsOdd, "") { return (arg % 2) != 0; }
}  

TEST(MatcherMacroTest, WorksInNamespace) {
Matcher<int> m = matcher_test::IsOdd();
EXPECT_FALSE(m.Matches(4));
EXPECT_TRUE(m.Matches(5));
}

MATCHER(IsPositiveOdd, "") {
return Value(arg, matcher_test::IsOdd()) && arg > 0;
}

TEST(MatcherMacroTest, CanBeComposedUsingValue) {
EXPECT_THAT(3, IsPositiveOdd());
EXPECT_THAT(4, Not(IsPositiveOdd()));
EXPECT_THAT(-1, Not(IsPositiveOdd()));
}


MATCHER_P(IsGreaterThan32And, n, "") { return arg > 32 && arg > n; }

TEST(MatcherPMacroTest, Works) {
const Matcher<int> m = IsGreaterThan32And(5);
EXPECT_TRUE(m.Matches(36));
EXPECT_FALSE(m.Matches(5));

EXPECT_EQ("is greater than 32 and 5", Describe(m));
EXPECT_EQ("not (is greater than 32 and 5)", DescribeNegation(m));
EXPECT_EQ("", Explain(m, 36));
EXPECT_EQ("", Explain(m, 5));
}

MATCHER_P(_is_Greater_Than32and_, n, "") { return arg > 32 && arg > n; }

TEST(MatcherPMacroTest, GeneratesCorrectDescription) {
const Matcher<int> m = _is_Greater_Than32and_(5);

EXPECT_EQ("is greater than 32 and 5", Describe(m));
EXPECT_EQ("not (is greater than 32 and 5)", DescribeNegation(m));
EXPECT_EQ("", Explain(m, 36));
EXPECT_EQ("", Explain(m, 5));
}


class UncopyableFoo {
public:
explicit UncopyableFoo(char value) : value_(value) { (void)value_; }

UncopyableFoo(const UncopyableFoo&) = delete;
void operator=(const UncopyableFoo&) = delete;

private:
char value_;
};

MATCHER_P(ReferencesUncopyable, variable, "") { return &arg == &variable; }

TEST(MatcherPMacroTest, WorksWhenExplicitlyInstantiatedWithReference) {
UncopyableFoo foo1('1'), foo2('2');
const Matcher<const UncopyableFoo&> m =
ReferencesUncopyable<const UncopyableFoo&>(foo1);

EXPECT_TRUE(m.Matches(foo1));
EXPECT_FALSE(m.Matches(foo2));

EXPECT_EQ("references uncopyable 1-byte object <31>", Describe(m));
}


MATCHER_P3(ParamTypesAreIntLongAndChar, foo, bar, baz, "") {
StaticAssertTypeEq<int, foo_type>();
StaticAssertTypeEq<long, bar_type>();  
StaticAssertTypeEq<char, baz_type>();
return arg == 0;
}

TEST(MatcherPnMacroTest, CanReferenceParamTypes) {
EXPECT_THAT(0, ParamTypesAreIntLongAndChar(10, 20L, 'a'));
}


MATCHER_P2(ReferencesAnyOf, variable1, variable2, "") {
return &arg == &variable1 || &arg == &variable2;
}

TEST(MatcherPnMacroTest, WorksWhenExplicitlyInstantiatedWithReferences) {
UncopyableFoo foo1('1'), foo2('2'), foo3('3');
const Matcher<const UncopyableFoo&> const_m =
ReferencesAnyOf<const UncopyableFoo&, const UncopyableFoo&>(foo1, foo2);

EXPECT_TRUE(const_m.Matches(foo1));
EXPECT_TRUE(const_m.Matches(foo2));
EXPECT_FALSE(const_m.Matches(foo3));

const Matcher<UncopyableFoo&> m =
ReferencesAnyOf<UncopyableFoo&, UncopyableFoo&>(foo1, foo2);

EXPECT_TRUE(m.Matches(foo1));
EXPECT_TRUE(m.Matches(foo2));
EXPECT_FALSE(m.Matches(foo3));
}

TEST(MatcherPnMacroTest,
GeneratesCorretDescriptionWhenExplicitlyInstantiatedWithReferences) {
UncopyableFoo foo1('1'), foo2('2');
const Matcher<const UncopyableFoo&> m =
ReferencesAnyOf<const UncopyableFoo&, const UncopyableFoo&>(foo1, foo2);

EXPECT_EQ("references any of (1-byte object <31>, 1-byte object <32>)",
Describe(m));
}


MATCHER_P2(IsNotInClosedRange, low, hi, "") { return arg < low || arg > hi; }

TEST(MatcherPnMacroTest, Works) {
const Matcher<const long&> m = IsNotInClosedRange(10, 20);  
EXPECT_TRUE(m.Matches(36L));
EXPECT_FALSE(m.Matches(15L));

EXPECT_EQ("is not in closed range (10, 20)", Describe(m));
EXPECT_EQ("not (is not in closed range (10, 20))", DescribeNegation(m));
EXPECT_EQ("", Explain(m, 36L));
EXPECT_EQ("", Explain(m, 15L));
}


MATCHER(EqualsSumOf, "") { return arg == 0; }
MATCHER_P(EqualsSumOf, a, "") { return arg == a; }
MATCHER_P2(EqualsSumOf, a, b, "") { return arg == a + b; }
MATCHER_P3(EqualsSumOf, a, b, c, "") { return arg == a + b + c; }
MATCHER_P4(EqualsSumOf, a, b, c, d, "") { return arg == a + b + c + d; }
MATCHER_P5(EqualsSumOf, a, b, c, d, e, "") { return arg == a + b + c + d + e; }
MATCHER_P6(EqualsSumOf, a, b, c, d, e, f, "") {
return arg == a + b + c + d + e + f;
}
MATCHER_P7(EqualsSumOf, a, b, c, d, e, f, g, "") {
return arg == a + b + c + d + e + f + g;
}
MATCHER_P8(EqualsSumOf, a, b, c, d, e, f, g, h, "") {
return arg == a + b + c + d + e + f + g + h;
}
MATCHER_P9(EqualsSumOf, a, b, c, d, e, f, g, h, i, "") {
return arg == a + b + c + d + e + f + g + h + i;
}
MATCHER_P10(EqualsSumOf, a, b, c, d, e, f, g, h, i, j, "") {
return arg == a + b + c + d + e + f + g + h + i + j;
}

TEST(MatcherPnMacroTest, CanBeOverloadedOnNumberOfParameters) {
EXPECT_THAT(0, EqualsSumOf());
EXPECT_THAT(1, EqualsSumOf(1));
EXPECT_THAT(12, EqualsSumOf(10, 2));
EXPECT_THAT(123, EqualsSumOf(100, 20, 3));
EXPECT_THAT(1234, EqualsSumOf(1000, 200, 30, 4));
EXPECT_THAT(12345, EqualsSumOf(10000, 2000, 300, 40, 5));
EXPECT_THAT("abcdef",
EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e", 'f'));
EXPECT_THAT("abcdefg",
EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e", 'f', 'g'));
EXPECT_THAT("abcdefgh", EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e",
'f', 'g', "h"));
EXPECT_THAT("abcdefghi", EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e",
'f', 'g', "h", 'i'));
EXPECT_THAT("abcdefghij",
EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e", 'f', 'g', "h",
'i', ::std::string("j")));

EXPECT_THAT(1, Not(EqualsSumOf()));
EXPECT_THAT(-1, Not(EqualsSumOf(1)));
EXPECT_THAT(-12, Not(EqualsSumOf(10, 2)));
EXPECT_THAT(-123, Not(EqualsSumOf(100, 20, 3)));
EXPECT_THAT(-1234, Not(EqualsSumOf(1000, 200, 30, 4)));
EXPECT_THAT(-12345, Not(EqualsSumOf(10000, 2000, 300, 40, 5)));
EXPECT_THAT("abcdef ",
Not(EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e", 'f')));
EXPECT_THAT("abcdefg ", Not(EqualsSumOf(::std::string("a"), 'b', 'c', "d",
"e", 'f', 'g')));
EXPECT_THAT("abcdefgh ", Not(EqualsSumOf(::std::string("a"), 'b', 'c', "d",
"e", 'f', 'g', "h")));
EXPECT_THAT("abcdefghi ", Not(EqualsSumOf(::std::string("a"), 'b', 'c', "d",
"e", 'f', 'g', "h", 'i')));
EXPECT_THAT("abcdefghij ",
Not(EqualsSumOf(::std::string("a"), 'b', 'c', "d", "e", 'f', 'g',
"h", 'i', ::std::string("j"))));
}

TEST(MatcherPnMacroTest, WorksForDifferentParameterTypes) {
EXPECT_THAT(123, EqualsSumOf(100L, 20, static_cast<char>(3)));
EXPECT_THAT("abcd", EqualsSumOf(::std::string("a"), "b", 'c', "d"));

EXPECT_THAT(124, Not(EqualsSumOf(100L, 20, static_cast<char>(3))));
EXPECT_THAT("abcde", Not(EqualsSumOf(::std::string("a"), "b", 'c', "d")));
}


MATCHER_P2(EqConcat, prefix, suffix, "") {
std::string prefix_str(prefix);
char suffix_char = static_cast<char>(suffix);
return arg == prefix_str + suffix_char;
}

TEST(MatcherPnMacroTest, SimpleTypePromotion) {
Matcher<std::string> no_promo = EqConcat(std::string("foo"), 't');
Matcher<const std::string&> promo = EqConcat("foo", static_cast<int>('t'));
EXPECT_FALSE(no_promo.Matches("fool"));
EXPECT_FALSE(promo.Matches("fool"));
EXPECT_TRUE(no_promo.Matches("foot"));
EXPECT_TRUE(promo.Matches("foot"));
}


TEST(MatcherPnMacroTest, TypesAreCorrect) {
EqualsSumOfMatcher a0 = EqualsSumOf();

EqualsSumOfMatcherP<int> a1 = EqualsSumOf(1);

EqualsSumOfMatcherP2<int, char> a2 = EqualsSumOf(1, '2');
EqualsSumOfMatcherP3<int, int, char> a3 = EqualsSumOf(1, 2, '3');
EqualsSumOfMatcherP4<int, int, int, char> a4 = EqualsSumOf(1, 2, 3, '4');
EqualsSumOfMatcherP5<int, int, int, int, char> a5 =
EqualsSumOf(1, 2, 3, 4, '5');
EqualsSumOfMatcherP6<int, int, int, int, int, char> a6 =
EqualsSumOf(1, 2, 3, 4, 5, '6');
EqualsSumOfMatcherP7<int, int, int, int, int, int, char> a7 =
EqualsSumOf(1, 2, 3, 4, 5, 6, '7');
EqualsSumOfMatcherP8<int, int, int, int, int, int, int, char> a8 =
EqualsSumOf(1, 2, 3, 4, 5, 6, 7, '8');
EqualsSumOfMatcherP9<int, int, int, int, int, int, int, int, char> a9 =
EqualsSumOf(1, 2, 3, 4, 5, 6, 7, 8, '9');
EqualsSumOfMatcherP10<int, int, int, int, int, int, int, int, int, char> a10 =
EqualsSumOf(1, 2, 3, 4, 5, 6, 7, 8, 9, '0');

(void)a0;
(void)a1;
(void)a2;
(void)a3;
(void)a4;
(void)a5;
(void)a6;
(void)a7;
(void)a8;
(void)a9;
(void)a10;
}


MATCHER_P3(TwoOf, m1, m2, m3, "") {
const int count = static_cast<int>(Value(arg, m1)) +
static_cast<int>(Value(arg, m2)) +
static_cast<int>(Value(arg, m3));
return count == 2;
}

TEST(MatcherPnMacroTest, CanUseMatcherTypedParameterInValue) {
EXPECT_THAT(42, TwoOf(Gt(0), Lt(50), Eq(10)));
EXPECT_THAT(0, Not(TwoOf(Gt(-1), Lt(1), Eq(0))));
}


TEST(ContainsTest, ListMatchesWhenElementIsInContainer) {
list<int> some_list;
some_list.push_back(3);
some_list.push_back(1);
some_list.push_back(2);
EXPECT_THAT(some_list, Contains(1));
EXPECT_THAT(some_list, Contains(Gt(2.5)));
EXPECT_THAT(some_list, Contains(Eq(2.0f)));

list<std::string> another_list;
another_list.push_back("fee");
another_list.push_back("fie");
another_list.push_back("foe");
another_list.push_back("fum");
EXPECT_THAT(another_list, Contains(std::string("fee")));
}

TEST(ContainsTest, ListDoesNotMatchWhenElementIsNotInContainer) {
list<int> some_list;
some_list.push_back(3);
some_list.push_back(1);
EXPECT_THAT(some_list, Not(Contains(4)));
}

TEST(ContainsTest, SetMatchesWhenElementIsInContainer) {
set<int> some_set;
some_set.insert(3);
some_set.insert(1);
some_set.insert(2);
EXPECT_THAT(some_set, Contains(Eq(1.0)));
EXPECT_THAT(some_set, Contains(Eq(3.0f)));
EXPECT_THAT(some_set, Contains(2));

set<std::string> another_set;
another_set.insert("fee");
another_set.insert("fie");
another_set.insert("foe");
another_set.insert("fum");
EXPECT_THAT(another_set, Contains(Eq(std::string("fum"))));
}

TEST(ContainsTest, SetDoesNotMatchWhenElementIsNotInContainer) {
set<int> some_set;
some_set.insert(3);
some_set.insert(1);
EXPECT_THAT(some_set, Not(Contains(4)));

set<std::string> c_string_set;
c_string_set.insert("hello");
EXPECT_THAT(c_string_set, Not(Contains(std::string("goodbye"))));
}

TEST(ContainsTest, ExplainsMatchResultCorrectly) {
const int a[2] = {1, 2};
Matcher<const int(&)[2]> m = Contains(2);
EXPECT_EQ("whose element #1 matches", Explain(m, a));

m = Contains(3);
EXPECT_EQ("", Explain(m, a));

m = Contains(GreaterThan(0));
EXPECT_EQ("whose element #0 matches, which is 1 more than 0", Explain(m, a));

m = Contains(GreaterThan(10));
EXPECT_EQ("", Explain(m, a));
}

TEST(ContainsTest, DescribesItselfCorrectly) {
Matcher<vector<int>> m = Contains(1);
EXPECT_EQ("contains at least one element that is equal to 1", Describe(m));

Matcher<vector<int>> m2 = Not(m);
EXPECT_EQ("doesn't contain any element that is equal to 1", Describe(m2));
}

TEST(ContainsTest, MapMatchesWhenElementIsInContainer) {
map<std::string, int> my_map;
const char* bar = "a string";
my_map[bar] = 2;
EXPECT_THAT(my_map, Contains(pair<const char* const, int>(bar, 2)));

map<std::string, int> another_map;
another_map["fee"] = 1;
another_map["fie"] = 2;
another_map["foe"] = 3;
another_map["fum"] = 4;
EXPECT_THAT(another_map,
Contains(pair<const std::string, int>(std::string("fee"), 1)));
EXPECT_THAT(another_map, Contains(pair<const std::string, int>("fie", 2)));
}

TEST(ContainsTest, MapDoesNotMatchWhenElementIsNotInContainer) {
map<int, int> some_map;
some_map[1] = 11;
some_map[2] = 22;
EXPECT_THAT(some_map, Not(Contains(pair<const int, int>(2, 23))));
}

TEST(ContainsTest, ArrayMatchesWhenElementIsInContainer) {
const char* string_array[] = {"fee", "fie", "foe", "fum"};
EXPECT_THAT(string_array, Contains(Eq(std::string("fum"))));
}

TEST(ContainsTest, ArrayDoesNotMatchWhenElementIsNotInContainer) {
int int_array[] = {1, 2, 3, 4};
EXPECT_THAT(int_array, Not(Contains(5)));
}

TEST(ContainsTest, AcceptsMatcher) {
const int a[] = {1, 2, 3};
EXPECT_THAT(a, Contains(Gt(2)));
EXPECT_THAT(a, Not(Contains(Gt(4))));
}

TEST(ContainsTest, WorksForNativeArrayAsTuple) {
const int a[] = {1, 2};
const int* const pointer = a;
EXPECT_THAT(std::make_tuple(pointer, 2), Contains(1));
EXPECT_THAT(std::make_tuple(pointer, 2), Not(Contains(Gt(3))));
}

TEST(ContainsTest, WorksForTwoDimensionalNativeArray) {
int a[][3] = {{1, 2, 3}, {4, 5, 6}};
EXPECT_THAT(a, Contains(ElementsAre(4, 5, 6)));
EXPECT_THAT(a, Contains(Contains(5)));
EXPECT_THAT(a, Not(Contains(ElementsAre(3, 4, 5))));
EXPECT_THAT(a, Contains(Not(Contains(5))));
}

TEST(AllOfArrayTest, BasicForms) {
std::vector<int> v0{};
std::vector<int> v1{1};
std::vector<int> v2{2, 3};
std::vector<int> v3{4, 4, 4};
EXPECT_THAT(0, AllOfArray(v0.begin(), v0.end()));
EXPECT_THAT(1, AllOfArray(v1.begin(), v1.end()));
EXPECT_THAT(2, Not(AllOfArray(v1.begin(), v1.end())));
EXPECT_THAT(3, Not(AllOfArray(v2.begin(), v2.end())));
EXPECT_THAT(4, AllOfArray(v3.begin(), v3.end()));
int ar[6] = {1, 2, 3, 4, 4, 4};
EXPECT_THAT(0, AllOfArray(ar, 0));
EXPECT_THAT(1, AllOfArray(ar, 1));
EXPECT_THAT(2, Not(AllOfArray(ar, 1)));
EXPECT_THAT(3, Not(AllOfArray(ar + 1, 3)));
EXPECT_THAT(4, AllOfArray(ar + 3, 3));
int ar1[1] = {1};
int ar2[2] = {2, 3};
int ar3[3] = {4, 4, 4};
EXPECT_THAT(1, AllOfArray(ar1));
EXPECT_THAT(2, Not(AllOfArray(ar1)));
EXPECT_THAT(3, Not(AllOfArray(ar2)));
EXPECT_THAT(4, AllOfArray(ar3));
EXPECT_THAT(0, AllOfArray(v0));
EXPECT_THAT(1, AllOfArray(v1));
EXPECT_THAT(2, Not(AllOfArray(v1)));
EXPECT_THAT(3, Not(AllOfArray(v2)));
EXPECT_THAT(4, AllOfArray(v3));
EXPECT_THAT(0, AllOfArray<int>({}));  
EXPECT_THAT(1, AllOfArray({1}));
EXPECT_THAT(2, Not(AllOfArray({1})));
EXPECT_THAT(3, Not(AllOfArray({2, 3})));
EXPECT_THAT(4, AllOfArray({4, 4, 4}));
}

TEST(AllOfArrayTest, Matchers) {
std::vector<Matcher<int>> matchers{Ge(1), Lt(2)};
EXPECT_THAT(0, Not(AllOfArray(matchers)));
EXPECT_THAT(1, AllOfArray(matchers));
EXPECT_THAT(2, Not(AllOfArray(matchers)));
EXPECT_THAT(0, Not(AllOfArray({Ge(0), Ge(1)})));
EXPECT_THAT(1, AllOfArray({Ge(0), Ge(1)}));
}

TEST(AnyOfArrayTest, BasicForms) {
std::vector<int> v0{};
std::vector<int> v1{1};
std::vector<int> v2{2, 3};
EXPECT_THAT(0, Not(AnyOfArray(v0.begin(), v0.end())));
EXPECT_THAT(1, AnyOfArray(v1.begin(), v1.end()));
EXPECT_THAT(2, Not(AnyOfArray(v1.begin(), v1.end())));
EXPECT_THAT(3, AnyOfArray(v2.begin(), v2.end()));
EXPECT_THAT(4, Not(AnyOfArray(v2.begin(), v2.end())));
int ar[3] = {1, 2, 3};
EXPECT_THAT(0, Not(AnyOfArray(ar, 0)));
EXPECT_THAT(1, AnyOfArray(ar, 1));
EXPECT_THAT(2, Not(AnyOfArray(ar, 1)));
EXPECT_THAT(3, AnyOfArray(ar + 1, 2));
EXPECT_THAT(4, Not(AnyOfArray(ar + 1, 2)));
int ar1[1] = {1};
int ar2[2] = {2, 3};
EXPECT_THAT(1, AnyOfArray(ar1));
EXPECT_THAT(2, Not(AnyOfArray(ar1)));
EXPECT_THAT(3, AnyOfArray(ar2));
EXPECT_THAT(4, Not(AnyOfArray(ar2)));
EXPECT_THAT(0, Not(AnyOfArray(v0)));
EXPECT_THAT(1, AnyOfArray(v1));
EXPECT_THAT(2, Not(AnyOfArray(v1)));
EXPECT_THAT(3, AnyOfArray(v2));
EXPECT_THAT(4, Not(AnyOfArray(v2)));
EXPECT_THAT(0, Not(AnyOfArray<int>({})));  
EXPECT_THAT(1, AnyOfArray({1}));
EXPECT_THAT(2, Not(AnyOfArray({1})));
EXPECT_THAT(3, AnyOfArray({2, 3}));
EXPECT_THAT(4, Not(AnyOfArray({2, 3})));
}

TEST(AnyOfArrayTest, Matchers) {
std::vector<Matcher<int>> matchers{Lt(1), Ge(2)};
EXPECT_THAT(0, AnyOfArray(matchers));
EXPECT_THAT(1, Not(AnyOfArray(matchers)));
EXPECT_THAT(2, AnyOfArray(matchers));
EXPECT_THAT(0, AnyOfArray({Lt(0), Lt(1)}));
EXPECT_THAT(1, Not(AllOfArray({Lt(0), Lt(1)})));
}

TEST(AnyOfArrayTest, ExplainsMatchResultCorrectly) {
const std::vector<int> v0{};
const std::vector<int> v1{1};
const std::vector<int> v2{2, 3};
const Matcher<int> m0 = AnyOfArray(v0);
const Matcher<int> m1 = AnyOfArray(v1);
const Matcher<int> m2 = AnyOfArray(v2);
EXPECT_EQ("", Explain(m0, 0));
EXPECT_EQ("", Explain(m1, 1));
EXPECT_EQ("", Explain(m1, 2));
EXPECT_EQ("", Explain(m2, 3));
EXPECT_EQ("", Explain(m2, 4));
EXPECT_EQ("()", Describe(m0));
EXPECT_EQ("(is equal to 1)", Describe(m1));
EXPECT_EQ("(is equal to 2) or (is equal to 3)", Describe(m2));
EXPECT_EQ("()", DescribeNegation(m0));
EXPECT_EQ("(isn't equal to 1)", DescribeNegation(m1));
EXPECT_EQ("(isn't equal to 2) and (isn't equal to 3)", DescribeNegation(m2));
const Matcher<int> g1 = AnyOfArray({GreaterThan(1)});
const Matcher<int> g2 = AnyOfArray({GreaterThan(1), GreaterThan(2)});
EXPECT_EQ("which is 1 less than 1", Explain(g1, 0));
EXPECT_EQ("which is the same as 1", Explain(g1, 1));
EXPECT_EQ("which is 1 more than 1", Explain(g1, 2));
EXPECT_EQ("which is 1 less than 1, and which is 2 less than 2",
Explain(g2, 0));
EXPECT_EQ("which is the same as 1, and which is 1 less than 2",
Explain(g2, 1));
EXPECT_EQ("which is 1 more than 1",  
Explain(g2, 2));
}

TEST(AllOfTest, HugeMatcher) {
EXPECT_THAT(0, testing::AllOf(_, _, _, _, _, _, _, _, _,
testing::AllOf(_, _, _, _, _, _, _, _, _, _)));
}

TEST(AnyOfTest, HugeMatcher) {
EXPECT_THAT(0, testing::AnyOf(_, _, _, _, _, _, _, _, _,
testing::AnyOf(_, _, _, _, _, _, _, _, _, _)));
}

namespace adl_test {


MATCHER(M, "") {
(void)arg;
return true;
}

template <typename T1, typename T2>
bool AllOf(const T1& , const T2& ) {
return true;
}

TEST(AllOfTest, DoesNotCallAllOfUnqualified) {
EXPECT_THAT(42,
testing::AllOf(M(), M(), M(), M(), M(), M(), M(), M(), M(), M()));
}

template <typename T1, typename T2>
bool AnyOf(const T1&, const T2&) {
return true;
}

TEST(AnyOfTest, DoesNotCallAnyOfUnqualified) {
EXPECT_THAT(42,
testing::AnyOf(M(), M(), M(), M(), M(), M(), M(), M(), M(), M()));
}

}  

TEST(AllOfTest, WorksOnMoveOnlyType) {
std::unique_ptr<int> p(new int(3));
EXPECT_THAT(p, AllOf(Pointee(Eq(3)), Pointee(Gt(0)), Pointee(Lt(5))));
EXPECT_THAT(p, Not(AllOf(Pointee(Eq(3)), Pointee(Gt(0)), Pointee(Lt(3)))));
}

TEST(AnyOfTest, WorksOnMoveOnlyType) {
std::unique_ptr<int> p(new int(3));
EXPECT_THAT(p, AnyOf(Pointee(Eq(5)), Pointee(Lt(0)), Pointee(Lt(5))));
EXPECT_THAT(p, Not(AnyOf(Pointee(Eq(5)), Pointee(Lt(0)), Pointee(Gt(5)))));
}

MATCHER(IsNotNull, "") { return arg != nullptr; }

TEST(MatcherMacroTest, WorksOnMoveOnlyType) {
std::unique_ptr<int> p(new int(3));
EXPECT_THAT(p, IsNotNull());
EXPECT_THAT(std::unique_ptr<int>(), Not(IsNotNull()));
}

MATCHER_P(UniquePointee, pointee, "") { return *arg == pointee; }

TEST(MatcherPMacroTest, WorksOnMoveOnlyType) {
std::unique_ptr<int> p(new int(3));
EXPECT_THAT(p, UniquePointee(3));
EXPECT_THAT(p, Not(UniquePointee(2)));
}

}  
}  
}  

#ifdef _MSC_VER
# pragma warning(pop)
#endif
