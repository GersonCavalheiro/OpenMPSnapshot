

#ifndef GMOCK_TEST_GMOCK_LINK_TEST_H_
#define GMOCK_TEST_GMOCK_LINK_TEST_H_

#include "gmock/gmock.h"

#if !GTEST_OS_WINDOWS_MOBILE
# include <errno.h>
#endif

#include "gmock/internal/gmock-port.h"
#include "gtest/gtest.h"
#include <iostream>
#include <vector>

using testing::_;
using testing::A;
using testing::AllOf;
using testing::AnyOf;
using testing::Assign;
using testing::ContainerEq;
using testing::DoAll;
using testing::DoDefault;
using testing::DoubleEq;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::EndsWith;
using testing::Eq;
using testing::Field;
using testing::FloatEq;
using testing::Ge;
using testing::Gt;
using testing::HasSubstr;
using testing::IgnoreResult;
using testing::Invoke;
using testing::InvokeArgument;
using testing::InvokeWithoutArgs;
using testing::IsNull;
using testing::Le;
using testing::Lt;
using testing::Matcher;
using testing::MatcherCast;
using testing::NanSensitiveDoubleEq;
using testing::NanSensitiveFloatEq;
using testing::Ne;
using testing::Not;
using testing::NotNull;
using testing::Pointee;
using testing::Property;
using testing::Ref;
using testing::ResultOf;
using testing::Return;
using testing::ReturnNull;
using testing::ReturnRef;
using testing::SetArgPointee;
using testing::SetArrayArgument;
using testing::StartsWith;
using testing::StrCaseEq;
using testing::StrCaseNe;
using testing::StrEq;
using testing::StrNe;
using testing::Truly;
using testing::TypedEq;
using testing::WithArg;
using testing::WithArgs;
using testing::WithoutArgs;

#if !GTEST_OS_WINDOWS_MOBILE
using testing::SetErrnoAndReturn;
#endif

#if GTEST_HAS_EXCEPTIONS
using testing::Throw;
#endif

using testing::ContainsRegex;
using testing::MatchesRegex;

class Interface {
public:
virtual ~Interface() {}
virtual void VoidFromString(char* str) = 0;
virtual char* StringFromString(char* str) = 0;
virtual int IntFromString(char* str) = 0;
virtual int& IntRefFromString(char* str) = 0;
virtual void VoidFromFunc(void(*func)(char* str)) = 0;
virtual void VoidFromIntRef(int& n) = 0;  
virtual void VoidFromFloat(float n) = 0;
virtual void VoidFromDouble(double n) = 0;
virtual void VoidFromVector(const std::vector<int>& v) = 0;
};

class Mock: public Interface {
public:
Mock() {}

MOCK_METHOD1(VoidFromString, void(char* str));
MOCK_METHOD1(StringFromString, char*(char* str));
MOCK_METHOD1(IntFromString, int(char* str));
MOCK_METHOD1(IntRefFromString, int&(char* str));
MOCK_METHOD1(VoidFromFunc, void(void(*func)(char* str)));
MOCK_METHOD1(VoidFromIntRef, void(int& n));  
MOCK_METHOD1(VoidFromFloat, void(float n));
MOCK_METHOD1(VoidFromDouble, void(double n));
MOCK_METHOD1(VoidFromVector, void(const std::vector<int>& v));

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(Mock);
};

class InvokeHelper {
public:
static void StaticVoidFromVoid() {}
void VoidFromVoid() {}
static void StaticVoidFromString(char* ) {}
void VoidFromString(char* ) {}
static int StaticIntFromString(char* ) { return 1; }
static bool StaticBoolFromString(const char* ) { return true; }
};

class FieldHelper {
public:
explicit FieldHelper(int a_field) : field_(a_field) {}
int field() const { return field_; }
int field_;  
};

TEST(LinkTest, TestReturnVoid) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(Return());
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestReturn) {
Mock mock;
char ch = 'x';

EXPECT_CALL(mock, StringFromString(_)).WillOnce(Return(&ch));
mock.StringFromString(NULL);
}

TEST(LinkTest, TestReturnNull) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(Return());
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestReturnRef) {
Mock mock;
int n = 42;

EXPECT_CALL(mock, IntRefFromString(_)).WillOnce(ReturnRef(n));
mock.IntRefFromString(NULL);
}

TEST(LinkTest, TestAssign) {
Mock mock;
char ch = 'x';

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(Assign(&ch, 'y'));
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestSetArgPointee) {
Mock mock;
char ch = 'x';

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(SetArgPointee<0>('y'));
mock.VoidFromString(&ch);
}

TEST(LinkTest, TestSetArrayArgument) {
Mock mock;
char ch = 'x';
char ch2 = 'y';

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(SetArrayArgument<0>(&ch2,
&ch2 + 1));
mock.VoidFromString(&ch);
}

#if !GTEST_OS_WINDOWS_MOBILE

TEST(LinkTest, TestSetErrnoAndReturn) {
Mock mock;

int saved_errno = errno;
EXPECT_CALL(mock, IntFromString(_)).WillOnce(SetErrnoAndReturn(1, -1));
mock.IntFromString(NULL);
errno = saved_errno;
}

#endif  

TEST(LinkTest, TestInvoke) {
Mock mock;
InvokeHelper test_invoke_helper;

EXPECT_CALL(mock, VoidFromString(_))
.WillOnce(Invoke(&InvokeHelper::StaticVoidFromString))
.WillOnce(Invoke(&test_invoke_helper, &InvokeHelper::VoidFromString));
mock.VoidFromString(NULL);
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestInvokeWithoutArgs) {
Mock mock;
InvokeHelper test_invoke_helper;

EXPECT_CALL(mock, VoidFromString(_))
.WillOnce(InvokeWithoutArgs(&InvokeHelper::StaticVoidFromVoid))
.WillOnce(InvokeWithoutArgs(&test_invoke_helper,
&InvokeHelper::VoidFromVoid));
mock.VoidFromString(NULL);
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestInvokeArgument) {
Mock mock;
char ch = 'x';

EXPECT_CALL(mock, VoidFromFunc(_)).WillOnce(InvokeArgument<0>(&ch));
mock.VoidFromFunc(InvokeHelper::StaticVoidFromString);
}

TEST(LinkTest, TestWithArg) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_))
.WillOnce(WithArg<0>(Invoke(&InvokeHelper::StaticVoidFromString)));
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestWithArgs) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_))
.WillOnce(WithArgs<0>(Invoke(&InvokeHelper::StaticVoidFromString)));
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestWithoutArgs) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(WithoutArgs(Return()));
mock.VoidFromString(NULL);
}

TEST(LinkTest, TestDoAll) {
Mock mock;
char ch = 'x';

EXPECT_CALL(mock, VoidFromString(_))
.WillOnce(DoAll(SetArgPointee<0>('y'), Return()));
mock.VoidFromString(&ch);
}

TEST(LinkTest, TestDoDefault) {
Mock mock;
char ch = 'x';

ON_CALL(mock, VoidFromString(_)).WillByDefault(Return());
EXPECT_CALL(mock, VoidFromString(_)).WillOnce(DoDefault());
mock.VoidFromString(&ch);
}

TEST(LinkTest, TestIgnoreResult) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(IgnoreResult(Return(42)));
mock.VoidFromString(NULL);
}

#if GTEST_HAS_EXCEPTIONS
TEST(LinkTest, TestThrow) {
Mock mock;

EXPECT_CALL(mock, VoidFromString(_)).WillOnce(Throw(42));
EXPECT_THROW(mock.VoidFromString(NULL), int);
}
#endif  

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif

namespace {
ACTION(Return1) { return 1; }
}

TEST(LinkTest, TestActionMacro) {
Mock mock;

EXPECT_CALL(mock, IntFromString(_)).WillOnce(Return1());
mock.IntFromString(NULL);
}

namespace {
ACTION_P(ReturnArgument, ret_value) { return ret_value; }
}

TEST(LinkTest, TestActionPMacro) {
Mock mock;

EXPECT_CALL(mock, IntFromString(_)).WillOnce(ReturnArgument(42));
mock.IntFromString(NULL);
}

namespace {
ACTION_P2(ReturnEqualsEitherOf, first, second) {
return arg0 == first || arg0 == second;
}
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

TEST(LinkTest, TestActionP2Macro) {
Mock mock;
char ch = 'x';

EXPECT_CALL(mock, IntFromString(_))
.WillOnce(ReturnEqualsEitherOf("one", "two"));
mock.IntFromString(&ch);
}

TEST(LinkTest, TestMatcherAnything) {
Mock mock;

ON_CALL(mock, VoidFromString(_)).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherA) {
Mock mock;

ON_CALL(mock, VoidFromString(A<char*>())).WillByDefault(Return());
}

TEST(LinkTest, TestMatchersEq) {
Mock mock;
const char* p = "x";

ON_CALL(mock, VoidFromString(Eq(p))).WillByDefault(Return());
ON_CALL(mock, VoidFromString(const_cast<char*>("y")))
.WillByDefault(Return());
}

TEST(LinkTest, TestMatchersRelations) {
Mock mock;

ON_CALL(mock, VoidFromFloat(Lt(1.0f))).WillByDefault(Return());
ON_CALL(mock, VoidFromFloat(Gt(1.0f))).WillByDefault(Return());
ON_CALL(mock, VoidFromFloat(Le(1.0f))).WillByDefault(Return());
ON_CALL(mock, VoidFromFloat(Ge(1.0f))).WillByDefault(Return());
ON_CALL(mock, VoidFromFloat(Ne(1.0f))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherNotNull) {
Mock mock;

ON_CALL(mock, VoidFromString(NotNull())).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherIsNull) {
Mock mock;

ON_CALL(mock, VoidFromString(IsNull())).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherRef) {
Mock mock;
int a = 0;

ON_CALL(mock, VoidFromIntRef(Ref(a))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherTypedEq) {
Mock mock;
long a = 0;

ON_CALL(mock, VoidFromIntRef(TypedEq<int&>(a))).WillByDefault(Return());
}

TEST(LinkTest, TestMatchersFloatingPoint) {
Mock mock;
float a = 0;

ON_CALL(mock, VoidFromFloat(FloatEq(a))).WillByDefault(Return());
ON_CALL(mock, VoidFromDouble(DoubleEq(a))).WillByDefault(Return());
ON_CALL(mock, VoidFromFloat(NanSensitiveFloatEq(a))).WillByDefault(Return());
ON_CALL(mock, VoidFromDouble(NanSensitiveDoubleEq(a)))
.WillByDefault(Return());
}

TEST(LinkTest, TestMatcherContainsRegex) {
Mock mock;

ON_CALL(mock, VoidFromString(ContainsRegex(".*"))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherMatchesRegex) {
Mock mock;

ON_CALL(mock, VoidFromString(MatchesRegex(".*"))).WillByDefault(Return());
}

TEST(LinkTest, TestMatchersSubstrings) {
Mock mock;

ON_CALL(mock, VoidFromString(StartsWith("a"))).WillByDefault(Return());
ON_CALL(mock, VoidFromString(EndsWith("c"))).WillByDefault(Return());
ON_CALL(mock, VoidFromString(HasSubstr("b"))).WillByDefault(Return());
}

TEST(LinkTest, TestMatchersStringEquality) {
Mock mock;
ON_CALL(mock, VoidFromString(StrEq("a"))).WillByDefault(Return());
ON_CALL(mock, VoidFromString(StrNe("a"))).WillByDefault(Return());
ON_CALL(mock, VoidFromString(StrCaseEq("a"))).WillByDefault(Return());
ON_CALL(mock, VoidFromString(StrCaseNe("a"))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherElementsAre) {
Mock mock;

ON_CALL(mock, VoidFromVector(ElementsAre('a', _))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherElementsAreArray) {
Mock mock;
char arr[] = { 'a', 'b' };

ON_CALL(mock, VoidFromVector(ElementsAreArray(arr))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherContainerEq) {
Mock mock;
std::vector<int> v;

ON_CALL(mock, VoidFromVector(ContainerEq(v))).WillByDefault(Return());
}

TEST(LinkTest, TestMatcherField) {
FieldHelper helper(0);

Matcher<const FieldHelper&> m = Field(&FieldHelper::field_, Eq(0));
EXPECT_TRUE(m.Matches(helper));

Matcher<const FieldHelper*> m2 = Field(&FieldHelper::field_, Eq(0));
EXPECT_TRUE(m2.Matches(&helper));
}

TEST(LinkTest, TestMatcherProperty) {
FieldHelper helper(0);

Matcher<const FieldHelper&> m = Property(&FieldHelper::field, Eq(0));
EXPECT_TRUE(m.Matches(helper));

Matcher<const FieldHelper*> m2 = Property(&FieldHelper::field, Eq(0));
EXPECT_TRUE(m2.Matches(&helper));
}

TEST(LinkTest, TestMatcherResultOf) {
Matcher<char*> m = ResultOf(&InvokeHelper::StaticIntFromString, Eq(1));
EXPECT_TRUE(m.Matches(NULL));
}

TEST(LinkTest, TestMatcherPointee) {
int n = 1;

Matcher<int*> m = Pointee(Eq(1));
EXPECT_TRUE(m.Matches(&n));
}

TEST(LinkTest, TestMatcherTruly) {
Matcher<const char*> m = Truly(&InvokeHelper::StaticBoolFromString);
EXPECT_TRUE(m.Matches(NULL));
}

TEST(LinkTest, TestMatcherAllOf) {
Matcher<int> m = AllOf(_, Eq(1));
EXPECT_TRUE(m.Matches(1));
}

TEST(LinkTest, TestMatcherAnyOf) {
Matcher<int> m = AnyOf(_, Eq(1));
EXPECT_TRUE(m.Matches(1));
}

TEST(LinkTest, TestMatcherNot) {
Matcher<int> m = Not(_);
EXPECT_FALSE(m.Matches(1));
}

TEST(LinkTest, TestMatcherCast) {
Matcher<const char*> m = MatcherCast<const char*>(_);
EXPECT_TRUE(m.Matches(NULL));
}

#endif  
