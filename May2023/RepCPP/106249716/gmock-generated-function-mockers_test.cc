

#include "gmock/gmock-generated-function-mockers.h"

#if GTEST_OS_WINDOWS
# include <objbase.h>
#endif  

#include <map>
#include <string>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#if !GTEST_OS_WINDOWS || (_MSC_VER >= 1500)
# define GMOCK_ALLOWS_CONST_PARAM_FUNCTIONS
#endif  

namespace testing {
namespace gmock_generated_function_mockers_test {

using testing::internal::string;
using testing::_;
using testing::A;
using testing::An;
using testing::AnyNumber;
using testing::Const;
using testing::DoDefault;
using testing::Eq;
using testing::Lt;
using testing::MockFunction;
using testing::Ref;
using testing::Return;
using testing::ReturnRef;
using testing::TypedEq;

class FooInterface {
public:
virtual ~FooInterface() {}

virtual void VoidReturning(int x) = 0;

virtual int Nullary() = 0;
virtual bool Unary(int x) = 0;
virtual long Binary(short x, int y) = 0;  
virtual int Decimal(bool b, char c, short d, int e, long f,  
float g, double h, unsigned i, char* j, const string& k)
= 0;

virtual bool TakesNonConstReference(int& n) = 0;  
virtual string TakesConstReference(const int& n) = 0;
#ifdef GMOCK_ALLOWS_CONST_PARAM_FUNCTIONS
virtual bool TakesConst(const int x) = 0;
#endif  

virtual int OverloadedOnArgumentNumber() = 0;
virtual int OverloadedOnArgumentNumber(int n) = 0;

virtual int OverloadedOnArgumentType(int n) = 0;
virtual char OverloadedOnArgumentType(char c) = 0;

virtual int OverloadedOnConstness() = 0;
virtual char OverloadedOnConstness() const = 0;

virtual int TypeWithHole(int (*func)()) = 0;
virtual int TypeWithComma(const std::map<int, string>& a_map) = 0;

#if GTEST_OS_WINDOWS
STDMETHOD_(int, CTNullary)() = 0;
STDMETHOD_(bool, CTUnary)(int x) = 0;
STDMETHOD_(int, CTDecimal)(bool b, char c, short d, int e, long f,  
float g, double h, unsigned i, char* j, const string& k) = 0;
STDMETHOD_(char, CTConst)(int x) const = 0;
#endif  
};

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable : 4373)
#endif
class MockFoo : public FooInterface {
public:
MockFoo() {}

MOCK_METHOD1(VoidReturning, void(int n));  

MOCK_METHOD0(Nullary, int());  

MOCK_METHOD1(Unary, bool(int));  
MOCK_METHOD2(Binary, long(short, int));  
MOCK_METHOD10(Decimal, int(bool, char, short, int, long, float,  
double, unsigned, char*, const string& str));

MOCK_METHOD1(TakesNonConstReference, bool(int&));  
MOCK_METHOD1(TakesConstReference, string(const int&));

#ifdef GMOCK_ALLOWS_CONST_PARAM_FUNCTIONS
MOCK_METHOD1(TakesConst, bool(const int));  
#endif

MOCK_METHOD0(ReturnTypeWithComma, std::map<int, string>());
MOCK_CONST_METHOD1(ReturnTypeWithComma,
std::map<int, string>(int));  

MOCK_METHOD0(OverloadedOnArgumentNumber, int());  
MOCK_METHOD1(OverloadedOnArgumentNumber, int(int));  

MOCK_METHOD1(OverloadedOnArgumentType, int(int));  
MOCK_METHOD1(OverloadedOnArgumentType, char(char));  

MOCK_METHOD0(OverloadedOnConstness, int());  
MOCK_CONST_METHOD0(OverloadedOnConstness, char());  

MOCK_METHOD1(TypeWithHole, int(int (*)()));  
MOCK_METHOD1(TypeWithComma, int(const std::map<int, string>&));  

#if GTEST_OS_WINDOWS
MOCK_METHOD0_WITH_CALLTYPE(STDMETHODCALLTYPE, CTNullary, int());
MOCK_METHOD1_WITH_CALLTYPE(STDMETHODCALLTYPE, CTUnary, bool(int));
MOCK_METHOD10_WITH_CALLTYPE(STDMETHODCALLTYPE, CTDecimal, int(bool b, char c,
short d, int e, long f, float g, double h, unsigned i, char* j,
const string& k));
MOCK_CONST_METHOD1_WITH_CALLTYPE(STDMETHODCALLTYPE, CTConst, char(int));

MOCK_METHOD0_WITH_CALLTYPE(STDMETHODCALLTYPE, CTReturnTypeWithComma,
std::map<int, string>());
#endif  

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockFoo);
};
#ifdef _MSC_VER
# pragma warning(pop)
#endif

class FunctionMockerTest : public testing::Test {
protected:
FunctionMockerTest() : foo_(&mock_foo_) {}

FooInterface* const foo_;
MockFoo mock_foo_;
};

TEST_F(FunctionMockerTest, MocksVoidFunction) {
EXPECT_CALL(mock_foo_, VoidReturning(Lt(100)));
foo_->VoidReturning(0);
}

TEST_F(FunctionMockerTest, MocksNullaryFunction) {
EXPECT_CALL(mock_foo_, Nullary())
.WillOnce(DoDefault())
.WillOnce(Return(1));

EXPECT_EQ(0, foo_->Nullary());
EXPECT_EQ(1, foo_->Nullary());
}

TEST_F(FunctionMockerTest, MocksUnaryFunction) {
EXPECT_CALL(mock_foo_, Unary(Eq(2)))
.Times(2)
.WillOnce(Return(true));

EXPECT_TRUE(foo_->Unary(2));
EXPECT_FALSE(foo_->Unary(2));
}

TEST_F(FunctionMockerTest, MocksBinaryFunction) {
EXPECT_CALL(mock_foo_, Binary(2, _))
.WillOnce(Return(3));

EXPECT_EQ(3, foo_->Binary(2, 1));
}

TEST_F(FunctionMockerTest, MocksDecimalFunction) {
EXPECT_CALL(mock_foo_, Decimal(true, 'a', 0, 0, 1L, A<float>(),
Lt(100), 5U, NULL, "hi"))
.WillOnce(Return(5));

EXPECT_EQ(5, foo_->Decimal(true, 'a', 0, 0, 1, 0, 0, 5, NULL, "hi"));
}

TEST_F(FunctionMockerTest, MocksFunctionWithNonConstReferenceArgument) {
int a = 0;
EXPECT_CALL(mock_foo_, TakesNonConstReference(Ref(a)))
.WillOnce(Return(true));

EXPECT_TRUE(foo_->TakesNonConstReference(a));
}

TEST_F(FunctionMockerTest, MocksFunctionWithConstReferenceArgument) {
int a = 0;
EXPECT_CALL(mock_foo_, TakesConstReference(Ref(a)))
.WillOnce(Return("Hello"));

EXPECT_EQ("Hello", foo_->TakesConstReference(a));
}

#ifdef GMOCK_ALLOWS_CONST_PARAM_FUNCTIONS
TEST_F(FunctionMockerTest, MocksFunctionWithConstArgument) {
EXPECT_CALL(mock_foo_, TakesConst(Lt(10)))
.WillOnce(DoDefault());

EXPECT_FALSE(foo_->TakesConst(5));
}
#endif  

TEST_F(FunctionMockerTest, MocksFunctionsOverloadedOnArgumentNumber) {
EXPECT_CALL(mock_foo_, OverloadedOnArgumentNumber())
.WillOnce(Return(1));
EXPECT_CALL(mock_foo_, OverloadedOnArgumentNumber(_))
.WillOnce(Return(2));

EXPECT_EQ(2, foo_->OverloadedOnArgumentNumber(1));
EXPECT_EQ(1, foo_->OverloadedOnArgumentNumber());
}

TEST_F(FunctionMockerTest, MocksFunctionsOverloadedOnArgumentType) {
EXPECT_CALL(mock_foo_, OverloadedOnArgumentType(An<int>()))
.WillOnce(Return(1));
EXPECT_CALL(mock_foo_, OverloadedOnArgumentType(TypedEq<char>('a')))
.WillOnce(Return('b'));

EXPECT_EQ(1, foo_->OverloadedOnArgumentType(0));
EXPECT_EQ('b', foo_->OverloadedOnArgumentType('a'));
}

TEST_F(FunctionMockerTest, MocksFunctionsOverloadedOnConstnessOfThis) {
EXPECT_CALL(mock_foo_, OverloadedOnConstness());
EXPECT_CALL(Const(mock_foo_), OverloadedOnConstness())
.WillOnce(Return('a'));

EXPECT_EQ(0, foo_->OverloadedOnConstness());
EXPECT_EQ('a', Const(*foo_).OverloadedOnConstness());
}

TEST_F(FunctionMockerTest, MocksReturnTypeWithComma) {
const std::map<int, string> a_map;
EXPECT_CALL(mock_foo_, ReturnTypeWithComma())
.WillOnce(Return(a_map));
EXPECT_CALL(mock_foo_, ReturnTypeWithComma(42))
.WillOnce(Return(a_map));

EXPECT_EQ(a_map, mock_foo_.ReturnTypeWithComma());
EXPECT_EQ(a_map, mock_foo_.ReturnTypeWithComma(42));
}

#if GTEST_OS_WINDOWS
TEST_F(FunctionMockerTest, MocksNullaryFunctionWithCallType) {
EXPECT_CALL(mock_foo_, CTNullary())
.WillOnce(Return(-1))
.WillOnce(Return(0));

EXPECT_EQ(-1, foo_->CTNullary());
EXPECT_EQ(0, foo_->CTNullary());
}

TEST_F(FunctionMockerTest, MocksUnaryFunctionWithCallType) {
EXPECT_CALL(mock_foo_, CTUnary(Eq(2)))
.Times(2)
.WillOnce(Return(true))
.WillOnce(Return(false));

EXPECT_TRUE(foo_->CTUnary(2));
EXPECT_FALSE(foo_->CTUnary(2));
}

TEST_F(FunctionMockerTest, MocksDecimalFunctionWithCallType) {
EXPECT_CALL(mock_foo_, CTDecimal(true, 'a', 0, 0, 1L, A<float>(),
Lt(100), 5U, NULL, "hi"))
.WillOnce(Return(10));

EXPECT_EQ(10, foo_->CTDecimal(true, 'a', 0, 0, 1, 0, 0, 5, NULL, "hi"));
}

TEST_F(FunctionMockerTest, MocksFunctionsConstFunctionWithCallType) {
EXPECT_CALL(Const(mock_foo_), CTConst(_))
.WillOnce(Return('a'));

EXPECT_EQ('a', Const(*foo_).CTConst(0));
}

TEST_F(FunctionMockerTest, MocksReturnTypeWithCommaAndCallType) {
const std::map<int, string> a_map;
EXPECT_CALL(mock_foo_, CTReturnTypeWithComma())
.WillOnce(Return(a_map));

EXPECT_EQ(a_map, mock_foo_.CTReturnTypeWithComma());
}

#endif  

class MockB {
public:
MockB() {}

MOCK_METHOD0(DoB, void());

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockB);
};

TEST(ExpectCallTest, UnmentionedFunctionCanBeCalledAnyNumberOfTimes) {
{
MockB b;
}

{
MockB b;
b.DoB();
}

{
MockB b;
b.DoB();
b.DoB();
}
}


template <typename T>
class StackInterface {
public:
virtual ~StackInterface() {}

virtual void Push(const T& value) = 0;
virtual void Pop() = 0;
virtual int GetSize() const = 0;
virtual const T& GetTop() const = 0;
};

template <typename T>
class MockStack : public StackInterface<T> {
public:
MockStack() {}

MOCK_METHOD1_T(Push, void(const T& elem));
MOCK_METHOD0_T(Pop, void());
MOCK_CONST_METHOD0_T(GetSize, int());  
MOCK_CONST_METHOD0_T(GetTop, const T&());

MOCK_METHOD0_T(ReturnTypeWithComma, std::map<int, int>());
MOCK_CONST_METHOD1_T(ReturnTypeWithComma, std::map<int, int>(int));  

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockStack);
};

TEST(TemplateMockTest, Works) {
MockStack<int> mock;

EXPECT_CALL(mock, GetSize())
.WillOnce(Return(0))
.WillOnce(Return(1))
.WillOnce(Return(0));
EXPECT_CALL(mock, Push(_));
int n = 5;
EXPECT_CALL(mock, GetTop())
.WillOnce(ReturnRef(n));
EXPECT_CALL(mock, Pop())
.Times(AnyNumber());

EXPECT_EQ(0, mock.GetSize());
mock.Push(5);
EXPECT_EQ(1, mock.GetSize());
EXPECT_EQ(5, mock.GetTop());
mock.Pop();
EXPECT_EQ(0, mock.GetSize());
}

TEST(TemplateMockTest, MethodWithCommaInReturnTypeWorks) {
MockStack<int> mock;

const std::map<int, int> a_map;
EXPECT_CALL(mock, ReturnTypeWithComma())
.WillOnce(Return(a_map));
EXPECT_CALL(mock, ReturnTypeWithComma(1))
.WillOnce(Return(a_map));

EXPECT_EQ(a_map, mock.ReturnTypeWithComma());
EXPECT_EQ(a_map, mock.ReturnTypeWithComma(1));
}

#if GTEST_OS_WINDOWS

template <typename T>
class StackInterfaceWithCallType {
public:
virtual ~StackInterfaceWithCallType() {}

STDMETHOD_(void, Push)(const T& value) = 0;
STDMETHOD_(void, Pop)() = 0;
STDMETHOD_(int, GetSize)() const = 0;
STDMETHOD_(const T&, GetTop)() const = 0;
};

template <typename T>
class MockStackWithCallType : public StackInterfaceWithCallType<T> {
public:
MockStackWithCallType() {}

MOCK_METHOD1_T_WITH_CALLTYPE(STDMETHODCALLTYPE, Push, void(const T& elem));
MOCK_METHOD0_T_WITH_CALLTYPE(STDMETHODCALLTYPE, Pop, void());
MOCK_CONST_METHOD0_T_WITH_CALLTYPE(STDMETHODCALLTYPE, GetSize, int());
MOCK_CONST_METHOD0_T_WITH_CALLTYPE(STDMETHODCALLTYPE, GetTop, const T&());

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockStackWithCallType);
};

TEST(TemplateMockTestWithCallType, Works) {
MockStackWithCallType<int> mock;

EXPECT_CALL(mock, GetSize())
.WillOnce(Return(0))
.WillOnce(Return(1))
.WillOnce(Return(0));
EXPECT_CALL(mock, Push(_));
int n = 5;
EXPECT_CALL(mock, GetTop())
.WillOnce(ReturnRef(n));
EXPECT_CALL(mock, Pop())
.Times(AnyNumber());

EXPECT_EQ(0, mock.GetSize());
mock.Push(5);
EXPECT_EQ(1, mock.GetSize());
EXPECT_EQ(5, mock.GetTop());
mock.Pop();
EXPECT_EQ(0, mock.GetSize());
}
#endif  

#define MY_MOCK_METHODS1_ \
MOCK_METHOD0(Overloaded, void()); \
MOCK_CONST_METHOD1(Overloaded, int(int n)); \
MOCK_METHOD2(Overloaded, bool(bool f, int n))

class MockOverloadedOnArgNumber {
public:
MockOverloadedOnArgNumber() {}

MY_MOCK_METHODS1_;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockOverloadedOnArgNumber);
};

TEST(OverloadedMockMethodTest, CanOverloadOnArgNumberInMacroBody) {
MockOverloadedOnArgNumber mock;
EXPECT_CALL(mock, Overloaded());
EXPECT_CALL(mock, Overloaded(1)).WillOnce(Return(2));
EXPECT_CALL(mock, Overloaded(true, 1)).WillOnce(Return(true));

mock.Overloaded();
EXPECT_EQ(2, mock.Overloaded(1));
EXPECT_TRUE(mock.Overloaded(true, 1));
}

#define MY_MOCK_METHODS2_ \
MOCK_CONST_METHOD1(Overloaded, int(int n)); \
MOCK_METHOD1(Overloaded, int(int n));

class MockOverloadedOnConstness {
public:
MockOverloadedOnConstness() {}

MY_MOCK_METHODS2_;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockOverloadedOnConstness);
};

TEST(OverloadedMockMethodTest, CanOverloadOnConstnessInMacroBody) {
MockOverloadedOnConstness mock;
const MockOverloadedOnConstness* const_mock = &mock;
EXPECT_CALL(mock, Overloaded(1)).WillOnce(Return(2));
EXPECT_CALL(*const_mock, Overloaded(1)).WillOnce(Return(3));

EXPECT_EQ(2, mock.Overloaded(1));
EXPECT_EQ(3, const_mock->Overloaded(1));
}

TEST(MockFunctionTest, WorksForVoidNullary) {
MockFunction<void()> foo;
EXPECT_CALL(foo, Call());
foo.Call();
}

TEST(MockFunctionTest, WorksForNonVoidNullary) {
MockFunction<int()> foo;
EXPECT_CALL(foo, Call())
.WillOnce(Return(1))
.WillOnce(Return(2));
EXPECT_EQ(1, foo.Call());
EXPECT_EQ(2, foo.Call());
}

TEST(MockFunctionTest, WorksForVoidUnary) {
MockFunction<void(int)> foo;
EXPECT_CALL(foo, Call(1));
foo.Call(1);
}

TEST(MockFunctionTest, WorksForNonVoidBinary) {
MockFunction<int(bool, int)> foo;
EXPECT_CALL(foo, Call(false, 42))
.WillOnce(Return(1))
.WillOnce(Return(2));
EXPECT_CALL(foo, Call(true, Ge(100)))
.WillOnce(Return(3));
EXPECT_EQ(1, foo.Call(false, 42));
EXPECT_EQ(2, foo.Call(false, 42));
EXPECT_EQ(3, foo.Call(true, 120));
}

TEST(MockFunctionTest, WorksFor10Arguments) {
MockFunction<int(bool a0, char a1, int a2, int a3, int a4,
int a5, int a6, char a7, int a8, bool a9)> foo;
EXPECT_CALL(foo, Call(_, 'a', _, _, _, _, _, _, _, _))
.WillOnce(Return(1))
.WillOnce(Return(2));
EXPECT_EQ(1, foo.Call(false, 'a', 0, 0, 0, 0, 0, 'b', 0, true));
EXPECT_EQ(2, foo.Call(true, 'a', 0, 0, 0, 0, 0, 'b', 1, false));
}

#if GTEST_HAS_STD_FUNCTION_
TEST(MockFunctionTest, AsStdFunction) {
MockFunction<int(int)> foo;
auto call = [](const std::function<int(int)> &f, int i) {
return f(i);
};
EXPECT_CALL(foo, Call(1)).WillOnce(Return(-1));
EXPECT_CALL(foo, Call(2)).WillOnce(Return(-2));
EXPECT_EQ(-1, call(foo.AsStdFunction(), 1));
EXPECT_EQ(-2, call(foo.AsStdFunction(), 2));
}

TEST(MockFunctionTest, AsStdFunctionReturnsReference) {
MockFunction<int&()> foo;
int value = 1;
EXPECT_CALL(foo, Call()).WillOnce(ReturnRef(value));
int& ref = foo.AsStdFunction()();
EXPECT_EQ(1, ref);
value = 2;
EXPECT_EQ(2, ref);
}
#endif  

}  
}  
