

#include "gmock/gmock-function-mocker.h"

#if GTEST_OS_WINDOWS
# include <objbase.h>
#endif  

#include <functional>
#include <map>
#include <string>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace testing {
namespace gmock_function_mocker_test {

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

template<typename T>
class TemplatedCopyable {
public:
TemplatedCopyable() {}

template <typename U>
TemplatedCopyable(const U& other) {}  
};

class FooInterface {
public:
virtual ~FooInterface() {}

virtual void VoidReturning(int x) = 0;

virtual int Nullary() = 0;
virtual bool Unary(int x) = 0;
virtual long Binary(short x, int y) = 0;  
virtual int Decimal(bool b, char c, short d, int e, long f,  
float g, double h, unsigned i, char* j,
const std::string& k) = 0;

virtual bool TakesNonConstReference(int& n) = 0;  
virtual std::string TakesConstReference(const int& n) = 0;
virtual bool TakesConst(const int x) = 0;

virtual int OverloadedOnArgumentNumber() = 0;
virtual int OverloadedOnArgumentNumber(int n) = 0;

virtual int OverloadedOnArgumentType(int n) = 0;
virtual char OverloadedOnArgumentType(char c) = 0;

virtual int OverloadedOnConstness() = 0;
virtual char OverloadedOnConstness() const = 0;

virtual int TypeWithHole(int (*func)()) = 0;
virtual int TypeWithComma(const std::map<int, std::string>& a_map) = 0;
virtual int TypeWithTemplatedCopyCtor(const TemplatedCopyable<int>&) = 0;

virtual int (*ReturnsFunctionPointer1(int))(bool) = 0;
using fn_ptr = int (*)(bool);
virtual fn_ptr ReturnsFunctionPointer2(int) = 0;

#if GTEST_OS_WINDOWS
STDMETHOD_(int, CTNullary)() = 0;
STDMETHOD_(bool, CTUnary)(int x) = 0;
STDMETHOD_(int, CTDecimal)
(bool b, char c, short d, int e, long f,  
float g, double h, unsigned i, char* j, const std::string& k) = 0;
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

MOCK_METHOD(void, VoidReturning, (int n));  

MOCK_METHOD(int, Nullary, ());  

MOCK_METHOD(bool, Unary, (int));          
MOCK_METHOD(long, Binary, (short, int));  
MOCK_METHOD(int, Decimal,
(bool, char, short, int, long, float,  
double, unsigned, char*, const std::string& str),
(override));

MOCK_METHOD(bool, TakesNonConstReference, (int&));  
MOCK_METHOD(std::string, TakesConstReference, (const int&));
MOCK_METHOD(bool, TakesConst, (const int));  

MOCK_METHOD((std::map<int, std::string>), ReturnTypeWithComma, (), ());
MOCK_METHOD((std::map<int, std::string>), ReturnTypeWithComma, (int),
(const));  

MOCK_METHOD(int, OverloadedOnArgumentNumber, ());     
MOCK_METHOD(int, OverloadedOnArgumentNumber, (int));  

MOCK_METHOD(int, OverloadedOnArgumentType, (int));    
MOCK_METHOD(char, OverloadedOnArgumentType, (char));  

MOCK_METHOD(int, OverloadedOnConstness, (), (override));          
MOCK_METHOD(char, OverloadedOnConstness, (), (override, const));  

MOCK_METHOD(int, TypeWithHole, (int (*)()), ());  
MOCK_METHOD(int, TypeWithComma, ((const std::map<int, std::string>&)));
MOCK_METHOD(int, TypeWithTemplatedCopyCtor,
(const TemplatedCopyable<int>&));  

MOCK_METHOD(int (*)(bool), ReturnsFunctionPointer1, (int), ());
MOCK_METHOD(fn_ptr, ReturnsFunctionPointer2, (int), ());

#if GTEST_OS_WINDOWS
MOCK_METHOD(int, CTNullary, (), (Calltype(STDMETHODCALLTYPE)));
MOCK_METHOD(bool, CTUnary, (int), (Calltype(STDMETHODCALLTYPE)));
MOCK_METHOD(int, CTDecimal,
(bool b, char c, short d, int e, long f, float g, double h,
unsigned i, char* j, const std::string& k),
(Calltype(STDMETHODCALLTYPE)));
MOCK_METHOD(char, CTConst, (int), (const, Calltype(STDMETHODCALLTYPE)));
MOCK_METHOD((std::map<int, std::string>), CTReturnTypeWithComma, (),
(Calltype(STDMETHODCALLTYPE)));
#endif  

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockFoo);
};

class LegacyMockFoo : public FooInterface {
public:
LegacyMockFoo() {}

MOCK_METHOD1(VoidReturning, void(int n));  

MOCK_METHOD0(Nullary, int());  

MOCK_METHOD1(Unary, bool(int));                                  
MOCK_METHOD2(Binary, long(short, int));                          
MOCK_METHOD10(Decimal, int(bool, char, short, int, long, float,  
double, unsigned, char*, const std::string& str));

MOCK_METHOD1(TakesNonConstReference, bool(int&));  
MOCK_METHOD1(TakesConstReference, std::string(const int&));
MOCK_METHOD1(TakesConst, bool(const int));  

MOCK_METHOD0(ReturnTypeWithComma, std::map<int, std::string>());
MOCK_CONST_METHOD1(ReturnTypeWithComma,
std::map<int, std::string>(int));  

MOCK_METHOD0(OverloadedOnArgumentNumber, int());     
MOCK_METHOD1(OverloadedOnArgumentNumber, int(int));  

MOCK_METHOD1(OverloadedOnArgumentType, int(int));    
MOCK_METHOD1(OverloadedOnArgumentType, char(char));  

MOCK_METHOD0(OverloadedOnConstness, int());         
MOCK_CONST_METHOD0(OverloadedOnConstness, char());  

MOCK_METHOD1(TypeWithHole, int(int (*)()));  
MOCK_METHOD1(TypeWithComma,
int(const std::map<int, std::string>&));  
MOCK_METHOD1(TypeWithTemplatedCopyCtor,
int(const TemplatedCopyable<int>&));  

MOCK_METHOD1(ReturnsFunctionPointer1, int (*(int))(bool));
MOCK_METHOD1(ReturnsFunctionPointer2, fn_ptr(int));

#if GTEST_OS_WINDOWS
MOCK_METHOD0_WITH_CALLTYPE(STDMETHODCALLTYPE, CTNullary, int());
MOCK_METHOD1_WITH_CALLTYPE(STDMETHODCALLTYPE, CTUnary, bool(int));  
MOCK_METHOD10_WITH_CALLTYPE(STDMETHODCALLTYPE, CTDecimal,
int(bool b, char c, short d, int e,  
long f, float g, double h,       
unsigned i, char* j, const std::string& k));
MOCK_CONST_METHOD1_WITH_CALLTYPE(STDMETHODCALLTYPE, CTConst,
char(int));  

MOCK_METHOD0_WITH_CALLTYPE(STDMETHODCALLTYPE, CTReturnTypeWithComma,
std::map<int, std::string>());
#endif  

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(LegacyMockFoo);
};

#ifdef _MSC_VER
# pragma warning(pop)
#endif

template <class T>
class FunctionMockerTest : public testing::Test {
protected:
FunctionMockerTest() : foo_(&mock_foo_) {}

FooInterface* const foo_;
T mock_foo_;
};
using FunctionMockerTestTypes = ::testing::Types<MockFoo, LegacyMockFoo>;
TYPED_TEST_SUITE(FunctionMockerTest, FunctionMockerTestTypes);

TYPED_TEST(FunctionMockerTest, MocksVoidFunction) {
EXPECT_CALL(this->mock_foo_, VoidReturning(Lt(100)));
this->foo_->VoidReturning(0);
}

TYPED_TEST(FunctionMockerTest, MocksNullaryFunction) {
EXPECT_CALL(this->mock_foo_, Nullary())
.WillOnce(DoDefault())
.WillOnce(Return(1));

EXPECT_EQ(0, this->foo_->Nullary());
EXPECT_EQ(1, this->foo_->Nullary());
}

TYPED_TEST(FunctionMockerTest, MocksUnaryFunction) {
EXPECT_CALL(this->mock_foo_, Unary(Eq(2))).Times(2).WillOnce(Return(true));

EXPECT_TRUE(this->foo_->Unary(2));
EXPECT_FALSE(this->foo_->Unary(2));
}

TYPED_TEST(FunctionMockerTest, MocksBinaryFunction) {
EXPECT_CALL(this->mock_foo_, Binary(2, _)).WillOnce(Return(3));

EXPECT_EQ(3, this->foo_->Binary(2, 1));
}

TYPED_TEST(FunctionMockerTest, MocksDecimalFunction) {
EXPECT_CALL(this->mock_foo_,
Decimal(true, 'a', 0, 0, 1L, A<float>(), Lt(100), 5U, NULL, "hi"))
.WillOnce(Return(5));

EXPECT_EQ(5, this->foo_->Decimal(true, 'a', 0, 0, 1, 0, 0, 5, nullptr, "hi"));
}

TYPED_TEST(FunctionMockerTest, MocksFunctionWithNonConstReferenceArgument) {
int a = 0;
EXPECT_CALL(this->mock_foo_, TakesNonConstReference(Ref(a)))
.WillOnce(Return(true));

EXPECT_TRUE(this->foo_->TakesNonConstReference(a));
}

TYPED_TEST(FunctionMockerTest, MocksFunctionWithConstReferenceArgument) {
int a = 0;
EXPECT_CALL(this->mock_foo_, TakesConstReference(Ref(a)))
.WillOnce(Return("Hello"));

EXPECT_EQ("Hello", this->foo_->TakesConstReference(a));
}

TYPED_TEST(FunctionMockerTest, MocksFunctionWithConstArgument) {
EXPECT_CALL(this->mock_foo_, TakesConst(Lt(10))).WillOnce(DoDefault());

EXPECT_FALSE(this->foo_->TakesConst(5));
}

TYPED_TEST(FunctionMockerTest, MocksFunctionsOverloadedOnArgumentNumber) {
EXPECT_CALL(this->mock_foo_, OverloadedOnArgumentNumber())
.WillOnce(Return(1));
EXPECT_CALL(this->mock_foo_, OverloadedOnArgumentNumber(_))
.WillOnce(Return(2));

EXPECT_EQ(2, this->foo_->OverloadedOnArgumentNumber(1));
EXPECT_EQ(1, this->foo_->OverloadedOnArgumentNumber());
}

TYPED_TEST(FunctionMockerTest, MocksFunctionsOverloadedOnArgumentType) {
EXPECT_CALL(this->mock_foo_, OverloadedOnArgumentType(An<int>()))
.WillOnce(Return(1));
EXPECT_CALL(this->mock_foo_, OverloadedOnArgumentType(TypedEq<char>('a')))
.WillOnce(Return('b'));

EXPECT_EQ(1, this->foo_->OverloadedOnArgumentType(0));
EXPECT_EQ('b', this->foo_->OverloadedOnArgumentType('a'));
}

TYPED_TEST(FunctionMockerTest, MocksFunctionsOverloadedOnConstnessOfThis) {
EXPECT_CALL(this->mock_foo_, OverloadedOnConstness());
EXPECT_CALL(Const(this->mock_foo_), OverloadedOnConstness())
.WillOnce(Return('a'));

EXPECT_EQ(0, this->foo_->OverloadedOnConstness());
EXPECT_EQ('a', Const(*this->foo_).OverloadedOnConstness());
}

TYPED_TEST(FunctionMockerTest, MocksReturnTypeWithComma) {
const std::map<int, std::string> a_map;
EXPECT_CALL(this->mock_foo_, ReturnTypeWithComma()).WillOnce(Return(a_map));
EXPECT_CALL(this->mock_foo_, ReturnTypeWithComma(42)).WillOnce(Return(a_map));

EXPECT_EQ(a_map, this->mock_foo_.ReturnTypeWithComma());
EXPECT_EQ(a_map, this->mock_foo_.ReturnTypeWithComma(42));
}

TYPED_TEST(FunctionMockerTest, MocksTypeWithTemplatedCopyCtor) {
EXPECT_CALL(this->mock_foo_, TypeWithTemplatedCopyCtor(_))
.WillOnce(Return(true));
EXPECT_TRUE(this->foo_->TypeWithTemplatedCopyCtor(TemplatedCopyable<int>()));
}

#if GTEST_OS_WINDOWS
TYPED_TEST(FunctionMockerTest, MocksNullaryFunctionWithCallType) {
EXPECT_CALL(this->mock_foo_, CTNullary())
.WillOnce(Return(-1))
.WillOnce(Return(0));

EXPECT_EQ(-1, this->foo_->CTNullary());
EXPECT_EQ(0, this->foo_->CTNullary());
}

TYPED_TEST(FunctionMockerTest, MocksUnaryFunctionWithCallType) {
EXPECT_CALL(this->mock_foo_, CTUnary(Eq(2)))
.Times(2)
.WillOnce(Return(true))
.WillOnce(Return(false));

EXPECT_TRUE(this->foo_->CTUnary(2));
EXPECT_FALSE(this->foo_->CTUnary(2));
}

TYPED_TEST(FunctionMockerTest, MocksDecimalFunctionWithCallType) {
EXPECT_CALL(this->mock_foo_, CTDecimal(true, 'a', 0, 0, 1L, A<float>(),
Lt(100), 5U, NULL, "hi"))
.WillOnce(Return(10));

EXPECT_EQ(10, this->foo_->CTDecimal(true, 'a', 0, 0, 1, 0, 0, 5, NULL, "hi"));
}

TYPED_TEST(FunctionMockerTest, MocksFunctionsConstFunctionWithCallType) {
EXPECT_CALL(Const(this->mock_foo_), CTConst(_)).WillOnce(Return('a'));

EXPECT_EQ('a', Const(*this->foo_).CTConst(0));
}

TYPED_TEST(FunctionMockerTest, MocksReturnTypeWithCommaAndCallType) {
const std::map<int, std::string> a_map;
EXPECT_CALL(this->mock_foo_, CTReturnTypeWithComma()).WillOnce(Return(a_map));

EXPECT_EQ(a_map, this->mock_foo_.CTReturnTypeWithComma());
}

#endif  

class MockB {
public:
MockB() {}

MOCK_METHOD(void, DoB, ());

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockB);
};

class LegacyMockB {
public:
LegacyMockB() {}

MOCK_METHOD0(DoB, void());

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(LegacyMockB);
};

template <typename T>
class ExpectCallTest : public ::testing::Test {};
using ExpectCallTestTypes = ::testing::Types<MockB, LegacyMockB>;
TYPED_TEST_SUITE(ExpectCallTest, ExpectCallTestTypes);

TYPED_TEST(ExpectCallTest, UnmentionedFunctionCanBeCalledAnyNumberOfTimes) {
{ TypeParam b; }

{
TypeParam b;
b.DoB();
}

{
TypeParam b;
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

MOCK_METHOD(void, Push, (const T& elem), ());
MOCK_METHOD(void, Pop, (), (final));
MOCK_METHOD(int, GetSize, (), (const, override));
MOCK_METHOD(const T&, GetTop, (), (const));

MOCK_METHOD((std::map<int, int>), ReturnTypeWithComma, (), ());
MOCK_METHOD((std::map<int, int>), ReturnTypeWithComma, (int), (const));

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockStack);
};

template <typename T>
class LegacyMockStack : public StackInterface<T> {
public:
LegacyMockStack() {}

MOCK_METHOD1_T(Push, void(const T& elem));
MOCK_METHOD0_T(Pop, void());
MOCK_CONST_METHOD0_T(GetSize, int());  
MOCK_CONST_METHOD0_T(GetTop, const T&());

MOCK_METHOD0_T(ReturnTypeWithComma, std::map<int, int>());
MOCK_CONST_METHOD1_T(ReturnTypeWithComma, std::map<int, int>(int));  

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(LegacyMockStack);
};

template <typename T>
class TemplateMockTest : public ::testing::Test {};
using TemplateMockTestTypes =
::testing::Types<MockStack<int>, LegacyMockStack<int>>;
TYPED_TEST_SUITE(TemplateMockTest, TemplateMockTestTypes);

TYPED_TEST(TemplateMockTest, Works) {
TypeParam mock;

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

TYPED_TEST(TemplateMockTest, MethodWithCommaInReturnTypeWorks) {
TypeParam mock;

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

MOCK_METHOD(void, Push, (const T& elem),
(Calltype(STDMETHODCALLTYPE), override));
MOCK_METHOD(void, Pop, (), (Calltype(STDMETHODCALLTYPE), override));
MOCK_METHOD(int, GetSize, (), (Calltype(STDMETHODCALLTYPE), override, const));
MOCK_METHOD(const T&, GetTop, (),
(Calltype(STDMETHODCALLTYPE), override, const));

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockStackWithCallType);
};

template <typename T>
class LegacyMockStackWithCallType : public StackInterfaceWithCallType<T> {
public:
LegacyMockStackWithCallType() {}

MOCK_METHOD1_T_WITH_CALLTYPE(STDMETHODCALLTYPE, Push, void(const T& elem));
MOCK_METHOD0_T_WITH_CALLTYPE(STDMETHODCALLTYPE, Pop, void());
MOCK_CONST_METHOD0_T_WITH_CALLTYPE(STDMETHODCALLTYPE, GetSize, int());
MOCK_CONST_METHOD0_T_WITH_CALLTYPE(STDMETHODCALLTYPE, GetTop, const T&());

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(LegacyMockStackWithCallType);
};

template <typename T>
class TemplateMockTestWithCallType : public ::testing::Test {};
using TemplateMockTestWithCallTypeTypes =
::testing::Types<MockStackWithCallType<int>,
LegacyMockStackWithCallType<int>>;
TYPED_TEST_SUITE(TemplateMockTestWithCallType,
TemplateMockTestWithCallTypeTypes);

TYPED_TEST(TemplateMockTestWithCallType, Works) {
TypeParam mock;

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

#define MY_MOCK_METHODS1_                       \
MOCK_METHOD(void, Overloaded, ());            \
MOCK_METHOD(int, Overloaded, (int), (const)); \
MOCK_METHOD(bool, Overloaded, (bool f, int n))

#define LEGACY_MY_MOCK_METHODS1_              \
MOCK_METHOD0(Overloaded, void());           \
MOCK_CONST_METHOD1(Overloaded, int(int n)); \
MOCK_METHOD2(Overloaded, bool(bool f, int n))

class MockOverloadedOnArgNumber {
public:
MockOverloadedOnArgNumber() {}

MY_MOCK_METHODS1_;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockOverloadedOnArgNumber);
};

class LegacyMockOverloadedOnArgNumber {
public:
LegacyMockOverloadedOnArgNumber() {}

LEGACY_MY_MOCK_METHODS1_;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(LegacyMockOverloadedOnArgNumber);
};

template <typename T>
class OverloadedMockMethodTest : public ::testing::Test {};
using OverloadedMockMethodTestTypes =
::testing::Types<MockOverloadedOnArgNumber,
LegacyMockOverloadedOnArgNumber>;
TYPED_TEST_SUITE(OverloadedMockMethodTest, OverloadedMockMethodTestTypes);

TYPED_TEST(OverloadedMockMethodTest, CanOverloadOnArgNumberInMacroBody) {
TypeParam mock;
EXPECT_CALL(mock, Overloaded());
EXPECT_CALL(mock, Overloaded(1)).WillOnce(Return(2));
EXPECT_CALL(mock, Overloaded(true, 1)).WillOnce(Return(true));

mock.Overloaded();
EXPECT_EQ(2, mock.Overloaded(1));
EXPECT_TRUE(mock.Overloaded(true, 1));
}

#define MY_MOCK_METHODS2_ \
MOCK_CONST_METHOD1(Overloaded, int(int n)); \
MOCK_METHOD1(Overloaded, int(int n))

class MockOverloadedOnConstness {
public:
MockOverloadedOnConstness() {}

MY_MOCK_METHODS2_;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(MockOverloadedOnConstness);
};

TEST(MockMethodOverloadedMockMethodTest, CanOverloadOnConstnessInMacroBody) {
MockOverloadedOnConstness mock;
const MockOverloadedOnConstness* const_mock = &mock;
EXPECT_CALL(mock, Overloaded(1)).WillOnce(Return(2));
EXPECT_CALL(*const_mock, Overloaded(1)).WillOnce(Return(3));

EXPECT_EQ(2, mock.Overloaded(1));
EXPECT_EQ(3, const_mock->Overloaded(1));
}

TEST(MockMethodMockFunctionTest, WorksForVoidNullary) {
MockFunction<void()> foo;
EXPECT_CALL(foo, Call());
foo.Call();
}

TEST(MockMethodMockFunctionTest, WorksForNonVoidNullary) {
MockFunction<int()> foo;
EXPECT_CALL(foo, Call())
.WillOnce(Return(1))
.WillOnce(Return(2));
EXPECT_EQ(1, foo.Call());
EXPECT_EQ(2, foo.Call());
}

TEST(MockMethodMockFunctionTest, WorksForVoidUnary) {
MockFunction<void(int)> foo;
EXPECT_CALL(foo, Call(1));
foo.Call(1);
}

TEST(MockMethodMockFunctionTest, WorksForNonVoidBinary) {
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

TEST(MockMethodMockFunctionTest, WorksFor10Arguments) {
MockFunction<int(bool a0, char a1, int a2, int a3, int a4,
int a5, int a6, char a7, int a8, bool a9)> foo;
EXPECT_CALL(foo, Call(_, 'a', _, _, _, _, _, _, _, _))
.WillOnce(Return(1))
.WillOnce(Return(2));
EXPECT_EQ(1, foo.Call(false, 'a', 0, 0, 0, 0, 0, 'b', 0, true));
EXPECT_EQ(2, foo.Call(true, 'a', 0, 0, 0, 0, 0, 'b', 1, false));
}

TEST(MockMethodMockFunctionTest, AsStdFunction) {
MockFunction<int(int)> foo;
auto call = [](const std::function<int(int)> &f, int i) {
return f(i);
};
EXPECT_CALL(foo, Call(1)).WillOnce(Return(-1));
EXPECT_CALL(foo, Call(2)).WillOnce(Return(-2));
EXPECT_EQ(-1, call(foo.AsStdFunction(), 1));
EXPECT_EQ(-2, call(foo.AsStdFunction(), 2));
}

TEST(MockMethodMockFunctionTest, AsStdFunctionReturnsReference) {
MockFunction<int&()> foo;
int value = 1;
EXPECT_CALL(foo, Call()).WillOnce(ReturnRef(value));
int& ref = foo.AsStdFunction()();
EXPECT_EQ(1, ref);
value = 2;
EXPECT_EQ(2, ref);
}

TEST(MockMethodMockFunctionTest, AsStdFunctionWithReferenceParameter) {
MockFunction<int(int &)> foo;
auto call = [](const std::function<int(int& )> &f, int &i) {
return f(i);
};
int i = 42;
EXPECT_CALL(foo, Call(i)).WillOnce(Return(-1));
EXPECT_EQ(-1, call(foo.AsStdFunction(), i));
}

namespace {

template <typename Expected, typename F>
static constexpr bool IsMockFunctionTemplateArgumentDeducedTo(
const MockFunction<F>&) {
return std::is_same<F, Expected>::value;
}

}  

template <typename F>
class MockMethodMockFunctionSignatureTest : public Test {};

using MockMethodMockFunctionSignatureTypes =
Types<void(), int(), void(int), int(int), int(bool, int),
int(bool, char, int, int, int, int, int, char, int, bool)>;
TYPED_TEST_SUITE(MockMethodMockFunctionSignatureTest,
MockMethodMockFunctionSignatureTypes);

TYPED_TEST(MockMethodMockFunctionSignatureTest,
IsMockFunctionTemplateArgumentDeducedForRawSignature) {
using Argument = TypeParam;
MockFunction<Argument> foo;
EXPECT_TRUE(IsMockFunctionTemplateArgumentDeducedTo<Argument>(foo));
}

TYPED_TEST(MockMethodMockFunctionSignatureTest,
IsMockFunctionTemplateArgumentDeducedForStdFunction) {
using Argument = std::function<TypeParam>;
MockFunction<Argument> foo;
EXPECT_TRUE(IsMockFunctionTemplateArgumentDeducedTo<Argument>(foo));
}

TYPED_TEST(
MockMethodMockFunctionSignatureTest,
IsMockFunctionCallMethodSignatureTheSameForRawSignatureAndStdFunction) {
using ForRawSignature = decltype(&MockFunction<TypeParam>::Call);
using ForStdFunction =
decltype(&MockFunction<std::function<TypeParam>>::Call);
EXPECT_TRUE((std::is_same<ForRawSignature, ForStdFunction>::value));
}

TYPED_TEST(
MockMethodMockFunctionSignatureTest,
IsMockFunctionAsStdFunctionMethodSignatureTheSameForRawSignatureAndStdFunction) {
using ForRawSignature = decltype(&MockFunction<TypeParam>::AsStdFunction);
using ForStdFunction =
decltype(&MockFunction<std::function<TypeParam>>::AsStdFunction);
EXPECT_TRUE((std::is_same<ForRawSignature, ForStdFunction>::value));
}

struct MockMethodSizes0 {
MOCK_METHOD(void, func, ());
};
struct MockMethodSizes1 {
MOCK_METHOD(void, func, (int));
};
struct MockMethodSizes2 {
MOCK_METHOD(void, func, (int, int));
};
struct MockMethodSizes3 {
MOCK_METHOD(void, func, (int, int, int));
};
struct MockMethodSizes4 {
MOCK_METHOD(void, func, (int, int, int, int));
};

struct LegacyMockMethodSizes0 {
MOCK_METHOD0(func, void());
};
struct LegacyMockMethodSizes1 {
MOCK_METHOD1(func, void(int));
};
struct LegacyMockMethodSizes2 {
MOCK_METHOD2(func, void(int, int));
};
struct LegacyMockMethodSizes3 {
MOCK_METHOD3(func, void(int, int, int));
};
struct LegacyMockMethodSizes4 {
MOCK_METHOD4(func, void(int, int, int, int));
};


TEST(MockMethodMockFunctionTest, MockMethodSizeOverhead) {
EXPECT_EQ(sizeof(MockMethodSizes0), sizeof(MockMethodSizes1));
EXPECT_EQ(sizeof(MockMethodSizes0), sizeof(MockMethodSizes2));
EXPECT_EQ(sizeof(MockMethodSizes0), sizeof(MockMethodSizes3));
EXPECT_EQ(sizeof(MockMethodSizes0), sizeof(MockMethodSizes4));

EXPECT_EQ(sizeof(LegacyMockMethodSizes0), sizeof(LegacyMockMethodSizes1));
EXPECT_EQ(sizeof(LegacyMockMethodSizes0), sizeof(LegacyMockMethodSizes2));
EXPECT_EQ(sizeof(LegacyMockMethodSizes0), sizeof(LegacyMockMethodSizes3));
EXPECT_EQ(sizeof(LegacyMockMethodSizes0), sizeof(LegacyMockMethodSizes4));

EXPECT_EQ(sizeof(LegacyMockMethodSizes0), sizeof(MockMethodSizes0));
}

void hasTwoParams(int, int);
void MaybeThrows();
void DoesntThrow() noexcept;
struct MockMethodNoexceptSpecifier {
MOCK_METHOD(void, func1, (), (noexcept));
MOCK_METHOD(void, func2, (), (noexcept(true)));
MOCK_METHOD(void, func3, (), (noexcept(false)));
MOCK_METHOD(void, func4, (), (noexcept(noexcept(MaybeThrows()))));
MOCK_METHOD(void, func5, (), (noexcept(noexcept(DoesntThrow()))));
MOCK_METHOD(void, func6, (), (noexcept(noexcept(DoesntThrow())), const));
MOCK_METHOD(void, func7, (), (const, noexcept(noexcept(DoesntThrow()))));
MOCK_METHOD(void, func8, (), (noexcept(noexcept(hasTwoParams(1, 2))), const));
};

TEST(MockMethodMockFunctionTest, NoexceptSpecifierPreserved) {
EXPECT_TRUE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func1()));
EXPECT_TRUE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func2()));
EXPECT_FALSE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func3()));
EXPECT_FALSE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func4()));
EXPECT_TRUE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func5()));
EXPECT_TRUE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func6()));
EXPECT_TRUE(noexcept(std::declval<MockMethodNoexceptSpecifier>().func7()));
EXPECT_EQ(noexcept(std::declval<MockMethodNoexceptSpecifier>().func8()),
noexcept(hasTwoParams(1, 2)));
}

}  
}  
