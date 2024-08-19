

#include "gmock/gmock-generated-actions.h"

#include <functional>
#include <sstream>
#include <string>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace testing {
namespace gmock_generated_actions_test {

using ::std::plus;
using ::std::string;
using testing::get;
using testing::make_tuple;
using testing::tuple;
using testing::tuple_element;
using testing::_;
using testing::Action;
using testing::ActionInterface;
using testing::ByRef;
using testing::DoAll;
using testing::Invoke;
using testing::Return;
using testing::ReturnNew;
using testing::SetArgPointee;
using testing::StaticAssertTypeEq;
using testing::Unused;
using testing::WithArgs;

inline short Short(short n) { return n; }  
inline char Char(char ch) { return ch; }

int Nullary() { return 1; }

class NullaryFunctor {
public:
int operator()() { return 2; }
};

bool g_done = false;

bool Unary(int x) { return x < 0; }

const char* Plus1(const char* s) { return s + 1; }

bool ByConstRef(const string& s) { return s == "Hi"; }

const double g_double = 0;
bool ReferencesGlobalDouble(const double& x) { return &x == &g_double; }

string ByNonConstRef(string& s) { return s += "+"; }  

struct UnaryFunctor {
int operator()(bool x) { return x ? 1 : -1; }
};

const char* Binary(const char* input, short n) { return input + n; }  

void VoidBinary(int, char) { g_done = true; }

int Ternary(int x, char y, short z) { return x + y + z; }  

void VoidTernary(int, char, bool) { g_done = true; }

int SumOf4(int a, int b, int c, int d) { return a + b + c + d; }

string Concat4(const char* s1, const char* s2, const char* s3,
const char* s4) {
return string(s1) + s2 + s3 + s4;
}

int SumOf5(int a, int b, int c, int d, int e) { return a + b + c + d + e; }

struct SumOf5Functor {
int operator()(int a, int b, int c, int d, int e) {
return a + b + c + d + e;
}
};

string Concat5(const char* s1, const char* s2, const char* s3,
const char* s4, const char* s5) {
return string(s1) + s2 + s3 + s4 + s5;
}

int SumOf6(int a, int b, int c, int d, int e, int f) {
return a + b + c + d + e + f;
}

struct SumOf6Functor {
int operator()(int a, int b, int c, int d, int e, int f) {
return a + b + c + d + e + f;
}
};

string Concat6(const char* s1, const char* s2, const char* s3,
const char* s4, const char* s5, const char* s6) {
return string(s1) + s2 + s3 + s4 + s5 + s6;
}

string Concat7(const char* s1, const char* s2, const char* s3,
const char* s4, const char* s5, const char* s6,
const char* s7) {
return string(s1) + s2 + s3 + s4 + s5 + s6 + s7;
}

string Concat8(const char* s1, const char* s2, const char* s3,
const char* s4, const char* s5, const char* s6,
const char* s7, const char* s8) {
return string(s1) + s2 + s3 + s4 + s5 + s6 + s7 + s8;
}

string Concat9(const char* s1, const char* s2, const char* s3,
const char* s4, const char* s5, const char* s6,
const char* s7, const char* s8, const char* s9) {
return string(s1) + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;
}

string Concat10(const char* s1, const char* s2, const char* s3,
const char* s4, const char* s5, const char* s6,
const char* s7, const char* s8, const char* s9,
const char* s10) {
return string(s1) + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10;
}

inline const char* CharPtr(const char* s) { return s; }


TEST(InvokeArgumentTest, Function0) {
Action<int(int, int(*)())> a = InvokeArgument<1>();  
EXPECT_EQ(1, a.Perform(make_tuple(2, &Nullary)));
}

TEST(InvokeArgumentTest, Functor1) {
Action<int(UnaryFunctor)> a = InvokeArgument<0>(true);  
EXPECT_EQ(1, a.Perform(make_tuple(UnaryFunctor())));
}

TEST(InvokeArgumentTest, Function5) {
Action<int(int(*)(int, int, int, int, int))> a =  
InvokeArgument<0>(10000, 2000, 300, 40, 5);
EXPECT_EQ(12345, a.Perform(make_tuple(&SumOf5)));
}

TEST(InvokeArgumentTest, Functor5) {
Action<int(SumOf5Functor)> a =  
InvokeArgument<0>(10000, 2000, 300, 40, 5);
EXPECT_EQ(12345, a.Perform(make_tuple(SumOf5Functor())));
}

TEST(InvokeArgumentTest, Function6) {
Action<int(int(*)(int, int, int, int, int, int))> a =  
InvokeArgument<0>(100000, 20000, 3000, 400, 50, 6);
EXPECT_EQ(123456, a.Perform(make_tuple(&SumOf6)));
}

TEST(InvokeArgumentTest, Functor6) {
Action<int(SumOf6Functor)> a =  
InvokeArgument<0>(100000, 20000, 3000, 400, 50, 6);
EXPECT_EQ(123456, a.Perform(make_tuple(SumOf6Functor())));
}

TEST(InvokeArgumentTest, Function7) {
Action<string(string(*)(const char*, const char*, const char*,
const char*, const char*, const char*,
const char*))> a =
InvokeArgument<0>("1", "2", "3", "4", "5", "6", "7");
EXPECT_EQ("1234567", a.Perform(make_tuple(&Concat7)));
}

TEST(InvokeArgumentTest, Function8) {
Action<string(string(*)(const char*, const char*, const char*,
const char*, const char*, const char*,
const char*, const char*))> a =
InvokeArgument<0>("1", "2", "3", "4", "5", "6", "7", "8");
EXPECT_EQ("12345678", a.Perform(make_tuple(&Concat8)));
}

TEST(InvokeArgumentTest, Function9) {
Action<string(string(*)(const char*, const char*, const char*,
const char*, const char*, const char*,
const char*, const char*, const char*))> a =
InvokeArgument<0>("1", "2", "3", "4", "5", "6", "7", "8", "9");
EXPECT_EQ("123456789", a.Perform(make_tuple(&Concat9)));
}

TEST(InvokeArgumentTest, Function10) {
Action<string(string(*)(const char*, const char*, const char*,
const char*, const char*, const char*,
const char*, const char*, const char*,
const char*))> a =
InvokeArgument<0>("1", "2", "3", "4", "5", "6", "7", "8", "9", "0");
EXPECT_EQ("1234567890", a.Perform(make_tuple(&Concat10)));
}

TEST(InvokeArgumentTest, ByPointerFunction) {
Action<const char*(const char*(*)(const char* input, short n))> a =  
InvokeArgument<0>(static_cast<const char*>("Hi"), Short(1));
EXPECT_STREQ("i", a.Perform(make_tuple(&Binary)));
}

TEST(InvokeArgumentTest, FunctionWithCStringLiteral) {
Action<const char*(const char*(*)(const char* input, short n))> a =  
InvokeArgument<0>("Hi", Short(1));
EXPECT_STREQ("i", a.Perform(make_tuple(&Binary)));
}

TEST(InvokeArgumentTest, ByConstReferenceFunction) {
Action<bool(bool(*function)(const string& s))> a =  
InvokeArgument<0>(string("Hi"));
EXPECT_TRUE(a.Perform(make_tuple(&ByConstRef)));
}

TEST(InvokeArgumentTest, ByExplicitConstReferenceFunction) {
Action<bool(bool(*)(const double& x))> a =  
InvokeArgument<0>(ByRef(g_double));
EXPECT_TRUE(a.Perform(make_tuple(&ReferencesGlobalDouble)));

double x = 0;
a = InvokeArgument<0>(ByRef(x));  
EXPECT_FALSE(a.Perform(make_tuple(&ReferencesGlobalDouble)));
}

TEST(WithArgsTest, OneArg) {
Action<bool(double x, int n)> a = WithArgs<1>(Invoke(Unary));  
EXPECT_TRUE(a.Perform(make_tuple(1.5, -1)));
EXPECT_FALSE(a.Perform(make_tuple(1.5, 1)));
}

TEST(WithArgsTest, TwoArgs) {
Action<const char*(const char* s, double x, short n)> a =
WithArgs<0, 2>(Invoke(Binary));
const char s[] = "Hello";
EXPECT_EQ(s + 2, a.Perform(make_tuple(CharPtr(s), 0.5, Short(2))));
}

TEST(WithArgsTest, ThreeArgs) {
Action<int(int, double, char, short)> a =  
WithArgs<0, 2, 3>(Invoke(Ternary));
EXPECT_EQ(123, a.Perform(make_tuple(100, 6.5, Char(20), Short(3))));
}

TEST(WithArgsTest, FourArgs) {
Action<string(const char*, const char*, double, const char*, const char*)> a =
WithArgs<4, 3, 1, 0>(Invoke(Concat4));
EXPECT_EQ("4310", a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), 2.5,
CharPtr("3"), CharPtr("4"))));
}

TEST(WithArgsTest, FiveArgs) {
Action<string(const char*, const char*, const char*,
const char*, const char*)> a =
WithArgs<4, 3, 2, 1, 0>(Invoke(Concat5));
EXPECT_EQ("43210",
a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), CharPtr("2"),
CharPtr("3"), CharPtr("4"))));
}

TEST(WithArgsTest, SixArgs) {
Action<string(const char*, const char*, const char*)> a =
WithArgs<0, 1, 2, 2, 1, 0>(Invoke(Concat6));
EXPECT_EQ("012210",
a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), CharPtr("2"))));
}

TEST(WithArgsTest, SevenArgs) {
Action<string(const char*, const char*, const char*, const char*)> a =
WithArgs<0, 1, 2, 3, 2, 1, 0>(Invoke(Concat7));
EXPECT_EQ("0123210",
a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), CharPtr("2"),
CharPtr("3"))));
}

TEST(WithArgsTest, EightArgs) {
Action<string(const char*, const char*, const char*, const char*)> a =
WithArgs<0, 1, 2, 3, 0, 1, 2, 3>(Invoke(Concat8));
EXPECT_EQ("01230123",
a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), CharPtr("2"),
CharPtr("3"))));
}

TEST(WithArgsTest, NineArgs) {
Action<string(const char*, const char*, const char*, const char*)> a =
WithArgs<0, 1, 2, 3, 1, 2, 3, 2, 3>(Invoke(Concat9));
EXPECT_EQ("012312323",
a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), CharPtr("2"),
CharPtr("3"))));
}

TEST(WithArgsTest, TenArgs) {
Action<string(const char*, const char*, const char*, const char*)> a =
WithArgs<0, 1, 2, 3, 2, 1, 0, 1, 2, 3>(Invoke(Concat10));
EXPECT_EQ("0123210123",
a.Perform(make_tuple(CharPtr("0"), CharPtr("1"), CharPtr("2"),
CharPtr("3"))));
}

class SubstractAction : public ActionInterface<int(int, int)> {  
public:
virtual int Perform(const tuple<int, int>& args) {
return get<0>(args) - get<1>(args);
}
};

TEST(WithArgsTest, NonInvokeAction) {
Action<int(const string&, int, int)> a =  
WithArgs<2, 1>(MakeAction(new SubstractAction));
string s("hello");
EXPECT_EQ(8, a.Perform(tuple<const string&, int, int>(s, 2, 10)));
}

TEST(WithArgsTest, Identity) {
Action<int(int x, char y, short z)> a =  
WithArgs<0, 1, 2>(Invoke(Ternary));
EXPECT_EQ(123, a.Perform(make_tuple(100, Char(20), Short(3))));
}

TEST(WithArgsTest, RepeatedArguments) {
Action<int(bool, int m, int n)> a =  
WithArgs<1, 1, 1, 1>(Invoke(SumOf4));
EXPECT_EQ(4, a.Perform(make_tuple(false, 1, 10)));
}

TEST(WithArgsTest, ReversedArgumentOrder) {
Action<const char*(short n, const char* input)> a =  
WithArgs<1, 0>(Invoke(Binary));
const char s[] = "Hello";
EXPECT_EQ(s + 2, a.Perform(make_tuple(Short(2), CharPtr(s))));
}

TEST(WithArgsTest, ArgsOfCompatibleTypes) {
Action<long(short x, char y, double z, char c)> a =  
WithArgs<0, 1, 3>(Invoke(Ternary));
EXPECT_EQ(123, a.Perform(make_tuple(Short(100), Char(20), 5.6, Char(3))));
}

TEST(WithArgsTest, VoidAction) {
Action<void(double x, char c, int n)> a = WithArgs<2, 1>(Invoke(VoidBinary));
g_done = false;
a.Perform(make_tuple(1.5, 'a', 3));
EXPECT_TRUE(g_done);
}

TEST(DoAllTest, TwoActions) {
int n = 0;
Action<int(int*)> a = DoAll(SetArgPointee<0>(1),  
Return(2));
EXPECT_EQ(2, a.Perform(make_tuple(&n)));
EXPECT_EQ(1, n);
}

TEST(DoAllTest, ThreeActions) {
int m = 0, n = 0;
Action<int(int*, int*)> a = DoAll(SetArgPointee<0>(1),  
SetArgPointee<1>(2),
Return(3));
EXPECT_EQ(3, a.Perform(make_tuple(&m, &n)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
}

TEST(DoAllTest, FourActions) {
int m = 0, n = 0;
char ch = '\0';
Action<int(int*, int*, char*)> a =  
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
Return(3));
EXPECT_EQ(3, a.Perform(make_tuple(&m, &n, &ch)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', ch);
}

TEST(DoAllTest, FiveActions) {
int m = 0, n = 0;
char a = '\0', b = '\0';
Action<int(int*, int*, char*, char*)> action =  
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
SetArgPointee<3>('b'),
Return(3));
EXPECT_EQ(3, action.Perform(make_tuple(&m, &n, &a, &b)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', a);
EXPECT_EQ('b', b);
}

TEST(DoAllTest, SixActions) {
int m = 0, n = 0;
char a = '\0', b = '\0', c = '\0';
Action<int(int*, int*, char*, char*, char*)> action =  
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
SetArgPointee<3>('b'),
SetArgPointee<4>('c'),
Return(3));
EXPECT_EQ(3, action.Perform(make_tuple(&m, &n, &a, &b, &c)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', a);
EXPECT_EQ('b', b);
EXPECT_EQ('c', c);
}

TEST(DoAllTest, SevenActions) {
int m = 0, n = 0;
char a = '\0', b = '\0', c = '\0', d = '\0';
Action<int(int*, int*, char*, char*, char*, char*)> action =  
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
SetArgPointee<3>('b'),
SetArgPointee<4>('c'),
SetArgPointee<5>('d'),
Return(3));
EXPECT_EQ(3, action.Perform(make_tuple(&m, &n, &a, &b, &c, &d)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', a);
EXPECT_EQ('b', b);
EXPECT_EQ('c', c);
EXPECT_EQ('d', d);
}

TEST(DoAllTest, EightActions) {
int m = 0, n = 0;
char a = '\0', b = '\0', c = '\0', d = '\0', e = '\0';
Action<int(int*, int*, char*, char*, char*, char*,  
char*)> action =
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
SetArgPointee<3>('b'),
SetArgPointee<4>('c'),
SetArgPointee<5>('d'),
SetArgPointee<6>('e'),
Return(3));
EXPECT_EQ(3, action.Perform(make_tuple(&m, &n, &a, &b, &c, &d, &e)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', a);
EXPECT_EQ('b', b);
EXPECT_EQ('c', c);
EXPECT_EQ('d', d);
EXPECT_EQ('e', e);
}

TEST(DoAllTest, NineActions) {
int m = 0, n = 0;
char a = '\0', b = '\0', c = '\0', d = '\0', e = '\0', f = '\0';
Action<int(int*, int*, char*, char*, char*, char*,  
char*, char*)> action =
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
SetArgPointee<3>('b'),
SetArgPointee<4>('c'),
SetArgPointee<5>('d'),
SetArgPointee<6>('e'),
SetArgPointee<7>('f'),
Return(3));
EXPECT_EQ(3, action.Perform(make_tuple(&m, &n, &a, &b, &c, &d, &e, &f)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', a);
EXPECT_EQ('b', b);
EXPECT_EQ('c', c);
EXPECT_EQ('d', d);
EXPECT_EQ('e', e);
EXPECT_EQ('f', f);
}

TEST(DoAllTest, TenActions) {
int m = 0, n = 0;
char a = '\0', b = '\0', c = '\0', d = '\0';
char e = '\0', f = '\0', g = '\0';
Action<int(int*, int*, char*, char*, char*, char*,  
char*, char*, char*)> action =
DoAll(SetArgPointee<0>(1),
SetArgPointee<1>(2),
SetArgPointee<2>('a'),
SetArgPointee<3>('b'),
SetArgPointee<4>('c'),
SetArgPointee<5>('d'),
SetArgPointee<6>('e'),
SetArgPointee<7>('f'),
SetArgPointee<8>('g'),
Return(3));
EXPECT_EQ(3, action.Perform(make_tuple(&m, &n, &a, &b, &c, &d, &e, &f, &g)));
EXPECT_EQ(1, m);
EXPECT_EQ(2, n);
EXPECT_EQ('a', a);
EXPECT_EQ('b', b);
EXPECT_EQ('c', c);
EXPECT_EQ('d', d);
EXPECT_EQ('e', e);
EXPECT_EQ('f', f);
EXPECT_EQ('g', g);
}

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif


ACTION(Return5) { return 5; }

TEST(ActionMacroTest, WorksWhenNotReferencingArguments) {
Action<double()> a1 = Return5();
EXPECT_DOUBLE_EQ(5, a1.Perform(make_tuple()));

Action<int(double, bool)> a2 = Return5();
EXPECT_EQ(5, a2.Perform(make_tuple(1, true)));
}

ACTION(IncrementArg1) { (*arg1)++; }

TEST(ActionMacroTest, WorksWhenReturningVoid) {
Action<void(int, int*)> a1 = IncrementArg1();
int n = 0;
a1.Perform(make_tuple(5, &n));
EXPECT_EQ(1, n);
}

ACTION(IncrementArg2) {
StaticAssertTypeEq<int*, arg2_type>();
arg2_type temp = arg2;
(*temp)++;
}

TEST(ActionMacroTest, CanReferenceArgumentType) {
Action<void(int, bool, int*)> a1 = IncrementArg2();
int n = 0;
a1.Perform(make_tuple(5, false, &n));
EXPECT_EQ(1, n);
}

ACTION(Sum2) {
StaticAssertTypeEq<tuple<int, char, int*>, args_type>();
args_type args_copy = args;
return get<0>(args_copy) + get<1>(args_copy);
}

TEST(ActionMacroTest, CanReferenceArgumentTuple) {
Action<int(int, char, int*)> a1 = Sum2();
int dummy = 0;
EXPECT_EQ(11, a1.Perform(make_tuple(5, Char(6), &dummy)));
}

int Dummy(bool flag) { return flag? 1 : 0; }

ACTION(InvokeDummy) {
StaticAssertTypeEq<int(bool), function_type>();
function_type* fp = &Dummy;
return (*fp)(true);
}

TEST(ActionMacroTest, CanReferenceMockFunctionType) {
Action<int(bool)> a1 = InvokeDummy();
EXPECT_EQ(1, a1.Perform(make_tuple(true)));
EXPECT_EQ(1, a1.Perform(make_tuple(false)));
}

ACTION(InvokeDummy2) {
StaticAssertTypeEq<int, return_type>();
return_type result = Dummy(true);
return result;
}

TEST(ActionMacroTest, CanReferenceMockFunctionReturnType) {
Action<int(bool)> a1 = InvokeDummy2();
EXPECT_EQ(1, a1.Perform(make_tuple(true)));
EXPECT_EQ(1, a1.Perform(make_tuple(false)));
}

ACTION(ReturnAddrOfConstBoolReferenceArg) {
StaticAssertTypeEq<const bool&, arg1_type>();
return &arg1;
}

TEST(ActionMacroTest, WorksForConstReferenceArg) {
Action<const bool*(int, const bool&)> a = ReturnAddrOfConstBoolReferenceArg();
const bool b = false;
EXPECT_EQ(&b, a.Perform(tuple<int, const bool&>(0, b)));
}

ACTION(ReturnAddrOfIntReferenceArg) {
StaticAssertTypeEq<int&, arg0_type>();
return &arg0;
}

TEST(ActionMacroTest, WorksForNonConstReferenceArg) {
Action<int*(int&, bool, int)> a = ReturnAddrOfIntReferenceArg();
int n = 0;
EXPECT_EQ(&n, a.Perform(tuple<int&, bool, int>(n, true, 1)));
}

namespace action_test {
ACTION(Sum) { return arg0 + arg1; }
}  

TEST(ActionMacroTest, WorksInNamespace) {
Action<int(int, int)> a1 = action_test::Sum();
EXPECT_EQ(3, a1.Perform(make_tuple(1, 2)));
}

ACTION(PlusTwo) { return arg0 + 2; }

TEST(ActionMacroTest, WorksForDifferentArgumentNumbers) {
Action<int(int)> a1 = PlusTwo();
EXPECT_EQ(4, a1.Perform(make_tuple(2)));

Action<double(float, void*)> a2 = PlusTwo();
int dummy;
EXPECT_DOUBLE_EQ(6, a2.Perform(make_tuple(4.0f, &dummy)));
}

ACTION_P(Plus, n) { return arg0 + n; }

TEST(ActionPMacroTest, DefinesParameterizedAction) {
Action<int(int m, bool t)> a1 = Plus(9);
EXPECT_EQ(10, a1.Perform(make_tuple(1, true)));
}

ACTION_P(TypedPlus, n) {
arg0_type t1 = arg0;
n_type t2 = n;
return t1 + t2;
}

TEST(ActionPMacroTest, CanReferenceArgumentAndParameterTypes) {
Action<int(char m, bool t)> a1 = TypedPlus(9);
EXPECT_EQ(10, a1.Perform(make_tuple(Char(1), true)));
}

TEST(ActionPMacroTest, WorksInCompatibleMockFunction) {
Action<std::string(const std::string& s)> a1 = Plus("tail");
const std::string re = "re";
EXPECT_EQ("retail", a1.Perform(tuple<const std::string&>(re)));
}


ACTION(OverloadedAction) { return arg0 ? arg1 : "hello"; }

ACTION_P(OverloadedAction, default_value) {
return arg0 ? arg1 : default_value;
}

ACTION_P2(OverloadedAction, true_value, false_value) {
return arg0 ? true_value : false_value;
}

TEST(ActionMacroTest, CanDefineOverloadedActions) {
typedef Action<const char*(bool, const char*)> MyAction;

const MyAction a1 = OverloadedAction();
EXPECT_STREQ("hello", a1.Perform(make_tuple(false, CharPtr("world"))));
EXPECT_STREQ("world", a1.Perform(make_tuple(true, CharPtr("world"))));

const MyAction a2 = OverloadedAction("hi");
EXPECT_STREQ("hi", a2.Perform(make_tuple(false, CharPtr("world"))));
EXPECT_STREQ("world", a2.Perform(make_tuple(true, CharPtr("world"))));

const MyAction a3 = OverloadedAction("hi", "you");
EXPECT_STREQ("hi", a3.Perform(make_tuple(true, CharPtr("world"))));
EXPECT_STREQ("you", a3.Perform(make_tuple(false, CharPtr("world"))));
}


ACTION_P3(Plus, m, n, k) { return arg0 + m + n + k; }

TEST(ActionPnMacroTest, WorksFor3Parameters) {
Action<double(int m, bool t)> a1 = Plus(100, 20, 3.4);
EXPECT_DOUBLE_EQ(3123.4, a1.Perform(make_tuple(3000, true)));

Action<std::string(const std::string& s)> a2 = Plus("tail", "-", ">");
const std::string re = "re";
EXPECT_EQ("retail->", a2.Perform(tuple<const std::string&>(re)));
}

ACTION_P4(Plus, p0, p1, p2, p3) { return arg0 + p0 + p1 + p2 + p3; }

TEST(ActionPnMacroTest, WorksFor4Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4);
EXPECT_EQ(10 + 1 + 2 + 3 + 4, a1.Perform(make_tuple(10)));
}

ACTION_P5(Plus, p0, p1, p2, p3, p4) { return arg0 + p0 + p1 + p2 + p3 + p4; }

TEST(ActionPnMacroTest, WorksFor5Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4, 5);
EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5, a1.Perform(make_tuple(10)));
}

ACTION_P6(Plus, p0, p1, p2, p3, p4, p5) {
return arg0 + p0 + p1 + p2 + p3 + p4 + p5;
}

TEST(ActionPnMacroTest, WorksFor6Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4, 5, 6);
EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6, a1.Perform(make_tuple(10)));
}

ACTION_P7(Plus, p0, p1, p2, p3, p4, p5, p6) {
return arg0 + p0 + p1 + p2 + p3 + p4 + p5 + p6;
}

TEST(ActionPnMacroTest, WorksFor7Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4, 5, 6, 7);
EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7, a1.Perform(make_tuple(10)));
}

ACTION_P8(Plus, p0, p1, p2, p3, p4, p5, p6, p7) {
return arg0 + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
}

TEST(ActionPnMacroTest, WorksFor8Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4, 5, 6, 7, 8);
EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, a1.Perform(make_tuple(10)));
}

ACTION_P9(Plus, p0, p1, p2, p3, p4, p5, p6, p7, p8) {
return arg0 + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
}

TEST(ActionPnMacroTest, WorksFor9Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4, 5, 6, 7, 8, 9);
EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, a1.Perform(make_tuple(10)));
}

ACTION_P10(Plus, p0, p1, p2, p3, p4, p5, p6, p7, p8, last_param) {
arg0_type t0 = arg0;
last_param_type t9 = last_param;
return t0 + p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + t9;
}

TEST(ActionPnMacroTest, WorksFor10Parameters) {
Action<int(int)> a1 = Plus(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10,
a1.Perform(make_tuple(10)));
}


ACTION_P2(PadArgument, prefix, suffix) {
std::string prefix_str(prefix);
char suffix_char = static_cast<char>(suffix);
return prefix_str + arg0 + suffix_char;
}

TEST(ActionPnMacroTest, SimpleTypePromotion) {
Action<std::string(const char*)> no_promo =
PadArgument(std::string("foo"), 'r');
Action<std::string(const char*)> promo =
PadArgument("foo", static_cast<int>('r'));
EXPECT_EQ("foobar", no_promo.Perform(make_tuple(CharPtr("ba"))));
EXPECT_EQ("foobar", promo.Perform(make_tuple(CharPtr("ba"))));
}


ACTION_P3(ConcatImpl, a, b, c) {
std::stringstream ss;
ss << a << b << c;
return ss.str();
}


template <typename T1, typename T2>
ConcatImplActionP3<std::string, T1, T2>
Concat(const std::string& a, T1 b, T2 c) {
GTEST_INTENTIONAL_CONST_COND_PUSH_()
if (true) {
GTEST_INTENTIONAL_CONST_COND_POP_()
return ConcatImpl(a, b, c);
} else {
return ConcatImpl<std::string, T1, T2>(a, b, c);
}
}

template <typename T1, typename T2>
ConcatImplActionP3<T1, int, T2>
Concat(T1 a, int b, T2 c) {
return ConcatImpl(a, b, c);
}

TEST(ActionPnMacroTest, CanPartiallyRestrictParameterTypes) {
Action<const std::string()> a1 = Concat("Hello", "1", 2);
EXPECT_EQ("Hello12", a1.Perform(make_tuple()));

a1 = Concat(1, 2, 3);
EXPECT_EQ("123", a1.Perform(make_tuple()));
}


ACTION(DoFoo) {}
ACTION_P(DoFoo, p) {}
ACTION_P2(DoFoo, p0, p1) {}

TEST(ActionPnMacroTest, TypesAreCorrect) {
DoFooAction a0 = DoFoo();

DoFooActionP<int> a1 = DoFoo(1);

DoFooActionP2<int, char> a2 = DoFoo(1, '2');
PlusActionP3<int, int, char> a3 = Plus(1, 2, '3');
PlusActionP4<int, int, int, char> a4 = Plus(1, 2, 3, '4');
PlusActionP5<int, int, int, int, char> a5 = Plus(1, 2, 3, 4, '5');
PlusActionP6<int, int, int, int, int, char> a6 = Plus(1, 2, 3, 4, 5, '6');
PlusActionP7<int, int, int, int, int, int, char> a7 =
Plus(1, 2, 3, 4, 5, 6, '7');
PlusActionP8<int, int, int, int, int, int, int, char> a8 =
Plus(1, 2, 3, 4, 5, 6, 7, '8');
PlusActionP9<int, int, int, int, int, int, int, int, char> a9 =
Plus(1, 2, 3, 4, 5, 6, 7, 8, '9');
PlusActionP10<int, int, int, int, int, int, int, int, int, char> a10 =
Plus(1, 2, 3, 4, 5, 6, 7, 8, 9, '0');

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


ACTION_P(Plus1, x) { return x; }
ACTION_P2(Plus2, x, y) { return x + y; }
ACTION_P3(Plus3, x, y, z) { return x + y + z; }
ACTION_P10(Plus10, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) {
return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
}

TEST(ActionPnMacroTest, CanExplicitlyInstantiateWithReferenceTypes) {
int x = 1, y = 2, z = 3;
const tuple<> empty = make_tuple();

Action<int()> a = Plus1<int&>(x);
EXPECT_EQ(1, a.Perform(empty));

a = Plus2<const int&, int&>(x, y);
EXPECT_EQ(3, a.Perform(empty));

a = Plus3<int&, const int&, int&>(x, y, z);
EXPECT_EQ(6, a.Perform(empty));

int n[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
a = Plus10<const int&, int&, const int&, int&, const int&, int&, const int&,
int&, const int&, int&>(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7],
n[8], n[9]);
EXPECT_EQ(55, a.Perform(empty));
}

class NullaryConstructorClass {
public:
NullaryConstructorClass() : value_(123) {}
int value_;
};

TEST(ReturnNewTest, NoArgs) {
Action<NullaryConstructorClass*()> a = ReturnNew<NullaryConstructorClass>();
NullaryConstructorClass* c = a.Perform(make_tuple());
EXPECT_EQ(123, c->value_);
delete c;
}

class UnaryConstructorClass {
public:
explicit UnaryConstructorClass(int value) : value_(value) {}
int value_;
};

TEST(ReturnNewTest, Unary) {
Action<UnaryConstructorClass*()> a = ReturnNew<UnaryConstructorClass>(4000);
UnaryConstructorClass* c = a.Perform(make_tuple());
EXPECT_EQ(4000, c->value_);
delete c;
}

TEST(ReturnNewTest, UnaryWorksWhenMockMethodHasArgs) {
Action<UnaryConstructorClass*(bool, int)> a =
ReturnNew<UnaryConstructorClass>(4000);
UnaryConstructorClass* c = a.Perform(make_tuple(false, 5));
EXPECT_EQ(4000, c->value_);
delete c;
}

TEST(ReturnNewTest, UnaryWorksWhenMockMethodReturnsPointerToConst) {
Action<const UnaryConstructorClass*()> a =
ReturnNew<UnaryConstructorClass>(4000);
const UnaryConstructorClass* c = a.Perform(make_tuple());
EXPECT_EQ(4000, c->value_);
delete c;
}

class TenArgConstructorClass {
public:
TenArgConstructorClass(int a1, int a2, int a3, int a4, int a5,
int a6, int a7, int a8, int a9, int a10)
: value_(a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10) {
}
int value_;
};

TEST(ReturnNewTest, ConstructorThatTakes10Arguments) {
Action<TenArgConstructorClass*()> a =
ReturnNew<TenArgConstructorClass>(1000000000, 200000000, 30000000,
4000000, 500000, 60000,
7000, 800, 90, 0);
TenArgConstructorClass* c = a.Perform(make_tuple());
EXPECT_EQ(1234567890, c->value_);
delete c;
}

ACTION_TEMPLATE(CreateNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_0_VALUE_PARAMS()) {
return new T;
}

TEST(ActionTemplateTest, WorksWithoutValueParam) {
const Action<int*()> a = CreateNew<int>();
int* p = a.Perform(make_tuple());
delete p;
}

ACTION_TEMPLATE(CreateNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_1_VALUE_PARAMS(a0)) {
return new T(a0);
}

TEST(ActionTemplateTest, WorksWithValueParams) {
const Action<int*()> a = CreateNew<int>(42);
int* p = a.Perform(make_tuple());
EXPECT_EQ(42, *p);
delete p;
}

ACTION_TEMPLATE(MyDeleteArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
delete get<k>(args);
}

class BoolResetter {
public:
explicit BoolResetter(bool* value) : value_(value) {}
~BoolResetter() { *value_ = false; }
private:
bool* value_;
};

TEST(ActionTemplateTest, WorksForIntegralTemplateParams) {
const Action<void(int*, BoolResetter*)> a = MyDeleteArg<1>();
int n = 0;
bool b = true;
BoolResetter* resetter = new BoolResetter(&b);
a.Perform(make_tuple(&n, resetter));
EXPECT_FALSE(b);  
}

ACTION_TEMPLATE(ReturnSmartPointer,
HAS_1_TEMPLATE_PARAMS(template <typename Pointee> class,
Pointer),
AND_1_VALUE_PARAMS(pointee)) {
return Pointer<pointee_type>(new pointee_type(pointee));
}

TEST(ActionTemplateTest, WorksForTemplateTemplateParameters) {
using ::testing::internal::linked_ptr;
const Action<linked_ptr<int>()> a = ReturnSmartPointer<linked_ptr>(42);
linked_ptr<int> p = a.Perform(make_tuple());
EXPECT_EQ(42, *p);
}

template <typename T1, typename T2, typename T3, int k4, bool k5,
unsigned int k6, typename T7, typename T8, typename T9>
struct GiantTemplate {
public:
explicit GiantTemplate(int a_value) : value(a_value) {}
int value;
};

ACTION_TEMPLATE(ReturnGiant,
HAS_10_TEMPLATE_PARAMS(
typename, T1,
typename, T2,
typename, T3,
int, k4,
bool, k5,
unsigned int, k6,
class, T7,
class, T8,
class, T9,
template <typename T> class, T10),
AND_1_VALUE_PARAMS(value)) {
return GiantTemplate<T10<T1>, T2, T3, k4, k5, k6, T7, T8, T9>(value);
}

TEST(ActionTemplateTest, WorksFor10TemplateParameters) {
using ::testing::internal::linked_ptr;
typedef GiantTemplate<linked_ptr<int>, bool, double, 5,
true, 6, char, unsigned, int> Giant;
const Action<Giant()> a = ReturnGiant<
int, bool, double, 5, true, 6, char, unsigned, int, linked_ptr>(42);
Giant giant = a.Perform(make_tuple());
EXPECT_EQ(42, giant.value);
}

ACTION_TEMPLATE(ReturnSum,
HAS_1_TEMPLATE_PARAMS(typename, Number),
AND_10_VALUE_PARAMS(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)) {
return static_cast<Number>(v1) + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10;
}

TEST(ActionTemplateTest, WorksFor10ValueParameters) {
const Action<int()> a = ReturnSum<int>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
EXPECT_EQ(55, a.Perform(make_tuple()));
}


ACTION(ReturnSum) { return 0; }

ACTION_P(ReturnSum, x) { return x; }

ACTION_TEMPLATE(ReturnSum,
HAS_1_TEMPLATE_PARAMS(typename, Number),
AND_2_VALUE_PARAMS(v1, v2)) {
return static_cast<Number>(v1) + v2;
}

ACTION_TEMPLATE(ReturnSum,
HAS_1_TEMPLATE_PARAMS(typename, Number),
AND_3_VALUE_PARAMS(v1, v2, v3)) {
return static_cast<Number>(v1) + v2 + v3;
}

ACTION_TEMPLATE(ReturnSum,
HAS_2_TEMPLATE_PARAMS(typename, Number, int, k),
AND_4_VALUE_PARAMS(v1, v2, v3, v4)) {
return static_cast<Number>(v1) + v2 + v3 + v4 + k;
}

TEST(ActionTemplateTest, CanBeOverloadedOnNumberOfValueParameters) {
const Action<int()> a0 = ReturnSum();
const Action<int()> a1 = ReturnSum(1);
const Action<int()> a2 = ReturnSum<int>(1, 2);
const Action<int()> a3 = ReturnSum<int>(1, 2, 3);
const Action<int()> a4 = ReturnSum<int, 10000>(2000, 300, 40, 5);
EXPECT_EQ(0, a0.Perform(make_tuple()));
EXPECT_EQ(1, a1.Perform(make_tuple()));
EXPECT_EQ(3, a2.Perform(make_tuple()));
EXPECT_EQ(6, a3.Perform(make_tuple()));
EXPECT_EQ(12345, a4.Perform(make_tuple()));
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

}  
}  
