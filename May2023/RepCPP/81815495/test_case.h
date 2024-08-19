
#pragma once

#include <string>
#include <iostream>


#include "testing/tester.h"
#include "testing/test_case_result.h"

namespace Kratos::Testing
{
namespace Internals
{
template <typename TestType> class RegisterThisTest {
public:
explicit RegisterThisTest(bool IsDisabled=false)
{
TestType* p_test = new TestType;
if (IsDisabled)
p_test->Disable();
Tester::AddTestCase(p_test);
}
};

}




class KRATOS_API(KRATOS_CORE) TestCase
{
public:


TestCase() = delete;

TestCase(TestCase const& rOther) = delete;

TestCase(std::string const& Name);

virtual ~TestCase();


TestCase& operator=(TestCase const& rOther) = delete;


virtual void Reset();

virtual void ResetResult();

virtual void Setup();

virtual void Run();

virtual void Profile();

virtual void TearDown();

virtual void Enable();

virtual void Disable();

virtual void Select();

virtual void UnSelect();


const std::string& Name() const;

const TestCaseResult& GetResult() const;

void SetResult(TestCaseResult const& TheResult);

void SetResultOutput(std::string const& TheResultOutput);


virtual bool IsEnabled() const;

virtual bool IsDisabled() const;

virtual bool IsSelected() const;


virtual std::string Info() const;

virtual void PrintInfo(std::ostream& rOStream) const;

virtual void PrintData(std::ostream& rOStream) const;

private:


const std::string mName;

bool mIsEnabled;

bool mIsSelected;

TestCaseResult mResult;


virtual void TestFunction() = 0;

}; 



inline std::ostream& operator << (std::ostream& rOStream,
const TestCase& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


#define KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) \
Test##TestCaseName

#define KRATOS_TESTING_CONVERT_TO_STRING(Name) #Name

#define KRATOS_TESTING_TEST_CASE_CLASS_BODY(TestCaseName,ParentName) \
public:\
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)() : ParentName(KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName)) {}\
private: \
void TestFunction() override; \
static const Internals::RegisterThisTest< KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) > mDummy;

#define KRATOS_TESTING_TEST_CASE_CLASS(TestCaseName,ParentName) \
class KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) : public ParentName \
{\
KRATOS_TESTING_TEST_CASE_CLASS_BODY(TestCaseName,ParentName) \
};


#define KRATOS_TEST_CASE(TestCaseName) \
KRATOS_TESTING_TEST_CASE_CLASS(TestCaseName, TestCase) \
const Kratos::Testing::Internals::RegisterThisTest< KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) > \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy; \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_DISABLED_TEST_CASE(TestCaseName) \
KRATOS_TESTING_TEST_CASE_CLASS(TestCaseName, TestCase) \
const Kratos::Testing::Internals::RegisterThisTest< KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) > \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy(true); \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_TEST_FIXTURE(TestFixtureName) \
class TestFixtureName : public TestCase \
{\
public:\
TestFixtureName(std::string const& Name) : TestCase(Name) {}\
private: \
void Setup() override; \
void TearDown() override; \
};

#define KRATOS_TEST_FIXTURE_SETUP(TestFixtureName) \
void TestFixtureName::Setup()

#define KRATOS_TEST_FIXTURE_TEAR_DOWN(TestFixtureName) \
void TestFixtureName::TearDown()

#define KRATOS_TEST_CASE_WITH_FIXTURE(TestCaseName,TestFixtureName) \
KRATOS_TESTING_TEST_CASE_CLASS(TestCaseName, TestFixtureName)  \
const Kratos::Testing::Internals::RegisterThisTest< KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) > \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy; \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_DISABLED_TEST_CASE_WITH_FIXTURE(TestCaseName,TestFixtureName) \
KRATOS_TESTING_TEST_CASE_CLASS(TestCaseName, TestFixtureName)  \
const Kratos::Testing::Internals::RegisterThisTest< KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) > \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy(true); \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()


}  
