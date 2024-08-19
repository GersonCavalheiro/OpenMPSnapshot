
#pragma once

#include <vector>


#include "testing/test_case.h"
#include "testing/distributed_test_case.h"

namespace Kratos::Testing {
namespace Internals {
class AddThisTestToTestSuite {
public:
AddThisTestToTestSuite(
std::string const& TestName, std::string const& TestSuiteName) {
Tester::AddTestToTestSuite(TestName, TestSuiteName);
}
};
}




class KRATOS_API(KRATOS_CORE) TestSuite : public TestCase {
public:

typedef std::vector<TestCase*> TestCasesContainerType;


TestSuite() = delete;

TestSuite(TestSuite const& rOther) = delete;

explicit TestSuite(std::string const& Name);

virtual ~TestSuite();


TestSuite& operator=(TestSuite const& rOther) = delete;


void AddTestCase(TestCase* pTestCase);

void Reset() override;

void ResetResult() override;

void Run() override;

void Profile() override;

void Enable() override;

void Disable() override;

void Select() override;

void UnSelect() override;




virtual std::string Info() const override;

virtual void PrintInfo(std::ostream& rOStream) const override;

virtual void PrintData(std::ostream& rOStream) const override;

private:

TestCasesContainerType mTestCases;


void TestFunction() override;


};  



inline std::istream& operator>>(std::istream& rIStream, TestSuite& rThis);

inline std::ostream& operator<<(
std::ostream& rOStream, const TestSuite& rThis) {
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

#define KRATOS_TESTING_CONCATENATE(a, b) a##b

#define KRATOS_TESTING_CREATE_DUMMY_VARIABLE_NAME(prefix, UniqueNumber) \
KRATOS_TESTING_CONCATENATE(prefix, UniqueNumber)

#define KRATOS_TESTING_ADD_TEST_TO_TEST_SUITE(TestName, TestSuiteName) \
Kratos::Testing::Internals::AddThisTestToTestSuite                 \
KRATOS_TESTING_CREATE_DUMMY_VARIABLE_NAME(dummy, __LINE__)(    \
TestName, TestSuiteName)

#define KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, ParentName)      \
class KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName) : public ParentName { \
KRATOS_TESTING_TEST_CASE_CLASS_BODY(TestCaseName, ParentName)          \
static const Kratos::Testing::Internals::AddThisTestToTestSuite        \
mAnotherDummy;                                                     \
};

#define KRATOS_TEST_CASE_IN_SUITE(TestCaseName, TestSuiteName)          \
KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, TestCase)     \
const Kratos::Testing::Internals::RegisterThisTest<                 \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)>                 \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy;         \
const Kratos::Testing::Internals::AddThisTestToTestSuite            \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mAnotherDummy = \
Kratos::Testing::Internals::AddThisTestToTestSuite(         \
KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName),   \
KRATOS_TESTING_CONVERT_TO_STRING(TestSuiteName));       \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_DISABLED_TEST_CASE_IN_SUITE(TestCaseName, TestSuiteName) \
KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, TestCase)     \
const Kratos::Testing::Internals::RegisterThisTest<                 \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)>                 \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy(true);   \
const Kratos::Testing::Internals::AddThisTestToTestSuite            \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mAnotherDummy = \
Kratos::Testing::Internals::AddThisTestToTestSuite(         \
KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName),   \
KRATOS_TESTING_CONVERT_TO_STRING(TestSuiteName));       \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_DISTRIBUTED_TEST_CASE_IN_SUITE(TestCaseName, TestSuiteName)     \
KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, DistributedTestCase) \
const Kratos::Testing::Internals::RegisterThisTest<                        \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)>                        \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy;                \
const Kratos::Testing::Internals::AddThisTestToTestSuite                   \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mAnotherDummy =        \
Kratos::Testing::Internals::AddThisTestToTestSuite(                \
KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName),          \
KRATOS_TESTING_CONVERT_TO_STRING(TestSuiteName));              \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_DISABLED_DISTRIBUTED_TEST_CASE_IN_SUITE(TestCaseName, TestSuiteName) \
KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, DistributedTestCase)      \
const Kratos::Testing::Internals::RegisterThisTest<                             \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)>                             \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy(true);               \
const Kratos::Testing::Internals::AddThisTestToTestSuite                        \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mAnotherDummy =             \
Kratos::Testing::Internals::AddThisTestToTestSuite(                     \
KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName),               \
KRATOS_TESTING_CONVERT_TO_STRING(TestSuiteName));                   \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_TEST_CASE_WITH_FIXTURE_IN_SUITE(                            \
TestCaseName, TestFixtureName, TestSuiteName)                          \
KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, TestFixtureName) \
const Kratos::Testing::Internals::RegisterThisTest<                    \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)>                    \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy;            \
const Kratos::Testing::Internals::AddThisTestToTestSuite               \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mAnotherDummy =    \
Kratos::Testing::Internals::AddThisTestToTestSuite(            \
KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName),      \
KRATOS_TESTING_CONVERT_TO_STRING(TestSuiteName));          \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()

#define KRATOS_DISABLED_TEST_CASE_WITH_FIXTURE_IN_SUITE(                   \
TestCaseName, TestFixtureName, TestSuiteName)                          \
KRATOS_TESTING_TEST_CASE_IN_SUITE_CLASS(TestCaseName, TestFixtureName) \
const Kratos::Testing::Internals::RegisterThisTest<                    \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)>                    \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mDummy(true);      \
const Kratos::Testing::Internals::AddThisTestToTestSuite               \
KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::mAnotherDummy =    \
Kratos::Testing::Internals::AddThisTestToTestSuite(            \
KRATOS_TESTING_CONVERT_TO_STRING(Test##TestCaseName),      \
KRATOS_TESTING_CONVERT_TO_STRING(TestSuiteName));          \
\
void KRATOS_TESTING_CREATE_CLASS_NAME(TestCaseName)::TestFunction()


}  
