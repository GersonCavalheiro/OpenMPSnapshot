
#pragma once

#include <string>
#include <iostream>
#include <map>


#include "includes/kratos_export_api.h"

namespace Kratos::Testing
{



class TestCase;  
class TestSuite; 


class KRATOS_API(KRATOS_CORE) Tester
{
public:

using TestCasesContainerType = std::map<std::string, TestCase*>;

using TestSuitesContainerType = std::map<std::string, TestSuite*>;


enum class Verbosity {QUITE, PROGRESS, TESTS_LIST, FAILED_TESTS_OUTPUTS, TESTS_OUTPUTS};


Tester(Tester const& rOther) = delete;

virtual ~Tester();


Tester& operator=(Tester const& rOther) = delete;


static void ResetAllTestCasesResults();

static int RunAllTestCases();

static int ProfileAllTestCases();

static int RunTestSuite(std::string const& TestSuiteName);

static int ProfileTestSuite(std::string const& TestSuiteName);

static int RunTestCases(std::string const& TestCasesNamePattern);

static int ProfileTestCases(std::string const& TestCasesNamePattern);

static std::size_t NumberOfFailedTestCases();

static std::size_t NumberOfSkippedTestCases();

static void AddTestCase(TestCase* pHeapAllocatedTestCase);

static TestCase& GetTestCase(std::string const& TestCaseName);

static TestCase* pGetTestCase(std::string const& TestCaseName);

static TestSuite* CreateTestSuite(std::string const& TestSuiteName);

static TestSuite* CreateNewTestSuite(std::string const& TestSuiteName);

static void AddTestSuite(TestSuite* pHeapAllocatedTestSuite);

static TestSuite& GetTestSuite(std::string const& TestSuiteName);

static TestSuite* pGetTestSuite(std::string const& TestSuiteName);

static void AddTestToTestSuite(std::string const& TestName, std::string const& TestSuiteName);


static Tester& GetInstance();

static void SetVerbosity(Verbosity TheVerbosity);


static bool HasTestCase(std::string const& TestCaseName);

static bool HasTestSuite(std::string const& TestSuiteName);


virtual std::string Info() const;

virtual void PrintInfo(std::ostream& rOStream) const;

virtual void PrintData(std::ostream& rOStream) const;

private:

Tester();


TestCasesContainerType mTestCases;

TestSuitesContainerType mTestSuites;

Verbosity mVerbosity;



static void UnSelectAllTestCases();


static void SelectOnlyEnabledTestCases();


static void SelectTestCasesByPattern(std::string const& rTestCasesNamePattern);


static int RunSelectedTestCases();


static int ProfileSelectedTestCases();


static std::size_t NumberOfSelectedTestCases();


static void StartShowProgress(
const std::size_t Current,
const std::size_t Total,
const TestCase* pTheTestCase
);


static void EndShowProgress(
const std::size_t Current,
const std::size_t Total,
const TestCase* pTheTestCase
);


static int ReportResults(
std::ostream& rOStream,
const std::size_t NumberOfRunTests,
const double ElapsedTime
);


static void ReportFailures(std::ostream& rOStream);


static void ReportDistributedFailureDetails(
std::ostream& rOStream,
const TestCase* pTheTestCase
);

}; 



inline std::ostream& operator << (std::ostream& rOStream,
const Tester& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 
