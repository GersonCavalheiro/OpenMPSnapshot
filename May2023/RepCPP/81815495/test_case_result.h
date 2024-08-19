
#pragma once

#include <string>
#include <iostream>


#include "includes/kratos_export_api.h"

namespace Kratos::Testing
{



class KRATOS_API(KRATOS_CORE) TestCaseResult
{
public:


TestCaseResult();

TestCaseResult(TestCaseResult const& rOther);

virtual ~TestCaseResult();



TestCaseResult& operator=(TestCaseResult const& rOther);


virtual void Reset();



void SetToSucceed();

void SetToFailed();

void SetToSkipped();

void SetOutput(const std::string& TheOutput);

const std::string& GetOutput() const;

void SetErrorMessage(const std::string& TheMessage);

const std::string& GetErrorMessage() const;

void SetSetupElapsedTime(double ElapsedTime);

double GetSetupElapsedTime() const;

void SetRunElapsedTime(double ElapsedTime);

double GetRunElapsedTime() const;

void SetTearDownElapsedTime(double ElapsedTime);

double GetTearDownElapsedTime() const;

void SetElapsedTime(double ElapsedTime);

double GetElapsedTime() const;




bool IsSucceed() const;

bool IsFailed() const;

bool IsSkipped() const;

bool IsRun() const;


virtual std::string Info() const;

virtual void PrintInfo(std::ostream& rOStream) const;

virtual void PrintData(std::ostream& rOStream) const;






private:


enum class Result {
DidNotRun,
Passed,
Failed,
Skipped
};




Result mStatus;
std::string mOutput;
std::string mErrorMessage;
double mSetupElapsedTime;
double mRunElapsedTime;
double mTearDownElapsedTime;
double mElapsedTime;



}; 



inline std::ostream& operator << (std::ostream& rOStream,
const TestCaseResult& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
