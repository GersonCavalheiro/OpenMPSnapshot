
#if !defined(KRATOS_MANAGE_TIME_STEP_PROCESS_H_INCLUDED)
#define  KRATOS_MANAGE_TIME_STEP_PROCESS_H_INCLUDED




#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class ManageTimeStepProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ManageTimeStepProcess);



ManageTimeStepProcess(ModelPart& rModelPart,
double& rMinDeltaTime, double& rMaxDeltaTime,
double& rReductionFactor, double& rIncreaseFactor,
double& rErrorTolerance, unsigned int& rMinIterations,
unsigned int& rMaxIterations, unsigned int& rNumberOfConstantSteps) : Process(Flags()),  mrModelPart(rModelPart), mMinDeltaTime(rMinDeltaTime), mMaxDeltaTime(rMaxDeltaTime), mReductionFactor(rReductionFactor), mIncreaseFactor(rIncreaseFactor), mErrorTolerance(rErrorTolerance), mMinIterations(rMinIterations), mMaxIterations(rMaxIterations), mNumberOfConstantSteps(rNumberOfConstantSteps)
{
}

ManageTimeStepProcess( ModelPart& rModelPart,
Parameters CustomParameters )
: mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters DefaultParameters( R"(
{
"time_step": 1.0,
"start_time": 0.0,
"end_time": 1.0,
"adaptive_time_step":{
"minimum_time_step": 0.1,
"maximum_time_step": 1.0,
"reduction_factor": 2.0,
"increase_factor": 1.0,
"error_tolerance": 1e-4,
"minimum_iterations": 2,
"maximum_iterations": 10,
"number_constant_steps": 4
}
}  )" );


mAdaptiveTimeStep = false;
if( CustomParameters.Has("adaptive_time_step") )
mAdaptiveTimeStep = true;

CustomParameters.ValidateAndAssignDefaults(DefaultParameters);

ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();

bool restarted = false;
if( rCurrentProcessInfo.Has(IS_RESTARTED) ){
if( rCurrentProcessInfo[IS_RESTARTED] == true ){
restarted = true;
}
}

if( !restarted ){
rCurrentProcessInfo.SetValue(DELTA_TIME, CustomParameters["time_step"].GetDouble());
rCurrentProcessInfo.SetValue(TIME, CustomParameters["start_time"].GetDouble());
}

mTime = rCurrentProcessInfo[TIME];
mStep = rCurrentProcessInfo[STEP];
mEndTime = CustomParameters["end_time"].GetDouble();

if( mAdaptiveTimeStep )
SetAdaptiveTimeParameters(CustomParameters["adaptive_time_step"]);

KRATOS_CATCH(" ")

}

virtual ~ManageTimeStepProcess() {}



void operator()()
{
Execute();
}




void Execute() override
{
KRATOS_TRY;







KRATOS_CATCH("");
}

void ExecuteInitialize() override
{
}

void ExecuteBeforeSolutionLoop() override
{
}


void ExecuteInitializeSolutionStep() override
{
ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();

if( mAdaptiveTimeStep )
PredictTimeStep();

double mTime = rCurrentProcessInfo[TIME] + rCurrentProcessInfo[DELTA_TIME];

mrModelPart.ProcessInfo[STEP] = (++mStep);

mrModelPart.CloneTimeStep(mTime);
}

void ExecuteFinalizeSolutionStep() override
{
}


void ExecuteBeforeOutputStep() override
{
}


void ExecuteAfterOutputStep() override
{
}


void ExecuteFinalize() override
{
}







std::string Info() const override
{
return "ManageTimeStepProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ManageTimeStepProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


ManageTimeStepProcess(ManageTimeStepProcess const& rOther);


private:


ModelPart& mrModelPart;

bool mAdaptiveTimStep;

double mTime;
double mStep;
double mEndTime;

double mMinDeltaTime;
double mMaxDeltaTime;

double mReductionFactor;
double mIncreaseFactor;

double mErrorTolerance;

unsigned int mMinIterations;
unsigned int mMaxIterations;
unsigned int mNumberOfConstantSteps;


void SetAdaptiveTimeParameters(Parameters CustomParameters)
{
KRATOS_TRY

Parameters DefaultParameters( R"(
{
"minimum_time_step": 0.1,
"maximum_time_step": 1.0,
"reduction_factor": 2.0,
"increase_factor": 1.0,
"error_tolerance": 1e-4,
"minimum_iterations": 2,
"maximum_iterations": 10,
"number_constant_steps": 4
}  )" );

CustomParameters.ValidateAndAssignDefaults(DefaultParameters);


mMinDeltaTime = CustomParameters["minimum_time_step"].GetDouble();
mMaxDeltaTime = CustomParameters["maximum_time_step"].GetDouble();

mReductionFactor = CustomParameters["reduction_factor"].GetDouble();
mIncreaseFactor = CustomParameters["increase_factor"].GetDouble();

mErrorTolerance = CustomParameters["error_tolerance"].GetDouble();

mMinIterations = CustomParameters["minimum_iterations"].GetInt();
mMaxIterations = CustomParameters["maximum_iterations"].GetInt();
mNumberOfConstantSteps = CustomParameters["number_constant_steps"].GetInt();

KRATOS_CATCH(" ")
}

ManageTimeStepProcess& operator=(ManageTimeStepProcess const& rOther);


}; 







inline std::istream& operator >> (std::istream& rIStream,
ManageTimeStepProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ManageTimeStepProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
