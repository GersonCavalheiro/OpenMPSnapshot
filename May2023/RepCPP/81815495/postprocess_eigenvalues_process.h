
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "includes/kratos_parameters.h"

namespace Kratos {




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) PostprocessEigenvaluesProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(PostprocessEigenvaluesProcess);

typedef std::size_t SizeType;

typedef ModelPart::NodeType::DofsContainerType DofsContainerType;


PostprocessEigenvaluesProcess(
Model& rModel,
Parameters OutputParameters);


void ExecuteFinalizeSolutionStep() override;


const Parameters GetDefaultParameters() const override;


virtual std::string Info() const override {
return "PostprocessEigenvaluesProcess";
}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << "PostprocessEigenvaluesProcess";
}

void PrintData(std::ostream& rOStream) const override {
}


private:

ModelPart* mpModelPart;
Parameters mOutputParameters;


std::string GetLabel(const int NumberOfEigenValue,
const int NumberOfEigenvalues,
const double EigenValueSolution) const;

void GetVariables(std::vector<const Variable<double>*>& rRequestedDoubleResults,
std::vector<const Variable<array_1d<double,3>>*>& rRequestedVectorResults) const;


}; 


}  
