
#pragma once



#include "input_output/vtk_output.h"

namespace Kratos
{

class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) VtkEigenOutput : public VtkOutput
{
public:

KRATOS_CLASS_POINTER_DEFINITION(VtkEigenOutput);


explicit VtkEigenOutput(
ModelPart& rModelPart,
Parameters EigenOutputParameters,
Parameters VtkParameters)
: VtkOutput(rModelPart, VtkParameters),
mEigenOutputSettings(EigenOutputParameters) {};

virtual ~VtkEigenOutput() = default;


void PrintEigenOutput(
const std::string& rLabel,
const int AnimationStep,
const std::vector<const Variable<double>*>& rRequestedDoubleResults,
const std::vector<const Variable<array_1d<double,3>>*>& rRequestedVectorResults);


std::string Info() const override
{
return " VtkEigenOutput object ";
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << " VtkEigenOutput object " << std::endl;
}

void PrintData(std::ostream& rOStream) const override
{
}

private:

int mLastWrittenAnimationStepIndex = -1;
Parameters mEigenOutputSettings;


void OpenOutputFile(
const std::string& rFileName,
const std::ios::openmode OpenModeFlags,
std::ofstream& rOutputFile) const;

std::string GetEigenOutputFileName(const int AnimationStep) const;

void WriteScalarEigenVariable(
const ModelPart::NodesContainerType& rNodes,
const Variable<double>& rVariable,
const std::string& rLabel,
std::ofstream& rFileStream) const;

void WriteVectorEigenVariable(
const ModelPart::NodesContainerType& rNodes,
const Variable<array_1d<double, 3>>& rVariable,
const std::string& rLabel,
std::ofstream& rFileStream) const;

};

} 
