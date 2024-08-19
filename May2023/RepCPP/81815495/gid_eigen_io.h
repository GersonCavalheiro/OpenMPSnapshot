
#pragma once



#include "includes/gid_io.h"

namespace Kratos
{







class GidEigenIO : public GidIO<>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(GidEigenIO);

typedef std::size_t SizeType;


GidEigenIO( const std::string& rDatafilename,
GiD_PostMode Mode,
MultiFileFlag use_multiple_files_flag,
WriteDeformedMeshFlag write_deformed_flag,
WriteConditionsFlag write_conditions_flag) :
GidIO<>(rDatafilename,
Mode,
use_multiple_files_flag,
write_deformed_flag,
write_conditions_flag) { }





void WriteEigenResults( ModelPart& rModelPart,
const Variable<double>& rVariable,
std::string Label,
const SizeType NumberOfAnimationStep )
{
Label += "_" + rVariable.Name();
GiD_fBeginResult( mResultFile, (char*)Label.c_str() , "EigenVector_Animation",
NumberOfAnimationStep, GiD_Scalar,
GiD_OnNodes, NULL, NULL, 0, NULL );

for (const auto& r_node : rModelPart.Nodes())
{
const double& nodal_result = r_node.FastGetSolutionStepValue(rVariable);
GiD_fWriteScalar( mResultFile, r_node.Id(), nodal_result );
}

GiD_fEndResult(mResultFile);
}


void WriteEigenResults( ModelPart& rModelPart,
const Variable<array_1d<double, 3>>& rVariable,
std::string Label,
const SizeType NumberOfAnimationStep)
{
Label += "_" + rVariable.Name();
GiD_fBeginResult( mResultFile, (char*)Label.c_str() , "EigenVector_Animation",
NumberOfAnimationStep, GiD_Vector,
GiD_OnNodes, NULL, NULL, 0, NULL );

for (auto& r_node : rModelPart.Nodes())
{
const array_1d<double, 3>& nodal_result = r_node.FastGetSolutionStepValue(rVariable);
GiD_fWriteVector(mResultFile, r_node.Id(), nodal_result[0], nodal_result[1], nodal_result[2]);
}

GiD_fEndResult(mResultFile);
}







std::string Info() const override
{
std::stringstream buffer;
buffer << "GidEigenIO" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override {rOStream << "GidEigenIO";}

void PrintData(std::ostream& rOStream) const override {}





protected:















private:















}; 








}  

