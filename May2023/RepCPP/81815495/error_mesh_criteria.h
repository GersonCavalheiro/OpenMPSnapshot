
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "utilities/color_utilities.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

#include "custom_processes/spr_error_process.h"

namespace Kratos
{







template<class TSparseSpace, class TDenseSpace>
class ErrorMeshCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ErrorMeshCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >        BaseType;

typedef TSparseSpace                                     SparseSpaceType;

typedef typename BaseType::TDataType                           TDataType;

typedef typename BaseType::DofsArrayType                   DofsArrayType;

typedef typename BaseType::TSystemMatrixType           TSystemMatrixType;

typedef typename BaseType::TSystemVectorType           TSystemVectorType;

typedef ModelPart::ConditionsContainerType           ConditionsArrayType;

typedef ModelPart::NodesContainerType                     NodesArrayType;

typedef std::size_t                                              KeyType;

typedef std::size_t                                             SizeType;



explicit ErrorMeshCriteria(Parameters ThisParameters = Parameters(R"({})"))
: ConvergenceCriteria< TSparseSpace, TDenseSpace >(),
mThisParameters(ThisParameters)
{

Parameters default_parameters = Parameters(R"(
{
"error_mesh_tolerance" : 5.0e-3,
"error_mesh_constant"  : 5.0e-3,
"compute_error_extra_parameters":
{
"echo_level"                          : 0
}
})" );

mThisParameters.ValidateAndAssignDefaults(default_parameters);

mErrorTolerance = mThisParameters["error_mesh_tolerance"].GetDouble();
mConstantError = mThisParameters["error_mesh_constant"].GetDouble();

}

ErrorMeshCriteria( ErrorMeshCriteria const& rOther )
:BaseType(rOther)
,mErrorTolerance(rOther.mErrorTolerance)
,mConstantError(rOther.mConstantError)
{
}

~ErrorMeshCriteria() override = default;



void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& A,
const TSystemVectorType& Dx,
const TSystemVectorType& b
) override
{
const ProcessInfo& process_info = rModelPart.GetProcessInfo();

if (process_info[DOMAIN_SIZE] == 2) {
SPRErrorProcess<2> compute_error_process = SPRErrorProcess<2>(rModelPart, mThisParameters["compute_error_extra_parameters"]);
compute_error_process.Execute();
} else {
SPRErrorProcess<3> compute_error_process = SPRErrorProcess<3>(rModelPart, mThisParameters["compute_error_extra_parameters"]);
compute_error_process.Execute();
}

const double estimated_error = process_info[ERROR_RATIO];

const bool converged_error = (estimated_error > mErrorTolerance) ? false : true;

if (converged_error) {
KRATOS_INFO_IF("ErrorMeshCriteria", rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) << "NL ITERATION: " << process_info[NL_ITERATION_NUMBER] << "\tThe error due to the mesh size: " << estimated_error << " is under the tolerance prescribed: " << mErrorTolerance << ". " << BOLDFONT(FGRN("No remeshing required")) << std::endl;
} else {
KRATOS_INFO_IF("ErrorMeshCriteria", rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0)
<< "NL ITERATION: " << process_info[NL_ITERATION_NUMBER] << "\tThe error due to the mesh size: " << estimated_error << " is bigger than the tolerance prescribed: " << mErrorTolerance << ". "<< BOLDFONT(FRED("Remeshing required")) << std::endl;
}

return converged_error;
}





protected:








private:


Parameters mThisParameters; 

double mErrorTolerance;     
double mConstantError;      







}; 


}  

