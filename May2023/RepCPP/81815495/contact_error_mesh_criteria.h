
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "utilities/variable_utils.h"
#include "utilities/color_utilities.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

#include "custom_processes/contact_spr_error_process.h"

namespace Kratos
{







template<class TSparseSpace, class TDenseSpace>
class ContactErrorMeshCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ContactErrorMeshCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >                       BaseType;

typedef ContactErrorMeshCriteria< TSparseSpace, TDenseSpace >                 ClassType;

typedef typename BaseType::DofsArrayType                                  DofsArrayType;

typedef typename BaseType::TSystemMatrixType                          TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                          TSystemVectorType;




explicit ContactErrorMeshCriteria()
: BaseType()
{
}

explicit ContactErrorMeshCriteria(Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}

ContactErrorMeshCriteria( ContactErrorMeshCriteria const& rOther )
:BaseType(rOther)
,mErrorTolerance(rOther.mErrorTolerance)
,mConstantError(rOther.mConstantError)
{
}

~ContactErrorMeshCriteria() override = default;




typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);

ModelPart& r_contact_model_part = rModelPart.GetSubModelPart("Contact");
VariableUtils().SetFlag(CONTACT, true, r_contact_model_part.Nodes());
VariableUtils().SetFlag(CONTACT, true, r_contact_model_part.Conditions());
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
const ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (r_process_info[DOMAIN_SIZE] == 2) {
auto compute_error_process = ContactSPRErrorProcess<2>(rModelPart, mThisParameters["compute_error_extra_parameters"]);
compute_error_process.Execute();
} else {
auto compute_error_process = ContactSPRErrorProcess<3>(rModelPart, mThisParameters["compute_error_extra_parameters"]);
compute_error_process.Execute();
}

const double estimated_error = r_process_info[ERROR_RATIO];

const bool converged_error = (estimated_error > mErrorTolerance) ? false : true;

if (converged_error) {
KRATOS_INFO_IF("ContactErrorMeshCriteria", rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) << "NL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << "\tThe error due to the mesh size: " << estimated_error << " is under the tolerance prescribed: " << mErrorTolerance << ". " << BOLDFONT(FGRN("No remeshing required")) << std::endl;
} else {
KRATOS_INFO_IF("ContactErrorMeshCriteria", rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0)
<< "NL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << "\tThe error due to the mesh size: " << estimated_error << " is bigger than the tolerance prescribed: " << mErrorTolerance << ". "<< BOLDFONT(FRED("Remeshing required")) << std::endl;
}

return converged_error;
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                 : "contact_error_mesh_criteria",
"error_mesh_tolerance" : 5.0e-3,
"error_mesh_constant"  : 5.0e-3,
"compute_error_extra_parameters":
{
"penalty_normal"       : 1.0e4,
"penalty_tangential"   : 1.0e4,
"echo_level"           : 0
}
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "contact_error_mesh_criteria";
}




std::string Info() const override
{
return "ContactErrorMeshCriteria";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info();
}


protected:






void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
mThisParameters = ThisParameters;
mErrorTolerance = mThisParameters["error_mesh_tolerance"].GetDouble();
mConstantError = mThisParameters["error_mesh_constant"].GetDouble();
}




private:


Parameters mThisParameters; 

double mErrorTolerance;     
double mConstantError;      







}; 


}  

