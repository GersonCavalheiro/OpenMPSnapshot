
#pragma once



#include "includes/model_part.h"
#include "includes/define.h"
#include "utilities/constraint_utilities.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

namespace Kratos
{






template<class TSparseSpace,
class TDenseSpace
>
class ResidualCriteria
: public  ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ResidualCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef ResidualCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;


/
explicit ResidualCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);

this->mActualizeRHSIsNeeded = true;
}

/
typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
const SizeType size_b = TSparseSpace::Size(rb);
if (size_b != 0) { 

SizeType size_residual;
CalculateResidualNorm(rModelPart, mCurrentResidualNorm, size_residual, rDofSet, rb);

TDataType ratio{};
if(mInitialResidualNorm < std::numeric_limits<TDataType>::epsilon()) {
ratio = 0.0;
} else {
ratio = mCurrentResidualNorm/mInitialResidualNorm;
}

const TDataType float_size_residual = static_cast<TDataType>(size_residual);
const TDataType absolute_norm = (mCurrentResidualNorm/float_size_residual);

KRATOS_INFO_IF("RESIDUAL CRITERION", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << " :: [ Initial residual norm = " << mInitialResidualNorm << "; Current residual norm =  " << mCurrentResidualNorm << "]" << std::endl;
KRATOS_INFO_IF("RESIDUAL CRITERION", this->GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0) << " :: [ Obtained ratio = " << ratio << "; Expected ratio = " << mRatioTolerance << "; Absolute norm = " << absolute_norm << "; Expected norm =  " << mAlwaysConvergedNorm << "]" << std::endl;

rModelPart.GetProcessInfo()[CONVERGENCE_RATIO] = ratio;
rModelPart.GetProcessInfo()[RESIDUAL_NORM] = absolute_norm;

if (ratio <= mRatioTolerance || absolute_norm < mAlwaysConvergedNorm) {
KRATOS_INFO_IF("RESIDUAL CRITERION", this->GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0) << "Convergence is achieved" << std::endl;
return true;
} else {
return false;
}
} else {
return true;
}
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);
KRATOS_ERROR_IF(rModelPart.IsDistributed() && rModelPart.NumberOfMasterSlaveConstraints() > 0) << "This Criteria does not yet support constraints in MPI!" << std::endl;
}


void InitializeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::InitializeSolutionStep(rModelPart, rDofSet, rA, rDx, rb);

if (rModelPart.NumberOfMasterSlaveConstraints() > 0) {
ConstraintUtilities::ComputeActiveDofs(rModelPart, mActiveDofs, rDofSet);
}

SizeType size_residual;
CalculateResidualNorm(rModelPart, mInitialResidualNorm, size_residual, rDofSet, rb);
}


void FinalizeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::FinalizeSolutionStep(rModelPart, rDofSet, rA, rDx, rb);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                        : "residual_criteria",
"residual_absolute_tolerance" : 1.0e-4,
"residual_relative_tolerance" : 1.0e-9
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "residual_criteria";
}




std::string Info() const override
{
return "ResidualCriteria";
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





virtual void CalculateResidualNorm(
ModelPart& rModelPart,
TDataType& rResidualSolutionNorm,
SizeType& rDofNum,
DofsArrayType& rDofSet,
const TSystemVectorType& rb
)
{
TDataType residual_solution_norm = TDataType();
SizeType dof_num = 0;

TDataType residual_dof_value{};
const auto it_dof_begin = rDofSet.begin();
const int number_of_dof = static_cast<int>(rDofSet.size());

if (rModelPart.NumberOfMasterSlaveConstraints() > 0) {
#pragma omp parallel for firstprivate(residual_dof_value) reduction(+:residual_solution_norm, dof_num)
for (int i = 0; i < number_of_dof; i++) {
auto it_dof = it_dof_begin + i;

const IndexType dof_id = it_dof->EquationId();

if (mActiveDofs[dof_id] == 1) {
residual_dof_value = TSparseSpace::GetValue(rb,dof_id);
residual_solution_norm += std::pow(residual_dof_value, 2);
dof_num++;
}
}
} else {
#pragma omp parallel for firstprivate(residual_dof_value) reduction(+:residual_solution_norm, dof_num)
for (int i = 0; i < number_of_dof; i++) {
auto it_dof = it_dof_begin + i;

if (!it_dof->IsFixed()) {
const IndexType dof_id = it_dof->EquationId();
residual_dof_value = TSparseSpace::GetValue(rb,dof_id);
residual_solution_norm += std::pow(residual_dof_value, 2);
dof_num++;
}
}
}

rDofNum = dof_num;
rResidualSolutionNorm = std::sqrt(residual_solution_norm);
}


void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
mAlwaysConvergedNorm = ThisParameters["residual_absolute_tolerance"].GetDouble();
mRatioTolerance = ThisParameters["residual_relative_tolerance"].GetDouble();
}





private:


TDataType mRatioTolerance{};      

TDataType mInitialResidualNorm{}; 

TDataType mCurrentResidualNorm{}; 

TDataType mAlwaysConvergedNorm{}; 

TDataType mReferenceDispNorm{};   

std::vector<int> mActiveDofs;     







}; 




}  
