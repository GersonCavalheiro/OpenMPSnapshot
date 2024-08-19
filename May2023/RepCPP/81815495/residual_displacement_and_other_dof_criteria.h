
#pragma once



#include "includes/model_part.h"
#include "includes/define.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace
>
class ResidualDisplacementAndOtherDoFCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ResidualDisplacementAndOtherDoFCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef TSparseSpace SparseSpaceType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;




ResidualDisplacementAndOtherDoFCriteria(
TDataType RatioTolerance,
TDataType AbsoluteTolerance,
const std::string& OtherDoFName = "ROTATION"
)
: ConvergenceCriteria< TSparseSpace, TDenseSpace >(),
mOtherDoFName(OtherDoFName),
mRatioTolerance(RatioTolerance),
mAbsoluteTolerance(AbsoluteTolerance)
{
this->mActualizeRHSIsNeeded = true;
}

/

bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
if (TSparseSpace::Size(rb) != 0) { 
TDataType ratio_displacement = 0.0;
TDataType ratio_other_dof     = 0.0;

SizeType disp_size;
CalculateResidualNorm(rModelPart, mCurrentResidualDispNorm, mCurrentResidualOtherDoFNorm, disp_size, rDofSet, rb);

if (mInitialResidualDispNorm == 0.0) {
ratio_displacement = 0.0;
} else {
ratio_displacement = mCurrentResidualDispNorm/mInitialResidualDispNorm;
}

if (mInitialResidualOtherDoFNorm == 0.0) {
ratio_other_dof = 0.0;
} else {
ratio_other_dof = mCurrentResidualOtherDoFNorm/mInitialResidualOtherDoFNorm;
}

const std::size_t system_size = TSparseSpace::Size(rb);
const TDataType absolute_norm_disp      = (mCurrentResidualDispNorm/static_cast<TDataType>(disp_size));
const TDataType absolute_norm_other_dof = (mCurrentResidualOtherDoFNorm/static_cast<TDataType>(system_size - disp_size));

KRATOS_INFO_IF("ResidualDisplacementAndOtherDoFCriteria", this->GetEchoLevel() > 0) << "RESIDUAL DISPLACEMENT CRITERION :: Ratio = "<< ratio_displacement  << ";  Norm = " << absolute_norm_disp << std::endl;
KRATOS_INFO_IF("ResidualDisplacementAndOtherDoFCriteria", this->GetEchoLevel() > 0) << "RESIDUAL " << mOtherDoFName << " CRITERION :: Ratio = "<< ratio_other_dof  << ";  Norm = " << absolute_norm_other_dof << std::endl;

rModelPart.GetProcessInfo()[CONVERGENCE_RATIO] = ratio_displacement;
rModelPart.GetProcessInfo()[RESIDUAL_NORM] = absolute_norm_disp;

if ((ratio_displacement <= mRatioTolerance || absolute_norm_disp < mAbsoluteTolerance) && (ratio_other_dof <= mRatioTolerance || absolute_norm_other_dof < mAbsoluteTolerance)) {
KRATOS_INFO_IF("ResidualDisplacementAndOtherDoFCriteria", this->GetEchoLevel() > 0) << "Convergence is achieved" << std::endl;
return true;
} else {
return false;
}
} else {
return true;
}
}



void Initialize(
ModelPart& rModelPart
) override
{
BaseType::mConvergenceCriteriaIsInitialized = true;
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
mActiveDofs.resize(rDofSet.size());

#pragma omp parallel for
for(int i=0; i<static_cast<int>(mActiveDofs.size()); ++i) {
mActiveDofs[i] = true;
}

#pragma omp parallel for
for (int i=0; i<static_cast<int>(rDofSet.size()); ++i) {
const auto it_dof = rDofSet.begin() + i;
if (it_dof->IsFixed()) {
mActiveDofs[it_dof->EquationId()] = false;
}
}

for (const auto& r_mpc : rModelPart.MasterSlaveConstraints()) {
for (const auto& r_dof : r_mpc.GetMasterDofsVector()) {
mActiveDofs[r_dof->EquationId()] = false;
}
for (const auto& r_dof : r_mpc.GetSlaveDofsVector()) {
mActiveDofs[r_dof->EquationId()] = false;
}
}
}

SizeType size_residual;
CalculateResidualNorm(rModelPart, mInitialResidualDispNorm, mInitialResidualOtherDoFNorm, size_residual, rDofSet, rb);
}






protected:








private:


std::string mOtherDoFName;                

TDataType mInitialResidualDispNorm;       
TDataType mCurrentResidualDispNorm;       
TDataType mInitialResidualOtherDoFNorm;   
TDataType mCurrentResidualOtherDoFNorm;   

TDataType mRatioTolerance;                
TDataType mAbsoluteTolerance;             

std::vector<bool> mActiveDofs;  




virtual void CalculateResidualNorm(
ModelPart& rModelPart,
TDataType& rResidualSolutionNormDisp,
TDataType& rResidualSolutionNormOtherDof,
SizeType& rDofNumDisp,
DofsArrayType& rDofSet,
const TSystemVectorType& rb
)
{
TDataType residual_solution_norm_disp = TDataType();
TDataType residual_solution_norm_other_dof = TDataType();
SizeType disp_dof_num = 0;

TDataType residual_dof_value = 0.0;
const auto it_dof_begin = rDofSet.begin();
const int number_of_dof = static_cast<int>(rDofSet.size());

if (rModelPart.NumberOfMasterSlaveConstraints() > 0) {
#pragma omp parallel for firstprivate(residual_dof_value) reduction(+:residual_solution_norm_disp, residual_solution_norm_other_dof, disp_dof_num)
for (int i = 0; i < number_of_dof; i++) {
auto it_dof = it_dof_begin + i;

const IndexType dof_id = it_dof->EquationId();
residual_dof_value = TSparseSpace::GetValue(rb,dof_id);

if (mActiveDofs[dof_id]) {
if (it_dof->GetVariable() == DISPLACEMENT_X || it_dof->GetVariable() == DISPLACEMENT_Y || it_dof->GetVariable() == DISPLACEMENT_Z) {
residual_solution_norm_disp += std::pow(residual_dof_value, 2);
disp_dof_num++;
} else {
residual_solution_norm_other_dof += std::pow(residual_dof_value, 2);
}
}
}
} else {
#pragma omp parallel for firstprivate(residual_dof_value) reduction(+:residual_solution_norm_disp, residual_solution_norm_other_dof, disp_dof_num)
for (int i = 0; i < number_of_dof; i++) {
auto it_dof = it_dof_begin + i;

if (!it_dof->IsFixed()) {
const IndexType dof_id = it_dof->EquationId();
residual_dof_value = TSparseSpace::GetValue(rb,dof_id);

if (it_dof->GetVariable() == DISPLACEMENT_X || it_dof->GetVariable() == DISPLACEMENT_Y || it_dof->GetVariable() == DISPLACEMENT_Z) {
residual_solution_norm_disp += std::pow(residual_dof_value, 2);
disp_dof_num++;
} else {
residual_solution_norm_other_dof += std::pow(residual_dof_value, 2);
}
}
}
}

rDofNumDisp = disp_dof_num;
rResidualSolutionNormDisp = std::sqrt(residual_solution_norm_disp);
rResidualSolutionNormOtherDof = std::sqrt(residual_solution_norm_other_dof);
}





}; 





}  

