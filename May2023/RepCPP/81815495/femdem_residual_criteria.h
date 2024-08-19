
#if !defined(KRATOS_FEMDEM_RESIDUAL_CRITERIA )
#define  KRATOS_FEMDEM_RESIDUAL_CRITERIA



#include "includes/model_part.h"
#include "includes/define.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

namespace Kratos
{






template<class TSparseSpace,
class TDenseSpace
>
class FemDemResidualCriteria
: public  ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( FemDemResidualCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;


/
bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
if (rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] == 1) {
KRATOS_INFO("") << "___________________________________________________________________" << std::endl;
KRATOS_INFO("") << "|    ITER     |     RATIO      |    ABS_NORM    |    CONVERGED    |" << std::endl;
}
const SizeType size_b = TSparseSpace::Size(rb);
if (size_b != 0) { 

SizeType size_residual;
CalculateResidualNorm(rModelPart, mCurrentResidualNorm, size_residual, rDofSet, rb);

TDataType ratio = 0.0;
if(mInitialResidualNorm < std::numeric_limits<TDataType>::epsilon()) {
ratio = 0.0;
} else {
ratio = mCurrentResidualNorm/mInitialResidualNorm;
}

const TDataType float_size_residual = static_cast<TDataType>(size_residual);
const TDataType absolute_norm = (mCurrentResidualNorm / float_size_residual);

rModelPart.GetProcessInfo()[CONVERGENCE_RATIO] = ratio;
rModelPart.GetProcessInfo()[RESIDUAL_NORM] = absolute_norm;

if (ratio <= mRatioTolerance || absolute_norm < mAlwaysConvergedNorm) { 
if (rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] < 10) {
std::cout <<"|      " << rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] << "      |  " 
<< std::scientific << ratio << "  |  " 
<< absolute_norm << "  |" << "      TRUE       |"<< std::endl;                
} else {
std::cout <<"|      " << rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] << "     |  "
<< std::scientific << ratio << "  |  " 
<< absolute_norm << "  |" << "      TRUE       |"<< std::endl;  
}
return true;
} else {
if (rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] < 10) {
std::cout <<"|      " << rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] << "      |  " 
<< std::scientific << ratio << "  |  " 
<< absolute_norm << "  |" << "      FALSE      |"<< std::endl;                
} else {
std::cout <<"|      " << rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] << "     |  " 
<< std::scientific << ratio << "  |  " 
<< absolute_norm << "  |" << "      FALSE      |"<< std::endl;  
}
if (rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] == mMaxIterations) {
KRATOS_INFO("") << " ATTENTION! SOLUTION STEP NOT CONVERGED AFTER " <<  mMaxIterations << "ITERATIONS" << std::endl;
}
return false;
}
} else {
return true;
}
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);
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
for (int i = 0; i<static_cast<int>(rDofSet.size()); ++i) {
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
KRATOS_INFO("") << "|_____________|________________|________________|_________________|" << std::endl;
KRATOS_INFO("") << "" << std::endl;
}





std::string Info() const override
{
return "FemDemResidualCriteria";
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

TDataType residual_dof_value = 0.0;
const auto it_dof_begin = rDofSet.begin();
const int number_of_dof = static_cast<int>(rDofSet.size());

if (rModelPart.NumberOfMasterSlaveConstraints() > 0) {
#pragma omp parallel for firstprivate(residual_dof_value) reduction(+:residual_solution_norm, dof_num)
for (int i = 0; i < number_of_dof; i++) {
auto it_dof = it_dof_begin + i;

const IndexType dof_id = it_dof->EquationId();

if (mActiveDofs[dof_id]) {
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





private:


TDataType mRatioTolerance;      

TDataType mInitialResidualNorm; 

TDataType mCurrentResidualNorm; 

TDataType mAlwaysConvergedNorm; 

TDataType mReferenceDispNorm;   

std::vector<bool> mActiveDofs;  

int mMaxIterations;







}; 





}  

#endif 
