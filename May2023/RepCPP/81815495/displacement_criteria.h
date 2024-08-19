
#if !defined(KRATOS_DISPLACEMENT_CRITERIA )
#define  KRATOS_DISPLACEMENT_CRITERIA






#include "includes/model_part.h"
#include "includes/define.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

namespace Kratos
{






template<class TSparseSpace,
class TDenseSpace
>
class DisplacementCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( DisplacementCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef DisplacementCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef TSparseSpace SparseSpaceType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;


/
explicit DisplacementCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit DisplacementCriteria(
TDataType NewRatioTolerance,
TDataType AlwaysConvergedNorm)
: BaseType(),
mRatioTolerance(NewRatioTolerance),
mAlwaysConvergedNorm(AlwaysConvergedNorm)
{
}


explicit DisplacementCriteria( DisplacementCriteria const& rOther )
:BaseType(rOther)
,mRatioTolerance(rOther.mRatioTolerance)
,mAlwaysConvergedNorm(rOther.mAlwaysConvergedNorm)
,mReferenceDispNorm(rOther.mReferenceDispNorm)
{
}


~DisplacementCriteria() override {}





typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& A,
const TSystemVectorType& Dx,
const TSystemVectorType& b
) override
{
const TDataType approx_zero_tolerance = std::numeric_limits<TDataType>::epsilon();
const SizeType size_Dx = Dx.size();
if (size_Dx != 0) { 
SizeType size_solution;
TDataType final_correction_norm = CalculateFinalCorrectionNorm(size_solution, rDofSet, Dx);

TDataType ratio = 0.0;

CalculateReferenceNorm(rDofSet);
if (mReferenceDispNorm < approx_zero_tolerance) {
KRATOS_WARNING("DisplacementCriteria") << "NaN norm is detected. Setting reference to convergence criteria" << std::endl;
mReferenceDispNorm = final_correction_norm;
}

if(final_correction_norm < approx_zero_tolerance) {
ratio = 0.0;
} else {
ratio = final_correction_norm/mReferenceDispNorm;
}

const TDataType float_size_solution = static_cast<TDataType>(size_solution);

const TDataType absolute_norm = (final_correction_norm/std::sqrt(float_size_solution));

KRATOS_INFO_IF("DISPLACEMENT CRITERION", this->GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0) << " :: [ Obtained ratio = " << ratio << "; Expected ratio = " << mRatioTolerance << "; Absolute norm = " << absolute_norm << "; Expected norm =  " << mAlwaysConvergedNorm << "]" << std::endl;

rModelPart.GetProcessInfo()[CONVERGENCE_RATIO] = ratio;
rModelPart.GetProcessInfo()[RESIDUAL_NORM] = absolute_norm;

if ( ratio <= mRatioTolerance  ||  absolute_norm<mAlwaysConvergedNorm )  { 
KRATOS_INFO_IF("DISPLACEMENT CRITERION", this->GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0) << "Convergence is achieved" << std::endl;
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
const TSystemMatrixType& A,
const TSystemVectorType& Dx,
const TSystemVectorType& b
) override
{
BaseType::InitializeSolutionStep(rModelPart, rDofSet, A, Dx, b);
}


void FinalizeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& A,
const TSystemVectorType& Dx,
const TSystemVectorType& b
) override
{
BaseType::FinalizeSolutionStep(rModelPart, rDofSet, A, Dx, b);
}


static std::string Name()
{
return "displacement_criteria";
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                            : "displacement_criteria",
"displacement_relative_tolerance" : 1.0e-4,
"displacement_absolute_tolerance" : 1.0e-9
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}





std::string Info() const override
{
return "DisplacementCriteria";
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
mAlwaysConvergedNorm = ThisParameters["displacement_absolute_tolerance"].GetDouble();
mRatioTolerance = ThisParameters["displacement_relative_tolerance"].GetDouble();
}







private:



TDataType mRatioTolerance;      

TDataType mAlwaysConvergedNorm; 

TDataType mReferenceDispNorm;   




void CalculateReferenceNorm(DofsArrayType& rDofSet)
{
TDataType reference_disp_norm = TDataType();
TDataType dof_value;

#pragma omp parallel for reduction(+:reference_disp_norm)
for (int i = 0; i < static_cast<int>(rDofSet.size()); i++) {
auto it_dof = rDofSet.begin() + i;

if(it_dof->IsFree()) {
dof_value = it_dof->GetSolutionStepValue();
reference_disp_norm += dof_value * dof_value;
}
}
mReferenceDispNorm = std::sqrt(reference_disp_norm);
}


TDataType CalculateFinalCorrectionNorm(
SizeType& rDofNum,
DofsArrayType& rDofSet,
const TSystemVectorType& Dx
)
{
TDataType final_correction_norm = TDataType();
SizeType dof_num = 0;

#pragma omp parallel for reduction(+:final_correction_norm,dof_num)
for (int i = 0; i < static_cast<int>(rDofSet.size()); i++) {
auto it_dof = rDofSet.begin() + i;

IndexType dof_id;
TDataType variation_dof_value;

if (it_dof->IsFree()) {
dof_id = it_dof->EquationId();
variation_dof_value = Dx[dof_id];
final_correction_norm += std::pow(variation_dof_value, 2);
dof_num++;
}
}

rDofNum = dof_num;
return std::sqrt(final_correction_norm);
}










}; 





}  

#endif 

