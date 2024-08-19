
#pragma once



#include "includes/define.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "containers/array_1d.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{



























template<class TSparseSpace,
class TDenseSpace 
>
class ResidualBasedRelaxationScheme : public Scheme<TSparseSpace, TDenseSpace>
{
public:



KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedRelaxationScheme );

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename Element::DofsVectorType DofsVectorType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;







ResidualBasedRelaxationScheme(double NewAlphaBossak, double damping_factor)
: Scheme<TSparseSpace, TDenseSpace>()
{
mAlphaBossak = NewAlphaBossak;
mBetaNewmark = 0.25 * pow((1.00 - mAlphaBossak), 2);
mGammaNewmark = 0.5 - mAlphaBossak;

mdamping_factor = damping_factor;

const int num_threads = ParallelUtilities::GetNumThreads();
mMass.resize(num_threads);
mDamp.resize(num_threads);
mvel.resize(num_threads);

}


~ResidualBasedRelaxationScheme() override
{
}








/
void CalculateSystemContributions(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo
) override
{
KRATOS_TRY
int k = OpenMPUtils::ThisThread();
rCurrentElement.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);
rCurrentElement.CalculateMassMatrix(mMass[k], CurrentProcessInfo);
rCurrentElement.CalculateDampingMatrix(mDamp[k], CurrentProcessInfo);
rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);

AddDynamicsToLHS(LHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

AddDynamicsToRHS(rCurrentElement, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

KRATOS_CATCH( "" )

}

void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
int k = OpenMPUtils::ThisThread();

rCurrentElement.CalculateRightHandSide(RHS_Contribution, CurrentProcessInfo);
rCurrentElement.CalculateMassMatrix(mMass[k], CurrentProcessInfo);
rCurrentElement.CalculateDampingMatrix(mDamp[k], CurrentProcessInfo);
rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);

AddDynamicsToRHS(rCurrentElement, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

}


void CalculateSystemContributions(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY
int k = OpenMPUtils::ThisThread();
rCurrentCondition.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);
rCurrentCondition.CalculateMassMatrix(mMass[k], CurrentProcessInfo);
rCurrentCondition.CalculateDampingMatrix(mDamp[k], CurrentProcessInfo);
rCurrentCondition.EquationIdVector(EquationId, CurrentProcessInfo);

AddDynamicsToLHS(LHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

AddDynamicsToRHS(rCurrentCondition, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

KRATOS_CATCH( "" )
}

void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY
int k = OpenMPUtils::ThisThread();

rCurrentCondition.CalculateRightHandSide(RHS_Contribution, CurrentProcessInfo);
rCurrentCondition.CalculateMassMatrix(mMass[k], CurrentProcessInfo);
rCurrentCondition.CalculateDampingMatrix(mDamp[k], CurrentProcessInfo);
rCurrentCondition.EquationIdVector(EquationId, CurrentProcessInfo);


AddDynamicsToRHS(rCurrentCondition, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

KRATOS_CATCH( "" )
}

void InitializeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
ProcessInfo& CurrentProcessInfo = r_model_part.GetProcessInfo();

Scheme<TSparseSpace, TDenseSpace>::InitializeSolutionStep(r_model_part, A, Dx, b);

double DeltaTime = CurrentProcessInfo[DELTA_TIME];

if (DeltaTime == 0)
KRATOS_THROW_ERROR( std::logic_error, "detected delta_time = 0 in the Bossak Scheme ... check if the time step is created correctly for the current model part", "" )

ma0 = 1.0 / (mBetaNewmark * pow(DeltaTime, 2));
ma1 = mGammaNewmark / (mBetaNewmark * DeltaTime);
ma2 = 1.0 / (mBetaNewmark * DeltaTime);
ma3 = 1.0 / (2.0 * mBetaNewmark) - 1.0;
ma4 = mGammaNewmark / mBetaNewmark - 1.0;
ma5 = DeltaTime * 0.5 * (mGammaNewmark / mBetaNewmark - 2.0);
mam = (1.0 - mAlphaBossak) / (mBetaNewmark * pow(DeltaTime, 2));
}


int Check(const ModelPart& r_model_part) const override
{
KRATOS_TRY

int err = Scheme<TSparseSpace, TDenseSpace>::Check(r_model_part);
if (err != 0) return err;

for (const auto& r_node : r_model_part.Nodes())
{
if (r_node.SolutionStepsDataHas(DISPLACEMENT) == false)
KRATOS_THROW_ERROR( std::logic_error, "DISPLACEMENT variable is not allocated for node ", r_node.Id() )
if (r_node.SolutionStepsDataHas(VELOCITY) == false)
KRATOS_THROW_ERROR( std::logic_error, "DISPLACEMENT variable is not allocated for node ", r_node.Id() )
if (r_node.SolutionStepsDataHas(ACCELERATION) == false)
KRATOS_THROW_ERROR( std::logic_error, "DISPLACEMENT variable is not allocated for node ", r_node.Id() )
}

for (const auto& r_node : r_model_part.Nodes())
{
if (r_node.HasDofFor(DISPLACEMENT_X) == false)
KRATOS_THROW_ERROR( std::invalid_argument, "missing DISPLACEMENT_X dof on node ", r_node.Id() )
if (r_node.HasDofFor(DISPLACEMENT_Y) == false)
KRATOS_THROW_ERROR( std::invalid_argument, "missing DISPLACEMENT_Y dof on node ", r_node.Id() )
if (r_node.HasDofFor(DISPLACEMENT_Z) == false)
KRATOS_THROW_ERROR( std::invalid_argument, "missing DISPLACEMENT_Z dof on node ", r_node.Id() )
}


if (mAlphaBossak > 0.0 || mAlphaBossak < -0.3)
KRATOS_THROW_ERROR( std::logic_error, "Value not admissible for AlphaBossak. Admissible values should be between 0.0 and -0.3. Current value is ", mAlphaBossak )

if (r_model_part.GetBufferSize() < 2)
KRATOS_THROW_ERROR( std::logic_error, "insufficient buffer size. Buffer size should be greater than 2. Current size is", r_model_part.GetBufferSize() )


return 0;
KRATOS_CATCH( "" )
}























protected:







double mAlphaBossak;
double mBetaNewmark;
double mGammaNewmark;

double ma0;
double ma1;
double ma2;
double ma3;
double ma4;
double ma5;
double mam;

std::vector< Matrix >mMass;
std::vector< Matrix >mDamp;
std::vector< Vector >mvel;

double mdamping_factor;






/
void AddDynamicsToLHS(
LocalSystemMatrixType& LHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& CurrentProcessInfo)
{
if (M.size1() != 0) 
{
noalias(LHS_Contribution) += (mdamping_factor * ma1) * M;
}
}

/
void AddDynamicsToRHS(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& CurrentProcessInfo)
{
if (M.size1() != 0)
{
int k = OpenMPUtils::ThisThread();
const auto& r_const_elem_ref = rCurrentElement;
r_const_elem_ref.GetFirstDerivativesVector(mvel[k], 0);
noalias(RHS_Contribution) -= mdamping_factor * prod(M, mvel[k]);
}

}

void AddDynamicsToRHS(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& CurrentProcessInfo)
{
if (M.size1() != 0)
{
int k = OpenMPUtils::ThisThread();
const auto& r_const_cond_ref = rCurrentCondition;
r_const_cond_ref.GetFirstDerivativesVector(mvel[k], 0);
noalias(RHS_Contribution) -= mdamping_factor * prod(M, mvel[k]);
}

}
























private:


































}; 









} 

