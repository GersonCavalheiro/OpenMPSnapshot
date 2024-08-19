
#pragma once




#include "includes/define.h"
#include "includes/element.h"
#include "includes/condition.h"
#include "includes/process_info.h"
#include "includes/ublas_interface.h"
#include "solving_strategies/schemes/scheme.h"

#include "structural_mechanics_application_variables.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace
>
class EigensolverDynamicScheme : public Scheme<TSparseSpace,TDenseSpace>
{
public:

KRATOS_CLASS_POINTER_DEFINITION( EigensolverDynamicScheme );

typedef Scheme<TSparseSpace,TDenseSpace> BaseType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;


EigensolverDynamicScheme() : Scheme<TSparseSpace,TDenseSpace>() {}

~EigensolverDynamicScheme() override {}



void CalculateSystemContributions(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo
) override
{
KRATOS_TRY

if (CurrentProcessInfo[BUILD_LEVEL] == 1)
{ 
rCurrentElement.CalculateMassMatrix(LHS_Contribution,CurrentProcessInfo);
std::size_t LocalSize = LHS_Contribution.size1();
if (RHS_Contribution.size() != LocalSize)
RHS_Contribution.resize(LocalSize,false);
noalias(RHS_Contribution) = ZeroVector(LocalSize);
}
else if (CurrentProcessInfo[BUILD_LEVEL] == 2) 
{
rCurrentElement.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,CurrentProcessInfo);
}
else
{
KRATOS_ERROR <<"Invalid BUILD_LEVEL" << std::endl;
}

rCurrentElement.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH("")
}

void CalculateLHSContribution(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

LocalSystemVectorType RHS_Contribution;
RHS_Contribution.resize(LHS_Contribution.size1(), false);
CalculateSystemContributions(
rCurrentElement,
LHS_Contribution,
RHS_Contribution,
EquationId,
CurrentProcessInfo);

KRATOS_CATCH("")
}

void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentElement.CalculateRightHandSide(RHS_Contribution,CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH("")
}

void CalculateSystemContributions(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Condition::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

if (CurrentProcessInfo[BUILD_LEVEL] == 1)
{ 
rCurrentCondition.CalculateMassMatrix(LHS_Contribution,CurrentProcessInfo);
std::size_t LocalSize = LHS_Contribution.size1();
if (RHS_Contribution.size() != LocalSize)
{
RHS_Contribution.resize(LocalSize,false);
}
noalias(RHS_Contribution) = ZeroVector(LocalSize);
}
else if (CurrentProcessInfo[BUILD_LEVEL] == 2) 
{
rCurrentCondition.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,CurrentProcessInfo);
}
else
{
KRATOS_ERROR <<"Invalid BUILD_LEVEL" << std::endl;
}

rCurrentCondition.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH("")
}

void CalculateLHSContribution(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
Condition::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

LocalSystemVectorType RHS_Contribution;
RHS_Contribution.resize(LHS_Contribution.size1(), false);
CalculateSystemContributions(
rCurrentCondition,
LHS_Contribution,
RHS_Contribution,
EquationId,
CurrentProcessInfo);

KRATOS_CATCH("")
}

void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentCondition.CalculateRightHandSide(RHS_Contribution,CurrentProcessInfo);

rCurrentCondition.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH("")
}


}; 




}  

