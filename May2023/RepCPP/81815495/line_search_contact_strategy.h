
#pragma once



#include "includes/kratos_parameters.h"
#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/variables.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "solving_strategies/strategies/line_search_strategy.h"
#include "utilities/parallel_utilities.h"
#include "utilities/variable_utils.h"
#include "utilities/atomic_utilities.h"


#include "solving_strategies/convergencecriterias/convergence_criteria.h"


#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"


namespace Kratos
{








template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class LineSearchContactStrategy :
public LineSearchStrategy< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:
typedef ConvergenceCriteria<TSparseSpace, TDenseSpace> TConvergenceCriteriaType;


KRATOS_CLASS_POINTER_DEFINITION( LineSearchContactStrategy );

typedef SolvingStrategy<TSparseSpace, TDenseSpace>                          SolvingStrategyType;

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>      StrategyBaseType;

typedef ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> NRBaseType;

typedef LineSearchStrategy<TSparseSpace, TDenseSpace, TLinearSolver>                   BaseType;

typedef LineSearchContactStrategy<TSparseSpace, TDenseSpace, TLinearSolver>           ClassType;

typedef typename BaseType::TBuilderAndSolverType                          TBuilderAndSolverType;

typedef typename BaseType::TDataType                                                  TDataType;

typedef TSparseSpace                                                            SparseSpaceType;

typedef typename BaseType::TSchemeType                                              TSchemeType;

typedef typename BaseType::DofsArrayType                                          DofsArrayType;

typedef typename BaseType::TSystemMatrixType                                  TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                                  TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType                          LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType                          LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType                    TSystemMatrixPointerType;

typedef typename BaseType::TSystemVectorPointerType                    TSystemVectorPointerType;

typedef ModelPart::NodesContainerType                                            NodesArrayType;

typedef ModelPart::ConditionsContainerType                                  ConditionsArrayType;

typedef std::size_t                                                                   IndexType;


explicit LineSearchContactStrategy()
{
}


explicit LineSearchContactStrategy(ModelPart& rModelPart, Parameters ThisParameters)
: BaseType(rModelPart, BaseType::GetDefaultParameters())
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


LineSearchContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})")
)
: BaseType(rModelPart, pScheme, pNewLinearSolver, pNewConvergenceCriteria, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag)
{
KRATOS_TRY;

Parameters default_parameters = this->GetDefaultParameters();

ThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}



LineSearchContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})")
)
: BaseType(rModelPart, pScheme, pNewLinearSolver, pNewConvergenceCriteria, pNewBuilderAndSolver, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag )
{
KRATOS_TRY;

Parameters default_parameters = this->GetDefaultParameters();

ThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}



~LineSearchContactStrategy() override
= default;




typename SolvingStrategyType::Pointer Create(
ModelPart& rModelPart,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(rModelPart, ThisParameters);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "line_search_contact_strategy"
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "line_search_contact_strategy";
}




std::string Info() const override
{
return "LineSearchContactStrategy";
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



bool mRecalculateFactor;         




void InitializeSolutionStep() override
{
BaseType::InitializeSolutionStep();

}


void UpdateDatabase(
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b,
const bool MoveMesh
) override
{
typename TSchemeType::Pointer pScheme = this->GetScheme();
typename TBuilderAndSolverType::Pointer pBuilderAndSolver = this->GetBuilderAndSolver(); 

TSystemVectorType aux(b.size()); 
TSparseSpace::Assign(aux, 0.5, Dx);

TSystemVectorType DxDisp(b.size());
TSystemVectorType DxLM(b.size());
ComputeSplitDx(Dx, DxDisp, DxLM);

TSparseSpace::SetToZero(b);
pBuilderAndSolver->BuildRHS(pScheme, BaseType::GetModelPart(), b );
double roDisp;
double roLM;
ComputeMixedResidual(b, roDisp, roLM);

NRBaseType::UpdateDatabase(A,aux,b,MoveMesh);
TSparseSpace::SetToZero(b);
pBuilderAndSolver->BuildRHS(pScheme, BaseType::GetModelPart(), b );
double rhDisp;
double rhLM;
ComputeMixedResidual(b, rhDisp, rhLM);

NRBaseType::UpdateDatabase(A,aux,b,MoveMesh);
TSparseSpace::SetToZero(b);
pBuilderAndSolver->BuildRHS(pScheme, BaseType::GetModelPart(), b );
double rfDisp;
double rfLM;
ComputeMixedResidual(b, rfDisp, rfLM);

double XminDisp = 1e-3;
double XmaxDisp = 1.0;
double XminLM = 1e-3;
double XmaxLM = 1.0;

ComputeParabola(XminDisp, XmaxDisp, rfDisp, roDisp, rhDisp);
ComputeParabola(XminLM, XmaxLM, rfLM, roLM, rhLM);

TSparseSpace::Assign(aux,-(1.0 - XmaxDisp), DxDisp);
TSparseSpace::UnaliasedAdd(aux,-(1.0 - XmaxLM), DxLM);
NRBaseType::UpdateDatabase(A,aux,b,MoveMesh);
}


void ComputeSplitDx(
TSystemVectorType& Dx,
TSystemVectorType& DxDisp,
TSystemVectorType& DxLM
)
{
NodesArrayType& r_nodes_array = StrategyBaseType::GetModelPart().Nodes();
block_for_each(r_nodes_array, [&](Node& rNode) {
for(auto itDoF = rNode.GetDofs().begin() ; itDoF != rNode.GetDofs().end() ; itDoF++) {
const int j = (**itDoF).EquationId();
const std::size_t CurrVar = (**itDoF).GetVariable().Key();

if ((CurrVar == DISPLACEMENT_X) || (CurrVar == DISPLACEMENT_Y) || (CurrVar == DISPLACEMENT_Z)) {
DxDisp[j] = Dx[j];
DxLM[j] = 0.0;
} else { 
DxDisp[j] = 0.0;
DxLM[j] = Dx[j];
}
}
});
}


void ComputeMixedResidual(
TSystemVectorType& b,
double& normDisp,
double& normLM
)
{
NodesArrayType& r_nodes_array = StrategyBaseType::GetModelPart().Nodes();
block_for_each(r_nodes_array, [&](Node& rNode) {
for(auto itDoF = rNode.GetDofs().begin() ; itDoF != rNode.GetDofs().end() ; itDoF++) {
const int j = (**itDoF).EquationId();
const std::size_t CurrVar = (**itDoF).GetVariable().Key();

if ((CurrVar == DISPLACEMENT_X) || (CurrVar == DISPLACEMENT_Y) || (CurrVar == DISPLACEMENT_Z)) {
AtomicAdd(normDisp, b[j] * b[j]);
} else { 
AtomicAdd(normLM, b[j] * b[j]);
}
}
});

normDisp = std::sqrt(normDisp);
normLM = std::sqrt(normLM);
}


void ComputeParabola(
double& Xmax,
double& Xmin,
const double rf,
const double ro,
const double rh
)
{

const double parabole_a = 2 * rf + 2 * ro - 4 * rh;
const double parabole_b = 4 * rh - rf - 3 * ro;

if( parabole_a > 0.0) 
{
Xmax = -0.5 * parabole_b/parabole_a; 
if( Xmax > 1.0)
Xmax = 1.0;
else if(Xmax < -1.0)
Xmax = -1.0;
}
else 
{
if(rf < ro)
Xmax = 1.0;
else
Xmax = Xmin; 
}
}


void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
}






LineSearchContactStrategy(const LineSearchContactStrategy& Other)
{
};

private:








}; 
}  
