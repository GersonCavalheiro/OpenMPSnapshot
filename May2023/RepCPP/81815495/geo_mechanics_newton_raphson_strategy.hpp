
#if !defined(KRATOS_GEO_MECHANICS_NEWTON_RAPHSON_STRATEGY)
#define KRATOS_GEO_MECHANICS_NEWTON_RAPHSON_STRATEGY

#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"

#include "geo_mechanics_application_variables.h"

namespace Kratos
{

template<class TSparseSpace, class TDenseSpace, class TLinearSolver>
class GeoMechanicsNewtonRaphsonStrategy :
public ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{

public:

KRATOS_CLASS_POINTER_DEFINITION(GeoMechanicsNewtonRaphsonStrategy);

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>              BaseType;
typedef ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> MotherType;
typedef ConvergenceCriteria<TSparseSpace, TDenseSpace>                 TConvergenceCriteriaType;
typedef typename BaseType::TBuilderAndSolverType                          TBuilderAndSolverType;
typedef typename BaseType::TSchemeType                                              TSchemeType;
typedef typename BaseType::DofsArrayType                                          DofsArrayType;
typedef typename BaseType::TSystemMatrixType                                  TSystemMatrixType;
typedef typename BaseType::TSystemVectorType                                  TSystemVectorType;
using MotherType::mpScheme;
using MotherType::mpBuilderAndSolver;
using MotherType::mpA; 
using MotherType::mpb; 
using MotherType::mpDx; 
using MotherType::mMaxIterationNumber;
using MotherType::mInitializeWasPerformed;


GeoMechanicsNewtonRaphsonStrategy(
ModelPart& model_part,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
Parameters& rParameters,
int MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false
) : ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(model_part,
pScheme,

pNewConvergenceCriteria,
pNewBuilderAndSolver,
MaxIterations,
CalculateReactions,
ReformDofSetAtEachStep,
MoveMeshFlag)
{
Parameters default_parameters( R"(
{
"min_iteration":    2,
"number_cycles":    5,
"increase_factor":  2.0,
"reduction_factor": 0.5,
"end_time": 1.0,
"max_piping_iterations": 50,
"desired_iterations": 4,
"max_radius_factor": 20.0,
"min_radius_factor": 0.5,
"search_neighbours_step": false,
"body_domain_sub_model_part_list": [],
"loads_sub_model_part_list": [],
"loads_variable_list" : [],
"rebuild_level": 2
}  )" );

rParameters.ValidateAndAssignDefaults(default_parameters);

mpParameters = &rParameters;

if (rParameters["loads_sub_model_part_list"].size() > 0)
{
mSubModelPartList.resize(rParameters["loads_sub_model_part_list"].size());
mVariableNames.resize(rParameters["loads_variable_list"].size());

if ( mSubModelPartList.size() != mVariableNames.size() )
KRATOS_ERROR << "For each SubModelPart there must be a corresponding nodal Variable"
<< std::endl;

for (unsigned int i = 0; i < mVariableNames.size(); ++i) {
mSubModelPartList[i] = &( model_part.GetSubModelPart(rParameters["loads_sub_model_part_list"][i].GetString()) );
mVariableNames[i] = rParameters["loads_variable_list"][i].GetString();
}
}
}


~GeoMechanicsNewtonRaphsonStrategy() override {}


protected:

Parameters* mpParameters;
std::vector<ModelPart*> mSubModelPartList; 
std::vector<std::string> mVariableNames; 


int Check() override
{
KRATOS_TRY

return MotherType::Check();

KRATOS_CATCH( "" )
}


double CalculateReferenceDofsNorm(DofsArrayType& rDofSet)
{
double ReferenceDofsNorm = 0.0;

int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector DofSetPartition;
OpenMPUtils::DivideInPartitions(rDofSet.size(), NumThreads, DofSetPartition);

#pragma omp parallel reduction(+:ReferenceDofsNorm)
{
int k = OpenMPUtils::ThisThread();

typename DofsArrayType::iterator DofsBegin = rDofSet.begin() + DofSetPartition[k];
typename DofsArrayType::iterator DofsEnd = rDofSet.begin() + DofSetPartition[k+1];

for (typename DofsArrayType::iterator itDof = DofsBegin; itDof != DofsEnd; ++itDof)
{
if (itDof->IsFree())
{
const double& temp = itDof->GetSolutionStepValue();
ReferenceDofsNorm += temp*temp;
}
}
}

return sqrt(ReferenceDofsNorm);
}

private:



}; 

} 

#endif 
