
#if !defined(CHIMERA_DISTANCE_CALCULATION_UTILITY )
#define  CHIMERA_DISTANCE_CALCULATION_UTILITY





#include "includes/define.h"
#include "processes/variational_distance_calculation_process.h"
#include "processes/parallel_distance_calculation_process.h"
#include "processes/calculate_distance_to_skin_process.h"
#include "utilities/variable_utils.h"


namespace Kratos
{


template <int TDim>
class ChimeraDistanceCalculationUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ChimeraDistanceCalculationUtility);


ChimeraDistanceCalculationUtility() = delete;

ChimeraDistanceCalculationUtility(const ChimeraDistanceCalculationUtility& rOther) = delete;






static inline void CalculateDistance(ModelPart &rBackgroundModelPart, ModelPart &rSkinModelPart)
{
typedef CalculateDistanceToSkinProcess<TDim> CalculateDistanceToSkinProcessType;
const int nnodes = static_cast<int>(rBackgroundModelPart.NumberOfNodes());

#pragma omp parallel for
for (int i_node = 0; i_node < nnodes; ++i_node)
{
auto it_node = rBackgroundModelPart.NodesBegin() + i_node;
it_node->FastGetSolutionStepValue(DISTANCE, 0) = 0.0;
it_node->FastGetSolutionStepValue(DISTANCE, 1) = 0.0;
it_node->SetValue(DISTANCE, 0.0);
}

CalculateDistanceToSkinProcessType(rBackgroundModelPart, rSkinModelPart).Execute();

Parameters parallel_redistance_settings(R"({
"max_levels" : 100,
"max_distance" : 200.0
})");
auto p_distance_smoother = Kratos::make_shared<ParallelDistanceCalculationProcess<TDim>>(
rBackgroundModelPart,
parallel_redistance_settings
);
p_distance_smoother->Execute();

VariableUtils().CopyVariable<double>(DISTANCE, CHIMERA_DISTANCE, rBackgroundModelPart.Nodes());
}









}; 







}  

#endif 
