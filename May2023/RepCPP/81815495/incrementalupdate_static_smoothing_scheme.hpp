
#if !defined(KRATOS_INCREMENTAL_UPDATE_STATIC_SMOOTHING_SCHEME )
#define  KRATOS_INCREMENTAL_UPDATE_STATIC_SMOOTHING_SCHEME

#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme.h"
#include "dam_application_variables.h"

namespace Kratos
{

template<class TSparseSpace, class TDenseSpace>

class IncrementalUpdateStaticSmoothingScheme : public ResidualBasedIncrementalUpdateStaticScheme<TSparseSpace,TDenseSpace>
{

public:

KRATOS_CLASS_POINTER_DEFINITION( IncrementalUpdateStaticSmoothingScheme );

typedef Scheme<TSparseSpace,TDenseSpace>     BaseType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;


IncrementalUpdateStaticSmoothingScheme()
: ResidualBasedIncrementalUpdateStaticScheme<TSparseSpace,TDenseSpace>() {}


virtual ~IncrementalUpdateStaticSmoothingScheme() {}


void FinalizeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

unsigned int Dim = rModelPart.GetProcessInfo()[DOMAIN_SIZE];

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;
Matrix& rNodalStress = itNode->FastGetSolutionStepValue(NODAL_CAUCHY_STRESS_TENSOR);
if(rNodalStress.size1() != Dim)
rNodalStress.resize(Dim,Dim,false);
noalias(rNodalStress) = ZeroMatrix(Dim,Dim);
itNode->FastGetSolutionStepValue(NODAL_JOINT_AREA) = 0.0;
itNode->FastGetSolutionStepValue(NODAL_JOINT_WIDTH) = 0.0;
}
}

BaseType::FinalizeSolutionStep(rModelPart,A,Dx,b);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
const double& NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
if (NodalArea>1.0e-15)
{
const double InvNodalArea = 1.0/(NodalArea);
Matrix& rNodalStress = itNode->FastGetSolutionStepValue(NODAL_CAUCHY_STRESS_TENSOR);
for(unsigned int i = 0; i<Dim; i++)
{
for(unsigned int j = 0; j<Dim; j++)
{
rNodalStress(i,j) *= InvNodalArea;
}
}
}

const double& NodalJointArea = itNode->FastGetSolutionStepValue(NODAL_JOINT_AREA);
if (NodalJointArea>1.0e-15)
{
double& NodalJointWidth = itNode->FastGetSolutionStepValue(NODAL_JOINT_WIDTH);
NodalJointWidth = NodalJointWidth/NodalJointArea;
}
}
}

KRATOS_CATCH("")
}


}; 
}  

#endif 
