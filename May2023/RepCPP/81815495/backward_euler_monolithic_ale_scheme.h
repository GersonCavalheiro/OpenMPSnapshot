

#ifndef KRATOS_BACKWARD_EULER_MONOLITHIC_ALE_SCHEME
#define KRATOS_BACKWARD_EULER_MONOLITHIC_ALE_SCHEME






#include "boost/smart_ptr.hpp"


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "includes/cfd_variables.h"
#include "containers/array_1d.h"
#include "utilities/openmp_utils.h"
#include "utilities/dof_updater.h"
#include "utilities/coordinate_transformation_utilities.h"
#include "processes/process.h"
#include "../../FluidDynamicsApplication/custom_strategies/schemes/residualbased_predictorcorrector_velocity_bossak_scheme_turbulent.h"

namespace Kratos {







template<class TSparseSpace, class TDenseSpace >
class BackwardEulerMonolithicAleScheme : public ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace> {

public:

KRATOS_CLASS_POINTER_DEFINITION(BackwardEulerMonolithicAleScheme);

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace> BossakType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename Element::DofsVectorType DofsVectorType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef Element::GeometryType  GeometryType;


BackwardEulerMonolithicAleScheme(unsigned int DomainSize, bool IsLagrangian = true)
: ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace>(0.0, 0.0, DomainSize)
{
mIsLagrangian = IsLagrangian;
BossakType::mGammaNewmark = 1.0;
}

~BackwardEulerMonolithicAleScheme() override {}


void Update(ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dv,
TSystemVectorType& b) override
{
BossakType::Update(rModelPart, rDofSet, A, Dv, b);
this->Pfem2AdditionalUpdateOperations(rModelPart, rDofSet, A, Dv, b);
}

void Predict(ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dv,
TSystemVectorType& b) override
{
BossakType::Predict(rModelPart, rDofSet, A, Dv, b);
this->Pfem2AdditionalUpdateOperations(rModelPart, rDofSet, A, Dv, b);
}

void Pfem2AdditionalUpdateOperations(ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dv,
TSystemVectorType& b)
{
if (mIsLagrangian) {
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rModelPart.NumberOfNodes()); ++i)
{
auto it_node = rModelPart.NodesBegin() + i;
noalias(it_node->FastGetSolutionStepValue(MESH_VELOCITY)) = it_node->FastGetSolutionStepValue(VELOCITY);
}
}
}


protected:

bool mIsLagrangian;


}; 





} 

#endif 
