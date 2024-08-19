

#if !defined(KRATOS_PFEM2_MONOLITHIC_SLIP_SCHEME )
#define  KRATOS_PFEM2_MONOLITHIC_SLIP_SCHEME






#include "boost/smart_ptr.hpp"


#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "includes/deprecated_variables.h"

#include "containers/array_1d.h"
#include "utilities/openmp_utils.h"
#include "utilities/coordinate_transformation_utilities.h"
#include "processes/process.h"

namespace Kratos {


























template<class TSparseSpace,
class TDenseSpace 
>
class PFEM2MonolithicSlipScheme : public Scheme<TSparseSpace, TDenseSpace> {
public:




KRATOS_CLASS_POINTER_DEFINITION(PFEM2MonolithicSlipScheme);

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename Element::DofsVectorType DofsVectorType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef Element::GeometryType  GeometryType;







PFEM2MonolithicSlipScheme(unsigned int DomainSize)
:
Scheme<TSparseSpace, TDenseSpace>(),
mRotationTool(DomainSize,DomainSize+1,SLIP)
{
}



virtual ~PFEM2MonolithicSlipScheme() {}







/
void CalculateSystemContributions(Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentElement.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);

mRotationTool.Rotate(LHS_Contribution,RHS_Contribution,rCurrentElement.GetGeometry());
mRotationTool.ApplySlipCondition(LHS_Contribution,RHS_Contribution,rCurrentElement.GetGeometry());

KRATOS_CATCH("")
}

void CalculateRHSContribution(Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{


rCurrentElement.CalculateRightHandSide(RHS_Contribution, CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);


mRotationTool.Rotate(RHS_Contribution,rCurrentElement.GetGeometry());
mRotationTool.ApplySlipCondition(RHS_Contribution,rCurrentElement.GetGeometry());
}


virtual void CalculateSystemContributions(Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentCondition.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);
rCurrentCondition.EquationIdVector(EquationId, CurrentProcessInfo);


KRATOS_CATCH("")
}

virtual void CalculateRHSContribution(Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_TRY;



rCurrentCondition.CalculateRightHandSide(RHS_Contribution,rCurrentProcessInfo);

rCurrentCondition.EquationIdVector(EquationId,rCurrentProcessInfo);


mRotationTool.Rotate(RHS_Contribution,rCurrentCondition.GetGeometry());
mRotationTool.ApplySlipCondition(RHS_Contribution,rCurrentCondition.GetGeometry());

KRATOS_CATCH("");
}
/
}

void FinalizeSolutionStep(ModelPart &rModelPart, TSystemMatrixType &A, TSystemVectorType &Dx, TSystemVectorType &b) override
{

Scheme<TSparseSpace, TDenseSpace>::FinalizeSolutionStep(rModelPart, A, Dx, b);
}

/





















protected:































private:








CoordinateTransformationUtils<LocalSystemMatrixType,LocalSystemVectorType,double> mRotationTool;




























}; 









} 

#endif 
