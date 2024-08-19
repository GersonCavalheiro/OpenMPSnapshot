
#pragma once


#include <set>




#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/kratos_parameters.h"

namespace Kratos
{








template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class BuilderAndSolver
{
public:

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef typename TSparseSpace::DataType TDataType;

typedef typename TSparseSpace::MatrixType TSystemMatrixType;

typedef typename TSparseSpace::VectorType TSystemVectorType;

typedef typename TSparseSpace::MatrixPointerType TSystemMatrixPointerType;

typedef typename TSparseSpace::VectorPointerType TSystemVectorPointerType;

typedef typename TDenseSpace::MatrixType LocalSystemMatrixType;

typedef typename TDenseSpace::VectorType LocalSystemVectorType;

typedef Scheme<TSparseSpace, TDenseSpace> TSchemeType;

typedef ModelPart::DofType TDofType;

typedef ModelPart::DofsArrayType DofsArrayType;

typedef ModelPart::NodesContainerType NodesArrayType;
typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::ConditionsContainerType ConditionsArrayType;

typedef PointerVectorSet<Element, IndexedObject> ElementsContainerType;

KRATOS_CLASS_POINTER_DEFINITION(BuilderAndSolver);



explicit BuilderAndSolver()
{
}


explicit BuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);

mpLinearSystemSolver = pNewLinearSystemSolver;
}


explicit BuilderAndSolver(typename TLinearSolver::Pointer pNewLinearSystemSolver)
{
mpLinearSystemSolver = pNewLinearSystemSolver;
}


virtual ~BuilderAndSolver()
{
}


virtual ClassType::Pointer Create(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) const
{
return Kratos::make_shared<ClassType>(pNewLinearSystemSolver,ThisParameters);
}



bool GetCalculateReactionsFlag() const
{
return mCalculateReactionsFlag;
}


void SetCalculateReactionsFlag(bool flag)
{
mCalculateReactionsFlag = flag;
}


bool GetDofSetIsInitializedFlag() const
{
return mDofSetIsInitialized;
}


void SetDofSetIsInitializedFlag(bool DofSetIsInitialized)
{
mDofSetIsInitialized = DofSetIsInitialized;
}


bool GetReshapeMatrixFlag() const
{
return mReshapeMatrixFlag;
}


void SetReshapeMatrixFlag(bool ReshapeMatrixFlag)
{
mReshapeMatrixFlag = ReshapeMatrixFlag;
}


unsigned int GetEquationSystemSize() const
{
return mEquationSystemSize;
}


typename TLinearSolver::Pointer GetLinearSystemSolver() const
{
return mpLinearSystemSolver;
}


void SetLinearSystemSolver(typename TLinearSolver::Pointer pLinearSystemSolver)
{
mpLinearSystemSolver = pLinearSystemSolver;
}


virtual void BuildLHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA
)
{
}


virtual void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
)
{
}


virtual void Build(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb
)
{
}


virtual void BuildLHS_CompleteOnFreeRows(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA
)
{
}


virtual void BuildLHS_Complete(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA
)
{
}


virtual void SystemSolve(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb)
{
}


virtual void BuildAndSolveLinearizedOnPreviousIteration(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
const bool MoveMesh
)
{
KRATOS_ERROR << "No special implementation available for "
<< "BuildAndSolveLinearizedOnPreviousIteration "
<< " please use UseOldStiffnessInFirstIterationFlag=false in the settings of the strategy "
<< std::endl;
}


virtual void BuildRHSAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void ApplyDirichletConditions_LHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx
)
{
}


virtual void ApplyDirichletConditions_RHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void ApplyRHSConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
)
{
}


virtual void ApplyConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb
)
{
}


virtual void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
)
{
}


virtual DofsArrayType& GetDofSet()
{
return mDofSet;
}


virtual const DofsArrayType& GetDofSet() const
{
return mDofSet;
}


virtual void SetUpSystem(ModelPart& rModelPart)
{
}


virtual void ResizeAndInitializeVectors(
typename TSchemeType::Pointer pScheme,
TSystemMatrixPointerType& pA,
TSystemVectorPointerType& pDx,
TSystemVectorPointerType& pb,
ModelPart& rModelPart
)
{
}


virtual void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void FinalizeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void CalculateReactions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
}


virtual void Clear()
{
this->mDofSet = DofsArrayType();
this->mpReactionsVector.reset();
if (this->mpLinearSystemSolver != nullptr) this->mpLinearSystemSolver->Clear();

KRATOS_INFO_IF("BuilderAndSolver", this->GetEchoLevel() > 0) << "Clear Function called" << std::endl;
}


virtual int Check(ModelPart& rModelPart)
{
KRATOS_TRY

return 0;
KRATOS_CATCH("");
}


virtual Parameters GetDefaultParameters() const
{
const Parameters default_parameters = Parameters(R"(
{
"name"       : "builder_and_solver",
"echo_level" : 1
})" );
return default_parameters;
}


static std::string Name()
{
return "builder_and_solver";
}




void SetEchoLevel(int Level)
{
mEchoLevel = Level;
}


int GetEchoLevel() const
{
return mEchoLevel;
}


virtual typename TSparseSpace::MatrixType& GetConstraintRelationMatrix()
{
KRATOS_ERROR << "GetConstraintRelationMatrix is not implemented in base BuilderAndSolver" << std::endl;
}


virtual typename TSparseSpace::VectorType& GetConstraintConstantVector()
{
KRATOS_ERROR << "GetConstraintConstantVector is not implemented in base BuilderAndSolver" << std::endl;
}



virtual std::string Info() const
{
return "BuilderAndSolver";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << Info();
}

virtual void PrintData(std::ostream& rOStream) const
{
rOStream << Info();
}



protected:


typename TLinearSolver::Pointer mpLinearSystemSolver = nullptr; 

DofsArrayType mDofSet; 

bool mReshapeMatrixFlag = false;  

bool mDofSetIsInitialized = false; 

bool mCalculateReactionsFlag = false; 

unsigned int mEquationSystemSize; 

int mEchoLevel = 0;

TSystemVectorPointerType mpReactionsVector;




virtual Parameters ValidateAndAssignParameters(
Parameters ThisParameters,
const Parameters DefaultParameters
) const
{
ThisParameters.ValidateAndAssignDefaults(DefaultParameters);
return ThisParameters;
}


virtual void AssignSettings(const Parameters ThisParameters)
{
mEchoLevel = ThisParameters["echo_level"].GetInt();
}





private:









}; 





} 
