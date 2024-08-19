
#if !defined(KRATOS_SOLUTION_SCHEME_H_INCLUDED)
#define KRATOS_SOLUTION_SCHEME_H_INCLUDED



#include "includes/checks.h"
#include "includes/model_part.h"
#include "custom_processes/solver_process.hpp"
#include "custom_solvers/time_integration_methods/time_integration_method.hpp"

namespace Kratos
{








template<class TSparseSpace,
class TDenseSpace 
>
class SolutionScheme : public Flags
{
public:

typedef SolutionScheme<TSparseSpace,TDenseSpace>              SolutionSchemeType;
typedef typename SolutionSchemeType::Pointer           SolutionSchemePointerType;
typedef SolverLocalFlags                                           LocalFlagType;
typedef typename ModelPart::DofsArrayType                          DofsArrayType;

typedef typename TSparseSpace::MatrixType                       SystemMatrixType;
typedef typename TSparseSpace::VectorType                       SystemVectorType;
typedef typename TDenseSpace::MatrixType                   LocalSystemMatrixType;
typedef typename TDenseSpace::VectorType                   LocalSystemVectorType;

typedef ModelPart::NodesContainerType                         NodesContainerType;
typedef ModelPart::ElementsContainerType                   ElementsContainerType;
typedef ModelPart::ConditionsContainerType               ConditionsContainerType;

typedef ModelPart::NodeType                                                NodeType;
typedef array_1d<double, 3>                                              VectorType;
typedef Variable<VectorType>                                     VariableVectorType;
typedef TimeIntegrationMethod<VariableVectorType,VectorType>  IntegrationVectorType;
typedef typename IntegrationVectorType::Pointer        IntegrationVectorPointerType;
typedef std::vector<IntegrationVectorPointerType>      IntegrationMethodsVectorType;

typedef Variable<double>                                         VariableScalarType;
typedef TimeIntegrationMethod<VariableScalarType, double>     IntegrationScalarType;
typedef typename IntegrationScalarType::Pointer        IntegrationScalarPointerType;
typedef std::vector<IntegrationScalarPointerType>      IntegrationMethodsScalarType;

typedef typename SolverProcess::Pointer                          ProcessPointerType;
typedef std::vector<ProcessPointerType>                    ProcessPointerVectorType;

KRATOS_CLASS_POINTER_DEFINITION(SolutionScheme);


SolutionScheme() : Flags() {SetDefaultFlags();}

SolutionScheme(Flags& rOptions) : Flags(), mOptions(rOptions) {SetDefaultFlags();}

SolutionScheme(IntegrationMethodsVectorType& rTimeVectorIntegrationMethods, Flags& rOptions) : Flags(), mOptions(rOptions), mTimeVectorIntegrationMethods(rTimeVectorIntegrationMethods) {SetDefaultFlags();}

SolutionScheme(IntegrationMethodsVectorType& rTimeVectorIntegrationMethods) : Flags(), mTimeVectorIntegrationMethods(rTimeVectorIntegrationMethods) {SetDefaultFlags();}

SolutionScheme(IntegrationMethodsScalarType& rTimeScalarIntegrationMethods, Flags& rOptions) : Flags(), mOptions(rOptions), mTimeScalarIntegrationMethods(rTimeScalarIntegrationMethods) {SetDefaultFlags();}

SolutionScheme(IntegrationMethodsScalarType& rTimeScalarIntegrationMethods) : Flags(), mTimeScalarIntegrationMethods(rTimeScalarIntegrationMethods) {SetDefaultFlags();}

SolutionScheme(IntegrationMethodsVectorType& rTimeVectorIntegrationMethods,
IntegrationMethodsScalarType& rTimeScalarIntegrationMethods,
Flags& rOptions)
: Flags(), mOptions(rOptions), mTimeVectorIntegrationMethods(rTimeVectorIntegrationMethods), mTimeScalarIntegrationMethods(rTimeScalarIntegrationMethods) {SetDefaultFlags();}

SolutionScheme(IntegrationMethodsVectorType& rTimeVectorIntegrationMethods,
IntegrationMethodsScalarType& rTimeScalarIntegrationMethods)
: Flags(), mTimeVectorIntegrationMethods(rTimeVectorIntegrationMethods), mTimeScalarIntegrationMethods(rTimeScalarIntegrationMethods) {SetDefaultFlags();}

SolutionScheme(SolutionScheme& rOther) : mOptions(rOther.mOptions)
, mProcesses(rOther.mProcesses)
{
std::copy(std::begin(rOther.mTimeVectorIntegrationMethods), std::end(rOther.mTimeVectorIntegrationMethods), std::back_inserter(mTimeVectorIntegrationMethods));
}

virtual SolutionSchemePointerType Clone()
{
return SolutionSchemePointerType( new SolutionScheme(*this) );
}

~SolutionScheme() override {}




void SetDefaultFlags()
{
KRATOS_TRY

if( this->mOptions.IsNotDefined(LocalFlagType::MOVE_MESH) )
mOptions.Set(LocalFlagType::MOVE_MESH,true); 

if( this->mOptions.IsNotDefined(LocalFlagType::UPDATE_VARIABLES) )
mOptions.Set(LocalFlagType::UPDATE_VARIABLES,true); 

if( this->mOptions.IsNotDefined(LocalFlagType::INCREMENTAL_SOLUTION) )
mOptions.Set(LocalFlagType::INCREMENTAL_SOLUTION,true); 

KRATOS_CATCH("")
}


virtual void Initialize(ModelPart& rModelPart)
{
KRATOS_TRY

for(typename IntegrationMethodsVectorType::iterator it=mTimeVectorIntegrationMethods.begin();
it!=mTimeVectorIntegrationMethods.end(); ++it)
(*it)->SetParameters(rModelPart.GetProcessInfo());

for(typename IntegrationMethodsScalarType::iterator it=mTimeScalarIntegrationMethods.begin();
it!=mTimeScalarIntegrationMethods.end(); ++it)
(*it)->SetParameters(rModelPart.GetProcessInfo());

this->InitializeElements(rModelPart);

this->InitializeConditions(rModelPart);

this->Set(LocalFlagType::INITIALIZED, true);

KRATOS_CATCH("")
}


virtual void InitializeSolutionStep(ModelPart& rModelPart)
{
KRATOS_TRY

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

for(typename ProcessPointerVectorType::iterator it=mProcesses.begin(); it!=mProcesses.end(); ++it)
(*it)->ExecuteInitializeSolutionStep();

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Elements().size()); i++)
{
auto itElem = rModelPart.ElementsBegin() + i;
itElem->InitializeSolutionStep(rCurrentProcessInfo);
}

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Conditions().size()); i++)
{
auto itCond = rModelPart.ConditionsBegin() + i;
itCond->InitializeSolutionStep(rCurrentProcessInfo);
}

KRATOS_CATCH("")
}




virtual void FinalizeSolutionStep(ModelPart& rModelPart)
{
KRATOS_TRY

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

for(typename ProcessPointerVectorType::iterator it=mProcesses.begin(); it!=mProcesses.end(); ++it)
(*it)->ExecuteFinalizeSolutionStep();


#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Elements().size()); i++)
{
auto itElem = rModelPart.ElementsBegin() + i;
itElem->FinalizeSolutionStep(rCurrentProcessInfo);
}


#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Conditions().size()); i++)
{
auto itCond = rModelPart.ConditionsBegin() + i;
itCond->FinalizeSolutionStep(rCurrentProcessInfo);
}

KRATOS_CATCH("")
}


virtual void InitializeNonLinearIteration(ModelPart& rModelPart)
{
KRATOS_TRY

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

for(typename ProcessPointerVectorType::iterator it=mProcesses.begin(); it!=mProcesses.end(); ++it)
(*it)->ExecuteInitialize(); 

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Elements().size()); i++)
{
auto itElem = rModelPart.ElementsBegin() + i;
itElem->InitializeNonLinearIteration(rCurrentProcessInfo);
}


#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Conditions().size()); i++)
{
auto itCond = rModelPart.ConditionsBegin() + i;
itCond->InitializeNonLinearIteration(rCurrentProcessInfo);
}


KRATOS_CATCH("")
}



virtual void FinalizeNonLinearIteration(ModelPart& rModelPart)
{
KRATOS_TRY

for(typename ProcessPointerVectorType::iterator it=mProcesses.begin(); it!=mProcesses.end(); ++it)
(*it)->ExecuteFinalize(); 

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Elements().size()); i++)
{
auto itElem = rModelPart.ElementsBegin() + i;
itElem->FinalizeNonLinearIteration(rCurrentProcessInfo);
}


#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Conditions().size()); i++)
{
auto itCond = rModelPart.ConditionsBegin() + i;
itCond->FinalizeNonLinearIteration(rCurrentProcessInfo);
}

KRATOS_CATCH("")
}



virtual void Predict(ModelPart& rModelPart,
DofsArrayType& rDofSet,
SystemVectorType& rDx)
{
KRATOS_TRY

KRATOS_CATCH("")
}


virtual void Update(ModelPart& rModelPart,
DofsArrayType& rDofSet,
SystemVectorType& rDx)
{
KRATOS_TRY


KRATOS_CATCH("")
}



static inline void SetSolution(ModelPart& rModelPart,
DofsArrayType& rDofSet,
SystemVectorType& rDx)
{
KRATOS_TRY

const unsigned int NumThreads = ParallelUtilities::GetNumThreads();

OpenMPUtils::PartitionVector DofPartition;
OpenMPUtils::DivideInPartitions(rDofSet.size(), NumThreads, DofPartition);

const int ndof = static_cast<int>(rDofSet.size());
typename DofsArrayType::iterator DofBegin = rDofSet.begin();

#pragma omp parallel for firstprivate(DofBegin)
for(int i = 0;  i < ndof; i++)
{
typename DofsArrayType::iterator itDof = DofBegin + i;

if (itDof->IsFree() )
{
itDof->GetSolutionStepValue() = TSparseSpace::GetValue(rDx,itDof->EquationId());
}
}

KRATOS_CATCH("")
}



static inline void AddSolution(ModelPart& rModelPart,
DofsArrayType& rDofSet,
SystemVectorType& rDx)
{
KRATOS_TRY

const unsigned int NumThreads = ParallelUtilities::GetNumThreads();

OpenMPUtils::PartitionVector DofPartition;
OpenMPUtils::DivideInPartitions(rDofSet.size(), NumThreads, DofPartition);

const int ndof = static_cast<int>(rDofSet.size());
typename DofsArrayType::iterator DofBegin = rDofSet.begin();

#pragma omp parallel for firstprivate(DofBegin)
for(int i = 0;  i < ndof; i++)
{
typename DofsArrayType::iterator itDof = DofBegin + i;

if (itDof->IsFree() )
{
itDof->GetSolutionStepValue() += TSparseSpace::GetValue(rDx,itDof->EquationId());
}
}

KRATOS_CATCH("")
}


virtual void UpdateDofs(ModelPart& rModelPart,
DofsArrayType& rDofSet,
SystemVectorType& rDx)
{
KRATOS_TRY

if( mOptions.Is(LocalFlagType::INCREMENTAL_SOLUTION) )
AddSolution(rModelPart,rDofSet,rDx);  
else
SetSolution(rModelPart,rDofSet,rDx);  

KRATOS_CATCH("")
}




virtual void UpdateVariables(ModelPart& rModelPart)
{
KRATOS_TRY

if( this->mOptions.Is(LocalFlagType::UPDATE_VARIABLES) ){

const unsigned int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector NodePartition;
OpenMPUtils::DivideInPartitions(rModelPart.Nodes().size(), NumThreads, NodePartition);

const int nnodes = static_cast<int>(rModelPart.Nodes().size());
NodesContainerType::iterator NodeBegin = rModelPart.Nodes().begin();

#pragma omp parallel for firstprivate(NodeBegin)
for(int i = 0;  i < nnodes; i++)
{
NodesContainerType::iterator itNode = NodeBegin + i;

this->IntegrationMethodUpdate(*itNode);
}
}

KRATOS_CATCH("")
}


virtual void PredictVariables(ModelPart& rModelPart)
{
KRATOS_TRY

const unsigned int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector NodePartition;
OpenMPUtils::DivideInPartitions(rModelPart.Nodes().size(), NumThreads, NodePartition);

const int nnodes = static_cast<int>(rModelPart.Nodes().size());
NodesContainerType::iterator NodeBegin = rModelPart.Nodes().begin();

#pragma omp parallel for firstprivate(NodeBegin)
for(int i = 0;  i < nnodes; i++)
{
NodesContainerType::iterator itNode = NodeBegin + i;

this->IntegrationMethodPredict(*itNode);
}

KRATOS_CATCH("")
}



virtual void MoveMesh(ModelPart& rModelPart)
{
KRATOS_TRY

if( this->mOptions.Is(LocalFlagType::MOVE_MESH) ){

if (rModelPart.NodesBegin()->SolutionStepsDataHas(DISPLACEMENT_X) == false)
{
KRATOS_ERROR << "It is impossible to move the mesh since the DISPLACEMENT variable is not in the Model Part. Add DISPLACEMENT to the list of variables" << std::endl;
}

bool DisplacementIntegration = false;
for(typename IntegrationMethodsVectorType::iterator it=mTimeVectorIntegrationMethods.begin();
it!=mTimeVectorIntegrationMethods.end(); ++it)
{
if( "DISPLACEMENT" == (*it)->GetVariableName() ){
DisplacementIntegration = true;
break;
}
}

if(DisplacementIntegration == true){

const int nnodes = rModelPart.NumberOfNodes();
ModelPart::NodesContainerType::iterator it_begin = rModelPart.NodesBegin();

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it_node = it_begin + i;

noalias(it_node->Coordinates()) = it_node->GetInitialPosition().Coordinates();
noalias(it_node->Coordinates()) += it_node->FastGetSolutionStepValue(DISPLACEMENT);
}

}

}

KRATOS_CATCH("")
}


virtual void Clear(Element::Pointer rCurrentElement)
{
}


virtual void Clear(Condition::Pointer rCurrentCondition)
{
}


virtual void Clear() {}


virtual int Check(ModelPart& rModelPart)
{
KRATOS_TRY

for(ModelPart::ElementsContainerType::iterator it=rModelPart.ElementsBegin();
it!=rModelPart.ElementsEnd(); ++it)
{
it->Check(rModelPart.GetProcessInfo());
}

for(ModelPart::ConditionsContainerType::iterator it=rModelPart.ConditionsBegin();
it!=rModelPart.ConditionsEnd(); ++it)
{
it->Check(rModelPart.GetProcessInfo());
}

for(typename IntegrationMethodsVectorType::iterator it=mTimeVectorIntegrationMethods.begin();
it!=mTimeVectorIntegrationMethods.end(); ++it)
(*it)->Check(rModelPart.GetProcessInfo());

for(typename IntegrationMethodsScalarType::iterator it=mTimeScalarIntegrationMethods.begin();
it!=mTimeScalarIntegrationMethods.end(); ++it)
(*it)->Check(rModelPart.GetProcessInfo());

KRATOS_CATCH("")

return 0;
}




virtual void CalculateSystemContributions(Element::Pointer pCurrentElement,
LocalSystemMatrixType& rLHS_Contribution,
LocalSystemVectorType& rRHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentElement->CalculateLocalSystem(rLHS_Contribution, rRHS_Contribution, rCurrentProcessInfo);
pCurrentElement->EquationIdVector(rEquationId, rCurrentProcessInfo);
}


virtual void Calculate_RHS_Contribution(Element::Pointer pCurrentElement,
LocalSystemVectorType& rRHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentElement->CalculateRightHandSide(rRHS_Contribution, rCurrentProcessInfo);
pCurrentElement->EquationIdVector(rEquationId, rCurrentProcessInfo);
}

virtual void Calculate_LHS_Contribution(Element::Pointer pCurrentElement,
LocalSystemMatrixType& rLHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
std::cout<< " it is C_LHS "<<std::endl;
pCurrentElement->CalculateLeftHandSide(rLHS_Contribution, rCurrentProcessInfo);
pCurrentElement->EquationIdVector(rEquationId, rCurrentProcessInfo);
}

virtual void EquationId(Element::Pointer pCurrentElement,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
(pCurrentElement)->EquationIdVector(rEquationId, rCurrentProcessInfo);
}


virtual void Condition_CalculateSystemContributions(Condition::Pointer pCurrentCondition,
LocalSystemMatrixType& rLHS_Contribution,
LocalSystemVectorType& rRHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentCondition->CalculateLocalSystem(rLHS_Contribution, rRHS_Contribution, rCurrentProcessInfo);
pCurrentCondition->EquationIdVector(rEquationId, rCurrentProcessInfo);
}

virtual void Condition_Calculate_RHS_Contribution(Condition::Pointer pCurrentCondition,
LocalSystemVectorType& rRHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentCondition->CalculateRightHandSide(rRHS_Contribution, rCurrentProcessInfo);
pCurrentCondition->EquationIdVector(rEquationId, rCurrentProcessInfo);
}


virtual void Condition_Calculate_LHS_Contribution(Condition::Pointer pCurrentCondition,
LocalSystemMatrixType& rLHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentCondition->CalculateLeftHandSide(rLHS_Contribution, rCurrentProcessInfo);
pCurrentCondition->EquationIdVector(rEquationId, rCurrentProcessInfo);
}

virtual void Condition_EquationId(Condition::Pointer pCurrentCondition,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
(pCurrentCondition)->EquationIdVector(rEquationId, rCurrentProcessInfo);
}


virtual void GetElementalDofList(Element::Pointer pCurrentElement,
Element::DofsVectorType& rElementalDofList,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentElement->GetDofList(rElementalDofList, rCurrentProcessInfo);
}


virtual void GetConditionDofList(Condition::Pointer pCurrentCondition,
Element::DofsVectorType& rConditionDofList,
ProcessInfo& rCurrentProcessInfo)
{
pCurrentCondition->GetDofList(rConditionDofList, rCurrentProcessInfo);
}




void SetOptions(Flags& rOptions)
{
mOptions = rOptions;
}


Flags& GetOptions()
{
return mOptions;
}


void SetProcess( ProcessPointerType pProcess )
{
mProcesses.push_back(pProcess); 
}


void SetProcessVector( ProcessPointerVectorType& rProcessVector )
{
mProcesses = rProcessVector;
}



protected:


Flags mOptions;

IntegrationMethodsVectorType mTimeVectorIntegrationMethods;
IntegrationMethodsScalarType mTimeScalarIntegrationMethods;

ProcessPointerVectorType mProcesses;




virtual void InitializeElements(ModelPart& rModelPart)
{
KRATOS_TRY

if( this->IsNot(LocalFlagType::ELEMENTS_INITIALIZED) ){

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Elements().size()); i++)
{
auto itElem = rModelPart.ElementsBegin() + i;
itElem->Initialize(rModelPart.GetProcessInfo());
}

this->Set(LocalFlagType::ELEMENTS_INITIALIZED, true);

}

KRATOS_CATCH("")
}


virtual void InitializeConditions(ModelPart& rModelPart)
{
KRATOS_TRY

if( this->IsNot(LocalFlagType::ELEMENTS_INITIALIZED) )
KRATOS_ERROR << "Before initilizing Conditions, initialize Elements FIRST" << std::endl;

if( this->IsNot(LocalFlagType::CONDITIONS_INITIALIZED) ){

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rModelPart.Conditions().size()); i++)
{
auto itCond = rModelPart.ConditionsBegin() + i;
itCond->Initialize(rModelPart.GetProcessInfo());
}

this->Set(LocalFlagType::CONDITIONS_INITIALIZED, true);
}

KRATOS_CATCH("")
}


virtual void InitializeNonLinearIteration(Condition::Pointer rCurrentCondition,
ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY

rCurrentCondition->InitializeNonLinearIteration(rCurrentProcessInfo);

KRATOS_CATCH("")
}


virtual void InitializeNonLinearIteration(Element::Pointer rCurrentElement,
ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY

rCurrentElement->InitializeNonLinearIteration(rCurrentProcessInfo);

KRATOS_CATCH("")
}


virtual void IntegrationMethodUpdate(NodeType& rNode)
{
for(typename IntegrationMethodsVectorType::iterator it=mTimeVectorIntegrationMethods.begin();
it!=mTimeVectorIntegrationMethods.end(); ++it)
(*it)->Update(rNode);
for(typename IntegrationMethodsScalarType::iterator it=mTimeScalarIntegrationMethods.begin();
it!=mTimeScalarIntegrationMethods.end(); ++it)
(*it)->Update(rNode);
}

virtual void IntegrationMethodPredict(NodeType& rNode)
{
for(typename IntegrationMethodsVectorType::iterator it=mTimeVectorIntegrationMethods.begin();
it!=mTimeVectorIntegrationMethods.end(); ++it)
(*it)->Predict(rNode);
for(typename IntegrationMethodsScalarType::iterator it=mTimeScalarIntegrationMethods.begin();
it!=mTimeScalarIntegrationMethods.end(); ++it)
(*it)->Predict(rNode);
}





private:








}; 







}  

#endif 
