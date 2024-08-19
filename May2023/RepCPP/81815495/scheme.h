
#pragma once



#include "includes/model_part.h"
#include "utilities/openmp_utils.h" 
#include "includes/kratos_parameters.h"
#include "utilities/entities_utilities.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{


template<class TSparseSpace,
class TDenseSpace 
>
class Scheme
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Scheme);

using ClassType = Scheme<TSparseSpace, TDenseSpace>;

using TDataType = typename TSparseSpace::DataType;

using TSystemMatrixType = typename TSparseSpace::MatrixType;

using TSystemVectorType = typename TSparseSpace::VectorType;

using LocalSystemMatrixType = typename TDenseSpace::MatrixType;

using LocalSystemVectorType = typename TDenseSpace::VectorType;

using TDofType = Dof<double>;

using DofsArrayType = ModelPart::DofsArrayType;

using ElementsArrayType = ModelPart::ElementsContainerType;

using ConditionsArrayType = ModelPart::ConditionsContainerType;



explicit Scheme()
{
mSchemeIsInitialized = false;
mElementsAreInitialized = false;
mConditionsAreInitialized = false;
}

explicit Scheme(Parameters ThisParameters)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);

mSchemeIsInitialized = false;
mElementsAreInitialized = false;
mConditionsAreInitialized = false;
}


explicit Scheme(Scheme& rOther)
:mSchemeIsInitialized(rOther.mSchemeIsInitialized)
,mElementsAreInitialized(rOther.mElementsAreInitialized)
,mConditionsAreInitialized(rOther.mConditionsAreInitialized)
{
}


virtual ~Scheme()
{
}




virtual typename ClassType::Pointer Create(Parameters ThisParameters) const
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


virtual Pointer Clone()
{
return Kratos::make_shared<Scheme>(*this) ;
}


virtual void Initialize(ModelPart& rModelPart)
{
KRATOS_TRY
mSchemeIsInitialized = true;
KRATOS_CATCH("")
}


bool SchemeIsInitialized()
{
return mSchemeIsInitialized;
}


void SetSchemeIsInitialized(bool SchemeIsInitializedFlag = true)
{
mSchemeIsInitialized = SchemeIsInitializedFlag;
}


bool ElementsAreInitialized()
{
return mElementsAreInitialized;
}


void SetElementsAreInitialized(bool ElementsAreInitializedFlag = true)
{
mElementsAreInitialized = ElementsAreInitializedFlag;
}


bool ConditionsAreInitialized()
{
return mConditionsAreInitialized;
}


void SetConditionsAreInitialized(bool ConditionsAreInitializedFlag = true)
{
mConditionsAreInitialized = ConditionsAreInitializedFlag;
}


virtual void InitializeElements( ModelPart& rModelPart)
{
KRATOS_TRY

EntitiesUtilities::InitializeEntities<Element>(rModelPart);

SetElementsAreInitialized();

KRATOS_CATCH("")
}


virtual void InitializeConditions(ModelPart& rModelPart)
{
KRATOS_TRY

KRATOS_ERROR_IF_NOT(mElementsAreInitialized) << "Before initializing Conditions, initialize Elements FIRST" << std::endl;

EntitiesUtilities::InitializeEntities<Condition>(rModelPart);

SetConditionsAreInitialized();

KRATOS_CATCH("")
}


virtual void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
)
{
KRATOS_TRY

EntitiesUtilities::InitializeSolutionStepAllEntities(rModelPart);

KRATOS_CATCH("")
}


virtual void FinalizeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b)
{
KRATOS_TRY

EntitiesUtilities::FinalizeSolutionStepAllEntities(rModelPart);

KRATOS_CATCH("")
}










virtual void InitializeNonLinIteration(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
)
{
KRATOS_TRY

EntitiesUtilities::InitializeNonLinearIterationAllEntities(rModelPart);

KRATOS_CATCH("")
}


virtual void FinalizeNonLinIteration(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
)
{
KRATOS_TRY

EntitiesUtilities::FinalizeNonLinearIterationAllEntities(rModelPart);

KRATOS_CATCH("")
}


virtual void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
)
{
KRATOS_TRY
KRATOS_CATCH("")
}


virtual void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
)
{
KRATOS_TRY
KRATOS_CATCH("")
}


virtual void CalculateOutputData(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
)
{
KRATOS_TRY
KRATOS_CATCH("")
}


virtual void CleanOutputData()
{
KRATOS_TRY
KRATOS_CATCH("")
}


virtual void Clean()
{
KRATOS_TRY
KRATOS_CATCH("")
}


virtual void Clear()
{
KRATOS_TRY
KRATOS_CATCH("")
}


virtual int Check(const ModelPart& rModelPart) const
{
KRATOS_TRY
return 0;
KRATOS_CATCH("");
}

virtual int Check(ModelPart& rModelPart)
{
const Scheme& r_const_this = *this;
const ModelPart& r_const_model_part = rModelPart;
return r_const_this.Check(r_const_model_part);
}


virtual void CalculateSystemContributions(
Element& rElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationIdVector,
const ProcessInfo& rCurrentProcessInfo
)
{
rElement.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, rCurrentProcessInfo);
}


virtual void CalculateSystemContributions(
Condition& rCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationIdVector,
const ProcessInfo& rCurrentProcessInfo
)
{
rCondition.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, rCurrentProcessInfo);
}


virtual void CalculateRHSContribution(
Element& rElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationIdVector,
const ProcessInfo& rCurrentProcessInfo
)
{
rElement.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);
}


virtual void CalculateRHSContribution(
Condition& rCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationIdVector,
const ProcessInfo& rCurrentProcessInfo
)
{
rCondition.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);
}


virtual void CalculateLHSContribution(
Element& rElement,
LocalSystemMatrixType& LHS_Contribution,
Element::EquationIdVectorType& rEquationIdVector,
const ProcessInfo& rCurrentProcessInfo
)
{
rElement.CalculateLeftHandSide(LHS_Contribution, rCurrentProcessInfo);
}


virtual void CalculateLHSContribution(
Condition& rCondition,
LocalSystemMatrixType& LHS_Contribution,
Element::EquationIdVectorType& rEquationIdVector,
const ProcessInfo& rCurrentProcessInfo
)
{
rCondition.CalculateLeftHandSide(LHS_Contribution, rCurrentProcessInfo);
}


virtual void EquationId(
const Element& rElement,
Element::EquationIdVectorType& rEquationId,
const ProcessInfo& rCurrentProcessInfo
)
{
rElement.EquationIdVector(rEquationId, rCurrentProcessInfo);
}


virtual void EquationId(
const Condition& rCondition,
Element::EquationIdVectorType& rEquationId,
const ProcessInfo& rCurrentProcessInfo
)
{
rCondition.EquationIdVector(rEquationId, rCurrentProcessInfo);
}


virtual void GetDofList(
const Element& rElement,
Element::DofsVectorType& rDofList,
const ProcessInfo& rCurrentProcessInfo
)
{
rElement.GetDofList(rDofList, rCurrentProcessInfo);
}


virtual void GetDofList(
const Condition& rCondition,
Element::DofsVectorType& rDofList,
const ProcessInfo& rCurrentProcessInfo
)
{
rCondition.GetDofList(rDofList, rCurrentProcessInfo);
}


virtual Parameters GetDefaultParameters() const
{
const Parameters default_parameters = Parameters(R"(
{
"name" : "scheme"
})" );
return default_parameters;
}


static std::string Name()
{
return "scheme";
}




virtual std::string Info() const
{
return "Scheme";
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


bool mSchemeIsInitialized;      
bool mElementsAreInitialized;   
bool mConditionsAreInitialized; 




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
}





private:








}; 

} 
