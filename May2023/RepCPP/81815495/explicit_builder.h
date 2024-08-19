
#if !defined(KRATOS_EXPLICIT_BUILDER)
#define  KRATOS_EXPLICIT_BUILDER

#include <set>
#include <unordered_set>


#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/parallel_utilities.h"
#include "utilities/constraint_utilities.h"
#include "includes/kratos_parameters.h"
#include "factories/factory.h"
#include "utilities/atomic_utilities.h"

namespace Kratos
{










template<class TSparseSpace, class TDenseSpace >
class ExplicitBuilder
{
public:

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef typename TSparseSpace::DataType TDataType;

typedef typename TSparseSpace::MatrixType TSystemMatrixType;

typedef typename TSparseSpace::VectorType TSystemVectorType;

typedef typename TSparseSpace::MatrixPointerType TSystemMatrixPointerType;

typedef typename TSparseSpace::VectorPointerType TSystemVectorPointerType;

typedef typename TDenseSpace::MatrixType LocalSystemMatrixType;

typedef typename TDenseSpace::VectorType LocalSystemVectorType;

typedef ModelPart::DofType DofType;

typedef ModelPart::DofsArrayType DofsArrayType;

typedef ModelPart::DofsVectorType DofsVectorType;

typedef typename std::unordered_set<DofType::Pointer, DofPointerHasher> DofSetType;

typedef ModelPart::NodesContainerType NodesArrayType;
typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::ConditionsContainerType ConditionsArrayType;

typedef PointerVectorSet<Element, IndexedObject> ElementsContainerType;

typedef ExplicitBuilder<TSparseSpace, TDenseSpace> ClassType;

KRATOS_CLASS_POINTER_DEFINITION(ExplicitBuilder);



explicit ExplicitBuilder(Parameters ThisParameters)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ExplicitBuilder() = default;


virtual ~ExplicitBuilder() = default;



virtual typename ClassType::Pointer Create(Parameters ThisParameters) const
{
return Kratos::make_shared<ClassType>(ThisParameters);
}





bool GetCalculateReactionsFlag() const
{
return mCalculateReactionsFlag;
}


void SetCalculateReactionsFlag(bool CalculateReactionsFlag)
{
mCalculateReactionsFlag = CalculateReactionsFlag;
}


bool GetDofSetIsInitializedFlag() const
{
return mDofSetIsInitialized;
}


void SetDofSetIsInitializedFlag(bool DofSetIsInitialized)
{
mDofSetIsInitialized = DofSetIsInitialized;
}


bool GetResetDofSetFlag() const
{
return mResetDofSetFlag;
}


void SetResetDofSetFlag(bool ResetDofSetFlag)
{
mResetDofSetFlag = ResetDofSetFlag;
}


bool GetResetLumpedMassVectorFlag() const
{
return mResetLumpedMassVectorFlag;
}


void SetResetLumpedMassVectorFlag(bool ResetLumpedMassVectorFlag)
{
mResetLumpedMassVectorFlag = ResetLumpedMassVectorFlag;
}


unsigned int GetEquationSystemSize() const
{
return mEquationSystemSize;
}


DofsArrayType& GetDofSet()
{
return mDofSet;
}


const DofsArrayType& GetDofSet() const
{
return mDofSet;
}


TSystemVectorPointerType& pGetLumpedMassMatrixVector()
{
return mpLumpedMassVector;
}


TSystemVectorType& GetLumpedMassMatrixVector()
{
KRATOS_ERROR_IF_NOT(mpLumpedMassVector) << "Lumped mass matrix vector is not initialized!" << std::endl;
return (*mpLumpedMassVector);
}


virtual void BuildRHS(ModelPart& rModelPart)
{
KRATOS_TRY

BuildRHSNoDirichlet(rModelPart);

KRATOS_CATCH("")
}


virtual void BuildRHSNoDirichlet(ModelPart& rModelPart)
{
KRATOS_TRY

InitializeDofSetReactions();

const auto &r_elements_array = rModelPart.Elements();
const auto &r_conditions_array = rModelPart.Conditions();
const int n_elems = static_cast<int>(r_elements_array.size());
const int n_conds = static_cast<int>(r_conditions_array.size());

const auto& r_process_info = rModelPart.GetProcessInfo();

#pragma omp parallel firstprivate(n_elems, n_conds)
{
#pragma omp for schedule(guided, 512) nowait
for (int i_elem = 0; i_elem < n_elems; ++i_elem) {
auto it_elem = r_elements_array.begin() + i_elem;
if (it_elem->IsActive()) {
it_elem->AddExplicitContribution(r_process_info);
}
}

#pragma omp for schedule(guided, 512)
for (int i_cond = 0; i_cond < n_conds; ++i_cond) {
auto it_cond = r_conditions_array.begin() + i_cond;
if (it_cond->IsActive()) {
it_cond->AddExplicitContribution(r_process_info);
}
}
}

KRATOS_CATCH("")
}


virtual void ApplyConstraints(ModelPart& rModelPart)
{
ConstraintUtilities::ResetSlaveDofs(rModelPart);

ConstraintUtilities::ApplyConstraints(rModelPart);
}


virtual void Initialize(ModelPart& rModelPart)
{
if (!mDofSetIsInitialized) {
this->SetUpDofSet(rModelPart);
this->SetUpDofSetEquationIds();
this->SetUpLumpedMassVector(rModelPart);
} else if (!mLumpedMassVectorIsInitialized) {
KRATOS_WARNING("ExplicitBuilder") << "Calling Initialize() with already initialized DOF set. Initializing lumped mass vector." << std::endl;;
this->SetUpLumpedMassVector(rModelPart);
} else {
KRATOS_WARNING("ExplicitBuilder") << "Calling Initialize() with already initialized DOF set and lumped mass vector." << std::endl;;
}
}


virtual void InitializeSolutionStep(ModelPart& rModelPart)
{
if (mResetDofSetFlag) {
this->SetUpDofSet(rModelPart);
this->SetUpDofSetEquationIds();
this->SetUpLumpedMassVector(rModelPart);
} else if (mResetLumpedMassVectorFlag) {
this->SetUpLumpedMassVector(rModelPart);
}

this->InitializeDofSetReactions();
}


virtual void FinalizeSolutionStep(ModelPart& rModelPart)
{
if (mCalculateReactionsFlag) {
this->CalculateReactions();
}
}


virtual void Clear()
{
this->mDofSet = DofsArrayType();
this->mpLumpedMassVector.reset();

KRATOS_INFO_IF("ExplicitBuilder", this->GetEchoLevel() > 0) << "Clear Function called" << std::endl;
}


virtual int Check(const ModelPart& rModelPart) const
{
KRATOS_TRY

return 0;

KRATOS_CATCH("");
}


virtual Parameters GetDefaultParameters() const
{
const Parameters default_parameters = Parameters(R"(
{
"name" : "explicit_builder"
})");
return default_parameters;
}


static std::string Name()
{
return "explicit_builder";
}


void SetEchoLevel(int Level)
{
mEchoLevel = Level;
}


int GetEchoLevel() const
{
return mEchoLevel;
}






virtual std::string Info() const
{
return "ExplicitBuilder";
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



DofsArrayType mDofSet; 

TSystemVectorPointerType mpLumpedMassVector; 

bool mResetDofSetFlag = false;  

bool mResetLumpedMassVectorFlag = false;  

bool mDofSetIsInitialized = false; 

bool mLumpedMassVectorIsInitialized = false; 

bool mCalculateReactionsFlag = false; 

unsigned int mEquationSystemSize; 

int mEchoLevel = 0;





virtual void SetUpDofSet(const ModelPart& rModelPart)
{
KRATOS_TRY;

KRATOS_INFO_IF("ExplicitBuilder", this->GetEchoLevel() > 1) << "Setting up the dofs" << std::endl;

const auto &r_elements_array = rModelPart.Elements();
const auto &r_conditions_array = rModelPart.Conditions();
const auto &r_constraints_array = rModelPart.MasterSlaveConstraints();
const int n_elems = static_cast<int>(r_elements_array.size());
const int n_conds = static_cast<int>(r_conditions_array.size());
const int n_constraints = static_cast<int>(r_constraints_array.size());

DofSetType dof_global_set;
dof_global_set.reserve(n_elems*20);

DofsVectorType dof_list;
DofsVectorType second_dof_list; 

#pragma omp parallel firstprivate(dof_list, second_dof_list)
{
const auto& r_process_info = rModelPart.GetProcessInfo();

DofSetType dofs_tmp_set;
dofs_tmp_set.reserve(20000);

#pragma omp for schedule(guided, 512) nowait
for (int i_elem = 0; i_elem < n_elems; ++i_elem) {
const auto it_elem = r_elements_array.begin() + i_elem;
it_elem->GetDofList(dof_list, r_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
}

#pragma omp for schedule(guided, 512) nowait
for (int i_cond = 0; i_cond < n_conds; ++i_cond) {
const auto it_cond = r_conditions_array.begin() + i_cond;
it_cond->GetDofList(dof_list, r_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
}

#pragma omp for  schedule(guided, 512) nowait
for (int i_const = 0; i_const < n_constraints; ++i_const) {
auto it_const = r_constraints_array.begin() + i_const;
it_const->GetDofList(dof_list, second_dof_list, r_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
dofs_tmp_set.insert(second_dof_list.begin(), second_dof_list.end());
}

#pragma omp critical
{
dof_global_set.insert(dofs_tmp_set.begin(), dofs_tmp_set.end());
}
}

KRATOS_INFO_IF("ExplicitBuilder", ( this->GetEchoLevel() > 2)) << "Initializing ordered array filling\n" << std::endl;

mDofSet = DofsArrayType();
DofsArrayType temp_dof_set;
temp_dof_set.reserve(dof_global_set.size());
for (auto it_dof = dof_global_set.begin(); it_dof != dof_global_set.end(); ++it_dof) {
temp_dof_set.push_back(*it_dof);
}
temp_dof_set.Sort();
mDofSet = temp_dof_set;
mEquationSystemSize = mDofSet.size();

KRATOS_ERROR_IF(mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

for (auto it_dof = mDofSet.begin(); it_dof != mDofSet.end(); ++it_dof) {
KRATOS_ERROR_IF_NOT(it_dof->HasReaction()) << "Reaction variable not set for the following : " << std::endl
<< "Node : " << it_dof->Id() << std::endl
<< "Dof : " << (*it_dof) << std::endl << "Not possible to calculate reactions." << std::endl;
}

mDofSetIsInitialized = true;

KRATOS_INFO_IF("ExplicitBuilder", ( this->GetEchoLevel() > 2)) << "Number of degrees of freedom:" << mDofSet.size() << std::endl;
KRATOS_INFO_IF("ExplicitBuilder", ( this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0)) << "Finished setting up the dofs" << std::endl;
KRATOS_INFO_IF("ExplicitBuilder", ( this->GetEchoLevel() > 2)) << "End of setup dof set\n" << std::endl;

KRATOS_CATCH("");
}


virtual void SetUpDofSetEquationIds()
{
KRATOS_ERROR_IF_NOT(mDofSetIsInitialized) << "Trying to set the equation ids. before initializing the DOF set. Please call the SetUpDofSet() before." << std::endl;
KRATOS_ERROR_IF(mEquationSystemSize == 0) << "Trying to set the equation ids. in an empty DOF set (equation system size is 0)." << std::endl;

IndexPartition<int>(mEquationSystemSize).for_each(
[&](int i_dof){
auto it_dof = mDofSet.begin() + i_dof;
it_dof->SetEquationId(i_dof);
}
);
}


virtual void SetUpLumpedMassVector(const ModelPart &rModelPart)
{
KRATOS_TRY;

KRATOS_INFO_IF("ExplicitBuilder", this->GetEchoLevel() > 1) << "Setting up the lumped mass matrix vector" << std::endl;

mpLumpedMassVector = TSystemVectorPointerType(new TSystemVectorType(GetDofSet().size()));
TDenseSpace::SetToZero(*mpLumpedMassVector);

LocalSystemVectorType elem_mass_vector;
std::vector<std::size_t> elem_equation_id;
const auto &r_elements_array = rModelPart.Elements();
const auto &r_process_info = rModelPart.GetProcessInfo();
const int n_elems = static_cast<int>(r_elements_array.size());

#pragma omp for private(elem_mass_vector) schedule(guided, 512) nowait
for (int i_elem = 0; i_elem < n_elems; ++i_elem) {
const auto it_elem = r_elements_array.begin() + i_elem;

it_elem->CalculateLumpedMassVector(elem_mass_vector, r_process_info);
it_elem->EquationIdVector(elem_equation_id, r_process_info);

for (IndexType i = 0; i < elem_equation_id.size(); ++i) {
AtomicAdd((*mpLumpedMassVector)[elem_equation_id[i]], elem_mass_vector(i));
}
}

mLumpedMassVectorIsInitialized = true;

KRATOS_CATCH("");
}


virtual void InitializeDofSetReactions()
{
KRATOS_ERROR_IF_NOT(mDofSetIsInitialized) << "Trying to initialize the explicit residual but the DOFs set is not initialized yet." << std::endl;
KRATOS_ERROR_IF(mEquationSystemSize == 0) << "Trying to set the equation ids. in an empty DOF set (equation system size is 0)." << std::endl;

block_for_each(
mDofSet,
[](DofType& rDof){
rDof.GetSolutionStepReactionValue() = 0.0;
}
);
}


virtual void CalculateReactions()
{
if (mCalculateReactionsFlag) {
KRATOS_ERROR_IF_NOT(mDofSetIsInitialized) << "Trying to initialize the explicit residual but the DOFs set is not initialized yet." << std::endl;
KRATOS_ERROR_IF(mEquationSystemSize == 0) << "Trying to set the equation ids. in an empty DOF set (equation system size is 0)." << std::endl;

block_for_each(
mDofSet,
[](DofType& rDof){
auto& r_reaction_value = rDof.GetSolutionStepReactionValue();
r_reaction_value *= -1.0;
}
);
}
}


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

static std::vector<Internals::RegisteredPrototypeBase<ClassType>> msPrototypes;












}; 


} 

#endif 
