
#pragma once




#include "concurrentqueue/concurrentqueue.h"


#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "custom_strategies/rom_builder_and_solver.h"
#include "utilities/builtin_timer.h"
#include "utilities/reduction_utilities.h"
#include "utilities/dense_householder_qr_decomposition.h"


#include "rom_application_variables.h"
#include "custom_utilities/rom_auxiliary_utilities.h"

namespace Kratos
{












template <class TSparseSpace, class TDenseSpace, class TLinearSolver>
class PetrovGalerkinROMBuilderAndSolver : public ROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:



KRATOS_CLASS_POINTER_DEFINITION(PetrovGalerkinROMBuilderAndSolver);

typedef std::size_t SizeType;
typedef std::size_t IndexType;

typedef PetrovGalerkinROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef ROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;
typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;
typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef typename ModelPart::MasterSlaveConstraintContainerType MasterSlaveConstraintContainerType;
typedef Element::EquationIdVectorType EquationIdVectorType;
typedef Element::DofsVectorType DofsVectorType;

typedef LocalSystemMatrixType RomSystemMatrixType;
typedef LocalSystemVectorType RomSystemVectorType;

typedef RomSystemMatrixType PetrovGalerkinSystemMatrixType; 
typedef RomSystemVectorType PetrovGalerkinSystemVectorType;

typedef Node NodeType;
typedef typename NodeType::DofType DofType;
typedef typename DofType::Pointer DofPointerType;
typedef moodycamel::ConcurrentQueue<DofType::Pointer> DofQueue;


explicit PetrovGalerkinROMBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters)
: ROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver) 
{
Parameters this_parameters_copy = ThisParameters.Clone();
this_parameters_copy = this->ValidateAndAssignParameters(this_parameters_copy, this->GetDefaultParameters());
this->AssignSettings(this_parameters_copy);
}

~PetrovGalerkinROMBuilderAndSolver() = default;




void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 1)) << "Setting up the dofs" << std::endl;
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Number of threads" << ParallelUtilities::GetNumThreads() << "\n" << std::endl;
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Initializing element loop" << std::endl;

if (this->mHromWeightsInitialized == false) {
this->InitializeHROMWeights(rModelPart);
}

auto dof_queue = this->ExtractDofSet(pScheme, rModelPart);

KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Initializing ordered array filling\n" << std::endl;
auto dof_array = this->SortAndRemoveDuplicateDofs(dof_queue);

BaseType::GetDofSet().swap(dof_array);
BaseType::SetDofSetIsInitializedFlag(true);

KRATOS_ERROR_IF(BaseType::GetDofSet().size() == 0) << "No degrees of freedom!" << std::endl;
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Number of degrees of freedom:" << BaseType::GetDofSet().size() << std::endl;
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Finished setting up the dofs" << std::endl;

#ifdef KRATOS_DEBUG
if (BaseType::GetCalculateReactionsFlag())
{
for (const auto& r_dof: BaseType::GetDofSet())
{
KRATOS_ERROR_IF_NOT(r_dof.HasReaction())
<< "Reaction variable not set for the following :\n"
<< "Node : " << r_dof.Id() << '\n'
<< "Dof  : " << r_dof      << '\n'
<< "Not possible to calculate reactions." << std::endl;
}
}
#endif
KRATOS_CATCH("");
} 

void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
PetrovGalerkinSystemMatrixType Arom = ZeroMatrix(mNumberOfPetrovGalerkinRomModes, this->GetNumberOfROMModes());
PetrovGalerkinSystemVectorType brom = ZeroVector(mNumberOfPetrovGalerkinRomModes);
BuildROM(pScheme, rModelPart, Arom, brom);
SolveROM(rModelPart, Arom, brom, Dx);


}

Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "petrov_galerkin_rom_builder_and_solver",
"nodal_unknowns" : [],
"number_of_rom_dofs" : 10,
"petrov_galerkin_number_of_rom_dofs" : 10
})");
default_parameters.AddMissingParameters(BaseType::GetDefaultParameters());

return default_parameters;
}

static std::string Name() 
{
return "petrov_galerkin_rom_builder_and_solver";
}






virtual std::string Info() const override
{
return "PetrovGalerkinROMBuilderAndSolver";
}

virtual void PrintInfo(std::ostream &rOStream) const override
{
rOStream << Info();
}

virtual void PrintData(std::ostream &rOStream) const override
{
rOStream << Info();
}



protected:



SizeType mNumberOfPetrovGalerkinRomModes;




void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mNumberOfPetrovGalerkinRomModes = ThisParameters["petrov_galerkin_number_of_rom_dofs"].GetInt();
}


struct AssemblyTLS
{
AssemblyTLS(SizeType NRomModes, SizeType NPetrovGalerkinRomModes)
: romA(ZeroMatrix(NPetrovGalerkinRomModes, NRomModes)),
romB(ZeroVector(NPetrovGalerkinRomModes))
{ }
AssemblyTLS() = delete;

Matrix phiE = {};                
Matrix psiE = {};                
LocalSystemMatrixType lhs = {};  
LocalSystemVectorType rhs = {};  
EquationIdVectorType eq_id = {}; 
DofsVectorType dofs = {};        
PetrovGalerkinSystemMatrixType romA;        
PetrovGalerkinSystemVectorType romB;        
RomSystemMatrixType aux = {};    
};


template<typename T>
struct NonTrivialSumReduction
{
typedef T value_type;
typedef T return_type;

T mValue;
bool mInitialized = false;

void Init(const value_type& first_value)
{
mValue = first_value;
mInitialized = true;
}

return_type GetValue() const
{
return mValue;
}

void LocalReduce(const value_type& value)
{
if(!mInitialized) {
Init(value);
} else {
noalias(mValue) += value;
}
}

void ThreadSafeReduce(const NonTrivialSumReduction& rOther)
{
if(!rOther.mInitialized) return;

const std::lock_guard<LockObject> scope_lock(ParallelUtilities::GetGlobalLock());
LocalReduce(rOther.mValue);
}
};


template<typename TMatrix>
static void ResizeIfNeeded(TMatrix& mat, const SizeType rows, const SizeType cols)
{
if(mat.size1() != rows || mat.size2() != cols) {
mat.resize(rows, cols, false);
}
};


void BuildROM(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
PetrovGalerkinSystemMatrixType &rA,
PetrovGalerkinSystemVectorType &rb) override
{
rA = ZeroMatrix(mNumberOfPetrovGalerkinRomModes, this->GetNumberOfROMModes());
rb = ZeroVector(mNumberOfPetrovGalerkinRomModes);

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const auto& r_current_process_info = rModelPart.GetProcessInfo();


const auto assembling_timer = BuiltinTimer();

using SystemSumReducer = CombinedReduction<NonTrivialSumReduction<PetrovGalerkinSystemMatrixType>, NonTrivialSumReduction<PetrovGalerkinSystemVectorType>>;
AssemblyTLS assembly_tls_container(this->GetNumberOfROMModes(), mNumberOfPetrovGalerkinRomModes);

const auto& r_elements = this->mHromSimulation ? this->mSelectedElements : rModelPart.Elements();

if(!r_elements.empty())
{
std::tie(rA, rb) =
block_for_each<SystemSumReducer>(r_elements, assembly_tls_container, 
[&](Element& r_element, AssemblyTLS& r_thread_prealloc)
{
return CalculateLocalContributionPetrovGalerkin(r_element, rA, rb, r_thread_prealloc, *pScheme, r_current_process_info);
});
}


const auto& r_conditions = this->mHromSimulation ? this->mSelectedConditions : rModelPart.Conditions();

if(!r_conditions.empty())
{
PetrovGalerkinSystemMatrixType aconditions;
PetrovGalerkinSystemVectorType bconditions;

std::tie(aconditions, bconditions) =
block_for_each<SystemSumReducer>(r_conditions, assembly_tls_container, 
[&](Condition& r_condition, AssemblyTLS& r_thread_prealloc)
{
return CalculateLocalContributionPetrovGalerkin(r_condition, rA, rb, r_thread_prealloc, *pScheme, r_current_process_info);
});

rA += aconditions;
rb += bconditions;
}

KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Build time: " << assembling_timer.ElapsedSeconds() << std::endl;
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Finished parallel building" << std::endl;

}


void SolveROM(
ModelPart &rModelPart,
PetrovGalerkinSystemMatrixType &rA,
PetrovGalerkinSystemVectorType &rb,
TSystemVectorType &rDx) override
{
KRATOS_TRY

RomSystemVectorType dxrom(this->GetNumberOfROMModes());

const auto solving_timer = BuiltinTimer();
DenseHouseholderQRDecomposition<TDenseSpace> qr_decomposition;
qr_decomposition.Compute(rA);
qr_decomposition.Solve(rb, dxrom);
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Solve reduced system time: " << solving_timer.ElapsedSeconds() << std::endl;

auto& r_root_mp = rModelPart.GetRootModelPart();
noalias(r_root_mp.GetValue(ROM_SOLUTION_INCREMENT)) += dxrom;

const auto backward_projection_timer = BuiltinTimer();
this->ProjectToFineBasis(dxrom, rModelPart, rDx);
KRATOS_INFO_IF("PetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Project to fine basis time: " << backward_projection_timer.ElapsedSeconds() << std::endl;

KRATOS_CATCH("")
}






private:


template<typename TEntity>
std::tuple<LocalSystemMatrixType, LocalSystemVectorType> CalculateLocalContributionPetrovGalerkin(
TEntity& rEntity,
PetrovGalerkinSystemMatrixType& rAglobal,
PetrovGalerkinSystemVectorType& rBglobal,
AssemblyTLS& rPreAlloc,
TSchemeType& rScheme,
const ProcessInfo& rCurrentProcessInfo)
{
if (rEntity.IsDefined(ACTIVE) && rEntity.IsNot(ACTIVE))
{
rPreAlloc.romA = ZeroMatrix(mNumberOfPetrovGalerkinRomModes, this->GetNumberOfROMModes());
rPreAlloc.romB = ZeroVector(mNumberOfPetrovGalerkinRomModes);
return std::tie(rPreAlloc.romA, rPreAlloc.romB);
}

rScheme.CalculateSystemContributions(rEntity, rPreAlloc.lhs, rPreAlloc.rhs, rPreAlloc.eq_id, rCurrentProcessInfo);
rEntity.GetDofList(rPreAlloc.dofs, rCurrentProcessInfo);

const SizeType ndofs = rPreAlloc.dofs.size();
ResizeIfNeeded(rPreAlloc.phiE, ndofs, this->GetNumberOfROMModes());
ResizeIfNeeded(rPreAlloc.psiE, ndofs, mNumberOfPetrovGalerkinRomModes);
ResizeIfNeeded(rPreAlloc.aux, ndofs, this->GetNumberOfROMModes());

const auto &r_geom = rEntity.GetGeometry();
RomAuxiliaryUtilities::GetPhiElemental(rPreAlloc.phiE, rPreAlloc.dofs, r_geom, this->mMapPhi);
RomAuxiliaryUtilities::GetPsiElemental(rPreAlloc.psiE, rPreAlloc.dofs, r_geom, this->mMapPhi);

const double h_rom_weight = this->mHromSimulation ? rEntity.GetValue(HROM_WEIGHT) : 1.0;

noalias(rPreAlloc.aux) = prod(rPreAlloc.lhs, rPreAlloc.phiE);
noalias(rPreAlloc.romA) = prod(trans(rPreAlloc.psiE), rPreAlloc.aux) * h_rom_weight;
noalias(rPreAlloc.romB) = prod(trans(rPreAlloc.psiE), rPreAlloc.rhs) * h_rom_weight;

return std::tie(rPreAlloc.romA, rPreAlloc.romB);
}


}; 




} 

