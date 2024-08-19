
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
#include "custom_utilities/rom_residuals_utility.h"

namespace Kratos
{












template <class TSparseSpace, class TDenseSpace, class TLinearSolver>
class LeastSquaresPetrovGalerkinROMBuilderAndSolver : public ROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:


KRATOS_CLASS_POINTER_DEFINITION(LeastSquaresPetrovGalerkinROMBuilderAndSolver);

typedef std::size_t SizeType;
typedef std::size_t IndexType;

typedef LeastSquaresPetrovGalerkinROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

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

typedef RomSystemMatrixType LSPGSystemMatrixType; 
typedef RomSystemVectorType LSPGSystemVectorType;

typedef Node NodeType;
typedef typename NodeType::DofType DofType;
typedef typename DofType::Pointer DofPointerType;
typedef moodycamel::ConcurrentQueue<DofType::Pointer> DofQueue;


explicit LeastSquaresPetrovGalerkinROMBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters) : BaseType(pNewLinearSystemSolver) 
{
Parameters this_parameters_copy = ThisParameters.Clone();
this_parameters_copy = this->ValidateAndAssignParameters(this_parameters_copy, this->GetDefaultParameters());
this->AssignSettings(this_parameters_copy);
} 

~LeastSquaresPetrovGalerkinROMBuilderAndSolver() = default;




void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 1)) << "Setting up the dofs" << std::endl;
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Number of threads" << ParallelUtilities::GetNumThreads() << "\n" << std::endl;
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Initializing element loop" << std::endl;

if (this->mHromWeightsInitialized == false) {
this->InitializeHROMWeights(rModelPart);
}

auto dof_queue = this->ExtractDofSet(pScheme, rModelPart);

KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Initializing ordered array filling\n" << std::endl;
auto dof_array = this->SortAndRemoveDuplicateDofs(dof_queue);

BaseType::GetDofSet().swap(dof_array);
BaseType::SetDofSetIsInitializedFlag(true);

KRATOS_ERROR_IF(BaseType::GetDofSet().size() == 0) << "No degrees of freedom!" << std::endl;
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Number of degrees of freedom:" << BaseType::GetDofSet().size() << std::endl;
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Finished setting up the dofs" << std::endl;

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
KRATOS_TRY
LSPGSystemMatrixType Arom = ZeroMatrix(BaseType::GetEquationSystemSize(), this->GetNumberOfROMModes());
LSPGSystemVectorType brom = ZeroVector(BaseType::GetEquationSystemSize());
BuildROM(pScheme, rModelPart, Arom, brom);

if (mTrainPetrovGalerkinFlag){
TSystemVectorType r_residual;
GetAssembledResiduals(pScheme, rModelPart, r_residual);
}

SolveROM(rModelPart, Arom, brom, Dx);


KRATOS_CATCH("")
}

Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "lspg_rom_builder_and_solver",
"nodal_unknowns" : [],
"number_of_rom_dofs" : 10,
"train_petrov_galerkin" : false
})");
default_parameters.AddMissingParameters(BaseType::GetDefaultParameters());

return default_parameters;
}

static std::string Name() 
{
return "lspg_rom_builder_and_solver";
}






virtual std::string Info() const override
{
return "LeastSquaresPetrovGalerkinROMBuilderAndSolver";
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

SizeType mNodalDofs;
bool mTrainPetrovGalerkinFlag = false;







void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mTrainPetrovGalerkinFlag = ThisParameters["train_petrov_galerkin"].GetBool();
}


struct AssemblyTLS
{
Matrix phiE = {};                
LocalSystemMatrixType lhs = {};  
EquationIdVectorType eq_id = {}; 
DofsVectorType dofs = {};        
RomSystemMatrixType romA;        
RomSystemVectorType romB;        
};


template<typename TMatrix>
static void ResizeIfNeeded(TMatrix& rMat, const SizeType Rows, const SizeType Cols)

{
if(rMat.size1() != Rows || rMat.size2() != Cols) {
rMat.resize(Rows, Cols, false);
}
};


void BuildROM(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
LSPGSystemMatrixType &rA,
LSPGSystemVectorType &rb) override
{
KRATOS_TRY
rA = ZeroMatrix(BaseType::GetEquationSystemSize(), this->GetNumberOfROMModes());
rb = ZeroVector(BaseType::GetEquationSystemSize());

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const auto& r_current_process_info = rModelPart.GetProcessInfo();


const auto assembling_timer = BuiltinTimer();

AssemblyTLS assembly_tls_container;

const auto& r_elements = rModelPart.Elements();

if(!r_elements.empty())
{
block_for_each(r_elements, assembly_tls_container, 
[&](Element& r_element, AssemblyTLS& r_thread_prealloc)
{
CalculateLocalContributionLSPG(r_element, rA, rb, r_thread_prealloc, *pScheme, r_current_process_info);
});
}


const auto& r_conditions = rModelPart.Conditions();

if(!r_conditions.empty())
{
block_for_each(r_conditions, assembly_tls_container, 
[&](Condition& r_condition, AssemblyTLS& r_thread_prealloc)
{
CalculateLocalContributionLSPG(r_condition, rA, rb, r_thread_prealloc, *pScheme, r_current_process_info);
});
}

KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Build time: " << assembling_timer.ElapsedSeconds() << std::endl;
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Finished parallel building" << std::endl;

KRATOS_CATCH("")
}


void SolveROM(
ModelPart &rModelPart,
LSPGSystemMatrixType &rA,
LSPGSystemVectorType &rb,
TSystemVectorType &rDx) override
{
KRATOS_TRY

LSPGSystemVectorType dxrom(this->GetNumberOfROMModes());

const auto solving_timer = BuiltinTimer();
DenseHouseholderQRDecomposition<TDenseSpace> qr_decomposition;
qr_decomposition.Compute(rA);
qr_decomposition.Solve(rb, dxrom);
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Solve reduced system time: " << solving_timer.ElapsedSeconds() << std::endl;

auto& r_root_mp = rModelPart.GetRootModelPart();
noalias(r_root_mp.GetValue(ROM_SOLUTION_INCREMENT)) += dxrom;

const auto backward_projection_timer = BuiltinTimer();
this->ProjectToFineBasis(dxrom, rModelPart, rDx);
KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Project to fine basis time: " << backward_projection_timer.ElapsedSeconds() << std::endl;

KRATOS_CATCH("")
}


void GetAssembledResiduals(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
LSPGSystemVectorType &rb) 
{
KRATOS_TRY
const auto residual_writing_timer = BuiltinTimer();
rb = ZeroVector(BaseType::GetEquationSystemSize());

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const auto& r_current_process_info = rModelPart.GetProcessInfo();

AssemblyTLS assembly_tls_container;

const auto& r_elements = rModelPart.Elements();

if(!r_elements.empty())
{
block_for_each(r_elements, assembly_tls_container, 
[&](Element& r_element, AssemblyTLS& r_thread_prealloc)
{
CalculateAssembledResiduals(r_element, rb, r_thread_prealloc, *pScheme, r_current_process_info);
});
}


const auto& r_conditions = rModelPart.Conditions();

if(!r_conditions.empty())
{
block_for_each(r_conditions, assembly_tls_container, 
[&](Condition& r_condition, AssemblyTLS& r_thread_prealloc)
{
CalculateAssembledResiduals(r_condition, rb, r_thread_prealloc, *pScheme, r_current_process_info);
});
}

std::stringstream matrix_market_vector_name;
matrix_market_vector_name << "R_" << rModelPart.GetProcessInfo()[TIME] << "_" << rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] <<  ".res.mm";
SparseSpaceType::WriteMatrixMarketVector((matrix_market_vector_name.str()).c_str(), rb);

KRATOS_INFO_IF("LeastSquaresPetrovGalerkinROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Write residuals to train Petrov Galerkin time: " << residual_writing_timer.ElapsedSeconds() << std::endl;
KRATOS_CATCH("")
}






private:

SizeType mNumberOfRomModes;



template<typename TEntity>
void CalculateLocalContributionLSPG(
TEntity& rEntity,
LSPGSystemMatrixType& rAglobal,
LSPGSystemVectorType& rBglobal,
AssemblyTLS& rPreAlloc,
TSchemeType& rScheme,
const ProcessInfo& rCurrentProcessInfo)
{
if (rEntity.IsDefined(ACTIVE) && rEntity.IsNot(ACTIVE)) return;

rScheme.CalculateSystemContributions(rEntity, rPreAlloc.lhs, rPreAlloc.romB, rPreAlloc.eq_id, rCurrentProcessInfo);
rEntity.GetDofList(rPreAlloc.dofs, rCurrentProcessInfo);

const SizeType ndofs = rPreAlloc.dofs.size();
ResizeIfNeeded(rPreAlloc.phiE, ndofs, this->GetNumberOfROMModes());
ResizeIfNeeded(rPreAlloc.romA, ndofs, this->GetNumberOfROMModes());

const auto &r_geom = rEntity.GetGeometry();
RomAuxiliaryUtilities::GetPhiElemental(rPreAlloc.phiE, rPreAlloc.dofs, r_geom, this->mMapPhi);

noalias(rPreAlloc.romA) = prod(rPreAlloc.lhs, rPreAlloc.phiE);


for(SizeType row=0; row < ndofs; ++row)
{
const SizeType global_row = rPreAlloc.eq_id[row];

double& r_bi = rBglobal(global_row);
AtomicAdd(r_bi, rPreAlloc.romB[row]);

if(rPreAlloc.dofs[row]->IsFree())
{
for(SizeType col=0; col < this->GetNumberOfROMModes(); ++col)
{
const SizeType global_col = col;
double& r_aij = rAglobal(global_row, global_col);
AtomicAdd(r_aij, rPreAlloc.romA(row, col));
}
}
}
}


template<typename TEntity>
void CalculateAssembledResiduals(
TEntity& rEntity,
LSPGSystemVectorType& rBglobal,
AssemblyTLS& rPreAlloc,
TSchemeType& rScheme,
const ProcessInfo& rCurrentProcessInfo)
{
if (rEntity.IsDefined(ACTIVE) && rEntity.IsNot(ACTIVE)) return;

rScheme.CalculateRHSContribution(rEntity, rPreAlloc.romB, rPreAlloc.eq_id, rCurrentProcessInfo);
rEntity.GetDofList(rPreAlloc.dofs, rCurrentProcessInfo);

const SizeType ndofs = rPreAlloc.dofs.size();

for(SizeType row=0; row < ndofs; ++row)
{
const SizeType global_row = rPreAlloc.eq_id[row];

if(rPreAlloc.dofs[row]->IsFree())
{
double& r_bi = rBglobal(global_row);
AtomicAdd(r_bi, rPreAlloc.romB[row]);
}
}
}




}; 




} 

