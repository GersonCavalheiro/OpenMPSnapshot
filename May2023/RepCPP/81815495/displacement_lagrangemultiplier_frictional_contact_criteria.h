
#pragma once



#include "utilities/table_stream_utility.h"
#include "utilities/color_utilities.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "custom_utilities/active_set_utilities.h"
#include "utilities/constraint_utilities.h"
#include "custom_utilities/contact_utilities.h"

namespace Kratos
{







template<   class TSparseSpace,
class TDenseSpace >
class DisplacementLagrangeMultiplierFrictionalContactCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( DisplacementLagrangeMultiplierFrictionalContactCriteria );

KRATOS_DEFINE_LOCAL_FLAG( ENSURE_CONTACT );
KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );
KRATOS_DEFINE_LOCAL_FLAG( ROTATION_DOF_IS_CONSIDERED );
KRATOS_DEFINE_LOCAL_FLAG( PURE_SLIP );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >                                      BaseType;

typedef DisplacementLagrangeMultiplierFrictionalContactCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType                                                 DofsArrayType;

typedef typename BaseType::TSystemMatrixType                                         TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                                         TSystemVectorType;

typedef TSparseSpace                                                                   SparseSpaceType;

typedef TableStreamUtility::Pointer                                            TablePrinterPointerType;

typedef std::size_t                                                                          IndexType;

static constexpr double Tolerance = std::numeric_limits<double>::epsilon();



explicit DisplacementLagrangeMultiplierFrictionalContactCriteria()
: BaseType()
{
}


explicit DisplacementLagrangeMultiplierFrictionalContactCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit DisplacementLagrangeMultiplierFrictionalContactCriteria(
const double DispRatioTolerance,
const double DispAbsTolerance,
const double RotRatioTolerance,
const double RotAbsTolerance,
const double LMNormalRatioTolerance,
const double LMNormalAbsTolerance,
const double LMTangentStickRatioTolerance,
const double LMTangentStickAbsTolerance,
const double LMTangentSlipRatioTolerance,
const double LMTangentSlipAbsTolerance,
const double NormalTangentRatio,
const bool EnsureContact = false,
const bool PureSlip = false,
const bool PrintingOutput = false
)
: BaseType()
{
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::ENSURE_CONTACT, EnsureContact);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT, PrintingOutput);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::PURE_SLIP, PureSlip);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);

mDispRatioTolerance = DispRatioTolerance;
mDispAbsTolerance = DispAbsTolerance;

mRotRatioTolerance = RotRatioTolerance;
mRotAbsTolerance = RotAbsTolerance;

mLMNormalRatioTolerance = LMNormalRatioTolerance;
mLMNormalAbsTolerance = LMNormalAbsTolerance;

mLMTangentStickRatioTolerance = LMTangentStickRatioTolerance;
mLMTangentStickAbsTolerance = LMTangentStickAbsTolerance;
mLMTangentStickRatioTolerance = LMTangentSlipRatioTolerance;
mLMTangentStickAbsTolerance = LMTangentSlipAbsTolerance;

mNormalTangentRatio = NormalTangentRatio;
}

/
typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
if (SparseSpaceType::Size(rDx) != 0) { 

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

double disp_solution_norm = 0.0, rot_solution_norm = 0.0, normal_lm_solution_norm = 0.0, tangent_lm_stick_solution_norm = 0.0, tangent_lm_slip_solution_norm = 0.0, disp_increase_norm = 0.0, rot_increase_norm = 0.0, normal_lm_increase_norm = 0.0, tangent_lm_stick_increase_norm = 0.0, tangent_lm_slip_increase_norm = 0.0;
IndexType disp_dof_num(0), rot_dof_num(0), lm_dof_num(0), lm_stick_dof_num(0), lm_slip_dof_num(0);

auto& r_nodes_array = rModelPart.Nodes();

struct AuxValues {
std::size_t dof_id = 0;
double dof_value = 0.0, dof_incr = 0.0;
};
const bool pure_slip = mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::PURE_SLIP);

const std::size_t number_active_dofs = rb.size();

const std::function<bool(const VariableData&)> check_without_rot =
[](const VariableData& rCurrVar) -> bool {return true;};
const std::function<bool(const VariableData&)> check_with_rot =
[](const VariableData& rCurrVar) -> bool {return ((rCurrVar == DISPLACEMENT_X) || (rCurrVar == DISPLACEMENT_Y) || (rCurrVar == DISPLACEMENT_Z));};
const auto* p_check_disp = (mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? &check_with_rot : &check_without_rot;

using FifteenReduction = CombinedReduction<SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<double>,SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<IndexType>, SumReduction<IndexType>, SumReduction<IndexType>, SumReduction<IndexType>, SumReduction<IndexType>>;
std::tie(disp_solution_norm, rot_solution_norm, normal_lm_solution_norm, tangent_lm_slip_solution_norm, tangent_lm_stick_solution_norm, disp_increase_norm, rot_increase_norm, normal_lm_increase_norm, tangent_lm_slip_increase_norm, tangent_lm_stick_increase_norm, disp_dof_num, rot_dof_num, lm_dof_num, lm_slip_dof_num, lm_stick_dof_num) = block_for_each<FifteenReduction>(rDofSet, AuxValues(), [this,p_check_disp,&number_active_dofs,&pure_slip, &r_nodes_array,&rDx](Dof<double>& rDof, AuxValues& aux_values) {
aux_values.dof_id = rDof.EquationId();

if (aux_values.dof_id < number_active_dofs) {
if (mActiveDofs[aux_values.dof_id] == 1) {
aux_values.dof_value = rDof.GetSolutionStepValue(0);
aux_values.dof_incr = rDx[aux_values.dof_id];

const auto& r_curr_var = rDof.GetVariable();
if (r_curr_var == VECTOR_LAGRANGE_MULTIPLIER_X || r_curr_var == VECTOR_LAGRANGE_MULTIPLIER_Y || r_curr_var == VECTOR_LAGRANGE_MULTIPLIER_Z) {
const auto it_node = r_nodes_array.find(rDof.Id());

const double mu = it_node->GetValue(FRICTION_COEFFICIENT);
if (mu < std::numeric_limits<double>::epsilon()) {
return std::make_tuple(0.0,0.0,std::pow(aux_values.dof_value, 2),0.0,0.0,0.0,0.0,std::pow(aux_values.dof_incr, 2),0.0,0.0,0,0,1,0,0);
} else {
const double normal = it_node->FastGetSolutionStepValue(NORMAL)[r_curr_var.GetComponentIndex()];
const double normal_dof_value = aux_values.dof_value * normal;
const double normal_dof_incr = aux_values.dof_incr * normal;

if (it_node->Is(SLIP) || pure_slip) {
return std::make_tuple(0.0,0.0,std::pow(normal_dof_value, 2),std::pow(aux_values.dof_value - normal_dof_value, 2),0.0,0.0,0.0,std::pow(normal_dof_incr, 2),std::pow(aux_values.dof_incr - normal_dof_incr, 2),0.0,0,0,1,1,0);
} else {
return std::make_tuple(0.0,0.0,std::pow(normal_dof_value, 2),0.0,std::pow(aux_values.dof_value - normal_dof_value, 2),0.0,0.0,std::pow(normal_dof_incr, 2),0.0,std::pow(aux_values.dof_incr - normal_dof_incr, 2),0,0,1,0,1);
}
}
} else if ((*p_check_disp)(r_curr_var)) {
return std::make_tuple(std::pow(aux_values.dof_value, 2),0.0,0.0,0.0,0.0,std::pow(aux_values.dof_incr, 2),0.0,0.0,0.0,0.0,1,0,0,0,0);
} else { 
KRATOS_DEBUG_ERROR_IF_NOT((r_curr_var == ROTATION_X) || (r_curr_var == ROTATION_Y) || (r_curr_var == ROTATION_Z)) << "Variable must be a ROTATION and it is: " << r_curr_var.Name() << std::endl;
return std::make_tuple(0.0,std::pow(aux_values.dof_value, 2),0.0,0.0,0.0,0.0,std::pow(aux_values.dof_incr, 2),0.0,0.0,0.0,0,1,0,0,0);
}
}
}
return std::make_tuple(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0,0);
});

if(disp_increase_norm < Tolerance) disp_increase_norm = 1.0;
if(rot_increase_norm < Tolerance) rot_increase_norm = 1.0;
if(normal_lm_increase_norm < Tolerance) normal_lm_increase_norm = 1.0;
if(tangent_lm_stick_increase_norm < Tolerance) tangent_lm_stick_increase_norm = 1.0;
if(tangent_lm_slip_increase_norm < Tolerance) tangent_lm_slip_increase_norm = 1.0;
if(disp_solution_norm < Tolerance) disp_solution_norm = 1.0;

KRATOS_ERROR_IF(mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ENSURE_CONTACT) && normal_lm_solution_norm < Tolerance) << "WARNING::CONTACT LOST::ARE YOU SURE YOU ARE SUPPOSED TO HAVE CONTACT?" << std::endl;

const double disp_ratio = std::sqrt(disp_increase_norm/disp_solution_norm);
const double rot_ratio = std::sqrt(rot_increase_norm/rot_solution_norm);
const double normal_lm_ratio = normal_lm_solution_norm > Tolerance ? std::sqrt(normal_lm_increase_norm/normal_lm_solution_norm) : 0.0;
const double tangent_lm_stick_ratio = tangent_lm_stick_solution_norm > Tolerance ? std::sqrt(tangent_lm_stick_increase_norm/tangent_lm_stick_solution_norm) : 0.0;
const double tangent_lm_slip_ratio = tangent_lm_slip_solution_norm > Tolerance ? std::sqrt(tangent_lm_slip_increase_norm/tangent_lm_slip_solution_norm) : 0.0;

const double disp_abs = std::sqrt(disp_increase_norm)/ static_cast<double>(disp_dof_num);
const double rot_abs = std::sqrt(rot_increase_norm)/ static_cast<double>(rot_dof_num);
const double normal_lm_abs = std::sqrt(normal_lm_increase_norm)/ static_cast<double>(lm_dof_num);
const double tangent_lm_stick_abs = lm_stick_dof_num > 0 ?  std::sqrt(tangent_lm_stick_increase_norm)/ static_cast<double>(lm_stick_dof_num) : 0.0;
const double tangent_lm_slip_abs = lm_slip_dof_num > 0 ? std::sqrt(tangent_lm_slip_increase_norm)/ static_cast<double>(lm_slip_dof_num) : 0.0;

const double normal_tangent_stick_ratio = tangent_lm_stick_abs/normal_lm_abs;
const double normal_tangent_slip_ratio = tangent_lm_slip_abs/normal_lm_abs;

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
std::cout.precision(4);
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& Table = p_table->GetTable();
if (mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
Table << disp_ratio << mDispRatioTolerance << disp_abs << mDispAbsTolerance << rot_ratio << mRotRatioTolerance << rot_abs << mRotAbsTolerance << normal_lm_ratio << mLMNormalRatioTolerance << normal_lm_abs << mLMNormalAbsTolerance << tangent_lm_stick_ratio << mLMTangentStickRatioTolerance << tangent_lm_stick_abs << mLMTangentStickAbsTolerance << tangent_lm_slip_ratio << mLMTangentSlipRatioTolerance << tangent_lm_slip_abs << mLMTangentSlipAbsTolerance;
} else {
Table << disp_ratio << mDispRatioTolerance << disp_abs << mDispAbsTolerance << normal_lm_ratio << mLMNormalRatioTolerance << normal_lm_abs << mLMNormalAbsTolerance << tangent_lm_stick_ratio << mLMTangentStickRatioTolerance << tangent_lm_stick_abs << mLMTangentStickAbsTolerance << tangent_lm_slip_ratio << mLMTangentSlipRatioTolerance << tangent_lm_slip_abs << mLMTangentSlipAbsTolerance;
}
} else {
std::cout.precision(4);
if (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT)) {
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT("DoF ONVERGENCE CHECK") << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl;
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT("\tDISPLACEMENT: RATIO = ") << disp_ratio << BOLDFONT(" EXP.RATIO = ") << mDispRatioTolerance << BOLDFONT(" ABS = ") << disp_abs << BOLDFONT(" EXP.ABS = ") << mDispAbsTolerance << std::endl;
if (mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT("\tROTATION: RATIO = ") << rot_ratio << BOLDFONT(" EXP.RATIO = ") << mRotRatioTolerance << BOLDFONT(" ABS = ") << rot_abs << BOLDFONT(" EXP.ABS = ") << mRotAbsTolerance << std::endl;
}
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT(" NORMAL LAGRANGE MUL:\tRATIO = ") << normal_lm_ratio << BOLDFONT(" EXP.RATIO = ") << mLMNormalRatioTolerance << BOLDFONT(" ABS = ") << normal_lm_abs << BOLDFONT(" EXP.ABS = ") << mLMNormalAbsTolerance << std::endl;
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT(" STICK LAGRANGE MUL:\tRATIO = ") << tangent_lm_stick_ratio << BOLDFONT(" EXP.RATIO = ") << mLMTangentStickRatioTolerance << BOLDFONT(" ABS = ") << tangent_lm_stick_abs << BOLDFONT(" EXP.ABS = ") << mLMTangentStickAbsTolerance << std::endl;
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT(" SLIP LAGRANGE MUL:\tRATIO = ") << tangent_lm_slip_ratio << BOLDFONT(" EXP.RATIO = ") << mLMTangentSlipRatioTolerance << BOLDFONT(" ABS = ") << tangent_lm_slip_abs << BOLDFONT(" EXP.ABS = ") << mLMTangentSlipAbsTolerance << std::endl;
} else {
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << "DoF ONVERGENCE CHECK" << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl;
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << "\tDISPLACEMENT: RATIO = " << disp_ratio << " EXP.RATIO = " << mDispRatioTolerance << " ABS = " << disp_abs << " EXP.ABS = " << mDispAbsTolerance << std::endl;
if (mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << "\tROTATION: RATIO = " << rot_ratio << " EXP.RATIO = " << mRotRatioTolerance << " ABS = " << rot_abs << " EXP.ABS = " << mRotAbsTolerance << std::endl;
}
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << " NORMAL LAGRANGE MUL:\tRATIO = " << normal_lm_ratio << " EXP.RATIO = " << mLMNormalRatioTolerance << " ABS = " << normal_lm_abs << " EXP.ABS = " << mLMNormalAbsTolerance << std::endl;
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << " STICK LAGRANGE MUL:\tRATIO = " << tangent_lm_stick_ratio << " EXP.RATIO = " << mLMTangentStickRatioTolerance << " ABS = " << tangent_lm_stick_abs << " EXP.ABS = " << mLMTangentStickAbsTolerance << std::endl;
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << " SLIP LAGRANGE MUL:\tRATIO = " << tangent_lm_slip_ratio << " EXP.RATIO = " << mLMTangentSlipRatioTolerance << " ABS = " << tangent_lm_slip_abs << " EXP.ABS = " << mLMTangentSlipAbsTolerance << std::endl;
}
}
}

const bool disp_converged = (disp_ratio <= mDispRatioTolerance || disp_abs <= mDispAbsTolerance);
const bool rot_converged = (mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? (rot_ratio <= mRotRatioTolerance || rot_abs <= mRotAbsTolerance) : true;
const bool lm_converged = (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::ENSURE_CONTACT) && normal_lm_solution_norm < Tolerance) ? true : (normal_lm_ratio <= mLMNormalRatioTolerance || normal_lm_abs <= mLMNormalAbsTolerance) && (tangent_lm_stick_ratio <= mLMTangentStickRatioTolerance || tangent_lm_stick_abs <= mLMTangentStickAbsTolerance || normal_tangent_stick_ratio <= mNormalTangentRatio) && (tangent_lm_slip_ratio <= mLMTangentSlipRatioTolerance || tangent_lm_slip_abs <= mLMTangentSlipAbsTolerance || normal_tangent_slip_ratio <= mNormalTangentRatio);

if (disp_converged && rot_converged && lm_converged) {
if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FGRN("       Achieved"));
else
r_table << "Achieved";
} else {
if (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT))
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT("\tDoF") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
else
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << "\tDoF convergence is achieved" << std::endl;
}
}
return true;
} else {
if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FRED("   Not achieved"));
else
r_table << "Not achieved";
} else {
if (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT))
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << BOLDFONT("\tDoF") << " convergence is " << BOLDFONT(FRED(" not achieved")) << std::endl;
else
KRATOS_INFO("DisplacementLagrangeMultiplierFrictionalContactCriteria") << "\tDoF convergence is not achieved" << std::endl;
}
}
return false;
}
}
else 
return true;
}


void Initialize( ModelPart& rModelPart ) override
{
BaseType::mConvergenceCriteriaIsInitialized = true;

mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED, ContactUtilities::CheckModelPartHasRotationDoF(rModelPart));

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info.Has(TABLE_UTILITY) && mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::TABLE_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
r_table.AddColumn("DP RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
if (mOptions.Is(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
r_table.AddColumn("RT RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
}
r_table.AddColumn("N.LM RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
if (mOptions.IsNot(DisplacementLagrangeMultiplierFrictionalContactCriteria::PURE_SLIP)) {
r_table.AddColumn("STI. RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
}
r_table.AddColumn("SLIP RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
r_table.AddColumn("CONVERGENCE", 15);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::TABLE_IS_INITIALIZED, true);
}
}


void InitializeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
ConstraintUtilities::ComputeActiveDofs(rModelPart, mActiveDofs, rDofSet);
}


void FinalizeNonLinearIteration(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::FinalizeNonLinearIteration(rModelPart, rDofSet, rA, rDx, rb);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
r_process_info.SetValue(ACTIVE_SET_COMPUTED, false);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                                     : "displacement_lagrangemultiplier_frictional_contact_criteria",
"ensure_contact"                                           : false,
"pure_slip"                                                : false,
"print_convergence_criterion"                              : false,
"displacement_relative_tolerance"                          : 1.0e-4,
"displacement_absolute_tolerance"                          : 1.0e-9,
"rotation_relative_tolerance"                              : 1.0e-4,
"rotation_absolute_tolerance"                              : 1.0e-9,
"contact_displacement_relative_tolerance"                  : 1.0e-4,
"contact_displacement_absolute_tolerance"                  : 1.0e-9,
"frictional_stick_contact_displacement_relative_tolerance" : 1.0e-4,
"frictional_stick_contact_displacement_absolute_tolerance" : 1.0e-9,
"frictional_slip_contact_displacement_relative_tolerance"  : 1.0e-4,
"frictional_slip_contact_displacement_absolute_tolerance"  : 1.0e-9,
"ratio_normal_tangent_threshold"                           : 1.0e-4
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "displacement_lagrangemultiplier_frictional_contact_criteria";
}




std::string Info() const override
{
return "DisplacementLagrangeMultiplierFrictionalContactCriteria";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info();
}


protected:






void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mDispRatioTolerance = ThisParameters["displacement_relative_tolerance"].GetDouble();
mDispAbsTolerance = ThisParameters["displacement_absolute_tolerance"].GetDouble();

mRotRatioTolerance = ThisParameters["rotation_relative_tolerance"].GetDouble();
mRotAbsTolerance = ThisParameters["rotation_absolute_tolerance"].GetDouble();

mLMNormalRatioTolerance =  ThisParameters["contact_displacement_relative_tolerance"].GetDouble();
mLMNormalAbsTolerance =  ThisParameters["contact_displacement_absolute_tolerance"].GetDouble();

mLMTangentStickRatioTolerance =  ThisParameters["frictional_stick_contact_displacement_relative_tolerance"].GetDouble();
mLMTangentStickAbsTolerance =  ThisParameters["frictional_stick_contact_displacement_absolute_tolerance"].GetDouble();
mLMTangentSlipRatioTolerance =  ThisParameters["frictional_slip_contact_displacement_relative_tolerance"].GetDouble();
mLMTangentSlipAbsTolerance =  ThisParameters["frictional_slip_contact_displacement_absolute_tolerance"].GetDouble();

mNormalTangentRatio = ThisParameters["ratio_normal_tangent_threshold"].GetDouble();

mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::ENSURE_CONTACT, ThisParameters["ensure_contact"].GetBool());
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);
mOptions.Set(DisplacementLagrangeMultiplierFrictionalContactCriteria::PURE_SLIP, ThisParameters["pure_slip"].GetBool());
}




private:


Flags mOptions; 

double mDispRatioTolerance;      
double mDispAbsTolerance;        

double mRotRatioTolerance;      
double mRotAbsTolerance;        

double mLMNormalRatioTolerance;  
double mLMNormalAbsTolerance;    

double mLMTangentStickRatioTolerance; 
double mLMTangentStickAbsTolerance;   
double mLMTangentSlipRatioTolerance;  
double mLMTangentSlipAbsTolerance;    

double mNormalTangentRatio;      

std::vector<int> mActiveDofs;       






};  


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierFrictionalContactCriteria<TSparseSpace, TDenseSpace>::ENSURE_CONTACT(Kratos::Flags::Create(0));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierFrictionalContactCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierFrictionalContactCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(2));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierFrictionalContactCriteria<TSparseSpace, TDenseSpace>::ROTATION_DOF_IS_CONSIDERED(Kratos::Flags::Create(3));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierFrictionalContactCriteria<TSparseSpace, TDenseSpace>::PURE_SLIP(Kratos::Flags::Create(4));
}
