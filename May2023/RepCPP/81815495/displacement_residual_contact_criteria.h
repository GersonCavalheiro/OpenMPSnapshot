
#pragma once



#include "utilities/table_stream_utility.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/color_utilities.h"

namespace Kratos
{







template<   class TSparseSpace,
class TDenseSpace >
class DisplacementResidualContactCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( DisplacementResidualContactCriteria );

KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );
KRATOS_DEFINE_LOCAL_FLAG( ROTATION_DOF_IS_CONSIDERED );
KRATOS_DEFINE_LOCAL_FLAG( INITIAL_RESIDUAL_IS_SET );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >                  BaseType;

typedef DisplacementResidualContactCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType                             DofsArrayType;

typedef typename BaseType::TSystemMatrixType                     TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                     TSystemVectorType;

typedef TSparseSpace                                               SparseSpaceType;

typedef TableStreamUtility::Pointer                        TablePrinterPointerType;

typedef std::size_t                                                      IndexType;



explicit DisplacementResidualContactCriteria()
: BaseType()
{
}


explicit DisplacementResidualContactCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit DisplacementResidualContactCriteria(
const double DispRatioTolerance,
const double DispAbsTolerance,
const double RotRatioTolerance,
const double RotAbsTolerance,
const bool PrintingOutput = false
)
: BaseType()
{
mOptions.Set(DisplacementResidualContactCriteria::PRINTING_OUTPUT, PrintingOutput);
mOptions.Set(DisplacementResidualContactCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);
mOptions.Set(DisplacementResidualContactCriteria::INITIAL_RESIDUAL_IS_SET, false);

mDispRatioTolerance = DispRatioTolerance;
mDispAbsTolerance = DispAbsTolerance;

mRotRatioTolerance = RotRatioTolerance;
mRotAbsTolerance = RotAbsTolerance;
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
if (SparseSpaceType::Size(rb) != 0) { 
double disp_residual_solution_norm = 0.0;
IndexType disp_dof_num(0);
double rot_residual_solution_norm = 0.0;
IndexType rot_dof_num(0);

struct AuxValues {
std::size_t dof_id = 0;
double residual_dof_value = 0.0;
};

const std::function<bool(const VariableData&)> check_without_rot =
[](const VariableData& rCurrVar) -> bool {return true;};
const std::function<bool(const VariableData&)> check_with_rot =
[](const VariableData& rCurrVar) -> bool {return ((rCurrVar == DISPLACEMENT_X) || (rCurrVar == DISPLACEMENT_Y) || (rCurrVar == DISPLACEMENT_Z));};
const auto* p_check_disp = (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? &check_with_rot : &check_without_rot;

using FourReduction = CombinedReduction<SumReduction<double>, SumReduction<IndexType>, SumReduction<double>, SumReduction<IndexType>>;
std::tie(disp_residual_solution_norm,disp_dof_num,rot_residual_solution_norm,rot_dof_num) = block_for_each<FourReduction>(rDofSet, AuxValues(), [&](Dof<double>& rDof, AuxValues& aux_values) {
if (rDof.IsFree()) {
aux_values.dof_id = rDof.EquationId();
aux_values.residual_dof_value = rb[aux_values.dof_id];

const auto& r_curr_var = rDof.GetVariable();
if ((*p_check_disp)(r_curr_var)) {
return std::make_tuple(std::pow(aux_values.residual_dof_value, 2),1,0.0,0);
} else { 
KRATOS_DEBUG_ERROR_IF_NOT((r_curr_var == ROTATION_X) || (r_curr_var == ROTATION_Y) || (r_curr_var == ROTATION_Z)) << "Variable must be a ROTATION and it is: " << r_curr_var.Name() << std::endl;
return std::make_tuple(0.0,0,std::pow(aux_values.residual_dof_value, 2),1);
}
}
return std::make_tuple(0.0,0,0.0,0);
});

mDispCurrentResidualNorm = disp_residual_solution_norm;
mRotCurrentResidualNorm = rot_residual_solution_norm;

double residual_disp_ratio = 1.0;
double residual_rot_ratio = 1.0;

if (mOptions.IsNot(DisplacementResidualContactCriteria::INITIAL_RESIDUAL_IS_SET)) {
mDispInitialResidualNorm = (disp_residual_solution_norm == 0.0) ? 1.0 : disp_residual_solution_norm;
residual_disp_ratio = 1.0;
if (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
mRotInitialResidualNorm = (rot_residual_solution_norm == 0.0) ? 1.0 : rot_residual_solution_norm;
residual_rot_ratio = 1.0;
}
mOptions.Set(DisplacementResidualContactCriteria::INITIAL_RESIDUAL_IS_SET, true);
}

residual_disp_ratio = mDispCurrentResidualNorm/mDispInitialResidualNorm;
residual_rot_ratio = mRotCurrentResidualNorm/mRotInitialResidualNorm;

const double residual_disp_abs = mDispCurrentResidualNorm/disp_dof_num;
const double residual_rot_abs = mRotCurrentResidualNorm/rot_dof_num;

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
std::cout.precision(4);
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
r_table << residual_disp_ratio << mDispRatioTolerance << residual_disp_abs << mDispAbsTolerance << residual_rot_ratio << mRotRatioTolerance << residual_rot_abs << mRotAbsTolerance;
} else {
r_table << residual_disp_ratio << mDispRatioTolerance << residual_disp_abs << mDispAbsTolerance;
}
} else {
std::cout.precision(4);
if (mOptions.IsNot(DisplacementResidualContactCriteria::PRINTING_OUTPUT)) {
KRATOS_INFO("DisplacementResidualContactCriteria") << BOLDFONT("RESIDUAL CONVERGENCE CHECK") << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl << std::scientific;
KRATOS_INFO("DisplacementResidualContactCriteria") << BOLDFONT("\tDISPLACEMENT: RATIO = ") << residual_disp_ratio << BOLDFONT(" EXP.RATIO = ") << mDispRatioTolerance << BOLDFONT(" ABS = ") << residual_disp_abs << BOLDFONT(" EXP.ABS = ") << mDispAbsTolerance << std::endl;
if (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
KRATOS_INFO("DisplacementResidualContactCriteria") << BOLDFONT("\tDISPLACEMENT: RATIO = ") << residual_rot_ratio << BOLDFONT(" EXP.RATIO = ") << mRotRatioTolerance << BOLDFONT(" ABS = ") << residual_rot_abs << BOLDFONT(" EXP.ABS = ") << mRotAbsTolerance << std::endl;
}
} else {
KRATOS_INFO("DisplacementResidualContactCriteria") << "RESIDUAL CONVERGENCE CHECK" << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl << std::scientific;
KRATOS_INFO("DisplacementResidualContactCriteria") << "\tDISPLACEMENT: RATIO = " << residual_disp_ratio << " EXP.RATIO = " << mDispRatioTolerance << " ABS = " << residual_disp_abs << " EXP.ABS = " << mDispAbsTolerance << std::endl;
if (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
KRATOS_INFO("DisplacementResidualContactCriteria") << "\tDISPLACEMENT: RATIO = " << residual_rot_ratio << " EXP.RATIO = " << mRotRatioTolerance << " ABS = " << residual_rot_abs << " EXP.ABS = " << mRotAbsTolerance << std::endl;
}
}
}
}

r_process_info[CONVERGENCE_RATIO] = residual_disp_ratio;
r_process_info[RESIDUAL_NORM] = residual_disp_abs;

const bool disp_converged = (residual_disp_ratio <= mDispRatioTolerance || residual_disp_abs <= mDispAbsTolerance);
const bool rot_converged = (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? (residual_rot_ratio <= mRotRatioTolerance || residual_rot_abs <= mRotAbsTolerance) : true;

if (disp_converged && rot_converged) {
if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (mOptions.IsNot(DisplacementResidualContactCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FGRN("       Achieved"));
else
r_table << "Achieved";
} else {
if (mOptions.IsNot(DisplacementResidualContactCriteria::PRINTING_OUTPUT))
KRATOS_INFO("DisplacementResidualContactCriteria") << BOLDFONT("\tResidual") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
else
KRATOS_INFO("DisplacementResidualContactCriteria") << "\tResidual convergence is achieved" << std::endl;
}
}
return true;
} else {
if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (mOptions.IsNot(DisplacementResidualContactCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FRED("   Not achieved"));
else
r_table << "Not achieved";
} else {
if (mOptions.IsNot(DisplacementResidualContactCriteria::PRINTING_OUTPUT))
KRATOS_INFO("DisplacementResidualContactCriteria") << BOLDFONT("\tResidual") << " convergence is " << BOLDFONT(FRED(" not achieved")) << std::endl;
else
KRATOS_INFO("DisplacementResidualContactCriteria") << "\tResidual convergence is not achieved" << std::endl;
}
}
return false;
}
} else 
return true;
}


void Initialize( ModelPart& rModelPart) override
{
BaseType::mConvergenceCriteriaIsInitialized = true;

mOptions.Set(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED, ContactUtilities::CheckModelPartHasRotationDoF(rModelPart));

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info.Has(TABLE_UTILITY) && mOptions.IsNot(DisplacementResidualContactCriteria::TABLE_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
r_table.AddColumn("DP RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
if (mOptions.Is(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
r_table.AddColumn("RT RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
}
r_table.AddColumn("CONVERGENCE", 15);
mOptions.Set(DisplacementResidualContactCriteria::TABLE_IS_INITIALIZED, true);
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
mOptions.Set(DisplacementResidualContactCriteria::INITIAL_RESIDUAL_IS_SET, false);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                 : "displacement_residual_contact_criteria",
"ensure_contact"                       : false,
"print_convergence_criterion"          : false,
"residual_relative_tolerance"          : 1.0e-4,
"residual_absolute_tolerance"          : 1.0e-9,
"rotation_residual_relative_tolerance" : 1.0e-4,
"rotation_residual_absolute_tolerance" : 1.0e-9
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "displacement_residual_contact_criteria";
}




std::string Info() const override
{
return "DisplacementResidualContactCriteria";
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

mDispRatioTolerance = ThisParameters["residual_relative_tolerance"].GetDouble();
mDispAbsTolerance = ThisParameters["residual_absolute_tolerance"].GetDouble();

mRotRatioTolerance = ThisParameters["rotation_residual_relative_tolerance"].GetDouble();
mRotAbsTolerance = ThisParameters["rotation_residual_absolute_tolerance"].GetDouble();

mOptions.Set(DisplacementResidualContactCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
mOptions.Set(DisplacementResidualContactCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(DisplacementResidualContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);
mOptions.Set(DisplacementResidualContactCriteria::INITIAL_RESIDUAL_IS_SET, false);
}




private:


Flags mOptions; 

double mDispRatioTolerance;      
double mDispAbsTolerance;        
double mDispInitialResidualNorm; 
double mDispCurrentResidualNorm; 

double mRotRatioTolerance;      
double mRotAbsTolerance;        
double mRotInitialResidualNorm; 
double mRotCurrentResidualNorm; 






};  


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementResidualContactCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementResidualContactCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(2));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementResidualContactCriteria<TSparseSpace, TDenseSpace>::ROTATION_DOF_IS_CONSIDERED(Kratos::Flags::Create(3));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementResidualContactCriteria<TSparseSpace, TDenseSpace>::INITIAL_RESIDUAL_IS_SET(Kratos::Flags::Create(4));
}
