
#pragma once



#include "utilities/table_stream_utility.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/color_utilities.h"

namespace Kratos
{







template<   class TSparseSpace,
class TDenseSpace >
class DisplacementContactCriteria
: public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( DisplacementContactCriteria );

KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );
KRATOS_DEFINE_LOCAL_FLAG( ROTATION_DOF_IS_CONSIDERED );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >          BaseType;

typedef DisplacementContactCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType                     DofsArrayType;

typedef typename BaseType::TSystemMatrixType             TSystemMatrixType;

typedef typename BaseType::TSystemVectorType             TSystemVectorType;

typedef TSparseSpace                                       SparseSpaceType;

typedef TableStreamUtility::Pointer                TablePrinterPointerType;

typedef std::size_t                                              IndexType;



explicit DisplacementContactCriteria()
: BaseType()
{
}


explicit DisplacementContactCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit DisplacementContactCriteria(
const double DispRatioTolerance,
const double DispAbsTolerance,
const double RotRatioTolerance,
const double RotAbsTolerance,
const bool PrintingOutput = false
)
: BaseType()
{
mOptions.Set(DisplacementContactCriteria::PRINTING_OUTPUT, PrintingOutput);
mOptions.Set(DisplacementContactCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);

mDispRatioTolerance = DispRatioTolerance;
mDispAbsTolerance = DispAbsTolerance;

mRotRatioTolerance = RotRatioTolerance;
mRotAbsTolerance = RotAbsTolerance;
}

DisplacementContactCriteria( DisplacementContactCriteria const& rOther )
:BaseType(rOther)
,mOptions(rOther.mOptions)
,mDispRatioTolerance(rOther.mDispRatioTolerance)
,mDispAbsTolerance(rOther.mDispAbsTolerance)
,mRotRatioTolerance(rOther.mRotRatioTolerance)
,mRotAbsTolerance(rOther.mRotAbsTolerance)
{
}

~DisplacementContactCriteria() override = default;




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
double disp_solution_norm = 0.0, disp_increase_norm = 0.0;
IndexType disp_dof_num(0);
double rot_solution_norm = 0.0, rot_increase_norm = 0.0;
IndexType rot_dof_num(0);

struct AuxValues {
std::size_t dof_id = 0;
double dof_value = 0.0, dof_incr = 0.0;
};

const std::function<bool(const VariableData&)> check_without_rot =
[](const VariableData& rCurrVar) -> bool {return true;};
const std::function<bool(const VariableData&)> check_with_rot =
[](const VariableData& rCurrVar) -> bool {return ((rCurrVar == DISPLACEMENT_X) || (rCurrVar == DISPLACEMENT_Y) || (rCurrVar == DISPLACEMENT_Z));};
const auto* p_check_disp = (mOptions.Is(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? &check_with_rot : &check_without_rot;

using SixReduction = CombinedReduction<SumReduction<double>, SumReduction<double>, SumReduction<IndexType>, SumReduction<double>, SumReduction<double>, SumReduction<IndexType>>;
std::tie(disp_solution_norm,disp_increase_norm,disp_dof_num,rot_solution_norm,rot_increase_norm,rot_dof_num) = block_for_each<SixReduction>(rDofSet, AuxValues(), [p_check_disp,&rDx](Dof<double>& rDof, AuxValues& aux_values) {
if (rDof.IsFree()) {
aux_values.dof_id = rDof.EquationId();
aux_values.dof_value = rDof.GetSolutionStepValue(0);
aux_values.dof_incr = rDx[aux_values.dof_id];

const auto& r_curr_var = rDof.GetVariable();
if ((*p_check_disp)(r_curr_var)) {
return std::make_tuple(std::pow(aux_values.dof_value, 2),std::pow(aux_values.dof_incr, 2),1,0.0,0.0,0);
} else {
KRATOS_DEBUG_ERROR_IF_NOT((r_curr_var == ROTATION_X) || (r_curr_var == ROTATION_Y) || (r_curr_var == ROTATION_Z)) << "Variable must be a ROTATION and it is: " << r_curr_var.Name() << std::endl;
return std::make_tuple(0.0,0.0,0,std::pow(aux_values.dof_value, 2),std::pow(aux_values.dof_incr, 2),1);
}
}
return std::make_tuple(0.0,0.0,0,0.0,0.0,0);
});

if(disp_increase_norm == 0.0) disp_increase_norm = 1.0;
if(disp_solution_norm == 0.0) disp_solution_norm = 1.0;
if(rot_increase_norm == 0.0) rot_increase_norm = 1.0;
if(rot_solution_norm == 0.0) rot_solution_norm = 1.0;

const double disp_ratio = std::sqrt(disp_increase_norm/disp_solution_norm);
const double disp_abs = std::sqrt(disp_increase_norm)/ static_cast<double>(disp_dof_num);
const double rot_ratio = std::sqrt(rot_increase_norm/rot_solution_norm);
const double rot_abs = std::sqrt(rot_increase_norm)/ static_cast<double>(rot_dof_num);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
std::cout.precision(4);
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& Table = p_table->GetTable();
if (mOptions.Is(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
Table << disp_ratio << mDispRatioTolerance << disp_abs << mDispAbsTolerance << rot_ratio << mRotRatioTolerance << rot_abs << mRotAbsTolerance;
} else {
Table << disp_ratio << mDispRatioTolerance << disp_abs << mDispAbsTolerance;
}
} else {
std::cout.precision(4);
if (mOptions.IsNot(DisplacementContactCriteria::PRINTING_OUTPUT)) {
KRATOS_INFO("DisplacementContactCriteria") << BOLDFONT("DoF ONVERGENCE CHECK") << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl;
KRATOS_INFO("DisplacementContactCriteria") << BOLDFONT("\tDISPLACEMENT: RATIO = ") << disp_ratio << BOLDFONT(" EXP.RATIO = ") << mDispRatioTolerance << BOLDFONT(" ABS = ") << disp_abs << BOLDFONT(" EXP.ABS = ") << mDispAbsTolerance << std::endl;
if (mOptions.Is(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
KRATOS_INFO("DisplacementContactCriteria") << BOLDFONT("\tROTATION: RATIO = ") << rot_ratio << BOLDFONT(" EXP.RATIO = ") << mRotRatioTolerance << BOLDFONT(" ABS = ") << rot_abs << BOLDFONT(" EXP.ABS = ") << mRotAbsTolerance << std::endl;
}
} else {
KRATOS_INFO("DisplacementContactCriteria") << "DoF ONVERGENCE CHECK" << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl;
KRATOS_INFO("DisplacementContactCriteria") << "\tDISPLACEMENT: RATIO = " << disp_ratio << " EXP.RATIO = " << mDispRatioTolerance << " ABS = " << disp_abs << " EXP.ABS = " << mDispAbsTolerance << std::endl;
if (mOptions.Is(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
KRATOS_INFO("DisplacementContactCriteria") << "\tROTATION: RATIO = " << rot_ratio << " EXP.RATIO = " << mRotRatioTolerance << " ABS = " << rot_abs << " EXP.ABS = " << mRotAbsTolerance << std::endl;
}
}
}
}

const bool disp_converged = (disp_ratio <= mDispRatioTolerance || disp_abs <= mDispAbsTolerance);
const bool rot_converged = mOptions.Is(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED) ? (rot_ratio <= mRotRatioTolerance || rot_abs <= mRotAbsTolerance) : true;

if (disp_converged && rot_converged) {
if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& table = p_table->GetTable();
if (mOptions.IsNot(DisplacementContactCriteria::PRINTING_OUTPUT))
table << BOLDFONT(FGRN("       Achieved"));
else
table << "Achieved";
} else {
if (mOptions.IsNot(DisplacementContactCriteria::PRINTING_OUTPUT))
KRATOS_INFO("DisplacementContactCriteria") << BOLDFONT("\tDoF") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
else
KRATOS_INFO("DisplacementContactCriteria") << "\tDoF convergence is achieved" << std::endl;
}
}
return true;
} else {
if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& table = p_table->GetTable();
if (mOptions.IsNot(DisplacementContactCriteria::PRINTING_OUTPUT))
table << BOLDFONT(FRED("   Not achieved"));
else
table << "Not achieved";
} else {
if (mOptions.IsNot(DisplacementContactCriteria::PRINTING_OUTPUT))
KRATOS_INFO("DisplacementContactCriteria") << BOLDFONT("\tDoF") << " convergence is " << BOLDFONT(FRED(" not achieved")) << std::endl;
else
KRATOS_INFO("DisplacementContactCriteria") << "\tDoF convergence is not achieved" << std::endl;
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

mOptions.Set(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED, ContactUtilities::CheckModelPartHasRotationDoF(rModelPart));

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info.Has(TABLE_UTILITY) && mOptions.IsNot(DisplacementContactCriteria::TABLE_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
r_table.AddColumn("DP RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
if (mOptions.Is(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
r_table.AddColumn("RT RATIO", 10);
r_table.AddColumn("EXP. RAT", 10);
r_table.AddColumn("ABS", 10);
r_table.AddColumn("EXP. ABS", 10);
}
r_table.AddColumn("CONVERGENCE", 15);
mOptions.Set(DisplacementContactCriteria::TABLE_IS_INITIALIZED, true);
}
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                            : "displacement_contact_criteria",
"ensure_contact"                  : false,
"print_convergence_criterion"     : false,
"displacement_relative_tolerance" : 1.0e-4,
"displacement_absolute_tolerance" : 1.0e-9,
"rotation_relative_tolerance"     : 1.0e-4,
"rotation_absolute_tolerance"     : 1.0e-9
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "displacement_contact_criteria";
}




std::string Info() const override
{
return "DisplacementContactCriteria";
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

mOptions.Set(DisplacementContactCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
mOptions.Set(DisplacementContactCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(DisplacementContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);
}




private:


Flags mOptions; 

double mDispRatioTolerance; 
double mDispAbsTolerance;   

double mRotRatioTolerance; 
double mRotAbsTolerance;   






};  


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementContactCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementContactCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(2));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementContactCriteria<TSparseSpace, TDenseSpace>::ROTATION_DOF_IS_CONSIDERED(Kratos::Flags::Create(3));
}
