
#pragma once



#include "utilities/table_stream_utility.h"
#include "custom_strategies/custom_convergencecriterias/base_mortar_criteria.h"
#include "utilities/color_utilities.h"
#include "custom_utilities/active_set_utilities.h"

namespace Kratos
{







template<class TSparseSpace, class TDenseSpace>
class ALMFrictionalMortarConvergenceCriteria
: public BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ALMFrictionalMortarConvergenceCriteria );

KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >  ConvergenceCriteriaBaseType;

typedef BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace >           BaseType;

typedef ALMFrictionalMortarConvergenceCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType                                DofsArrayType;

typedef typename BaseType::TSystemMatrixType                        TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                        TSystemVectorType;

typedef TableStreamUtility::Pointer                           TablePrinterPointerType;

static constexpr double Tolerance = std::numeric_limits<double>::epsilon();


explicit ALMFrictionalMortarConvergenceCriteria(
const bool PureSlip = false,
const bool PrintingOutput = false,
const bool ComputeDynamicFactor = false,
const bool IODebug = false
) : BaseType(ComputeDynamicFactor, IODebug, PureSlip)
{
BaseType::mOptions.Set(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT, PrintingOutput);
BaseType::mOptions.Set(ALMFrictionalMortarConvergenceCriteria::TABLE_IS_INITIALIZED, false);
}


explicit ALMFrictionalMortarConvergenceCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}

ALMFrictionalMortarConvergenceCriteria( ALMFrictionalMortarConvergenceCriteria const& rOther )
:BaseType(rOther)
{
}

~ALMFrictionalMortarConvergenceCriteria() override = default;




typename ConvergenceCriteriaBaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


bool PreCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::PreCriteria(rModelPart, rDofSet, rA, rDx, rb);

return true;
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::PostCriteria(rModelPart, rDofSet, rA, rDx, rb);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (!r_process_info[ACTIVE_SET_COMPUTED]) {
const array_1d<std::size_t, 2> is_converged = ActiveSetUtilities::ComputeALMFrictionalActiveSet(rModelPart, BaseType::mOptions.Is(BaseType::PURE_SLIP), this->GetEchoLevel());

r_process_info[ACTIVE_SET_CONVERGED] = is_converged[0] == 0 ? true : false;
r_process_info[SLIP_SET_CONVERGED] = is_converged[1] == 0 ? true : false;
r_process_info[ACTIVE_SET_COMPUTED] = true;
}

const bool active_set_converged = r_process_info[ACTIVE_SET_CONVERGED];
const bool slip_set_converged = r_process_info[SLIP_SET_CONVERGED];

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (active_set_converged) {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FGRN("       Achieved"));
else
r_table << "Achieved";
} else {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FRED("   Not achieved"));
else
r_table << "Not achieved";
}
if (BaseType::mOptions.IsNot(BaseType::PURE_SLIP)) {
if (slip_set_converged) {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FGRN("       Achieved"));
else
r_table << "Achieved";
} else {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FRED("   Not achieved"));
else
r_table << "Not achieved";
}
}
} else {
if (active_set_converged) {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << BOLDFONT("\tActive set") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
else
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << "\tActive set convergence is achieved" << std::endl;
} else {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << BOLDFONT("\tActive set") << " convergence is " << BOLDFONT(FRED("not achieved")) << std::endl;
else
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << "\tActive set convergence is not achieved" << std::endl;
}

if (BaseType::mOptions.IsNot(BaseType::PURE_SLIP)) {
if (slip_set_converged) {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << BOLDFONT("\tSlip/stick set") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
else
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << "\tSlip/stick set convergence is achieved" << std::endl;
} else {
if (BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << BOLDFONT("\tSlip/stick set") << " convergence is " << BOLDFONT(FRED("not achieved")) << std::endl;
else
KRATOS_INFO("ALMFrictionalMortarConvergenceCriteria") << "\tSlip/stick set  convergence is not achieved" << std::endl;
}
}
}
}

return (active_set_converged && slip_set_converged);
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info.Has(TABLE_UTILITY) && BaseType::mOptions.IsNot(ALMFrictionalMortarConvergenceCriteria::TABLE_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
r_table.AddColumn("ACTIVE SET CONV", 15);
if (BaseType::mOptions.IsNot(BaseType::PURE_SLIP)) {
r_table.AddColumn("SLIP/STICK CONV", 15);
}
BaseType::mOptions.Set(ALMFrictionalMortarConvergenceCriteria::TABLE_IS_INITIALIZED, true);
}
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                        : "alm_frictional_mortar_criteria",
"print_convergence_criterion" : false
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "alm_frictional_mortar_criteria";
}




std::string Info() const override
{
return "ALMFrictionalMortarConvergenceCriteria";
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

BaseType::mOptions.Set(ALMFrictionalMortarConvergenceCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
BaseType::mOptions.Set(ALMFrictionalMortarConvergenceCriteria::TABLE_IS_INITIALIZED, false);
}


void ResetWeightedGap(ModelPart& rModelPart) override
{
const array_1d<double, 3> zero_array = ZeroVector(3);

auto& r_nodes_array = rModelPart.GetSubModelPart("Contact").Nodes();
VariableUtils().SetVariable(WEIGHTED_GAP, 0.0, r_nodes_array);
VariableUtils().SetVariable(WEIGHTED_SLIP, zero_array, r_nodes_array);
}




private:








}; 


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags ALMFrictionalMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(3));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags ALMFrictionalMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(4));

}  
