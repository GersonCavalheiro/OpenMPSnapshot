
#pragma once



#include "utilities/table_stream_utility.h"
#include "custom_strategies/custom_convergencecriterias/base_mortar_criteria.h"
#include "utilities/color_utilities.h"
#include "custom_utilities/active_set_utilities.h"

namespace Kratos
{







template<class TSparseSpace, class TDenseSpace>
class ALMFrictionlessMortarConvergenceCriteria
: public BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ALMFrictionlessMortarConvergenceCriteria );

KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >    ConvergenceCriteriaBaseType;

typedef BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace >             BaseType;

typedef ALMFrictionlessMortarConvergenceCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType                                  DofsArrayType;

typedef typename BaseType::TSystemMatrixType                          TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                          TSystemVectorType;

typedef TableStreamUtility::Pointer                             TablePrinterPointerType;

typedef std::size_t                                                           IndexType;


explicit ALMFrictionlessMortarConvergenceCriteria(
const bool PrintingOutput = false,
const bool ComputeDynamicFactor = false,
const bool GiDIODebug = false
) : BaseType(ComputeDynamicFactor, GiDIODebug)
{
BaseType::mOptions.Set(ALMFrictionlessMortarConvergenceCriteria::PRINTING_OUTPUT, PrintingOutput);
BaseType::mOptions.Set(ALMFrictionlessMortarConvergenceCriteria::TABLE_IS_INITIALIZED, false);
}


explicit ALMFrictionlessMortarConvergenceCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}

ALMFrictionlessMortarConvergenceCriteria( ALMFrictionlessMortarConvergenceCriteria const& rOther )
:BaseType(rOther)
{
}

~ALMFrictionlessMortarConvergenceCriteria() override = default;




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
const IndexType is_converged = ActiveSetUtilities::ComputeALMFrictionlessActiveSet(rModelPart);

r_process_info[ACTIVE_SET_CONVERGED] = is_converged == 0 ? true : false;
r_process_info[ACTIVE_SET_COMPUTED] = true;
}

const bool active_set_converged = r_process_info[ACTIVE_SET_CONVERGED];

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
if (active_set_converged) {
if (BaseType::mOptions.IsNot(ALMFrictionlessMortarConvergenceCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FGRN("       Achieved"));
else
r_table << "Achieved";
} else {
if (BaseType::mOptions.IsNot(ALMFrictionlessMortarConvergenceCriteria::PRINTING_OUTPUT))
r_table << BOLDFONT(FRED("   Not achieved"));
else
r_table << "Not achieved";
}
} else {
if (active_set_converged) {
if (BaseType::mOptions.IsNot(ALMFrictionlessMortarConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("ALMFrictionlessMortarConvergenceCriteria")  << BOLDFONT("\tActive set") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
} else {
if (BaseType::mOptions.IsNot(ALMFrictionlessMortarConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("ALMFrictionlessMortarConvergenceCriteria")  << BOLDFONT("\tActive set") << " convergence is " << BOLDFONT(FRED("not achieved")) << std::endl;
else
KRATOS_INFO("ALMFrictionlessMortarConvergenceCriteria")  << "\tActive set convergence is not achieved" << std::endl;
}
}
}

return active_set_converged;
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info.Has(TABLE_UTILITY) && BaseType::mOptions.IsNot(ALMFrictionlessMortarConvergenceCriteria::TABLE_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
r_table.AddColumn("ACTIVE SET CONV", 15);
BaseType::mOptions.Set(ALMFrictionlessMortarConvergenceCriteria::TABLE_IS_INITIALIZED, true);
}
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                        : "alm_frictionless_mortar_criteria",
"print_convergence_criterion" : false
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "alm_frictionless_mortar_criteria";
}




std::string Info() const override
{
return "ALMFrictionlessMortarConvergenceCriteria";
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

BaseType::mOptions.Set(ALMFrictionlessMortarConvergenceCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
BaseType::mOptions.Set(ALMFrictionlessMortarConvergenceCriteria::TABLE_IS_INITIALIZED, false);
}




private:








}; 


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags ALMFrictionlessMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(3));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags ALMFrictionlessMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(4));

}  
