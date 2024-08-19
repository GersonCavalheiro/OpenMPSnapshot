
#pragma once



#include "utilities/table_stream_utility.h"
#include "solving_strategies/convergencecriterias/and_criteria.h"
#include "utilities/color_utilities.h"
#include "utilities/condition_number_utility.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace
>
class MortarAndConvergenceCriteria
: public And_Criteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( MortarAndConvergenceCriteria );

KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );
KRATOS_DEFINE_LOCAL_FLAG( CONDITION_NUMBER_IS_INITIALIZED );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > ConvergenceCriteriaBaseType;

typedef And_Criteria< TSparseSpace, TDenseSpace >                           BaseType;

typedef MortarAndConvergenceCriteria< TSparseSpace, TDenseSpace >          ClassType;

typedef typename BaseType::DofsArrayType                               DofsArrayType;

typedef typename BaseType::TSystemMatrixType                       TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                       TSystemVectorType;

typedef TableStreamUtility::Pointer                          TablePrinterPointerType;

typedef std::size_t                                                        IndexType;

typedef ConditionNumberUtility::Pointer            ConditionNumberUtilityPointerType;



explicit MortarAndConvergenceCriteria()
: BaseType()
{
}


explicit MortarAndConvergenceCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit MortarAndConvergenceCriteria(
typename ConvergenceCriteriaBaseType::Pointer pFirstCriterion,
typename ConvergenceCriteriaBaseType::Pointer pSecondCriterion,
const bool PrintingOutput = false,
ConditionNumberUtilityPointerType pConditionNumberUtility = nullptr
)
:BaseType(pFirstCriterion, pSecondCriterion),
mpConditionNumberUtility(pConditionNumberUtility)
{
mOptions.Set(MortarAndConvergenceCriteria::PRINTING_OUTPUT, PrintingOutput);
mOptions.Set(MortarAndConvergenceCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(MortarAndConvergenceCriteria::CONDITION_NUMBER_IS_INITIALIZED, false);
}


MortarAndConvergenceCriteria(MortarAndConvergenceCriteria const& rOther)
:BaseType(rOther)
,mOptions(rOther.mOptions)
,mpConditionNumberUtility(rOther.mpConditionNumberUtility)
{
}


~MortarAndConvergenceCriteria () override = default;




typename ConvergenceCriteriaBaseType::Pointer Create(Parameters ThisParameters) const override
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
ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
p_table->AddToRow<IndexType>(rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER]);
}
}

bool criterion_result = BaseType::PostCriteria(rModelPart, rDofSet, rA, rDx, rb);

if (mpConditionNumberUtility != nullptr) {
TSystemMatrixType copy_A(rA); 
const double condition_number = mpConditionNumberUtility->GetConditionNumber(copy_A);

if (r_process_info.Has(TABLE_UTILITY)) {
std::cout.precision(4);
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
auto& r_table = p_table->GetTable();
r_table  << condition_number;
} else {
if (mOptions.IsNot(MortarAndConvergenceCriteria::PRINTING_OUTPUT))
KRATOS_INFO("MortarAndConvergenceCriteria") << "\n" << BOLDFONT("CONDITION NUMBER:") << "\t " << std::scientific << condition_number << std::endl;
else
KRATOS_INFO("MortarAndConvergenceCriteria") << "\n" << "CONDITION NUMBER:" << "\t" << std::scientific << condition_number << std::endl;
}
}

if (criterion_result == true && rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0)
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
p_table->PrintFooter();
}

return criterion_result;
}


void Initialize(ModelPart& rModelPart) override
{
ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (r_process_info.Has(TABLE_UTILITY) && mOptions.IsNot(MortarAndConvergenceCriteria::TABLE_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
(p_table->GetTable()).SetBold(mOptions.IsNot(MortarAndConvergenceCriteria::PRINTING_OUTPUT));
(p_table->GetTable()).AddColumn("ITER", 4);
}

mOptions.Set(MortarAndConvergenceCriteria::TABLE_IS_INITIALIZED, true);
BaseType::Initialize(rModelPart);

if (r_process_info.Has(TABLE_UTILITY) && mpConditionNumberUtility != nullptr
&& mOptions.IsNot(MortarAndConvergenceCriteria::CONDITION_NUMBER_IS_INITIALIZED)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
(p_table->GetTable()).AddColumn("COND.NUM.", 10);
mOptions.Set(MortarAndConvergenceCriteria::CONDITION_NUMBER_IS_INITIALIZED, true);
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
ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
std::cout.precision(4);
if (mOptions.IsNot(MortarAndConvergenceCriteria::PRINTING_OUTPUT))
std::cout << "\n\n" << BOLDFONT("CONVERGENCE CHECK") << "\tSTEP: " << rModelPart.GetProcessInfo()[STEP] << "\tTIME: " << std::scientific << rModelPart.GetProcessInfo()[TIME] << "\tDELTA TIME: " << std::scientific << rModelPart.GetProcessInfo()[DELTA_TIME] << std::endl;
else
std::cout << "\n\n" << "CONVERGENCE CHECK" << "\tSTEP: " << rModelPart.GetProcessInfo()[STEP] << "\tTIME: " << std::scientific << rModelPart.GetProcessInfo()[TIME] << "\tDELTA TIME: " << std::scientific << rModelPart.GetProcessInfo()[DELTA_TIME] << std::endl;

if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
p_table->PrintHeader();
}
}

BaseType::InitializeSolutionStep(rModelPart, rDofSet, rA, rDx, rb);
}


void FinalizeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::FinalizeSolutionStep(rModelPart,rDofSet, rA, rDx, rb);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                        : "mortar_and_criteria",
"print_convergence_criterion" : false
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "mortar_and_criteria";
}




std::string Info() const override
{
return "MortarAndConvergenceCriteria";
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

mpConditionNumberUtility = nullptr; 

mOptions.Set(MortarAndConvergenceCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
mOptions.Set(MortarAndConvergenceCriteria::TABLE_IS_INITIALIZED, false);
mOptions.Set(MortarAndConvergenceCriteria::CONDITION_NUMBER_IS_INITIALIZED, false);
}





private:



Flags mOptions; 

ConditionNumberUtilityPointerType mpConditionNumberUtility; 







};  


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags MortarAndConvergenceCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(0));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags MortarAndConvergenceCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags MortarAndConvergenceCriteria<TSparseSpace, TDenseSpace>::CONDITION_NUMBER_IS_INITIALIZED(Kratos::Flags::Create(2));

}  
