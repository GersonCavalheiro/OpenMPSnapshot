
#pragma once



#include "custom_strategies/custom_convergencecriterias/base_mortar_criteria.h"
#include "utilities/table_stream_utility.h"
#include "utilities/color_utilities.h"

namespace Kratos
{







template<class TSparseSpace, class TDenseSpace>
class MeshTyingMortarConvergenceCriteria
: public  BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( MeshTyingMortarConvergenceCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > ConvergenceCriteriaBaseType;

typedef BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace >          BaseType;

typedef MeshTyingMortarConvergenceCriteria< TSparseSpace, TDenseSpace >    ClassType;

typedef typename BaseType::DofsArrayType                               DofsArrayType;

typedef typename BaseType::TSystemMatrixType                       TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                       TSystemVectorType;

typedef TableStreamUtility::Pointer                          TablePrinterPointerType;



explicit MeshTyingMortarConvergenceCriteria()
: BaseType()
{
}


explicit MeshTyingMortarConvergenceCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}

MeshTyingMortarConvergenceCriteria( MeshTyingMortarConvergenceCriteria const& rOther )
:BaseType(rOther)
{
}

~MeshTyingMortarConvergenceCriteria() override = default;




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
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
}

return true;
}


void Initialize(ModelPart& rModelPart) override
{
ConvergenceCriteriaBaseType::mConvergenceCriteriaIsInitialized = true;

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info.Has(TABLE_UTILITY)) {
TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
}
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "mesh_tying_mortar_criteria"
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "mesh_tying_mortar_criteria";
}




std::string Info() const override
{
return "MeshTyingMortarConvergenceCriteria";
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
}




private:








}; 


}  
