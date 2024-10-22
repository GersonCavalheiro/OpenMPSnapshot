
#if !defined(KRATOS_TEMPORAL_MAX_METHOD_H_INCLUDED)
#define KRATOS_TEMPORAL_MAX_METHOD_H_INCLUDED

#include <limits>
#include <string>


#include "includes/define.h"
#include "includes/model_part.h"

#include "custom_methods/temporal_method.h"
#include "custom_utilities/method_utilities.h"
#include "custom_utilities/temporal_method_utilities.h"

namespace Kratos
{


namespace TemporalMethods
{
template <class TContainerType, class TContainerItemType, template <class T> class TDataRetrievalFunctor, template <class T> class TDataStorageFunctor>
class TemporalMaxMethod
{
public:
template <class TDataType>
class NormMethod : public TemporalMethod
{
public:
KRATOS_CLASS_POINTER_DEFINITION(NormMethod);

NormMethod(
ModelPart& rModelPart,
const std::string& rNormType,
const Variable<TDataType>& rInputVariable,
const int EchoLevel,
const Variable<double>& rOutputVariable,
const Variable<double>& rMaxTimeValueVariable)
: TemporalMethod(rModelPart, EchoLevel),
mNormType(rNormType),
mrInputVariable(rInputVariable),
mrOutputVariable(rOutputVariable),
mrMaxTimeValueVariable(rMaxTimeValueVariable)
{
KRATOS_TRY

KRATOS_ERROR_IF(rOutputVariable == rMaxTimeValueVariable)
<< "Same variable is given for maximum and its time value "
"variable in norm min method for input variable "
<< rInputVariable.Name()
<< ". Please provide two different "
"variables. [ variable = "
<< rOutputVariable.Name() << " ].\n";

KRATOS_CATCH("");
}

void CalculateStatistics() override
{
TContainerType& r_container =
MethodUtilities::GetDataContainer<TContainerType>(this->GetModelPart());

const auto& norm_method =
MethodUtilities::GetNormMethod(mrInputVariable, mNormType);

const double total_time = this->GetTotalTime();

const int number_of_items = r_container.size();
#pragma omp parallel for
for (int i = 0; i < number_of_items; ++i)
{
TContainerItemType& r_item = *(r_container.begin() + i);
const TDataType& r_input_value =
TDataRetrievalFunctor<TContainerItemType>()(r_item, mrInputVariable);
const double input_norm_value = norm_method(r_input_value);

double& r_output_value =
TDataStorageFunctor<TContainerItemType>()(r_item, mrOutputVariable);
double& r_max_time_value = TDataStorageFunctor<TContainerItemType>()(
r_item, mrMaxTimeValueVariable);

if (input_norm_value > r_output_value)
{
r_output_value = input_norm_value;
r_max_time_value = total_time;
}
}

KRATOS_INFO_IF("TemporalNormMaxMethod", this->GetEchoLevel() > 1)
<< "Calculated temporal norm max for " << mrInputVariable.Name()
<< " input variable with " << mrOutputVariable.Name()
<< " max variable and " << mrMaxTimeValueVariable.Name()
<< " time value variable in " << this->GetModelPart().Name() << ".\n";
}

void InitializeStatisticsVariables() override
{
TContainerType& r_container =
MethodUtilities::GetDataContainer<TContainerType>(this->GetModelPart());

auto& initializer_method =
TemporalMethodUtilities::InitializeVariables<TContainerType, TContainerItemType, TDataStorageFunctor>;
initializer_method(
r_container, mrOutputVariable, std::numeric_limits<double>::lowest());
initializer_method(r_container, mrMaxTimeValueVariable, 0.0);

KRATOS_INFO_IF("TemporalNormMaxMethod", this->GetEchoLevel() > 0)
<< "Initialized temporal norm max method for "
<< mrInputVariable.Name() << " input variable with "
<< mrOutputVariable.Name() << " max variable and "
<< mrMaxTimeValueVariable.Name() << " time value variable in "
<< this->GetModelPart().Name() << ".\n";
}

private:
const std::string mNormType;
const Variable<TDataType>& mrInputVariable;
const Variable<double>& mrOutputVariable;
const Variable<double>& mrMaxTimeValueVariable;
};

std::vector<TemporalMethod::Pointer> static CreateTemporalMethodObject(
ModelPart& rModelPart, const std::string& rNormType, const int EchoLevel, Parameters Params)
{
KRATOS_TRY

Parameters default_parameters = Parameters(R"(
{
"input_variables"            : [],
"output_variables"           : [],
"output_time_step_variables" : []
})");
Params.RecursivelyValidateAndAssignDefaults(default_parameters);

const std::vector<std::string>& input_variable_names_list =
Params["input_variables"].GetStringArray();
const std::vector<std::string>& output_variable_1_names_list =
Params["output_variables"].GetStringArray();
const std::vector<std::string>& output_variable_2_names_list =
Params["output_time_step_variables"].GetStringArray();

std::vector<TemporalMethod::Pointer> method_list;
if (rNormType == "none") 
{
KRATOS_ERROR << "none norm type is not defined for Min method.\n";
}
else 
{
MethodUtilities::CheckVariableType<double>(output_variable_1_names_list);
MethodUtilities::CheckVariableType<double>(output_variable_2_names_list);

const int number_of_variables = input_variable_names_list.size();
for (int i = 0; i < number_of_variables; ++i)
{
const std::string& r_variable_input_name = input_variable_names_list[i];
const std::string& r_variable_1_output_name =
output_variable_1_names_list[i];
const std::string& r_variable_2_output_name =
output_variable_2_names_list[i];
ADD_TEMPORAL_NORM_METHOD_TWO_OUTPUT_VARIABLE_OBJECT(
rModelPart, rNormType, r_variable_input_name, EchoLevel,
r_variable_1_output_name, r_variable_2_output_name, method_list, NormMethod)
}
}

return method_list;

KRATOS_CATCH("");
}
};
} 
} 

#endif 