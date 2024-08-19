



#include "custom_statistics/power_sums_statistics.h"


namespace Kratos
{


PowerSumsStatistics::PowerSumsStatistics(
ModelPart& rModelPart,
Parameters ThisParameters):
Process(),
mrModelPart(rModelPart)
{

Parameters default_parameters = Parameters(R"(
{
"reference_variable_name" : "PLEASE_SPECIFY_VARIABLE_NAME"
})"
);

ThisParameters.ValidateAndAssignDefaults(default_parameters);

mReferenceVariable = ThisParameters["reference_variable_name"].GetString();
}

std::string PowerSumsStatistics::Info() const
{
return "PowerSumsStatistics";
}

void PowerSumsStatistics::PrintInfo(std::ostream& rOStream) const
{
rOStream << "PowerSumsStatistics";
}

void PowerSumsStatistics::PrintData(std::ostream& rOStream) const
{
this->PrintInfo(rOStream);
}

void PowerSumsStatistics::ExecuteInitialize()
{
KRATOS_TRY;

auto& r_nodes_array = mrModelPart.Nodes();
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_1, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_2, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_3, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_4, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_5, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_6, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_7, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_8, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_9, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(POWER_SUM_10, r_nodes_array);

KRATOS_CATCH("");
}

void PowerSumsStatistics::ExecuteFinalizeSolutionStep()
{
KRATOS_TRY;

KRATOS_ERROR_IF(mrModelPart.GetCommunicator().GlobalNumberOfNodes() == 0) << "The number of nodes in the domain is zero. The power sums statistic cannot be applied."<< std::endl;
const unsigned int number_nodes = mrModelPart.NumberOfNodes();

if (KratosComponents<Variable<double>>::Has(mReferenceVariable)) {
const auto& r_reference_var = KratosComponents<Variable<double>>::Get(mReferenceVariable);
#pragma omp parallel for
for(int i_node = 0; i_node < static_cast<int>(number_nodes); ++i_node) {
auto it_node = mrModelPart.NodesBegin() + i_node;
double variable_current = it_node->GetSolutionStepValue(r_reference_var);
it_node->GetValue(POWER_SUM_1) += std::pow(variable_current,1);
it_node->GetValue(POWER_SUM_2) += std::pow(variable_current,2);
it_node->GetValue(POWER_SUM_3) += std::pow(variable_current,3);
it_node->GetValue(POWER_SUM_4) += std::pow(variable_current,4);
it_node->GetValue(POWER_SUM_5) += std::pow(variable_current,5);
it_node->GetValue(POWER_SUM_6) += std::pow(variable_current,6);
it_node->GetValue(POWER_SUM_7) += std::pow(variable_current,7);
it_node->GetValue(POWER_SUM_8) += std::pow(variable_current,8);
it_node->GetValue(POWER_SUM_9) += std::pow(variable_current,9);
it_node->GetValue(POWER_SUM_10) += std::pow(variable_current,10);
}
}
else {
KRATOS_ERROR << "Variable " << mReferenceVariable << " not found in the scalar variables list of Kratos.\n";
}

KRATOS_CATCH("");
}



inline std::ostream& operator << (
std::ostream& rOStream,
const PowerSumsStatistics& rThis)
{
rThis.PrintData(rOStream);
return rOStream;
}

}; 
