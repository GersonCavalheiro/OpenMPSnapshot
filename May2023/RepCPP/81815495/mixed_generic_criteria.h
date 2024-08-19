
#ifndef KRATOS_MIXED_GENERIC_CRITERIA_H
#define	KRATOS_MIXED_GENERIC_CRITERIA_H



#include "includes/define.h"
#include "includes/model_part.h"
#include "convergence_criteria.h"


namespace Kratos
{



template< class TSparseSpace, class TDenseSpace >
class MixedGenericCriteria : public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MixedGenericCriteria);

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef MixedGenericCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef std::vector<std::tuple<const VariableData*, TDataType, TDataType>> ConvergenceVariableListType;

typedef std::size_t KeyType;



explicit MixedGenericCriteria()
: BaseType(),
mVariableSize(0)
{
}


explicit MixedGenericCriteria(Kratos::Parameters ThisParameters)
: MixedGenericCriteria(GenerateConvergenceVariableListFromParameters(ThisParameters))
{
}


MixedGenericCriteria(const ConvergenceVariableListType& rConvergenceVariablesList)
: BaseType()
, mVariableSize([&] (const ConvergenceVariableListType& rList) -> int {return rList.size();} (rConvergenceVariablesList))
, mVariableDataVector([&] (const ConvergenceVariableListType& rList) -> std::vector<const VariableData*> {
int i = 0;
std::vector<const VariableData*> aux_vect(mVariableSize);
for (const auto &r_tup : rList) {
aux_vect[i++] = std::get<0>(r_tup);
}
return aux_vect;
} (rConvergenceVariablesList))
, mRatioToleranceVector([&] (const ConvergenceVariableListType& rList) -> std::vector<TDataType> {
int i = 0;
std::vector<TDataType> aux_vect(mVariableSize);
for (const auto &r_tup : rList) {
aux_vect[i++] = std::get<1>(r_tup);
}
return aux_vect;
} (rConvergenceVariablesList))
, mAbsToleranceVector([&] (const ConvergenceVariableListType& rList) -> std::vector<TDataType> {
int i = 0;
std::vector<TDataType> aux_vect(mVariableSize);
for (const auto &r_tup : rList) {
aux_vect[i++] = std::get<2>(r_tup);
}
return aux_vect;
} (rConvergenceVariablesList))
, mLocalKeyMap([&] (const ConvergenceVariableListType& rList) -> std::unordered_map<KeyType, KeyType> {
KeyType local_key = 0;
std::unordered_map<KeyType, KeyType> aux_map;
for (const auto &r_tup : rList) {
const auto *p_var_data = std::get<0>(r_tup);
if (aux_map.find(p_var_data->Key()) != aux_map.end()) {
KRATOS_ERROR << "Convergence variable " << p_var_data->Name() << " is repeated. Check the input convergence variable list." << std::endl;
} else {
KRATOS_ERROR_IF(p_var_data->IsComponent()) << "Trying to check convergence with the " << p_var_data->Name() << " component variable. Use the corresponding vector one." << std::endl;
aux_map[p_var_data->Key()] = local_key++;
}
}
return aux_map;
} (rConvergenceVariablesList))
{}

~MixedGenericCriteria() override
{}





typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& A,
const TSystemVectorType& Dx,
const TSystemVectorType& b) override
{
if (TSparseSpace::Size(Dx) != 0) {
const auto convergence_norms = CalculateConvergenceNorms(rModelPart, rDofSet, Dx);

OutputConvergenceStatus(convergence_norms);

return CheckConvergence(convergence_norms);
} else {
return true;
}
}


static std::string Name()
{
return "mixed_generic_criteria";
}






std::string Info() const override
{
return "MixedGenericCriteria";
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








int GetVariableSize() const
{
return mVariableSize;
}


std::vector<const VariableData*> GetVariableDataVector() const
{
return mVariableDataVector;
}


std::vector<TDataType> GetRatioToleranceVector() const
{
return mRatioToleranceVector;
}


std::vector<TDataType> GetAbsToleranceVector() const
{
return mAbsToleranceVector;
}


std::unordered_map<KeyType, KeyType>& GetLocalKeyMap()
{
return mLocalKeyMap;
}


std::tuple<std::vector<TDataType>, std::vector<TDataType>> CalculateConvergenceNorms(
const ModelPart& rModelPart,
const DofsArrayType& rDofSet,
const TSystemVectorType& rDx)
{
std::vector<int> dofs_count(mVariableSize, 0);
std::vector<TDataType> solution_norms_vector(mVariableSize, 0.0);
std::vector<TDataType> increase_norms_vector(mVariableSize, 0.0);

GetNormValues(rModelPart, rDofSet, rDx, dofs_count, solution_norms_vector, increase_norms_vector);

const auto& r_data_comm = rModelPart.GetCommunicator().GetDataCommunicator();
auto global_solution_norms_vector = r_data_comm.SumAll(solution_norms_vector);
auto global_increase_norms_vector = r_data_comm.SumAll(increase_norms_vector);
auto global_dofs_count = r_data_comm.SumAll(dofs_count);

const double zero_tol = 1.0e-12;
for(int i = 0; i < mVariableSize; i++) {
if (global_solution_norms_vector[i] < zero_tol) {
global_solution_norms_vector[i] = 1.0;
}
}

std::vector<TDataType> var_ratio(mVariableSize, 0.0);
std::vector<TDataType> var_abs(mVariableSize, 0.0);
for(int i = 0; i < mVariableSize; i++) {
var_ratio[i] = std::sqrt(global_increase_norms_vector[i] / global_solution_norms_vector[i]);
var_abs[i] = std::sqrt(global_increase_norms_vector[i]) / static_cast<TDataType>(global_dofs_count[i]);
}

return std::make_tuple(var_ratio, var_abs);
}


virtual void OutputConvergenceStatus(
const std::tuple<std::vector<TDataType>,
std::vector<TDataType>>& rConvergenceNorms)
{
const auto& var_ratio = std::get<0>(rConvergenceNorms);
const auto& var_abs = std::get<1>(rConvergenceNorms);

if (this->GetEchoLevel() > 0) {
std::ostringstream stringbuf;
stringbuf << "CONVERGENCE CHECK:\n";

const int max_length_var_name = (*std::max_element(mVariableDataVector.begin(), mVariableDataVector.end(), [](const VariableData* p_var_data_1, const VariableData* p_var_data_2){
return p_var_data_1->Name().length() < p_var_data_2->Name().length();
}))->Name().length();

for(int i = 0; i < mVariableSize; i++) {
const auto r_var_data = mVariableDataVector[i];
const int key_map = mLocalKeyMap[r_var_data->Key()];
const std::string space_str(max_length_var_name-r_var_data->Name().length(), ' ');
stringbuf << " " << r_var_data->Name() << space_str <<" : ratio = " << var_ratio[key_map] << "; exp.ratio = " << mRatioToleranceVector[key_map] << " abs = " << var_abs[key_map] << " exp.abs = " << mAbsToleranceVector[key_map] << "\n";
}
KRATOS_INFO("") << stringbuf.str();
}
}


bool CheckConvergence(
const std::tuple<std::vector<TDataType>,
std::vector<TDataType>>& rConvergenceNorms)
{
bool is_converged = true;
const auto& var_ratio = std::get<0>(rConvergenceNorms);
const auto& var_abs = std::get<1>(rConvergenceNorms);

for (int i = 0; i < mVariableSize; i++) {
const auto r_var_data = mVariableDataVector[i];
const int key_map = mLocalKeyMap[r_var_data->Key()];
is_converged &= var_ratio[key_map] <= mRatioToleranceVector[key_map] || var_abs[key_map] <= mAbsToleranceVector[key_map];
}

if (is_converged) {
KRATOS_INFO_IF("", this->GetEchoLevel() > 0) << "*** CONVERGENCE IS ACHIEVED ***" << std::endl;
return true;
} else {
return false;
}
}







private:



const int mVariableSize;
const std::vector<const VariableData*> mVariableDataVector;
const std::vector<TDataType> mRatioToleranceVector;
const std::vector<TDataType> mAbsToleranceVector;
std::unordered_map<KeyType, KeyType> mLocalKeyMap;





virtual void GetNormValues(
const ModelPart& rModelPart,
const DofsArrayType& rDofSet,
const TSystemVectorType& rDx,
std::vector<int>& rDofsCount,
std::vector<TDataType>& rSolutionNormsVector,
std::vector<TDataType>& rIncreaseNormsVector)
{
int n_dofs = rDofSet.size();

#pragma omp parallel
{
int dof_id;
TDataType dof_dx;
TDataType dof_value;

std::vector<TDataType> var_solution_norm_reduction(mVariableSize);
std::vector<TDataType> var_correction_norm_reduction(mVariableSize);
std::vector<int> dofs_counter_reduction(mVariableSize);
for (int i = 0; i < mVariableSize; i++) {
var_solution_norm_reduction[i] = 0.0;
var_correction_norm_reduction[i] = 0.0;
dofs_counter_reduction[i] = 0;
}

#pragma omp for
for (int i = 0; i < n_dofs; i++) {
auto it_dof = rDofSet.begin() + i;
if (it_dof->IsFree()) {
dof_id = it_dof->EquationId();
dof_value = it_dof->GetSolutionStepValue(0);
dof_dx = TSparseSpace::GetValue(rDx, dof_id);

const auto &r_current_variable = it_dof->GetVariable();
int var_local_key = mLocalKeyMap[r_current_variable.IsComponent() ? r_current_variable.GetSourceVariable().Key() : r_current_variable.Key()];

var_solution_norm_reduction[var_local_key] += dof_value * dof_value;
var_correction_norm_reduction[var_local_key] += dof_dx * dof_dx;
dofs_counter_reduction[var_local_key]++;
}
}

#pragma omp critical
{
for (int i = 0; i < mVariableSize; i++) {
rDofsCount[i] += dofs_counter_reduction[i];
rSolutionNormsVector[i] += var_solution_norm_reduction[i];
rIncreaseNormsVector[i] += var_correction_norm_reduction[i];
}
}
}
}


static ConvergenceVariableListType GenerateConvergenceVariableListFromParameters(Kratos::Parameters ThisParameters)
{
ConvergenceVariableListType aux_list;
if (!ThisParameters.Has("convergence_variables_list")) return aux_list;
Kratos::Parameters convergence_variables_list = ThisParameters["convergence_variables_list"];
for (auto param : convergence_variables_list) {
if (param.Has("variable")) {
const std::string& r_variable_name = param["variable"].GetString();

const VariableData* p_variable = KratosComponents<Variable<double>>::Has(r_variable_name) ? dynamic_cast<const VariableData*>(&KratosComponents<Variable<double>>::Get(r_variable_name)) : dynamic_cast<const VariableData*>(&KratosComponents<Variable<array_1d<double, 3>>>::Get(r_variable_name));

const double rel_tol = param.Has("relative_tolerance") ? param["relative_tolerance"].GetDouble() : 1.0e-4;
const double abs_tol = param.Has("absolute_tolerance") ? param["absolute_tolerance"].GetDouble() : 1.0e-9;

aux_list.push_back(std::make_tuple(p_variable, rel_tol, abs_tol));
}
}

return aux_list;
}







};

}

#endif 
