
#if !defined( ROM_RESIDUALS_UTILITY_H_INCLUDED )
#define  ROM_RESIDUALS_UTILITY_H_INCLUDED


#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "spaces/ublas_space.h"


#include "rom_application_variables.h"
#include "custom_utilities/rom_auxiliary_utilities.h"

namespace Kratos
{
typedef UblasSpace<double, CompressedMatrix, boost::numeric::ublas::vector<double>> SparseSpaceType;
typedef UblasSpace<double, Matrix, Vector> LocalSpaceType;
typedef Scheme<SparseSpaceType, LocalSpaceType> BaseSchemeType;

class RomResidualsUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(RomResidualsUtility);

RomResidualsUtility(
ModelPart& rModelPart,
Parameters ThisParameters,
BaseSchemeType::Pointer pScheme
): mrModelPart(rModelPart), mpScheme(pScheme){
Parameters default_parameters = Parameters(R"(
{
"nodal_unknowns" : [],
"number_of_rom_dofs" : 10,
"petrov_galerkin_number_of_rom_dofs" : 10
})" );

ThisParameters.ValidateAndAssignDefaults(default_parameters);

mNodalVariablesNames = ThisParameters["nodal_unknowns"].GetStringArray();

mNodalDofs = mNodalVariablesNames.size();
mRomDofs = ThisParameters["number_of_rom_dofs"].GetInt();
mPetrovGalerkinRomDofs = ThisParameters["petrov_galerkin_number_of_rom_dofs"].GetInt();

for(int k=0; k<mNodalDofs; k++){
if(KratosComponents<Variable<double>>::Has(mNodalVariablesNames[k]))
{
const auto& var = KratosComponents<Variable<double>>::Get(mNodalVariablesNames[k]);
MapPhi[var.Key()] = k;
}
else
KRATOS_ERROR << "variable \""<< mNodalVariablesNames[k] << "\" not valid" << std::endl;

}
}

~RomResidualsUtility()= default;

Matrix GetProjectedResidualsOntoPhi()
{
const int n_elements = static_cast<int>(mrModelPart.Elements().size());
const int n_conditions = static_cast<int>(mrModelPart.Conditions().size());

const auto& r_current_process_info = mrModelPart.GetProcessInfo();
const auto el_begin = mrModelPart.ElementsBegin();
const auto cond_begin = mrModelPart.ConditionsBegin();

Matrix lhs_contribution = ZeroMatrix(0, 0);
Vector rhs_contribution = ZeroVector(0);

Element::EquationIdVectorType equation_id;
Matrix matrix_residuals( (n_elements + n_conditions), mRomDofs); 
Matrix phi_elemental;

Element::DofsVectorType elem_dofs;
Condition::DofsVectorType cond_dofs;
#pragma omp parallel firstprivate(n_elements, n_conditions, lhs_contribution, rhs_contribution, equation_id, phi_elemental, el_begin, cond_begin, elem_dofs, cond_dofs)
{
#pragma omp for nowait
for (int k = 0; k < n_elements; k++){
const auto it_el = el_begin + k;
bool element_is_active = true;
if ((it_el)->IsDefined(ACTIVE))
element_is_active = (it_el)->Is(ACTIVE);
if (element_is_active){
mpScheme->CalculateSystemContributions(*it_el, lhs_contribution, rhs_contribution, equation_id, r_current_process_info);
it_el->GetDofList(elem_dofs, r_current_process_info);
const auto& r_geom = it_el->GetGeometry();
if(phi_elemental.size1() != elem_dofs.size() || phi_elemental.size2() != mRomDofs)
phi_elemental.resize(elem_dofs.size(), mRomDofs,false);
RomAuxiliaryUtilities::GetPhiElemental(phi_elemental, elem_dofs, r_geom, MapPhi);
noalias(row(matrix_residuals, k)) = prod(trans(phi_elemental), rhs_contribution); 
}

}

#pragma omp for nowait
for (int k = 0; k < n_conditions;  k++){
ModelPart::ConditionsContainerType::iterator it = cond_begin + k;
bool condition_is_active = true;
if ((it)->IsDefined(ACTIVE))
condition_is_active = (it)->Is(ACTIVE);
if (condition_is_active){
it->GetDofList(cond_dofs, r_current_process_info);
mpScheme->CalculateSystemContributions(*it, lhs_contribution, rhs_contribution, equation_id, r_current_process_info);
const auto& r_geom = it->GetGeometry();
if(phi_elemental.size1() != cond_dofs.size() || phi_elemental.size2() != mRomDofs)
phi_elemental.resize(cond_dofs.size(), mRomDofs,false);
RomAuxiliaryUtilities::GetPhiElemental(phi_elemental, cond_dofs, r_geom, MapPhi);
noalias(row(matrix_residuals, k+n_elements)) = prod(trans(phi_elemental), rhs_contribution); 
}
}
}
return matrix_residuals;
}

Matrix GetProjectedResidualsOntoPsi()
{
const int n_elements = static_cast<int>(mrModelPart.Elements().size());
const int n_conditions = static_cast<int>(mrModelPart.Conditions().size());

const auto& r_current_process_info = mrModelPart.GetProcessInfo();
const auto el_begin = mrModelPart.ElementsBegin();
const auto cond_begin = mrModelPart.ConditionsBegin();

Matrix lhs_contribution;
Vector rhs_contribution;

Element::EquationIdVectorType equation_id;
Matrix matrix_residuals( (n_elements + n_conditions), mPetrovGalerkinRomDofs); 
Matrix psi_elemental;

Element::DofsVectorType elem_dofs;
Condition::DofsVectorType cond_dofs;
#pragma omp parallel firstprivate(n_elements, n_conditions, lhs_contribution, rhs_contribution, equation_id, psi_elemental, el_begin, cond_begin, elem_dofs, cond_dofs)
{
#pragma omp for nowait
for (int k = 0; k < n_elements; k++){
const auto it_el = el_begin + k;
const bool element_is_active = it_el->IsDefined(ACTIVE) ? it_el->Is(ACTIVE) : true;
if (element_is_active){
mpScheme->CalculateSystemContributions(*it_el, lhs_contribution, rhs_contribution, equation_id, r_current_process_info);
it_el->GetDofList(elem_dofs, r_current_process_info);
const auto& r_geom = it_el->GetGeometry();
if(psi_elemental.size1() != elem_dofs.size() || psi_elemental.size2() != mPetrovGalerkinRomDofs)
psi_elemental.resize(elem_dofs.size(), mPetrovGalerkinRomDofs,false);
RomAuxiliaryUtilities::GetPsiElemental(psi_elemental, elem_dofs, r_geom, MapPhi);
noalias(row(matrix_residuals, k)) = prod(trans(psi_elemental), rhs_contribution); 
}

}

#pragma omp for nowait
for (int k = 0; k < n_conditions;  k++){
const auto it = cond_begin + k;
const bool condition_is_active = it->IsDefined(ACTIVE) ? it->Is(ACTIVE) : true;
if (condition_is_active){
it->GetDofList(cond_dofs, r_current_process_info);
mpScheme->CalculateSystemContributions(*it, lhs_contribution, rhs_contribution, equation_id, r_current_process_info);
const auto& r_geom = it->GetGeometry();
if(psi_elemental.size1() != cond_dofs.size() || psi_elemental.size2() != mPetrovGalerkinRomDofs)
psi_elemental.resize(cond_dofs.size(), mPetrovGalerkinRomDofs,false);
RomAuxiliaryUtilities::GetPsiElemental(psi_elemental, cond_dofs, r_geom, MapPhi);
noalias(row(matrix_residuals, k+n_elements)) = prod(trans(psi_elemental), rhs_contribution); 
}
}
}
return matrix_residuals;
}

Matrix GetProjectedGlobalLHS()
{
const int n_elements = static_cast<int>(mrModelPart.Elements().size());
const int n_conditions = static_cast<int>(mrModelPart.Conditions().size());
const auto& n_nodes = mrModelPart.NumberOfNodes();

const auto& r_current_process_info = mrModelPart.GetProcessInfo();

const int system_size = n_nodes*mNodalDofs;

const auto el_begin = mrModelPart.ElementsBegin();
const auto cond_begin = mrModelPart.ConditionsBegin();

Matrix lhs_contribution = ZeroMatrix(0,0);

Element::EquationIdVectorType equation_id;
Matrix a_phi = ZeroMatrix(system_size, mRomDofs);

Element::DofsVectorType elem_dofs;
Condition::DofsVectorType cond_dofs;

Matrix phi_elemental;
Matrix temp_a_phi = ZeroMatrix(system_size,mRomDofs);
Matrix aux;

#pragma omp parallel firstprivate(n_elements, n_conditions, lhs_contribution, equation_id, el_begin, cond_begin, elem_dofs, cond_dofs)
{

#pragma omp for nowait
for (int k = 0; k < static_cast<int>(n_elements); k++) {
const auto it_el = el_begin + k;

const bool element_is_active = it_el->IsDefined(ACTIVE) ? it_el->Is(ACTIVE) : true;

if (element_is_active){
mpScheme->CalculateLHSContribution(*it_el, lhs_contribution, equation_id, r_current_process_info);
it_el->GetDofList(elem_dofs, r_current_process_info);
const auto &r_geom = it_el->GetGeometry();
if(phi_elemental.size1() != elem_dofs.size() || phi_elemental.size2() != mRomDofs) {
phi_elemental.resize(elem_dofs.size(), mRomDofs,false);
}
if(aux.size1() != elem_dofs.size() || aux.size2() != mRomDofs) {
aux.resize(elem_dofs.size(), mRomDofs,false);
}
RomAuxiliaryUtilities::GetPhiElemental(phi_elemental, elem_dofs, r_geom, MapPhi);
noalias(aux) = prod(lhs_contribution, phi_elemental);
for(int d = 0; d < static_cast<int>(elem_dofs.size()); ++d){
if (elem_dofs[d]->IsFixed()==false){
row(temp_a_phi,elem_dofs[d]->EquationId()) += row(aux,d);
}
}
}
}

#pragma omp for nowait
for (int k = 0; k < static_cast<int>(n_conditions); k++){
const auto it = cond_begin + k;

const bool condition_is_active = it->IsDefined(ACTIVE) ? it->Is(ACTIVE) : true;

if (condition_is_active) {
it->GetDofList(cond_dofs, r_current_process_info);
mpScheme->CalculateLHSContribution(*it, lhs_contribution, equation_id, r_current_process_info);
const auto &r_geom = it->GetGeometry();
if(phi_elemental.size1() != cond_dofs.size() || phi_elemental.size2() != mRomDofs) {
phi_elemental.resize(cond_dofs.size(), mRomDofs,false);
}
if(aux.size1() != cond_dofs.size() || aux.size2() != mRomDofs) {
aux.resize(cond_dofs.size(), mRomDofs,false);
}
RomAuxiliaryUtilities::GetPhiElemental(phi_elemental, cond_dofs, r_geom, MapPhi);
noalias(aux) = prod(lhs_contribution, phi_elemental);
for(int d = 0; d < static_cast<int>(cond_dofs.size()); ++d){
if (cond_dofs[d]->IsFixed()==false){
row(temp_a_phi,cond_dofs[d]->EquationId()) += row(aux,d);
}
}
}
}

#pragma omp critical
{
noalias(a_phi) += temp_a_phi;
}

}
return a_phi;
}

protected:
std::vector< std::string > mNodalVariablesNames;
int mNodalDofs;
unsigned int mRomDofs;
unsigned int mPetrovGalerkinRomDofs;
ModelPart& mrModelPart;
BaseSchemeType::Pointer mpScheme;
std::unordered_map<Kratos::VariableData::KeyType, Matrix::size_type> MapPhi;
};


} 



#endif 