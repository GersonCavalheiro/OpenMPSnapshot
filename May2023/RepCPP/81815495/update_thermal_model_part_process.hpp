
#if !defined(UPDATE_THERMAL_MODEL_PART_PROCESS)
#define UPDATE_THERMAL_MODEL_PART_PROCESS

#include <algorithm>
#include <iostream>
#include <string>
#include "includes/define.h"
#include "includes/model_part.h"
#include "processes/process.h"
#include "utilities/variable_utils.h"

namespace Kratos {


class UpdateThermalModelPartProcess : public Process {
public:
typedef Element BaseType;

KRATOS_CLASS_POINTER_DEFINITION(UpdateThermalModelPartProcess);

UpdateThermalModelPartProcess(ModelPart& origin_model_part, ModelPart& destination_model_part,
ModelPart& computing_model_part, unsigned int DomainSize)
: rOriginModelPart(origin_model_part),
rDestinationModelPart(destination_model_part),
rComputingModelPart(computing_model_part) {
rDomainSize = DomainSize;
if (rDomainSize == 2) {
ReferenceElement = "EulerianConvDiff2D";
} else {
ReferenceElement = "EulerianConvDiff3D";
}
}

~UpdateThermalModelPartProcess() override {}

void operator()() { Execute(); }

void Execute() override {
KRATOS_TRY;

this->ResetDestinationModelPart();
this->CopyNodes();
this->DuplicateElements();
this->BuildThermalComputingDomain();
this->UpdateConditions();


KRATOS_CATCH("");
}

void ExecuteInitialize() override {}

void ExecuteInitializeSolutionStep() override {}

protected:
ModelPart& rOriginModelPart;
ModelPart& rDestinationModelPart;
ModelPart& rComputingModelPart;
std::string ReferenceElement;
unsigned int rDomainSize;

private:
void ResetDestinationModelPart() const {
rOriginModelPart.RemoveNodesFromAllLevels(TO_ERASE);
VariableUtils().SetFlag(TO_ERASE, true, rDestinationModelPart.Nodes());
VariableUtils().SetFlag(TO_ERASE, true, rDestinationModelPart.Elements());
rDestinationModelPart.RemoveNodesFromAllLevels(TO_ERASE);
rDestinationModelPart.RemoveElementsFromAllLevels(TO_ERASE);
VariableUtils().SetFlag(TO_ERASE, false, rOriginModelPart.Nodes());
}

void CopyNodes() const {
rDestinationModelPart.AddNodes(rOriginModelPart.NodesBegin(), rOriginModelPart.NodesEnd());

for (auto i_part = rOriginModelPart.SubModelPartsBegin(); i_part != rOriginModelPart.SubModelPartsEnd(); ++i_part) {
if (!rDestinationModelPart.HasSubModelPart(i_part->Name())) {
rDestinationModelPart.CreateSubModelPart(i_part->Name());
}
ModelPart& destination_part = rDestinationModelPart.GetSubModelPart(i_part->Name());
destination_part.AddNodes(i_part->NodesBegin(), i_part->NodesEnd());
}
}

void DuplicateElements() const {
const Element& mReferenceElement = KratosComponents<Element>::Get(ReferenceElement);
for (ModelPart::SubModelPartIterator i_mp = rOriginModelPart.SubModelPartsBegin();
i_mp != rOriginModelPart.SubModelPartsEnd(); i_mp++) {
if (i_mp->NumberOfElements()) {
ModelPart& destination_part = rDestinationModelPart.GetSubModelPart(i_mp->Name());
ModelPart::ElementsContainerType temp_elements;
temp_elements.reserve(i_mp->NumberOfElements());

if ((i_mp->Is(SOLID) && i_mp->IsNot(ACTIVE)) || (i_mp->Is(FLUID) && i_mp->IsNot(ACTIVE)) ||
(i_mp->Is(BOUNDARY) && i_mp->Is(RIGID))) {
for (auto it_elem = i_mp->ElementsBegin(); it_elem != i_mp->ElementsEnd(); ++it_elem) {
Properties::Pointer properties = it_elem->pGetProperties();

Element::Pointer p_element =
mReferenceElement.Create(it_elem->Id(), it_elem->GetGeometry(), properties);
temp_elements.push_back(p_element);
}
rDestinationModelPart.AddElements(temp_elements.begin(), temp_elements.end());
destination_part.AddElements(temp_elements.begin(), temp_elements.end());
}
}
}
}

void BuildThermalComputingDomain() const {
rComputingModelPart.AddNodes(rDestinationModelPart.NodesBegin(), rDestinationModelPart.NodesEnd());

std::vector<ModelPart::IndexType> ids;
ids.reserve(rDestinationModelPart.Elements().size());
const auto& it_elem_begin = rDestinationModelPart.ElementsBegin();

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rDestinationModelPart.Elements().size()); i++) {
auto it_elem = it_elem_begin + i;
#pragma omp critical
{ ids.push_back(it_elem->Id()); }
}

rComputingModelPart.AddElements(ids, 0);
}

void UpdateConditions() const
{
for (ModelPart::SubModelPartIterator i_mp = rOriginModelPart.SubModelPartsBegin(); i_mp != rOriginModelPart.SubModelPartsEnd(); i_mp++) {
if (i_mp->NumberOfConditions() && rDestinationModelPart.HasSubModelPart(i_mp->Name())) {
ModelPart& destination_part = rDestinationModelPart.GetSubModelPart(i_mp->Name());
VariableUtils().SetFlag(TO_ERASE, true, destination_part.Conditions());
destination_part.RemoveConditionsFromAllLevels(TO_ERASE);
}
}
unsigned int condition_id = 0;

for (ModelPart::SubModelPartIterator i_mp = rOriginModelPart.SubModelPartsBegin(); i_mp != rOriginModelPart.SubModelPartsEnd(); i_mp++)
{
if (i_mp->NumberOfConditions() && rDestinationModelPart.HasSubModelPart(i_mp->Name())) {
ModelPart& destination_part = rDestinationModelPart.GetSubModelPart(i_mp->Name());

for (auto i_cond(i_mp->ConditionsBegin()); i_cond != i_mp->ConditionsEnd(); ++i_cond)
{
Geometry<Node>& r_geometry = i_cond->GetGeometry();
Condition::NodesArrayType cond_nodes;
cond_nodes.reserve(r_geometry.size());
for (unsigned int i = 0; i < r_geometry.size(); i++)
cond_nodes.push_back(r_geometry(i));

Properties::Pointer p_property = i_cond->pGetProperties();

std::string condition_type;
if      (rDomainSize == 2) condition_type = "ThermalFace2D2N";
else if (rDomainSize == 3) condition_type = "ThermalFace3D3N";
const Condition& r_reference_condition = KratosComponents<Condition>::Get(condition_type);

Condition::Pointer p_condition = r_reference_condition.Create(++condition_id, cond_nodes, p_property);
destination_part.Conditions().push_back(p_condition);
rDestinationModelPart.Conditions().push_back(p_condition);
rComputingModelPart.Conditions().push_back(p_condition);
}
}
}
}

void FixDOFs() const {
for (auto i_cond(rComputingModelPart.ConditionsBegin()); i_cond != rComputingModelPart.ConditionsEnd(); ++i_cond)
{
Geometry<Node>& cond_geometry = i_cond->GetGeometry();
for (unsigned int i = 0; i < cond_geometry.size(); i++)
{
bool fix = true;
ElementWeakPtrVectorType& neighbour_elements = cond_geometry(i)->GetValue(NEIGHBOUR_ELEMENTS);
for (unsigned int j = 0; j < neighbour_elements.size(); j++)
{
GeometryType neighbour_elements_geom = neighbour_elements(j)->GetGeometry();
if (neighbour_elements_geom.size() != 2) {
fix = false;
break;
}
}
if (fix) {
const Node::DofType::Pointer dof = cond_geometry(i)->pGetDof(TEMPERATURE);
if (!dof->IsFixed())
dof->FixDof();
}
}
}
}

};  

inline std::istream& operator>>(std::istream& rIStream, UpdateThermalModelPartProcess& rThis);

inline std::ostream& operator<<(std::ostream& rOStream, const UpdateThermalModelPartProcess& rThis) {
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  

#endif 