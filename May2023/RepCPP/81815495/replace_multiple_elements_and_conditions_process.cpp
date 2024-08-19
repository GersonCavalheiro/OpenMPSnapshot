


#include "replace_multiple_elements_and_conditions_process.h"
#include "utilities/compare_elements_and_conditions_utility.h"

namespace Kratos
{

namespace {

template <class TEntityContainer>
void ReplaceEntities(TEntityContainer& rEntityContainer,
Parameters EntitySettings,
Parameters IgnoreEntities,
bool IgnoreUndefinedTypes)
{
typedef typename TEntityContainer::data_type EntityType;

std::map<std::string, const EntityType*> entities_table;
for(Parameters::iterator it=EntitySettings.begin(); it!=EntitySettings.end(); ++it){
entities_table[it.name()] = &KratosComponents<EntityType>::Get(
EntitySettings[it.name()].GetString()
);
}

std::set<std::string> ignore_entities;
for(const auto& it : IgnoreEntities){
ignore_entities.insert( it.GetString() );
}

#pragma omp parallel for
for(int i=0; i< static_cast<int>(rEntityContainer.size()); i++) {
auto it = rEntityContainer.begin() + i;

std::string current_name;
CompareElementsAndConditionsUtility::GetRegisteredName(*it, current_name);

auto it_reference_entity = entities_table.find(current_name);

if (it_reference_entity == entities_table.end()) {
if (ignore_entities.find(current_name) != ignore_entities.end()){
continue;
}
else if (!IgnoreUndefinedTypes){
KRATOS_ERROR << current_name
<< " was not defined in the replacement table!" << std::endl;
}
else{
KRATOS_WARNING_ONCE("ReplaceEntities") << "ignoring undefined entity '"
<< current_name  << "'!" << std::endl;
continue;
}

}

auto p_entity = it_reference_entity->second->Create(it->Id(),
it->pGetGeometry(),
it->pGetProperties()
);

p_entity->GetData() = it->GetData();
p_entity->Set(Flags(*it));

(*it.base()) = p_entity;
}
}

} 


void ReplaceMultipleElementsAndConditionsProcess::Execute()
{
ModelPart& r_root_model_part = mrModelPart.GetRootModelPart();

bool ignore_undefined_types = mSettings["ignore_undefined_types"].GetBool();

ReplaceEntities(mrModelPart.Elements(),
mSettings["element_name_table"],
mSettings["ignore_elements"],
ignore_undefined_types
);

ReplaceEntities(mrModelPart.Conditions(),
mSettings["condition_name_table"],
mSettings["ignore_conditions"],
ignore_undefined_types
);

for (auto& i_sub_model_part : r_root_model_part.SubModelParts()) {
UpdateSubModelPart( i_sub_model_part, r_root_model_part );
}
}

void ReplaceMultipleElementsAndConditionsProcess::UpdateSubModelPart(
ModelPart& rModelPart,
ModelPart& rRootModelPart
)
{
#pragma omp parallel for
for(int i=0; i< static_cast<int>(rModelPart.Elements().size()); i++) {
auto it_elem = rModelPart.ElementsBegin() + i;

(*it_elem.base()) = rRootModelPart.Elements()(it_elem->Id());
}

#pragma omp parallel for
for(int i=0; i< static_cast<int>(rModelPart.Conditions().size()); i++) {
auto it_cond = rModelPart.ConditionsBegin() + i;

(*it_cond.base()) = rRootModelPart.Conditions()(it_cond->Id());
}

for (auto& i_sub_model_part : rModelPart.SubModelParts()) {
UpdateSubModelPart( i_sub_model_part, rRootModelPart );
}
}

}  



