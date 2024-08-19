
#if !defined(KRATOS_ADD_DOFS_PROCESS_H_INCLUDED)
#define  KRATOS_ADD_DOFS_PROCESS_H_INCLUDED





#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class AddDofsProcess : public Process
{
public:

typedef Variable<array_1d<double, 3> >                                    VectorVariableType;
typedef Variable<double>                                                  ScalarVariableType;
typedef Variable<double>                                                       ComponentType;

KRATOS_CLASS_POINTER_DEFINITION(AddDofsProcess);

AddDofsProcess(ModelPart& model_part,
Parameters rParameters
) : Process() , mrModelPart(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variables_list": [],
"reactions_list": []

}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

if( rParameters["variables_list"].size() != rParameters["reactions_list"].size() )
KRATOS_ERROR << "variables_list and reactions_list has not the same number of components "<<std::endl;


for(unsigned int i=0; i<rParameters["variables_list"].size(); i++)
{
if( !rParameters["variables_list"][i].IsString() )
KRATOS_ERROR << "variables_list contains a non-string variable name "<<std::endl;

std::string variable_name = rParameters["variables_list"][i].GetString();

bool supplied_reaction = true;
if(rParameters["reactions_list"][i].IsNull())
supplied_reaction = false;

if( KratosComponents< VectorVariableType >::Has( variable_name ) ){ 

const VectorVariableType& VectorVariable = KratosComponents< VectorVariableType >::Get(variable_name);
if( model_part.GetNodalSolutionStepVariablesList().Has( VectorVariable ) == false ){
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
else{
for(unsigned int j=0; j<3; j++)
{
std::string component_name = variable_name;
component_name += ms_components[j];
const ComponentType& ComponentVariable = KratosComponents< ComponentType >::Get(component_name);

if(supplied_reaction){
std::string reaction_component_name = rParameters["reactions_list"][i].GetString();
reaction_component_name += ms_components[j];
const ComponentType& ReactionComponentVariable = KratosComponents< ComponentType >::Get(reaction_component_name);
m_component_variables_list.push_back(&ComponentVariable);
m_component_reactions_list.push_back(&ReactionComponentVariable);
}
else{
m_component_variables_no_reaction_list.push_back(&ComponentVariable);
}

}
}
}
else if( KratosComponents< ComponentType >::Has(variable_name) ){ 

const ComponentType& ComponentVariable = KratosComponents< ComponentType >::Get(variable_name);

if( model_part.GetNodalSolutionStepVariablesList().Has( ComponentVariable.GetSourceVariable() ) == false ){

KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
else{

if(supplied_reaction){
std::string reaction_name = rParameters["reactions_list"][i].GetString();
const ComponentType& ReactionComponentVariable = KratosComponents< ComponentType >::Get(reaction_name);
m_component_variables_list.push_back(&ComponentVariable);
m_component_reactions_list.push_back(&ReactionComponentVariable);
}
else{
m_component_variables_no_reaction_list.push_back(&ComponentVariable);
}

}


}
else if( KratosComponents< ScalarVariableType >::Has( variable_name ) ){ 

const ScalarVariableType& ScalarVariable = KratosComponents< ScalarVariableType >::Get( variable_name );
if( model_part.GetNodalSolutionStepVariablesList().Has( ScalarVariable ) ==  false ){
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
else{

if(supplied_reaction){
std::string reaction_name = rParameters["reactions_list"][i].GetString();
const ScalarVariableType& ReactionVariable = KratosComponents< ScalarVariableType >::Get(reaction_name);
m_scalar_variables_list.push_back(&ScalarVariable);
m_scalar_reactions_list.push_back(&ReactionVariable);
}
else{
m_scalar_variables_no_reaction_list.push_back(&ScalarVariable);
}

}

}
else{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
}


KRATOS_CATCH("")
}


AddDofsProcess(ModelPart& model_part,
const pybind11::list& rVariablesList,
const pybind11::list& rReactionsList
) : Process(), mrModelPart(model_part)
{
KRATOS_TRY

unsigned int number_variables = len(rVariablesList);
unsigned int number_reactions = len(rReactionsList);

if( number_variables != number_reactions )
KRATOS_ERROR << "variables_list and reactions_list has not the same number of components "<<std::endl;

for(unsigned int i=0; i<number_variables; i++)
{

std::string variable_name = pybind11::cast<std::string>(rVariablesList[i]);
std::string reaction_name = pybind11::cast<std::string>(rReactionsList[i]);

bool supplied_reaction = true;
if(reaction_name == "NOT_DEFINED")
supplied_reaction = false;

if( KratosComponents< VectorVariableType >::Has( variable_name ) ){ 

const VectorVariableType& VectorVariable = KratosComponents< VectorVariableType >::Get(variable_name);
if( model_part.GetNodalSolutionStepVariablesList().Has( VectorVariable ) == false ){
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
else{
for(unsigned int j=0; j<3; j++)
{
std::string component_name = variable_name;
component_name += ms_components[j];
const ComponentType& ComponentVariable = KratosComponents< ComponentType >::Get(component_name);

if(supplied_reaction){
std::string reaction_component_name = reaction_name;
reaction_component_name += ms_components[j];
const ComponentType& ReactionComponentVariable = KratosComponents< ComponentType >::Get(reaction_component_name);
m_component_variables_list.push_back(&ComponentVariable);
m_component_reactions_list.push_back(&ReactionComponentVariable);
}
else{
m_component_variables_no_reaction_list.push_back(&ComponentVariable);
}

}
}
}
else if( KratosComponents< ComponentType >::Has(variable_name) ){ 

const ComponentType& ComponentVariable = KratosComponents< ComponentType >::Get(variable_name);

if( model_part.GetNodalSolutionStepVariablesList().Has( ComponentVariable.GetSourceVariable() ) == false ){

KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
else{

if(supplied_reaction){
const ComponentType& ReactionComponentVariable = KratosComponents< ComponentType >::Get(reaction_name);
m_component_variables_list.push_back(&ComponentVariable);
m_component_reactions_list.push_back(&ReactionComponentVariable);
}
else{
m_component_variables_list.push_back(&ComponentVariable);
}

}

}
else if( KratosComponents< ScalarVariableType >::Has( variable_name ) ){ 

const ScalarVariableType& ScalarVariable = KratosComponents< ScalarVariableType >::Get( variable_name );
if( model_part.GetNodalSolutionStepVariablesList().Has( ScalarVariable ) ==  false ){
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
else{

if(supplied_reaction){
const ScalarVariableType& ReactionVariable = KratosComponents< ScalarVariableType >::Get(reaction_name);
m_scalar_variables_list.push_back(&ScalarVariable);
m_scalar_reactions_list.push_back(&ReactionVariable);
}
else{
m_scalar_variables_no_reaction_list.push_back(&ScalarVariable);
}

}

}
else{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is "<<variable_name<<std::endl;
}
}

KRATOS_CATCH("")
}


~AddDofsProcess() override {}



void operator()()
{
Execute();
}




void Execute() override
{

KRATOS_TRY;

int number_of_nodes = mrModelPart.NumberOfNodes();
ModelPart::NodeConstantIterator nodes_begin = mrModelPart.NodesBegin();



for (int k=0; k<number_of_nodes; k++)
{
ModelPart::NodeConstantIterator it = nodes_begin + k;
AddNodalDofs(it);
}





KRATOS_CATCH("");

}

void ExecuteInitialize() override
{
}

void ExecuteBeforeSolutionLoop() override
{
}


void ExecuteInitializeSolutionStep() override
{
}

void ExecuteFinalizeSolutionStep() override
{
}


void ExecuteBeforeOutputStep() override
{
}


void ExecuteAfterOutputStep() override
{
}


void ExecuteFinalize() override
{
}







std::string Info() const override
{
return "AddDofsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AddDofsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


AddDofsProcess(AddDofsProcess const& rOther);


private:


ModelPart& mrModelPart;

const std::vector<std::string> ms_components {"_X", "_Y", "_Z"};

std::vector<ComponentType const *> m_component_variables_list;
std::vector<ComponentType const *> m_component_reactions_list;
std::vector<ComponentType const *> m_component_variables_no_reaction_list;

std::vector<ScalarVariableType const *> m_scalar_variables_list;
std::vector<ScalarVariableType const *> m_scalar_reactions_list;
std::vector<ScalarVariableType const *> m_scalar_variables_no_reaction_list;




void AddNodalDofs()
{
KRATOS_TRY

int number_of_nodes = mrModelPart.NumberOfNodes();
ModelPart::NodeConstantIterator nodes_begin = mrModelPart.NodesBegin();

for( unsigned int i=0; i < m_component_variables_list.size(); i++ )
{
#pragma omp parallel for
for (int k=0; k<number_of_nodes; k++)
{
ModelPart::NodeConstantIterator it = nodes_begin + k;
it->pAddDof(*m_component_variables_list[i],*m_component_reactions_list[i]);
}
}

for( unsigned int j=0; j < m_component_variables_no_reaction_list.size(); j++ )
{
#pragma omp parallel for
for (int k=0; k<number_of_nodes; k++)
{
ModelPart::NodeConstantIterator it = nodes_begin + k;
it->pAddDof(*m_component_variables_no_reaction_list[j]);
}
}

for( unsigned int l=0; l < m_scalar_variables_list.size(); l++ )
{
#pragma omp parallel for
for (int k=0; k<number_of_nodes; k++)
{
ModelPart::NodeConstantIterator it = nodes_begin + k;
it->pAddDof(*m_scalar_variables_list[l],*m_scalar_reactions_list[l]);
}
}

for( unsigned int m=0; m < m_scalar_variables_no_reaction_list.size(); m++ )
{
#pragma omp parallel for
for (int k=0; k<number_of_nodes; k++)
{
ModelPart::NodeConstantIterator it = nodes_begin + k;
it->pAddDof(*m_scalar_variables_no_reaction_list[m]);
}
}

KRATOS_CATCH(" ")
}


void AddNodalDofs( ModelPart::NodeConstantIterator& node_it )
{
KRATOS_TRY

for( unsigned int i=0; i < m_component_variables_list.size(); i++ )
{
node_it->pAddDof(*m_component_variables_list[i],*m_component_reactions_list[i]);
}

for( unsigned int j=0; j < m_component_variables_no_reaction_list.size(); j++ )
{
node_it->pAddDof(*m_component_variables_no_reaction_list[j]);
}

for( unsigned int l=0; l < m_scalar_variables_list.size(); l++ )
{
node_it->pAddDof(*m_scalar_variables_list[l],*m_scalar_reactions_list[l]);
}

for( unsigned int m=0; m < m_scalar_variables_no_reaction_list.size(); m++ )
{
node_it->pAddDof(*m_scalar_variables_no_reaction_list[m]);
}

KRATOS_CATCH(" ")
}


void CheckNodalData( ModelPart::NodeConstantIterator& node_it )
{
KRATOS_TRY

std::cout<<" CHECK VARIABLES LIST KEYS "<<std::endl;

VariablesListDataValueContainer VariablesList = (node_it)->SolutionStepData();

std::cout<<" list size "<<VariablesList.pGetVariablesList()->size()<<std::endl;
std::cout<<" Variable: "<<(*VariablesList.pGetVariablesList())[0]<<std::endl;
std::cout<<" end "<<std::endl;

KRATOS_CATCH(" ")
}


AddDofsProcess& operator=(AddDofsProcess const& rOther);



}; 






inline std::istream& operator >> (std::istream& rIStream,
AddDofsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AddDofsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
