
#if !defined(KRATOS_ASSIGN_PROPERTIES_TO_NODES_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_PROPERTIES_TO_NODES_PROCESS_H_INCLUDED



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "pfem_application_variables.h"

namespace Kratos
{



class AssignPropertiesToNodesProcess : public Process
{
public:

typedef Node NodeType;

typedef PointerVectorSet<Properties, IndexedObject> PropertiesContainerType;
typedef typename PropertiesContainerType::Pointer   PropertiesContainerPointerType;

typedef GlobalPointersVector<Element> ElementWeakPtrVectorType;
KRATOS_CLASS_POINTER_DEFINITION(AssignPropertiesToNodesProcess);

AssignPropertiesToNodesProcess(ModelPart& model_part,
Parameters rParameters
) : Process() , mrModelPart(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"fluid_mixture": false,
"solid_mixture": false
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

mFluidMixture = rParameters["fluid_mixture"].GetBool();
mSolidMixture = rParameters["solid_mixture"].GetBool();

mpProperties  = mrModelPart.pProperties();

KRATOS_CATCH("");
}


virtual ~AssignPropertiesToNodesProcess() {}



void operator()()
{
Execute();
}




void Execute()  override
{
}

void ExecuteInitialize() override
{
KRATOS_TRY

this->AssignPropertiesToNodes();

KRATOS_CATCH("")
}

void ExecuteBeforeSolutionLoop() override
{
}


void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY

this->AssignMaterialPercentageToNodes();

KRATOS_CATCH("")
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
return "AssignPropertiesToNodesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignPropertiesToNodesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


AssignPropertiesToNodesProcess(AssignPropertiesToNodesProcess const& rOther);


private:


ModelPart& mrModelPart;

bool mFluidMixture;

bool mSolidMixture;

PropertiesContainerPointerType mpProperties;



void AssignPropertiesToNodes()
{
const int nnodes = mrModelPart.GetMesh().Nodes().size();

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh().NodesBegin();

for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->SetValue(PROPERTIES_VECTOR,mpProperties);
}
}
}

void AssignMaterialPercentageToNodes()
{
const int nnodes = mrModelPart.GetMesh().Nodes().size();

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh().NodesBegin();

for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
Vector MaterialPercentage;
this->CalculateMaterialPercentage(*it, MaterialPercentage);
it->SetValue(MATERIAL_PERCENTAGE, MaterialPercentage);
}
}
}


void CalculateMaterialPercentage(NodeType& rNode, Vector& MaterialPercentage)
{
KRATOS_TRY

unsigned int size = mpProperties->size();
MaterialPercentage.resize(size,false);
noalias(MaterialPercentage) = ZeroVector(size);

double counter = 0;
if( rNode.Is(FLUID) && mFluidMixture ){

ElementWeakPtrVectorType& nElements = rNode.GetValue(NEIGHBOUR_ELEMENTS);

for(auto& i_nelem : nElements)
{
if(i_nelem.Is(FLUID)){
unsigned int id = i_nelem.GetProperties().Id();
if( id < size ){
MaterialPercentage[id] += 1;
++counter;
}
}
}
}
else if( rNode.Is(SOLID) && mSolidMixture ){

ElementWeakPtrVectorType& nElements = rNode.GetValue(NEIGHBOUR_ELEMENTS);
for(auto& i_nelem : nElements)
{
if(i_nelem.Is(SOLID)){
unsigned int id = i_nelem.GetProperties().Id();
if( id < size ){
MaterialPercentage[id] += 1;
++counter;
}
}
}

}

double divider = 1.0;
if( counter != 0 )
divider = 1.0/counter;

for(unsigned int i=0; i<size; ++i)
MaterialPercentage[i] *= divider;

KRATOS_CATCH("")
}


AssignPropertiesToNodesProcess& operator=(AssignPropertiesToNodesProcess const& rOther);



}; 






inline std::istream& operator >> (std::istream& rIStream,
AssignPropertiesToNodesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignPropertiesToNodesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
