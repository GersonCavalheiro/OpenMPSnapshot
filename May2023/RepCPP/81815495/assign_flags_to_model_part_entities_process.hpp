
#if !defined(KRATOS_ASSIGN_FLAGS_TO_MODEL_PART_ENTITIES_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_FLAGS_TO_MODEL_PART_ENTITIES_PROCESS_H_INCLUDED




#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class AssignFlagsToModelPartEntitiesProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignFlagsToModelPartEntitiesProcess);



AssignFlagsToModelPartEntitiesProcess(ModelPart& rModelPart, const std::string EntityType,
const std::vector<Flags>& rAssignFlags) : Process(Flags()) , mrModelPart(rModelPart), mEntityType(EntityType), mrTransferFlags(std::vector<Flags>()), mrAssignFlags(rAssignFlags)
{
KRATOS_TRY

KRATOS_CATCH("");
}


AssignFlagsToModelPartEntitiesProcess(ModelPart& rModelPart, const std::string EntityType, const std::vector<Flags>& rAssignFlags,
const std::vector<Flags>& rTransferFlags) : Process(Flags()) , mrModelPart(rModelPart), mEntityType(EntityType), mrTransferFlags(rTransferFlags), mrAssignFlags(rAssignFlags)
{
KRATOS_TRY

KRATOS_CATCH("");
}


~AssignFlagsToModelPartEntitiesProcess() override {}



void operator()()
{
Execute();
}




void Execute() override
{
KRATOS_TRY;

if (mEntityType == "Nodes")
{
const int nnodes = mrModelPart.Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin =
mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if (this->MatchTransferFlags(*(it.base())))
{
this->AssignFlags(*(it.base()));
}
}
}
}
else if (mEntityType == "Elements")
{
const int nelements = mrModelPart.Elements().size();

if (nelements != 0)
{
ModelPart::ElementsContainerType::iterator it_begin =
mrModelPart.ElementsBegin();

#pragma omp parallel for
for (int i = 0; i < nelements; i++)
{
ModelPart::ElementsContainerType::iterator it = it_begin + i;

if (this->MatchTransferFlags(*(it.base())))
{
this->AssignFlags(*(it.base()));
}
}
}
}
else if (mEntityType == "Conditions")
{
const int nconditions = mrModelPart.Conditions().size();

if (nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin =
mrModelPart.ConditionsBegin();

for (int i = 0; i < nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

if (this->MatchTransferFlags(*(it.base())))
{
this->AssignFlags(*(it.base()));
}
}
}
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
return "AssignFlagsToModelPartEntitiesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignFlagsToModelPartEntitiesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


AssignFlagsToModelPartEntitiesProcess(AssignFlagsToModelPartEntitiesProcess const& rOther);


private:


ModelPart& mrModelPart;

const std::string mEntityType;

const std::vector<Flags> mrTransferFlags;
const std::vector<Flags> mrAssignFlags;


bool MatchTransferFlags(const Node::Pointer& pNode)
{

for(unsigned int i = 0; i<mrTransferFlags.size(); i++)
{
if( pNode->IsNot(mrTransferFlags[i]) )
return false;
}

return true;

}

void AssignFlags(const Node::Pointer& pNode)
{

for(unsigned int i = 0; i<mrAssignFlags.size(); i++)
pNode->Set(mrAssignFlags[i]);

}


bool MatchTransferFlags(const Element::Pointer& pElement)
{

for(unsigned int i = 0; i<mrTransferFlags.size(); i++)
{
if( pElement->IsNot(mrTransferFlags[i]) )
return false;
}

return true;

}

void AssignFlags(const Element::Pointer& pElement)
{

for(unsigned int i = 0; i<mrAssignFlags.size(); i++)
pElement->Set(mrAssignFlags[i]);

}


bool MatchTransferFlags(const Condition::Pointer& pCondition)
{

for(unsigned int i = 0; i<mrTransferFlags.size(); i++)
{
if( pCondition->IsNot(mrTransferFlags[i]) )
return false;
}

return true;

}

void AssignFlags(const Condition::Pointer& pCondition)
{

for(unsigned int i = 0; i<mrAssignFlags.size(); i++)
pCondition->Set(mrAssignFlags[i]);

}


AssignFlagsToModelPartEntitiesProcess& operator=(AssignFlagsToModelPartEntitiesProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignFlagsToModelPartEntitiesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignFlagsToModelPartEntitiesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
