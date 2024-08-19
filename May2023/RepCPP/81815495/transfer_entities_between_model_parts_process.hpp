
#if !defined(KRATOS_TRANSFER_ENTITIES_BETWEEN_MODEL_PARTS_PROCESS_H_INCLUDED)
#define  KRATOS_TRANSFER_ENTITIES_BETWEEN_MODEL_PARTS_PROCESS_H_INCLUDED





#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class TransferEntitiesBetweenModelPartsProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TransferEntitiesBetweenModelPartsProcess);


TransferEntitiesBetweenModelPartsProcess(ModelPart& rDestinationModelPart,
ModelPart& rOriginModelPart,
const std::string EntityType
) : Process(Flags()) , mrDestinationModelPart(rDestinationModelPart), mrOriginModelPart(rOriginModelPart), mEntityType(EntityType), mrTransferFlags(std::vector<Flags>()), mrAssignFlags(std::vector<Flags>())
{
KRATOS_TRY

KRATOS_CATCH("");
}


TransferEntitiesBetweenModelPartsProcess(ModelPart& rDestinationModelPart,
ModelPart& rOriginModelPart,
const std::string EntityType,
const std::vector<Flags>& rTransferFlags
) : Process(Flags()) , mrDestinationModelPart(rDestinationModelPart), mrOriginModelPart(rOriginModelPart), mEntityType(EntityType), mrTransferFlags(rTransferFlags), mrAssignFlags(std::vector<Flags>() )
{
KRATOS_TRY


KRATOS_CATCH("");
}


TransferEntitiesBetweenModelPartsProcess(ModelPart& rDestinationModelPart,
ModelPart& rOriginModelPart,
const std::string EntityType,
const std::vector<Flags>& rTransferFlags,
const std::vector<Flags>& rAssignFlags
) : Process(Flags()) , mrDestinationModelPart(rDestinationModelPart), mrOriginModelPart(rOriginModelPart), mEntityType(EntityType), mrTransferFlags(rTransferFlags), mrAssignFlags(rAssignFlags)
{
KRATOS_TRY

KRATOS_CATCH("");
}


~TransferEntitiesBetweenModelPartsProcess() override {}



void operator()()
{
Execute();
}




void Execute() override
{
KRATOS_TRY;

if (mEntityType == "Nodes")
{
const int nnodes = mrOriginModelPart.Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin =
mrOriginModelPart.NodesBegin();
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if (this->MatchTransferFlags(*(it.base())))
{
this->AssignFlags(*(it.base()));
mrDestinationModelPart.Nodes().push_back(*(it.base()));
}
}
mrDestinationModelPart.Nodes().Unique();
}
}
else if (mEntityType == "Elements")
{
const int nelements = mrOriginModelPart.Elements().size();

if (nelements != 0)
{
ModelPart::ElementsContainerType::iterator it_begin =
mrOriginModelPart.ElementsBegin();
for (int i = 0; i < nelements; i++)
{
ModelPart::ElementsContainerType::iterator it = it_begin + i;

if (this->MatchTransferFlags(*(it.base())))
{
this->AssignFlags(*(it.base()));
mrDestinationModelPart.Elements().push_back(*(it.base()));
}
}
mrDestinationModelPart.Elements().Unique();
}
}
else if (mEntityType == "Conditions")
{
const int nconditions = mrOriginModelPart.Conditions().size();

if (nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin =
mrOriginModelPart.ConditionsBegin();
for (int i = 0; i < nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

if (this->MatchTransferFlags(*(it.base())))
{
this->AssignFlags(*(it.base()));
mrDestinationModelPart.Conditions().push_back(*(it.base()));
}
}
mrDestinationModelPart.Conditions().Unique();
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
return "TransferEntitiesBetweenModelPartsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "TransferEntitiesBetweenModelPartsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


TransferEntitiesBetweenModelPartsProcess(TransferEntitiesBetweenModelPartsProcess const& rOther);


private:


ModelPart& mrDestinationModelPart;
ModelPart& mrOriginModelPart;

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


TransferEntitiesBetweenModelPartsProcess& operator=(TransferEntitiesBetweenModelPartsProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
TransferEntitiesBetweenModelPartsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const TransferEntitiesBetweenModelPartsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
