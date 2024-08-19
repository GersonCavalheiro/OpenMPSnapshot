
#if !defined(KRATOS_TRANSFER_MODEL_PART_ELEMENTS_PROCESS_H_INCLUDED)
#define KRATOS_TRANSFER_MODEL_PART_ELEMENTS_PROCESS_H_INCLUDED



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class TransferModelPartElementsProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TransferModelPartElementsProcess);

TransferModelPartElementsProcess(ModelPart &rHostModelPart,
ModelPart &rGuestModelPart) : mrHostModelPart(rHostModelPart), mrGuestModelPart(rGuestModelPart)
{
KRATOS_TRY

KRATOS_CATCH("");
}

virtual ~TransferModelPartElementsProcess() {}


void operator()()
{
Execute();
}


void Execute() override
{
KRATOS_TRY;

const int nel = mrGuestModelPart.Elements().size();

if (nel != 0)
{
ModelPart::ElementsContainerType::iterator el_begin = mrGuestModelPart.ElementsBegin();

for (int i = 0; i < nel; i++)
{
ModelPart::ElementsContainerType::iterator el = el_begin + i;

mrHostModelPart.Elements().push_back(*(el.base()));
}
}

KRATOS_CATCH("");
}




std::string Info() const override
{
return "TransferModelPartElementsProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "TransferModelPartElementsProcess";
}


protected:

TransferModelPartElementsProcess(TransferModelPartElementsProcess const &rOther);


private:

ModelPart &mrHostModelPart;
ModelPart &mrGuestModelPart;


TransferModelPartElementsProcess &operator=(TransferModelPartElementsProcess const &rOther);


}; 




inline std::istream &operator>>(std::istream &rIStream,
TransferModelPartElementsProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const TransferModelPartElementsProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
