
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputeDynamicFactorProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ComputeDynamicFactorProcess);

typedef Node                                          NodeType;

typedef Geometry<NodeType>                           GeometryType;

typedef ModelPart::NodesContainerType              NodesArrayType;

typedef ModelPart::ConditionsContainerType    ConditionsArrayType;


ComputeDynamicFactorProcess( ModelPart& rThisModelPart)
:mrThisModelPart(rThisModelPart)
{
KRATOS_TRY;

KRATOS_CATCH("");
}

~ComputeDynamicFactorProcess() override
= default;






void operator()()
{
Execute();
}



void Execute() override;






std::string Info() const override
{
return "ComputeDynamicFactorProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ComputeDynamicFactorProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:















private:



ModelPart& mrThisModelPart;  




static inline double ComputeLogisticFactor(
const double MaxGapThreshold,
const double CurrentGap
);






ComputeDynamicFactorProcess& operator=(ComputeDynamicFactorProcess const& rOther) = delete;




}; 





inline std::istream& operator >> (std::istream& rIStream,
ComputeDynamicFactorProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ComputeDynamicFactorProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}
