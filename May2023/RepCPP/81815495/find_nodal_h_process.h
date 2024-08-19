
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{







struct FindNodalHSettings
{
constexpr static bool SaveAsHistoricalVariable = true;
constexpr static bool SaveAsNonHistoricalVariable = false;
};


template<bool THistorical = true>
class KRATOS_API(KRATOS_CORE) FindNodalHProcess
: public Process
{
public:

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef Node NodeType;

typedef ModelPart::NodeIterator NodeIterator;

KRATOS_CLASS_POINTER_DEFINITION(FindNodalHProcess);


explicit FindNodalHProcess(ModelPart& rModelPart)
: mrModelPart(rModelPart)
{
}

~FindNodalHProcess() override = default;


void operator()()
{
Execute();
}


void Execute() override;




std::string Info() const override
{
return "FindNodalHProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "FindNodalHProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


private:


ModelPart& mrModelPart;  




double& GetHValue(NodeType& rNode);




FindNodalHProcess& operator=(FindNodalHProcess const& rOther);


}; 





template<bool THistorical>
inline std::istream& operator >> (std::istream& rIStream,
FindNodalHProcess<THistorical>& rThis);

template<bool THistorical>
inline std::ostream& operator << (std::ostream& rOStream,
const FindNodalHProcess<THistorical>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
