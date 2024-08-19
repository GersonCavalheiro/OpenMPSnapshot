
#pragma once



#include "custom_processes/base_contact_search_process.h"

namespace Kratos
{


typedef std::size_t SizeType;





template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) SimpleContactSearchProcess
: public BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>
{
public:

typedef BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster> BaseType;

typedef typename BaseType::NodesArrayType           NodesArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;
typedef typename BaseType::NodeType                       NodeType;
typedef typename BaseType::GeometryType               GeometryType;

typedef std::size_t IndexType;

static constexpr double GapThreshold = 2.0e-4;

KRATOS_CLASS_POINTER_DEFINITION( SimpleContactSearchProcess );




SimpleContactSearchProcess(
ModelPart& rMainModelPart,
Parameters ThisParameters =  Parameters(R"({})"),
Properties::Pointer pPairedProperties = nullptr
);

virtual ~SimpleContactSearchProcess()= default;









std::string Info() const override
{
return "SimpleContactSearchProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:






void SetActiveNode(
NodeType& rNode,
const double CommonEpsilon,
const double ScaleFactor = 1.0
) override;





private:








}; 








template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
inline std::istream& operator >> (std::istream& rIStream,
SimpleContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis);




template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
inline std::ostream& operator << (std::ostream& rOStream,
const SimpleContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis)
{
return rOStream;
}


}  
