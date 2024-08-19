
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"
#include "processes/simple_mortar_mapper_process.h"

namespace Kratos
{






template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) NormalGapProcess
: public Process
{
public:

typedef SimpleMortarMapperProcess<TDim, TNumNodes, Variable<array_1d<double, 3>>, TNumNodesMaster> MapperType;

typedef ModelPart::NodesContainerType                    NodesArrayType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( NormalGapProcess );




NormalGapProcess(
ModelPart& rMasterModelPart,
ModelPart& rSlaveModelPart,
const bool SearchOrientation = true
) : mrMasterModelPart(rMasterModelPart),
mrSlaveModelPart(rSlaveModelPart),
mSearchOrientation(SearchOrientation)
{
}

virtual ~NormalGapProcess()= default;;



void operator()()
{
Execute();
}



void Execute() override;







std::string Info() const override
{
return "NormalGapProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:



ModelPart& mrMasterModelPart;  
ModelPart& mrSlaveModelPart;   
const bool mSearchOrientation; 




static inline void SwitchFlagNodes(NodesArrayType& rNodes)
{
block_for_each(rNodes, [&](NodeType& rNode) {
rNode.Flip(SLAVE);
rNode.Flip(MASTER);
});
}


void ComputeNormalGap(NodesArrayType& rNodes);





private:








}; 








template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster>
inline std::istream& operator >> (std::istream& rIStream,
NormalGapProcess<TDim, TNumNodes, TNumNodesMaster>& rThis);




template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster>
inline std::ostream& operator << (std::ostream& rOStream,
const NormalGapProcess<TDim, TNumNodes, TNumNodesMaster>& rThis)
{
return rOStream;
}


}  
