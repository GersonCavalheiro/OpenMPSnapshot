
#pragma once





#include "containers/model.h"
#include "processes/process.h"
#include "includes/kratos_parameters.h"
#include "utilities/binbased_fast_point_locator.h"

namespace Kratos
{






class ModelPart;


template<std::size_t TDim>
class KRATOS_API(SHALLOW_WATER_APPLICATION) WriteFromSwAtInterfaceProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(WriteFromSwAtInterfaceProcess);

using NodeType = Node;

using GeometryType = Geometry<NodeType>;



WriteFromSwAtInterfaceProcess() = delete;


WriteFromSwAtInterfaceProcess(Model& rModel, Parameters ThisParameters = Parameters());


~WriteFromSwAtInterfaceProcess() override = default;




struct locator_tls {
Vector N;
typename BinBasedFastPointLocator<TDim>::ResultContainerType results;
locator_tls(const int max_results = 10000) {
N.resize(TDim+1);
results.resize(max_results);
}
};

void Execute() override;

int Check() override;

const Parameters GetDefaultParameters() const override;






std::string Info() const override {
std::stringstream buffer;
buffer << "WriteFromSwAtInterfaceProcess";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override {}




private:

ModelPart& mrVolumeModelPart;
ModelPart& mrInterfaceModelPart;
array_1d<double,3> mDirection;
bool mStoreHistorical;

bool mPrintVelocityProfile;

bool mExtrapolateBoundaries;
NodeType::Pointer mpFirstBoundaryNode;
NodeType::Pointer mpSecondBoundaryNode;
NodeType::Pointer mpFirstBoundaryNeighbor;
NodeType::Pointer mpSecondBoundaryNeighbor;





void ReadAndSetValues(
NodeType& rNode,
BinBasedFastPointLocator<TDim>& rLocator,
typename BinBasedFastPointLocator<TDim>::ResultContainerType& rResults);


template<class TDataType, class TVarType = Variable<TDataType>>
void SetValue(NodeType& rNode, const TVarType& rVariable, TDataType rValue)
{
if (mStoreHistorical)
rNode.FastGetSolutionStepValue(rVariable) = rValue;
else
rNode.GetValue(rVariable) = rValue;
}

template<class TDataType, class TVarType = Variable<TDataType>>
TDataType GetValue(const NodeType& rNode, const TVarType& rVariable)
{
if (mStoreHistorical)
return rNode.FastGetSolutionStepValue(rVariable);
else
return rNode.GetValue(rVariable);
}


void CopyValues(const NodeType& rOriginNode, NodeType& rDestinationNode);






WriteFromSwAtInterfaceProcess& operator=(WriteFromSwAtInterfaceProcess const& rOther) = delete;

WriteFromSwAtInterfaceProcess(WriteFromSwAtInterfaceProcess const& rOther) = delete;


}; 







}  

