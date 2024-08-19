
#pragma once



#include "custom_searching/interface_communicator.h"

namespace Kratos
{



class KRATOS_API(MAPPING_APPLICATION) InterfaceCommunicatorMPI : public InterfaceCommunicator
{
public:

KRATOS_CLASS_POINTER_DEFINITION(InterfaceCommunicatorMPI);

typedef std::vector<std::vector<double>> BufferTypeDouble;
typedef std::vector<std::vector<char>> BufferTypeChar;


InterfaceCommunicatorMPI(ModelPart& rModelPartOrigin,
MapperLocalSystemPointerVector& rMapperLocalSystems,
Parameters SearchSettings);

virtual ~InterfaceCommunicatorMPI()
{
}


std::string Info() const override
{
std::stringstream buffer;
buffer << "InterfaceCommunicatorMPI" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "InterfaceCommunicatorMPI";
}

void PrintData(std::ostream& rOStream) const override {}


protected:

void InitializeSearch(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo) override;

void InitializeSearchIteration(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo) override;

void FinalizeSearchIteration(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo) override;


private:

std::vector<double> mGlobalBoundingBoxes;

int mCommRank;
int mCommSize;

std::vector<int> mSendSizes;
std::vector<int> mRecvSizes;

BufferTypeDouble mSendBufferDouble;
BufferTypeDouble mRecvBufferDouble;

BufferTypeChar mSendBufferChar;
BufferTypeChar mRecvBufferChar;


std::size_t GetBufferSizeEstimate() const
{
return mrMapperLocalSystems.size() / mCommSize;
}

void ComputeGlobalBoundingBoxes();

template< typename TDataType >
int ExchangeDataAsync(
const std::vector<std::vector<TDataType>>& rSendBuffer,
std::vector<std::vector<TDataType>>& rRecvBuffer);


}; 



}  
