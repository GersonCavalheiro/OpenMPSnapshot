
#pragma once



#include "includes/define.h"
#include "mapper_interface_info.h"

namespace Kratos
{



class MapperLocalSystem
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MapperLocalSystem);

typedef Kratos::shared_ptr<MapperInterfaceInfo> MapperInterfaceInfoPointerType;
typedef Kratos::unique_ptr<MapperLocalSystem> MapperLocalSystemUniquePointer;

typedef typename MapperInterfaceInfo::CoordinatesArrayType CoordinatesArrayType;

typedef Matrix MatrixType;
typedef std::vector<int> EquationIdVectorType; 

typedef InterfaceObject::NodePointerType NodePointerType;
typedef InterfaceObject::GeometryPointerType GeometryPointerType;


enum class PairingStatus
{
NoInterfaceInfo,
Approximation,
InterfaceInfoFound
};


virtual ~MapperLocalSystem() = default;


void EquationIdVectors(EquationIdVectorType& rOriginIds,
EquationIdVectorType& rDestinationIds)
{
if (!mIsComputed) {
CalculateAll(mLocalMappingMatrix, mOriginIds, mDestinationIds, mPairingStatus);
mIsComputed = true;
}

rOriginIds      = mOriginIds;
rDestinationIds = mDestinationIds;
}

void CalculateLocalSystem(MatrixType& rLocalMappingMatrix,
EquationIdVectorType& rOriginIds,
EquationIdVectorType& rDestinationIds) const
{
if (mIsComputed) {
rLocalMappingMatrix = mLocalMappingMatrix;
rOriginIds      = mOriginIds;
rDestinationIds = mDestinationIds;
}
else {
CalculateAll(rLocalMappingMatrix, rOriginIds, rDestinationIds, mPairingStatus);
}
}


void ResizeToZero(MatrixType& rLocalMappingMatrix,
EquationIdVectorType& rOriginIds,
EquationIdVectorType& rDestinationIds,
MapperLocalSystem::PairingStatus& rPairingStatus) const
{
rPairingStatus = MapperLocalSystem::PairingStatus::NoInterfaceInfo;

rLocalMappingMatrix.resize(0, 0, false);
rOriginIds.resize(0);
rDestinationIds.resize(0);
}

virtual CoordinatesArrayType& Coordinates() const = 0;


void AddInterfaceInfo(MapperInterfaceInfoPointerType pInterfaceInfo) 
{
mInterfaceInfos.push_back(pInterfaceInfo);
}

bool HasInterfaceInfo() const
{
return mInterfaceInfos.size() > 0;
}

bool HasInterfaceInfoThatIsNotAnApproximation() const
{
for (const auto& r_info : mInterfaceInfos) {
if (!r_info->GetIsApproximation()) {
return true;
}
}
return false;
}

virtual bool IsDoneSearching() const
{
return HasInterfaceInfoThatIsNotAnApproximation();
}

virtual MapperLocalSystemUniquePointer Create(NodePointerType pNode) const
{
KRATOS_ERROR << "Create is not implemented for NodePointerType!" << std::endl;
}

virtual MapperLocalSystemUniquePointer Create(GeometryPointerType pGeometry) const
{
KRATOS_ERROR << "Create is not implemented for GeometryPointerType!" << std::endl;
}


virtual void Clear()
{
mInterfaceInfos.clear();
mLocalMappingMatrix.clear();
mOriginIds.clear();
mDestinationIds.clear();
mIsComputed = false;
}

PairingStatus GetPairingStatus() const
{
return mPairingStatus;
}

virtual void SetPairingStatusForPrinting()
{
KRATOS_ERROR << "SetPairingStatusForPrinting is not implemented!" << std::endl;
}


virtual void PairingInfo(std::ostream& rOStream, const int EchoLevel) const = 0;

virtual std::string Info() const {return "MapperLocalSystem";}

virtual void PrintInfo(std::ostream& rOStream) const {}

virtual void PrintData(std::ostream& rOStream) const {}


protected:

MapperLocalSystem() = default; 


std::vector<MapperInterfaceInfoPointerType> mInterfaceInfos;

bool mIsComputed = false;

MatrixType mLocalMappingMatrix;
EquationIdVectorType mOriginIds;
EquationIdVectorType mDestinationIds;

mutable PairingStatus mPairingStatus = PairingStatus::NoInterfaceInfo;


virtual void CalculateAll(MatrixType& rLocalMappingMatrix,
EquationIdVectorType& rOriginIds,
EquationIdVectorType& rDestinationIds,
MapperLocalSystem::PairingStatus& rPairingStatus) const = 0;


}; 



}  