
#pragma once



#include "includes/kratos_parameters.h"
#include "includes/model_part.h"
#include "mappers/mapper_flags.h"
#include "spatial_containers/bins_dynamic_objects.h"
#include "utilities/builtin_timer.h"
#include "custom_searching/custom_configures/interface_object_configure.h"
#include "custom_utilities/mapper_local_system.h"


namespace Kratos
{



class KRATOS_API(MAPPING_APPLICATION) InterfaceCommunicator
{
public:

KRATOS_CLASS_POINTER_DEFINITION(InterfaceCommunicator);

typedef Kratos::unique_ptr<MapperInterfaceInfo> MapperInterfaceInfoUniquePointerType;

typedef Kratos::shared_ptr<MapperInterfaceInfo> MapperInterfaceInfoPointerType;
typedef std::vector<std::vector<MapperInterfaceInfoPointerType>> MapperInterfaceInfoPointerVectorType;

typedef Kratos::unique_ptr<MapperLocalSystem> MapperLocalSystemPointer;
typedef std::vector<MapperLocalSystemPointer> MapperLocalSystemPointerVector;

typedef Kratos::unique_ptr<BinsObjectDynamic<InterfaceObjectConfigure>> BinsUniquePointerType;

typedef InterfaceObjectConfigure::ContainerType InterfaceObjectContainerType;
typedef Kratos::unique_ptr<InterfaceObjectContainerType> InterfaceObjectContainerUniquePointerType;


InterfaceCommunicator(ModelPart& rModelPartOrigin,
MapperLocalSystemPointerVector& rMapperLocalSystems,
Parameters SearchSettings);

virtual ~InterfaceCommunicator() = default;


void ExchangeInterfaceData(const Communicator& rComm,
const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo);


int AreMeshesConforming() {
return mMeshesAreConforming;
}


virtual std::string Info() const
{
std::stringstream buffer;
buffer << "InterfaceCommunicator" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "InterfaceCommunicator";
}

virtual void PrintData(std::ostream& rOStream) const {}


protected:

ModelPart& mrModelPartOrigin;
const MapperLocalSystemPointerVector& mrMapperLocalSystems;
MapperInterfaceInfoPointerVectorType mMapperInterfaceInfosContainer; 

BinsUniquePointerType mpLocalBinStructure;

InterfaceObjectContainerUniquePointerType mpInterfaceObjectsOrigin;

Parameters mSearchSettings;
double mSearchRadius = -1.0;

int mEchoLevel = 0;
int mMeshesAreConforming = 0;


virtual void InitializeSearch(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo);

virtual void FinalizeSearch();

virtual void InitializeSearchIteration(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo);

virtual void FinalizeSearchIteration(const MapperInterfaceInfoUniquePointerType& rpInterfaceInfo);

void FilterInterfaceInfosSuccessfulSearch();

void AssignInterfaceInfos();


private:

void ConductLocalSearch();

void CreateInterfaceObjectsOrigin(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo);

void UpdateInterfaceObjectsOrigin();

void InitializeBinsSearchStructure();

void ConductSearchIteration(const MapperInterfaceInfoUniquePointerType& rpRefInterfaceInfo);

bool AllNeighborsFound(const Communicator& rComm) const;

void PrintInfoAboutCurrentSearchSuccess(
const Communicator& rComm,
const BuiltinTimer& rTimer) const;


}; 




}  
