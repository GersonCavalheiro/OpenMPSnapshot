
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>


#include "includes/define.h"
#include "includes/io.h"
#include "containers/flags.h"

namespace Kratos
{







class KRATOS_API(KRATOS_CORE) ModelPartIO : public IO
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ModelPartIO);

typedef IO                                    BaseType;

typedef BaseType::NodeType                    NodeType;
typedef BaseType::MeshType                    MeshType;
typedef BaseType::NodesContainerType          NodesContainerType;
typedef BaseType::PropertiesContainerType     PropertiesContainerType;
typedef BaseType::ElementsContainerType       ElementsContainerType;
typedef BaseType::ConditionsContainerType     ConditionsContainerType;
typedef BaseType::ConnectivitiesContainerType ConnectivitiesContainerType;

typedef std::vector<std::ostream*>            OutputFilesContainerType;
typedef std::size_t                           SizeType;

using BaseType::WriteProperties;


ModelPartIO(
std::filesystem::path const& Filename,
const Flags Options = IO::READ | IO::IGNORE_VARIABLES_ERROR.AsFalse() | IO::SKIP_TIMER);

ModelPartIO(
Kratos::shared_ptr<std::iostream> Stream,
const Flags Options = IO::IGNORE_VARIABLES_ERROR.AsFalse() | IO::SKIP_TIMER);




~ModelPartIO() override;






bool ReadNode(NodeType& rThisNode) override;


bool ReadNodes(NodesContainerType& rThisNodes) override;


std::size_t ReadNodesNumber() override;


void WriteNodes(NodesContainerType const& rThisNodes) override;


void ReadProperties(Properties& rThisProperties) override;


void ReadProperties(PropertiesContainerType& rThisProperties) override;


void WriteProperties(PropertiesContainerType const& rThisProperties) override;


void ReadGeometry(
NodesContainerType& rThisNodes,
GeometryType::Pointer& pThisGeometry
) override;


void ReadGeometries(
NodesContainerType& rThisNodes,
GeometryContainerType& rThisGeometries
) override;


std::size_t ReadGeometriesConnectivities(ConnectivitiesContainerType& rGeometriesConnectivities) override;


void WriteGeometries(GeometryContainerType const& rThisGeometries) override;


void ReadElement(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
Element::Pointer& pThisElement
) override;


void ReadElements(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
ElementsContainerType& rThisElements
) override;


std::size_t  ReadElementsConnectivities(ConnectivitiesContainerType& rElementsConnectivities) override;


void WriteElements(ElementsContainerType const& rThisElements) override;


void ReadConditions(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
ConditionsContainerType& rThisConditions
) override;


std::size_t  ReadConditionsConnectivities(ConnectivitiesContainerType& rConditionsConnectivities) override;


void WriteConditions(ConditionsContainerType const& rThisConditions) override;


void ReadInitialValues(ModelPart& rThisModelPart) override;


void ReadMesh(MeshType & rThisMesh) override;


void WriteMesh(MeshType & rThisMesh) override;


void ReadModelPart(ModelPart & rThisModelPart) override;


void WriteModelPart(ModelPart & rThisModelPart) override;


std::size_t ReadNodalGraph(ConnectivitiesContainerType& rAuxConnectivities) override;


void DivideInputToPartitions(SizeType NumberOfPartitions,
const PartitioningInfo& rPartitioningInfo) override;


void DivideInputToPartitions(Kratos::shared_ptr<std::iostream> * pStreams,
SizeType NumberOfPartitions,
const PartitioningInfo& rPartitioningInfo) override;

void SwapStreamSource(Kratos::shared_ptr<std::iostream> newStream);

void ReadSubModelPartElementsAndConditionsIds(
std::string const& rModelPartName,
std::unordered_set<SizeType> &rElementsIds,
std::unordered_set<SizeType> &rConditionsIds) override;

std::size_t ReadNodalGraphFromEntitiesList(
ConnectivitiesContainerType& rAuxConnectivities,
std::unordered_set<SizeType> &rElementsIds,
std::unordered_set<SizeType> &rConditionsIds) override;







std::string Info() const override
{
return "ModelPartIO";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ModelPartIO";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:







virtual ModelPartIO::SizeType ReorderedNodeId(ModelPartIO::SizeType NodeId);
virtual ModelPartIO::SizeType ReorderedGeometryId(ModelPartIO::SizeType GeometryId);
virtual ModelPartIO::SizeType ReorderedElementId(ModelPartIO::SizeType ElementId);
virtual ModelPartIO::SizeType ReorderedConditionId(ModelPartIO::SizeType ConditionId);








private:



SizeType mNumberOfLines;

std::filesystem::path mBaseFilename;
Flags mOptions;

Kratos::shared_ptr<std::iostream> mpStream;





std::string& ReadBlockName(std::string& rBlockName);

void SkipBlock(std::string const& BlockName);

bool CheckEndBlock(std::string const& BlockName, std::string& rWord);

void ReadModelPartDataBlock(ModelPart& rModelPart, const bool is_submodelpart=false);

void WriteModelPartDataBlock(ModelPart& rModelPart, const bool is_submodelpart=false);

template<class TablesContainerType>
void ReadTableBlock(TablesContainerType& rTables);

void ReadTableBlock(ModelPart::TablesContainerType& rTables);

template<class TablesContainerType>
void WriteTableBlock(TablesContainerType& rTables);

void WriteTableBlock(ModelPart::TablesContainerType& rTables);

void ReadNodesBlock(NodesContainerType& rThisNodes);

void ReadNodesBlock(ModelPart& rModelPart);

std::size_t CountNodesInBlock();

void ReadPropertiesBlock(PropertiesContainerType& rThisProperties);

void ReadGeometriesBlock(ModelPart& rModelPart);

void ReadGeometriesBlock(NodesContainerType& rThisNodes, GeometryContainerType& rThisGeometries);

void ReadElementsBlock(ModelPart& rModelPart);

void ReadElementsBlock(NodesContainerType& rThisNodes, PropertiesContainerType& rThisProperties, ElementsContainerType& rThisElements);


void ReadConditionsBlock(ModelPart& rModelPart);

void ReadConditionsBlock(NodesContainerType& rThisNodes, PropertiesContainerType& rThisProperties, ConditionsContainerType& rThisConditions);


void ReadNodalDataBlock(ModelPart& rThisModelPart);

void WriteNodalDataBlock(ModelPart& rThisModelPart);

template<class TVariableType>
void ReadNodalDofVariableData(NodesContainerType& rThisNodes, const TVariableType& rVariable);


void ReadNodalFlags(NodesContainerType& rThisNodes, Flags const& rFlags);

template<class TVariableType>
void ReadNodalScalarVariableData(NodesContainerType& rThisNodes, const TVariableType& rVariable);



template<class TVariableType, class TDataType>
void ReadNodalVectorialVariableData(NodesContainerType& rThisNodes, const TVariableType& rVariable, TDataType Dummy);

void ReadElementalDataBlock(ElementsContainerType& rThisElements);
template<class TObjectsContainerType>
void WriteDataBlock(const TObjectsContainerType& rThisObjectContainer, const std::string& rObjectName);
template<class TVariableType, class TObjectsContainerType>
void WriteDataBlock(const TObjectsContainerType& rThisObjectContainer,const VariableData* rVariable, const std::string& rObjectName);

template<class TVariableType>
void ReadElementalScalarVariableData(ElementsContainerType& rThisElements, const TVariableType& rVariable);


template<class TVariableType, class TDataType>
void ReadElementalVectorialVariableData(ElementsContainerType& rThisElements, const TVariableType& rVariable, TDataType Dummy);
void ReadConditionalDataBlock(ConditionsContainerType& rThisConditions);

template<class TVariableType>
void ReadConditionalScalarVariableData(ConditionsContainerType& rThisConditions, const TVariableType& rVariable);


template<class TVariableType, class TDataType>
void ReadConditionalVectorialVariableData(ConditionsContainerType& rThisConditions, const TVariableType& rVariable, TDataType Dummy);

SizeType ReadGeometriesConnectivitiesBlock(ConnectivitiesContainerType& rThisConnectivities);

SizeType ReadElementsConnectivitiesBlock(ConnectivitiesContainerType& rThisConnectivities);

SizeType ReadConditionsConnectivitiesBlock(ConnectivitiesContainerType& rThisConnectivities);

void FillNodalConnectivitiesFromGeometryBlock(ConnectivitiesContainerType& rNodalConnectivities);

void FillNodalConnectivitiesFromElementBlock(ConnectivitiesContainerType& rNodalConnectivities);

void FillNodalConnectivitiesFromConditionBlock(ConnectivitiesContainerType& rNodalConnectivities);

void FillNodalConnectivitiesFromGeometryBlockInList(
ConnectivitiesContainerType& rNodalConnectivities,
std::unordered_set<SizeType>& rGeometriesIds);

void FillNodalConnectivitiesFromElementBlockInList(
ConnectivitiesContainerType& rNodalConnectivities,
std::unordered_set<SizeType>& rElementsIds);

void FillNodalConnectivitiesFromConditionBlockInList(
ConnectivitiesContainerType& rNodalConnectivities,
std::unordered_set<SizeType>& rConditionsIds);

void ReadCommunicatorDataBlock(Communicator& rThisCommunicator, NodesContainerType& rThisNodes);

void ReadCommunicatorLocalNodesBlock(Communicator& rThisCommunicator, NodesContainerType& rThisNodes);


void ReadCommunicatorGhostNodesBlock(Communicator& rThisCommunicator, NodesContainerType& rThisNodes);

void ReadMeshBlock(ModelPart& rModelPart);

void WriteMeshBlock(ModelPart& rModelPart);


void ReadMeshDataBlock(MeshType& rMesh);


void ReadMeshNodesBlock(ModelPart& rModelPart, MeshType& rMesh);

void ReadMeshElementsBlock(ModelPart& rModelPart, MeshType& rMesh);

void ReadMeshConditionsBlock(ModelPart& rModelPart, MeshType& rMesh);

void ReadMeshPropertiesBlock(ModelPart& rModelPart, MeshType& rMesh);

void ReadSubModelPartBlock(ModelPart& rMainModelPart, ModelPart& rParentModelPart);

void WriteSubModelPartBlock(ModelPart& rMainModelPart, const std::string& InitialTabulation);

void ReadSubModelPartDataBlock(ModelPart& rModelPart);

void ReadSubModelPartTablesBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

void ReadSubModelPartPropertiesBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

void ReadSubModelPartNodesBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

void ReadSubModelPartElementsBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

void ReadSubModelPartConditionsBlock(ModelPart& rMainModelPart, ModelPart& rSubModelPart);

void DivideInputToPartitionsImpl(
OutputFilesContainerType& rOutputFiles,
SizeType NumberOfPartitions,
const PartitioningInfo& rPartitioningInfo);

void DivideModelPartDataBlock(OutputFilesContainerType& OutputFiles);

void DivideTableBlock(OutputFilesContainerType& OutputFiles);

void DividePropertiesBlock(OutputFilesContainerType& OutputFiles);

void DivideNodesBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions);

void DivideGeometriesBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& GeometriesAllPartitions);

void DivideElementsBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ElementsAllPartitions);



void DivideConditionsBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ConditionsAllPartitions);


void DivideNodalDataBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions);

void DivideFlagVariableData(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions);

void DivideDofVariableData(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions);

template<class TValueType>
void DivideVectorialVariableData(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& EntitiesPartitions,
std::string BlockName);


void DivideElementalDataBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ElementsAllPartitions);

void DivideScalarVariableData(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& EntitiesPartitions,
std::string BlockName);


void DivideConditionalDataBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ConditionsAllPartitions);


void DivideMeshBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions,
PartitionIndicesContainerType const& ElementsAllPartitions,
PartitionIndicesContainerType const& ConditionsAllPartitions);

void DivideSubModelPartBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions,
PartitionIndicesContainerType const& ElementsAllPartitions,
PartitionIndicesContainerType const& ConditionsAllPartitions);

void DivideMeshDataBlock(OutputFilesContainerType& OutputFiles);


void DivideMeshNodesBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions);


void DivideMeshElementsBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ElementsAllPartitions);

void DivideMeshConditionsBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ConditionsAllPartitions);


void DivideSubModelPartDataBlock(OutputFilesContainerType& OutputFiles);

void DivideSubModelPartTableBlock(OutputFilesContainerType& OutputFiles);


void DivideSubModelPartNodesBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& NodesAllPartitions);


void DivideSubModelPartElementsBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ElementsAllPartitions);

void DivideSubModelPartConditionsBlock(OutputFilesContainerType& OutputFiles,
PartitionIndicesContainerType const& ConditionsAllPartitions);

void WritePartitionIndices(OutputFilesContainerType& OutputFiles, PartitionIndicesType const&  NodesPartitions, PartitionIndicesContainerType const& NodesAllPartitions);


void WriteCommunicatorData(OutputFilesContainerType& OutputFiles, SizeType NumberOfPartitions, GraphType const& DomainsColoredGraph,
PartitionIndicesType const& NodesPartitions,
PartitionIndicesType const& ElementsPartitions,
PartitionIndicesType const& ConditionsPartitions,
PartitionIndicesContainerType const& NodesAllPartitions,
PartitionIndicesContainerType const& ElementsAllPartitions,
PartitionIndicesContainerType const& ConditionsAllPartitions);

void WriteCommunicatorLocalNodes(OutputFilesContainerType& OutputFiles, SizeType NumberOfPartitions, PartitionIndicesType const& NodesPartitions, PartitionIndicesContainerType const& NodesAllPartitions);

void WriteInAllFiles(OutputFilesContainerType& OutputFiles, std::string const& ThisWord);


template<class TContainerType, class TKeyType>
typename TContainerType::iterator FindKey(TContainerType& ThisContainer , TKeyType ThisKey, std::string ComponentName);




template<class TValueType>
TValueType& ReadVectorialValue(TValueType& rValue);

template<class TValueType>
TValueType& ExtractValue(std::string rWord, TValueType & rValue);

bool& ExtractValue(std::string rWord, bool & rValue);

void ReadConstitutiveLawValue(ConstitutiveLaw::Pointer& rValue);

ModelPartIO& ReadWord(std::string& Word);

ModelPartIO& ReadBlock(std::string& Block, std::string const& BlockName);

char SkipWhiteSpaces();

bool IsWhiteSpace(char C);

char GetCharacter();

bool CheckStatement(std::string const& rStatement, std::string const& rGivenWord);

void ResetInput();

inline void CreatePartition(unsigned int NumberOfThreads,const int NumberOfRows, DenseVector<unsigned int>& partitions);


void ScanNodeBlock();






}; 










}  
