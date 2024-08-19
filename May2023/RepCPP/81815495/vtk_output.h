
#pragma once

#include <unordered_map>


#include "includes/kratos_parameters.h"
#include "includes/io.h"
#include "processes/integration_values_extrapolation_to_nodes_process.h"

namespace Kratos
{

class KRATOS_API(KRATOS_CORE) VtkOutput : public IO
{
public:

using SizeType = std::size_t;

using IndexType = std::size_t;

using NodeType = Node;

using GeometryType = Geometry<NodeType>;

KRATOS_CLASS_POINTER_DEFINITION(VtkOutput);



explicit VtkOutput(
ModelPart& rModelPart,
Parameters ThisParameters = Parameters(R"({})" )
);

virtual ~VtkOutput() = default;



static Parameters GetDefaultParameters();


void PrintOutput(const std::string& rOutputFilename = "");


std::string Info() const override
{
return " VtkOutput object ";
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << " VtkOutput object " << std::endl;
}

void PrintData(std::ostream& rOStream) const override
{
}

enum class FileFormat {
VTK_ASCII,
VTK_BINARY
};

protected:

ModelPart& mrModelPart;                        
VtkOutput::FileFormat mFileFormat;             

Parameters mOutputSettings;                    
unsigned int mDefaultPrecision;                
std::unordered_map<int, int> mKratosIdToVtkId; 
bool mShouldSwap = false;                      

IntegrationValuesExtrapolationToNodesProcess::UniquePointer mpGaussToNodesProcess;



void PrepareGaussPointResults();


void WriteModelPartToFile(const ModelPart& rModelPart, const bool IsSubModelPart, const std::string& rOutputFilename);


std::string GetOutputFileName(const ModelPart& rModelPart, const bool IsSubModelPart, const std::string& rOutputFilename);


void Initialize(const ModelPart& rModelPart);


void CreateMapFromKratosIdToVTKId(const ModelPart& rModelPart);


void WriteHeaderToFile(const ModelPart& rModelPart, std::ofstream& rFileStream) const;


void WriteMeshToFile(const ModelPart& rModelPart, std::ofstream& rFileStream) const;


void WriteNodesToFile(const ModelPart& rModelPart, std::ofstream& rFileStream) const;


void WriteConditionsAndElementsToFile(const ModelPart& rModelPart, std::ofstream& rFileStream) const;


template<typename TContainerType>
std::size_t DetermineVtkCellListSize(const TContainerType& rContainer) const;


template <typename TContainerType>
void WriteConnectivity(const TContainerType& rContainer, std::ofstream& rFileStream) const;


template <typename TContainerType>
void WriteCellType(const TContainerType& rContainer, std::ofstream& rFileStream) const;


bool IsCompatibleVariable(const std::string& rVariableName) const;


void WriteNodalResultsToFile(const ModelPart& rModelPart, std::ofstream& rFileStream);


void WriteElementResultsToFile(const ModelPart& rModelPart, std::ofstream& rFileStream);


void WriteConditionResultsToFile(const ModelPart& rModelPart, std::ofstream& rFileStream);


void WriteNodalContainerResults(
const std::string& rVariableName,
const ModelPart::NodesContainerType& rNodes,
const bool IsHistoricalValue,
std::ofstream& rFileStream) const;


template<typename TContainerType>
void WriteGeometricalContainerResults(const std::string& rVariableName,
const TContainerType& rContainer,
std::ofstream& rFileStream) const;


template<typename TContainerType>
void WriteGeometricalContainerIntegrationResults(const std::string& rVariableName,
const TContainerType& rContainer,
std::ofstream& rFileStream) const;


template<class TVarType>
void WriteNodalScalarValues(
const ModelPart::NodesContainerType& rNodes,
const TVarType& rVariable,
const bool IsHistoricalValue,
std::ofstream& rFileStream) const;


template<class TVarType>
void WriteNodalVectorValues(
const ModelPart::NodesContainerType& rNodes,
const TVarType& rVariable,
const bool IsHistoricalValue,
std::ofstream& rFileStream) const;


template<typename TContainerType, class TVarType>
void WriteScalarSolutionStepVariable(
const TContainerType& rContainer,
const TVarType& rVariable,
std::ofstream& rFileStream) const;


template<typename TContainerType, class TVarType>
void WriteVectorSolutionStepVariable(
const TContainerType& rContainer,
const TVarType& rVariable,
std::ofstream& rFileStream) const;


template<typename TContainerType>
void WriteFlagContainerVariable(
const TContainerType& rContainer,
const Flags Flag,
const std::string& rFlagName,
std::ofstream& rFileStream) const;


template<typename TContainerType, class TVarType>
void WriteScalarContainerVariable(
const TContainerType& rContainer,
const TVarType& rVariable,
std::ofstream& rFileStream) const;


template<typename TContainerType, class TVarType>
void WriteIntegrationScalarContainerVariable(
const TContainerType& rContainer,
const Variable<TVarType>& rVariable,
std::ofstream& rFileStream) const;


template<typename TContainerType, class TVarType>
void WriteVectorContainerVariable(
const TContainerType& rContainer,
const TVarType& rVariable,
std::ofstream& rFileStream) const;


template<typename TContainerType, class TVarType>
void WriteIntegrationVectorContainerVariable(
const TContainerType& rContainer,
const Variable<TVarType>& rVariable,
std::ofstream& rFileStream) const;


template <typename TData>
void WriteScalarDataToFile(const TData& rData, std::ofstream& rFileStream) const
{
if (mFileFormat == VtkOutput::FileFormat::VTK_ASCII) {
rFileStream << rData;
} else if (mFileFormat == VtkOutput::FileFormat::VTK_BINARY) {
TData data = rData;
ForceBigEndian(reinterpret_cast<unsigned char *>(&data));
rFileStream.write(reinterpret_cast<char *>(&data), sizeof(TData));
}
}


template <typename TData>
void WriteVectorDataToFile(const TData& rData, std::ofstream& rFileStream) const
{
if (mFileFormat == VtkOutput::FileFormat::VTK_ASCII) {
for (const auto& r_data_comp : rData) {
rFileStream << float(r_data_comp) << " ";
}
} else if (mFileFormat == VtkOutput::FileFormat::VTK_BINARY) {
for (const auto& r_data_comp : rData ) {
float data_comp_local = (float)r_data_comp; 
ForceBigEndian(reinterpret_cast<unsigned char *>(&data_comp_local));
rFileStream.write(reinterpret_cast<char *>(&data_comp_local), sizeof(float));
}
}
}


void ForceBigEndian(unsigned char* pBytes) const;


private:


GeometryType::Pointer ReorderConnectivity(GeometryType::Pointer& pGeometry) const;


template<typename TContainerType>
void WritePropertiesIdsToFile(
const TContainerType& rContainer,
std::ofstream& rFileStream) const;


template<typename TContainerType>
void WriteIdsToFile(
const TContainerType& rContainer,
const std::string& DataName,
std::ofstream& rFileStream) const;



void WriteModelPartWithoutNodesToFile(ModelPart& rModelPart, const std::string& rOutputFilename);

};

} 
