
#pragma once

#include <string>
#include <iostream>




#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"


namespace Kratos
{






class KRATOS_API(KRATOS_CORE) StructuredMeshGeneratorProcess : public Process
{

public:

using GeometryType = Geometry<Node >;

KRATOS_CLASS_POINTER_DEFINITION(StructuredMeshGeneratorProcess);


StructuredMeshGeneratorProcess() = delete;

StructuredMeshGeneratorProcess(
const GeometryType& rGeometry,
ModelPart& rOutputModelPart,
Parameters TheParameters);

StructuredMeshGeneratorProcess(
GeometryType::Pointer pGeometry,
ModelPart& rOutputModelPart,
Parameters TheParameters)
: StructuredMeshGeneratorProcess(
*pGeometry,
rOutputModelPart,
TheParameters)
{};

StructuredMeshGeneratorProcess(StructuredMeshGeneratorProcess const& rOther) = delete;

~StructuredMeshGeneratorProcess() override ;


StructuredMeshGeneratorProcess& operator=(StructuredMeshGeneratorProcess const& rOther) = delete;


void Execute() override;

int Check() override;


const Parameters GetDefaultParameters() const override;






std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;





private:



const GeometryType& mrGeometry;
std::size_t mNumberOfDivisions;
std::size_t mStartNodeId;
std::size_t mStartElementId;
std::size_t mStartConditionId;
std::size_t mElementPropertiesId;
std::size_t mConditiongPropertiesId;
std::string mElementName;
std::string mConditionName;
std::string mBodySubModelPartName;
std::string mSkinSubModelPartName;
bool mCreateSkinSubModelPart;
bool mCreateBodySubModelPart;
ModelPart& mrOutputModelPart;


void Generate2DMesh();

void Generate3DMesh();

void GenerateNodes2D(Point const& rMinPoint, Point const& rMaxPoint);

void GenerateNodes3D(Point const& rMinPoint, Point const& rMaxPoint);

void GenerateTriangularElements();

void GenerateTetrahedraElements();

void CreateCellTetrahedra(std::size_t I, std::size_t J, std::size_t K, Properties::Pointer pProperties);

std::size_t GetNodeId(std::size_t I, std::size_t J, std::size_t K);

void GetLocalCoordinatesRange(Point& rMinPoint, Point& rMaxPoint);

bool CheckDomainGeometry();

bool CheckDomainGeometryConnectivityForQuadrilateral2D4();

bool CheckDomainGeometryConnectivityForHexahedra3D8();







}; 





inline std::istream& operator >> (std::istream& rIStream,
StructuredMeshGeneratorProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const StructuredMeshGeneratorProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
