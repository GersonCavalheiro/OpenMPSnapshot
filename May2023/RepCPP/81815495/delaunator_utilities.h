
#pragma once




namespace Kratos
{




class ModelPart;
class Point;


namespace DelaunatorUtilities
{

void KRATOS_API(KRATOS_CORE) CreateTriangleMeshFromNodes(ModelPart& rModelPart);


std::vector<std::size_t> KRATOS_API(KRATOS_CORE) ComputeTrianglesConnectivity(const std::vector<double>& rCoordinates);


std::vector<std::size_t> KRATOS_API(KRATOS_CORE) ComputeTrianglesConnectivity(const std::vector<Point>& rPoints);


std::pair<std::vector<std::size_t>, std::vector<double>> KRATOS_API(KRATOS_CORE) ComputeTrianglesConnectivity(
const std::vector<double>& rCoordinates,
const std::vector<std::array<double,2>>& rSegments,
const double AreaConstraint = 0);

}; 
}  
