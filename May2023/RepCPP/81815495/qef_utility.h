
#pragma once



#include "includes/node.h"
#include "geometries/geometry.h"
#include "geometries/bounding_box.h"

namespace Kratos
{ 





class GeometricalObject;


class KRATOS_API(KRATOS_CORE) QuadraticErrorFunction
{
public:


using NodeType = Node;
using NodePtrType = Node::Pointer;
using GeometryType = Geometry<NodeType>;
using GeometryPtrType = GeometryType::Pointer;
using GeometryArrayType = GeometryType::GeometriesArrayType;
using PointsArrayType = GeometryType::PointsArrayType;

KRATOS_CLASS_POINTER_DEFINITION( QuadraticErrorFunction );




QuadraticErrorFunction(){}

virtual ~QuadraticErrorFunction(){}



static array_1d<double,3> QuadraticErrorFunctionPoint (
const GeometryType& rVoxel,  
const GeometryArrayType& rTriangles     
);


static array_1d<double,3> QuadraticErrorFunctionPoint (
const BoundingBox<Point>& rBox,  
const std::vector<GeometricalObject*>& rTriangles     
);


static array_1d<double,3> CalculateNormal(const GeometryType& rTriangle);

private:




static Point FirstEnd(int i, const BoundingBox<Point>& rBox);
static Point SecondEnd(int i, const BoundingBox<Point>& rBox);

}; 



}  