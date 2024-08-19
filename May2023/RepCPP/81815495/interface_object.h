
#pragma once



#include "includes/define.h"
#include "includes/node.h"
#include "geometries/geometry.h"

namespace Kratos
{



class InterfaceObject : public Point
{
public:

KRATOS_CLASS_POINTER_DEFINITION(InterfaceObject);

typedef Point BaseType;

typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;

typedef Node NodeType;
typedef NodeType* NodePointerType;

typedef Geometry<NodeType> GeometryType;
typedef GeometryType* GeometryPointerType;


enum class ConstructionType
{
Node_Coords,
Geometry_Center,
Element_Center,
Condition_Center
};


explicit InterfaceObject(const CoordinatesArrayType& rCoordinates)
: Point(rCoordinates) { }

virtual ~InterfaceObject() = default;




virtual NodePointerType pGetBaseNode() const
{
KRATOS_ERROR << "Base class function called!" << std::endl;
}

virtual GeometryPointerType pGetBaseGeometry() const
{
KRATOS_ERROR << "Base class function called!" << std::endl;
}



std::string Info() const override
{
std::stringstream buffer;
buffer << "InterfaceObject" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override{rOStream << "InterfaceObject";}

void PrintData(std::ostream& rOStream) const override{}



protected:

InterfaceObject() : Point(0.0, 0.0, 0.0)
{
}


private:

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_ERROR << "This object is not supposed to be used with serialization!" << std::endl;
}
void load(Serializer& rSerializer) override
{
KRATOS_ERROR << "This object is not supposed to be used with serialization!" << std::endl;
}


}; 




class InterfaceNode : public InterfaceObject
{
public:
typedef InterfaceObject::NodePointerType NodePointerType;

InterfaceNode() {}

explicit InterfaceNode(NodePointerType pNode)
: mpNode(pNode)
{
noalias(Coordinates()) = mpNode->Coordinates();
}

NodePointerType pGetBaseNode() const override
{
return mpNode;
}

private:
NodePointerType mpNode;

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_ERROR << "This object is not supposed to be used with serialization!" << std::endl;
}
void load(Serializer& rSerializer) override
{
KRATOS_ERROR << "This object is not supposed to be used with serialization!" << std::endl;
}
};

class InterfaceGeometryObject : public InterfaceObject
{
public:
typedef InterfaceObject::GeometryPointerType GeometryPointerType;

InterfaceGeometryObject() {}

explicit InterfaceGeometryObject(GeometryPointerType pGeometry)
: mpGeometry(pGeometry)
{
noalias(Coordinates()) = mpGeometry->Center();
}

GeometryPointerType pGetBaseGeometry() const override
{
return mpGeometry;
}

private:
GeometryPointerType mpGeometry;

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_ERROR << "This object is not supposed to be used with serialization!" << std::endl;
}
void load(Serializer& rSerializer) override
{
KRATOS_ERROR << "This object is not supposed to be used with serialization!" << std::endl;
}
};

}  


