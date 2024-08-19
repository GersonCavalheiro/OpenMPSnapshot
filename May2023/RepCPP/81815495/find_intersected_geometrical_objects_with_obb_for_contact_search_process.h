
#pragma once



#include "processes/find_intersected_geometrical_objects_with_obb_process.h"

namespace Kratos
{






class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) FindIntersectedGeometricalObjectsWithOBBContactSearchProcess
: public FindIntersectedGeometricalObjectsWithOBBProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(FindIntersectedGeometricalObjectsWithOBBContactSearchProcess);

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef Point PointType;

typedef FindIntersectedGeometricalObjectsProcess BaseProcessType;

typedef FindIntersectedGeometricalObjectsWithOBBProcess BaseType;

typedef typename BaseType::OctreeType OctreeType;

using NodeType = Node;

using GeometryType = Geometry<NodeType>;

typedef PointerVectorSet<Condition, IndexedObject> EntityContainerType;



FindIntersectedGeometricalObjectsWithOBBContactSearchProcess() = delete;


FindIntersectedGeometricalObjectsWithOBBContactSearchProcess(
ModelPart& rPart1,
ModelPart& rPart2,
const double BoundingBoxFactor = -1.0,
const Flags Options = BaseProcessType::INTERSECTING_CONDITIONS|
BaseProcessType::INTERSECTING_ELEMENTS|
BaseProcessType::INTERSECTED_CONDITIONS|
BaseProcessType::INTERSECTED_ELEMENTS|
BaseType::DEBUG_OBB.AsFalse()|
BaseType::SEPARATING_AXIS_THEOREM|
BaseType::BUILD_OBB_FROM_BB
);


FindIntersectedGeometricalObjectsWithOBBContactSearchProcess(
Model& rModel,
Parameters ThisParameters
);

FindIntersectedGeometricalObjectsWithOBBContactSearchProcess(FindIntersectedGeometricalObjectsWithOBBContactSearchProcess const& rOther) = delete;

~FindIntersectedGeometricalObjectsWithOBBContactSearchProcess() override {}


const Parameters GetDefaultParameters() const override;




std::string Info() const override {
return "FindIntersectedGeometricalObjectsWithOBBContactSearchProcess";
}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override  {
BaseType::PrintData(rOStream);
}


protected:





void SetOctreeBoundingBox() override;


void MarkIfIntersected(
GeometricalObject& rIntersectedGeometricalObject,
OtreeCellVectorType& rLeaves
) override;





private:


double mLowerBBCoefficient  = 0.0;
double mHigherBBCoefficient = 1.0;



FindIntersectedGeometricalObjectsWithOBBContactSearchProcess& operator=(FindIntersectedGeometricalObjectsWithOBBContactSearchProcess const& rOther);



}; 






inline std::istream& operator >> (std::istream& rIStream,
FindIntersectedGeometricalObjectsWithOBBContactSearchProcess& rThis);



}  
