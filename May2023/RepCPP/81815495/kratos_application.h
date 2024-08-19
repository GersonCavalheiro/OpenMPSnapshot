
#pragma once

#include <string>
#include <iostream>

#include "includes/element.h"
#include "includes/condition.h"
#include "includes/kratos_components.h"
#include "includes/geometrical_object.h"
#include "includes/periodic_condition.h"
#include "includes/master_slave_constraint.h"
#include "input_output/logger.h"
#include "utilities/quaternion.h"
#include "constraints/linear_master_slave_constraint.h"

#include "geometries/register_kratos_components_for_geometry.h"
#include "geometries/line_2d_2.h"
#include "geometries/line_2d_3.h"
#include "geometries/line_3d_2.h"
#include "geometries/line_3d_3.h"
#include "geometries/point.h"
#include "geometries/point_2d.h"
#include "geometries/point_3d.h"
#include "geometries/sphere_3d_1.h"
#include "geometries/triangle_2d_3.h"
#include "geometries/triangle_2d_6.h"
#include "geometries/triangle_3d_3.h"
#include "geometries/triangle_3d_6.h"
#include "geometries/quadrilateral_2d_4.h"
#include "geometries/quadrilateral_2d_8.h"
#include "geometries/quadrilateral_2d_9.h"
#include "geometries/quadrilateral_3d_4.h"
#include "geometries/quadrilateral_3d_8.h"
#include "geometries/quadrilateral_3d_9.h"
#include "geometries/tetrahedra_3d_4.h"
#include "geometries/tetrahedra_3d_10.h"
#include "geometries/prism_3d_6.h"
#include "geometries/prism_3d_15.h"
#include "geometries/pyramid_3d_5.h"
#include "geometries/pyramid_3d_13.h"
#include "geometries/hexahedra_3d_8.h"
#include "geometries/hexahedra_3d_20.h"
#include "geometries/hexahedra_3d_27.h"
#include "geometries/quadrature_point_geometry.h"

#include "elements/mesh_element.h"
#include "elements/distance_calculation_element_simplex.h"
#include "elements/edge_based_gradient_recovery_element.h"
#include "elements/levelset_convection_element_simplex.h"
#include "elements/levelset_convection_element_simplex_algebraic_stabilization.h"

#include "conditions/mesh_condition.h"

#include "modeler/modeler.h"
#include "modeler/cad_io_modeler.h"
#include "modeler/cad_tessellation_modeler.h"
#include "modeler/serial_model_part_combinator_modeler.h"
#include "modeler/combine_model_part_modeler.h"

namespace Kratos {


class KRATOS_API(KRATOS_CORE) KratosApplication {
public:

typedef Node NodeType;
typedef Geometry<NodeType> GeometryType;

KRATOS_CLASS_POINTER_DEFINITION(KratosApplication);


explicit KratosApplication(const std::string& ApplicationName);

KratosApplication() = delete;

KratosApplication(KratosApplication const& rOther)
: mpVariableData(rOther.mpVariableData),
mpIntVariables(rOther.mpIntVariables),
mpUnsignedIntVariables(rOther.mpUnsignedIntVariables),
mpDoubleVariables(rOther.mpDoubleVariables),
mpArray1DVariables(rOther.mpArray1DVariables),
mpArray1D4Variables(rOther.mpArray1D4Variables),
mpArray1D6Variables(rOther.mpArray1D6Variables),
mpArray1D9Variables(rOther.mpArray1D9Variables),
mpVectorVariables(rOther.mpVectorVariables),
mpMatrixVariables(rOther.mpMatrixVariables),
mpGeometries(rOther.mpGeometries),
mpElements(rOther.mpElements),
mpConditions(rOther.mpConditions),
mpMasterSlaveConstraints(rOther.mpMasterSlaveConstraints),
mpModelers(rOther.mpModelers) {}

virtual ~KratosApplication() {}


virtual void Register()
{
RegisterKratosCore();
}

void RegisterKratosCore();

void RegisterVariables();  
void RegisterDeprecatedVariables();           
void RegisterCFDVariables();                  
void RegisterALEVariables();                  
void RegisterMappingVariables();              
void RegisterDEMVariables();                  
void RegisterFSIVariables();                  
void RegisterMATVariables();                  
void RegisterGlobalPointerVariables();

const std::string& Name() const { return mApplicationName; }


KratosComponents<Variable<int> >::ComponentsContainerType& GetComponents(
Variable<int> const& rComponentType) {
return *mpIntVariables;
}

KratosComponents<Variable<unsigned int> >::ComponentsContainerType&
GetComponents(Variable<unsigned int> const& rComponentType) {
return *mpUnsignedIntVariables;
}

KratosComponents<Variable<double> >::ComponentsContainerType& GetComponents(
Variable<double> const& rComponentType) {
return *mpDoubleVariables;
}

KratosComponents<Variable<array_1d<double, 3> > >::ComponentsContainerType&
GetComponents(Variable<array_1d<double, 3> > const& rComponentType) {
return *mpArray1DVariables;
}

KratosComponents<Variable<array_1d<double, 4> > >::ComponentsContainerType&
GetComponents(Variable<array_1d<double, 4> > const& rComponentType) {
return *mpArray1D4Variables;
}

KratosComponents<Variable<array_1d<double, 6> > >::ComponentsContainerType&
GetComponents(Variable<array_1d<double, 6> > const& rComponentType) {
return *mpArray1D6Variables;
}

KratosComponents<Variable<array_1d<double, 9> > >::ComponentsContainerType&
GetComponents(Variable<array_1d<double, 9> > const& rComponentType) {
return *mpArray1D9Variables;
}

KratosComponents<Variable<Quaternion<double> > >::ComponentsContainerType&
GetComponents(Variable<Quaternion<double> > const& rComponentType) {
return *mpQuaternionVariables;
}

KratosComponents<Variable<Vector> >::ComponentsContainerType& GetComponents(
Variable<Vector> const& rComponentType) {
return *mpVectorVariables;
}

KratosComponents<Variable<Matrix> >::ComponentsContainerType& GetComponents(
Variable<Matrix> const& rComponentType) {
return *mpMatrixVariables;
}

KratosComponents<VariableData>::ComponentsContainerType& GetVariables() {
return *mpVariableData;
}

KratosComponents<Geometry<Node>>::ComponentsContainerType& GetGeometries() {
return *mpGeometries;
}

KratosComponents<Element>::ComponentsContainerType& GetElements() {
return *mpElements;
}

KratosComponents<Condition>::ComponentsContainerType& GetConditions() {
return *mpConditions;
}

KratosComponents<MasterSlaveConstraint>::ComponentsContainerType& GetMasterSlaveConstraints() {
return *mpMasterSlaveConstraints;
}

KratosComponents<Modeler>::ComponentsContainerType& GetModelers() {
return *mpModelers;
}

void SetComponents(
KratosComponents<VariableData>::ComponentsContainerType const&
VariableDataComponents)

{
for (auto it = mpVariableData->begin(); it != mpVariableData->end(); it++) {
std::string const& r_variable_name = it->second->Name();
auto it_variable = VariableDataComponents.find(r_variable_name);
KRATOS_ERROR_IF(it_variable == VariableDataComponents.end()) << "This variable is not registered in Kernel : " << *(it_variable->second) << std::endl;
}
}

void SetComponents(KratosComponents<Geometry<Node>>::ComponentsContainerType const& GeometryComponents)
{
mpGeometries->insert(GeometryComponents.begin(), GeometryComponents.end());
}

void SetComponents(KratosComponents<Element>::ComponentsContainerType const&
ElementComponents)

{
mpElements->insert(ElementComponents.begin(), ElementComponents.end());
}

void SetComponents(KratosComponents<MasterSlaveConstraint>::ComponentsContainerType const&
MasterSlaveConstraintComponents)

{
mpMasterSlaveConstraints->insert(MasterSlaveConstraintComponents.begin(), MasterSlaveConstraintComponents.end());
}

void SetComponents(KratosComponents<Modeler>::ComponentsContainerType const& ModelerComponents)
{
mpModelers->insert(ModelerComponents.begin(), ModelerComponents.end());
}

void SetComponents(
KratosComponents<Condition>::ComponentsContainerType const&
ConditionComponents)

{
mpConditions->insert(
ConditionComponents.begin(), ConditionComponents.end());
}

Serializer::RegisteredObjectsContainerType& GetRegisteredObjects() {
return *mpRegisteredObjects;
}

Serializer::RegisteredObjectsNameContainerType& GetRegisteredObjectsName() {
return *mpRegisteredObjectsName;
}








virtual std::string Info() const

{
return "KratosApplication";
}


virtual void PrintInfo(std::ostream& rOStream) const

{
rOStream << Info();
}


virtual void PrintData(std::ostream& rOStream) const

{
rOStream << "Variables:" << std::endl;

KratosComponents<VariableData>().PrintData(rOStream);

rOStream << std::endl;

rOStream << "Geometries:" << std::endl;

KratosComponents<Geometry<Node>>().PrintData(rOStream);

rOStream << "Elements:" << std::endl;

KratosComponents<Element>().PrintData(rOStream);

rOStream << std::endl;

rOStream << "Conditions:" << std::endl;

KratosComponents<Condition>().PrintData(rOStream);

rOStream << std::endl;

rOStream << "MasterSlaveConstraints:" << std::endl;

KratosComponents<MasterSlaveConstraint>().PrintData(rOStream);

rOStream << std::endl;

rOStream << "Modelers:" << std::endl;

KratosComponents<Modeler>().PrintData(rOStream);
}


protected:


std::string mApplicationName;

const Point mPointPrototype;
const Point2D<NodeType> mPoint2DPrototype = Point2D<NodeType>(GeometryType::PointsArrayType(1));
const Point3D<NodeType> mPoint3DPrototype = Point3D<NodeType>(GeometryType::PointsArrayType(1));
const Sphere3D1<NodeType> mSphere3D1Prototype = Sphere3D1<NodeType>(GeometryType::PointsArrayType(1));
const Line2D2<NodeType> mLine2D2Prototype = Line2D2<NodeType>(GeometryType::PointsArrayType(2));
const Line2D3<NodeType> mLine2D3Prototype = Line2D3<NodeType>(GeometryType::PointsArrayType(3));
const Line3D2<NodeType> mLine3D2Prototype = Line3D2<NodeType>(GeometryType::PointsArrayType(2));
const Line3D3<NodeType> mLine3D3Prototype = Line3D3<NodeType>(GeometryType::PointsArrayType(3));
const Triangle2D3<NodeType> mTriangle2D3Prototype = Triangle2D3<NodeType>(GeometryType::PointsArrayType(3));
const Triangle2D6<NodeType> mTriangle2D6Prototype = Triangle2D6<NodeType>(GeometryType::PointsArrayType(6));
const Triangle3D3<NodeType> mTriangle3D3Prototype = Triangle3D3<NodeType>(GeometryType::PointsArrayType(3));
const Triangle3D6<NodeType> mTriangle3D6Prototype = Triangle3D6<NodeType>( GeometryType::PointsArrayType(6));
const Quadrilateral2D4<NodeType> mQuadrilateral2D4Prototype = Quadrilateral2D4<NodeType>( GeometryType::PointsArrayType(4));
const Quadrilateral2D8<NodeType> mQuadrilateral2D8Prototype = Quadrilateral2D8<NodeType>( GeometryType::PointsArrayType(8));
const Quadrilateral2D9<NodeType> mQuadrilateral2D9Prototype = Quadrilateral2D9<NodeType>( GeometryType::PointsArrayType(9));
const Quadrilateral3D4<NodeType> mQuadrilateral3D4Prototype = Quadrilateral3D4<NodeType>( GeometryType::PointsArrayType(4));
const Quadrilateral3D8<NodeType> mQuadrilateral3D8Prototype = Quadrilateral3D8<NodeType>( GeometryType::PointsArrayType(8));
const Quadrilateral3D9<NodeType> mQuadrilateral3D9Prototype = Quadrilateral3D9<NodeType>( GeometryType::PointsArrayType(9));
const Tetrahedra3D4<NodeType> mTetrahedra3D4Prototype = Tetrahedra3D4<NodeType>( GeometryType::PointsArrayType(4));
const Tetrahedra3D10<NodeType> mTetrahedra3D10Prototype = Tetrahedra3D10<NodeType>( GeometryType::PointsArrayType(10));
const Prism3D6<NodeType> mPrism3D6Prototype = Prism3D6<NodeType>( GeometryType::PointsArrayType(6));
const Prism3D15<NodeType> mPrism3D15Prototype = Prism3D15<NodeType>( GeometryType::PointsArrayType(15));
const Pyramid3D5<NodeType> mPyramid3D5Prototype = Pyramid3D5<NodeType>( GeometryType::PointsArrayType(5));
const Pyramid3D13<NodeType> mPyramid3D13Prototype = Pyramid3D13<NodeType>( GeometryType::PointsArrayType(13));
const Hexahedra3D8<NodeType> mHexahedra3D8Prototype = Hexahedra3D8<NodeType>( GeometryType::PointsArrayType(8));
const Hexahedra3D20<NodeType> mHexahedra3D20Prototype = Hexahedra3D20<NodeType>( GeometryType::PointsArrayType(20));
const Hexahedra3D27<NodeType> mHexahedra3D27Prototype = Hexahedra3D27<NodeType>( GeometryType::PointsArrayType(27));
const QuadraturePointGeometry<Node,1> mQuadraturePointGeometryPoint1D = QuadraturePointGeometry<Node,1>(GeometryType::PointsArrayType(),
GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(GeometryData::IntegrationMethod::GI_GAUSS_1, IntegrationPoint<3>(), Matrix(), Matrix()));
const QuadraturePointGeometry<Node,2,1> mQuadraturePointGeometryPoint2D = QuadraturePointGeometry<Node,2,1>(GeometryType::PointsArrayType(),
GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(GeometryData::IntegrationMethod::GI_GAUSS_1, IntegrationPoint<3>(), Matrix(), Matrix()));
const QuadraturePointGeometry<Node,3,1> mQuadraturePointGeometryPoint3D = QuadraturePointGeometry<Node,3,1>(GeometryType::PointsArrayType(),
GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(GeometryData::IntegrationMethod::GI_GAUSS_1, IntegrationPoint<3>(), Matrix(), Matrix()));
const QuadraturePointGeometry<Node,2> mQuadraturePointGeometrySurface2D = QuadraturePointGeometry<Node,2>(GeometryType::PointsArrayType(),
GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(GeometryData::IntegrationMethod::GI_GAUSS_1, IntegrationPoint<3>(), Matrix(), Matrix()));
const QuadraturePointGeometry<Node,3,2> mQuadraturePointGeometrySurface3D = QuadraturePointGeometry<Node,3,2>(GeometryType::PointsArrayType(),
GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(GeometryData::IntegrationMethod::GI_GAUSS_1, IntegrationPoint<3>(), Matrix(), Matrix()));
const QuadraturePointGeometry<Node,3> mQuadraturePointGeometryVolume3D = QuadraturePointGeometry<Node,3>(GeometryType::PointsArrayType(),
GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(GeometryData::IntegrationMethod::GI_GAUSS_1, IntegrationPoint<3>(), Matrix(), Matrix()));

const MeshCondition mGenericCondition;
const MeshCondition mPointCondition2D1N;
const MeshCondition mPointCondition3D1N;
const MeshCondition mLineCondition2D2N;
const MeshCondition mLineCondition2D3N;
const MeshCondition mLineCondition3D2N;
const MeshCondition mLineCondition3D3N;
const MeshCondition mSurfaceCondition3D3N;
const MeshCondition mSurfaceCondition3D6N;
const MeshCondition mSurfaceCondition3D4N;
const MeshCondition mSurfaceCondition3D8N;
const MeshCondition mSurfaceCondition3D9N;
const MeshCondition mPrismCondition2D4N;
const MeshCondition mPrismCondition3D6N;


const MasterSlaveConstraint mMasterSlaveConstraint;
const LinearMasterSlaveConstraint mLinearMasterSlaveConstraint;

const PeriodicCondition mPeriodicCondition;
const PeriodicCondition mPeriodicConditionEdge;
const PeriodicCondition mPeriodicConditionCorner;

const MeshElement mGenericElement;

const MeshElement mElement2D1N;
const MeshElement mElement2D2N;
const MeshElement mElement2D3N;
const MeshElement mElement2D6N;
const MeshElement mElement2D4N;
const MeshElement mElement2D8N;
const MeshElement mElement2D9N;

const MeshElement mElement3D1N;
const MeshElement mElement3D2N;
const MeshElement mElement3D3N;
const MeshElement mElement3D4N;
const MeshElement mElement3D5N;
const MeshElement mElement3D6N;
const MeshElement mElement3D8N;
const MeshElement mElement3D10N;
const MeshElement mElement3D13N;
const MeshElement mElement3D15N;
const MeshElement mElement3D20N;
const MeshElement mElement3D27N;

const DistanceCalculationElementSimplex<2> mDistanceCalculationElementSimplex2D3N;
const DistanceCalculationElementSimplex<3> mDistanceCalculationElementSimplex3D4N;

const EdgeBasedGradientRecoveryElement<2> mEdgeBasedGradientRecoveryElement2D2N;
const EdgeBasedGradientRecoveryElement<3> mEdgeBasedGradientRecoveryElement3D2N;

const LevelSetConvectionElementSimplex<2,3> mLevelSetConvectionElementSimplex2D3N;
const LevelSetConvectionElementSimplex<3,4> mLevelSetConvectionElementSimplex3D4N;
const LevelSetConvectionElementSimplexAlgebraicStabilization<2,3> mLevelSetConvectionElementSimplexAlgebraicStabilization2D3N;
const LevelSetConvectionElementSimplexAlgebraicStabilization<3,4> mLevelSetConvectionElementSimplexAlgebraicStabilization3D4N;

const Modeler mModeler;
const CadIoModeler mCadIoModeler;
#if USE_TRIANGLE_NONFREE_TPL
const CadTessellationModeler mCadTessellationModeler;
#endif
const SerialModelPartCombinatorModeler mSerialModelPartCombinatorModeler;
const CombineModelPartModeler mCombineModelPartModeler;

const ConstitutiveLaw mConstitutiveLaw;

KratosComponents<VariableData>::ComponentsContainerType* mpVariableData;

KratosComponents<Variable<int> >::ComponentsContainerType* mpIntVariables;

KratosComponents<Variable<unsigned int> >::ComponentsContainerType* mpUnsignedIntVariables;

KratosComponents<Variable<double> >::ComponentsContainerType* mpDoubleVariables;

KratosComponents<Variable<array_1d<double, 3> > >::ComponentsContainerType* mpArray1DVariables;

KratosComponents<Variable<array_1d<double, 4> > >::ComponentsContainerType* mpArray1D4Variables;

KratosComponents<Variable<array_1d<double, 6> > >::ComponentsContainerType* mpArray1D6Variables;

KratosComponents<Variable<array_1d<double, 9> > >::ComponentsContainerType* mpArray1D9Variables;

KratosComponents<Variable<Quaternion<double> > >::ComponentsContainerType* mpQuaternionVariables;

KratosComponents<Variable<Vector> >::ComponentsContainerType* mpVectorVariables;

KratosComponents<Variable<Matrix> >::ComponentsContainerType* mpMatrixVariables;

KratosComponents<Geometry<Node>>::ComponentsContainerType* mpGeometries;

KratosComponents<Element>::ComponentsContainerType* mpElements;

KratosComponents<Condition>::ComponentsContainerType* mpConditions;

KratosComponents<MasterSlaveConstraint>::ComponentsContainerType* mpMasterSlaveConstraints;

KratosComponents<Modeler>::ComponentsContainerType* mpModelers;

Serializer::RegisteredObjectsContainerType* mpRegisteredObjects;

Serializer::RegisteredObjectsNameContainerType* mpRegisteredObjectsName;

















private:





















KratosApplication& operator=(KratosApplication const& rOther);


};  








inline std::istream& operator>>(std::istream& rIStream,

KratosApplication& rThis);


inline std::ostream& operator<<(std::ostream& rOStream,

const KratosApplication& rThis)

{
rThis.PrintInfo(rOStream);

rOStream << std::endl;

rThis.PrintData(rOStream);

return rOStream;
}


}  
