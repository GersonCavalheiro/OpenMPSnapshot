
#if !defined(KRATOS_SET_MATERIAL_PROPERTIES_TO_FLUID_NODES_PROCESS_H_INCLUDED)
#define KRATOS_SET_MATERIAL_PROPERTIES_TO_FLUID_NODES_PROCESS_H_INCLUDED




#include "spatial_containers/spatial_containers.h"

#include "custom_processes/set_material_properties_to_fluid_nodes_process.hpp"
#include "custom_utilities/mesher_utilities.hpp"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "utilities/math_utils.h"
#include "custom_processes/mesher_process.hpp"


namespace Kratos
{


typedef ModelPart::NodesContainerType NodesContainerType;
typedef ModelPart::ElementsContainerType ElementsContainerType;
typedef ModelPart::MeshType::GeometryType::PointsArrayType PointsArrayType;
typedef std::size_t SizeType;





class SetMaterialPropertiesToFluidNodesProcess
: public MesherProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SetMaterialPropertiesToFluidNodesProcess);


SetMaterialPropertiesToFluidNodesProcess(ModelPart &rModelPart)
: mrModelPart(rModelPart)
{
}

virtual ~SetMaterialPropertiesToFluidNodesProcess()
{
}

void operator()()
{
Execute();
}


void Execute() override{
KRATOS_TRY

#pragma omp parallel
{

ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(mrModelPart.Elements(), ElemBegin, ElemEnd);
for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
ModelPart::PropertiesType &elemProperties = itElem->GetProperties();

double flow_index = 1;
double yield_shear = 0;
double adaptive_exponent = 0;
double static_friction = 0;
double dynamic_friction = 0;
double inertial_number_zero = 0;
double grain_diameter = 0;
double grain_density = 0;
double regularization_coefficient = 0;
double friction_angle = 0;
double cohesion = 0;

double density = elemProperties[DENSITY];
double bulk_modulus = elemProperties[BULK_MODULUS];
double viscosity = elemProperties[DYNAMIC_VISCOSITY];
unsigned int elem_property_id = elemProperties.Id();

if (elemProperties.Has(YIELD_SHEAR)) 
{
flow_index = elemProperties[FLOW_INDEX];
yield_shear = elemProperties[YIELD_SHEAR];
adaptive_exponent = elemProperties[ADAPTIVE_EXPONENT];
}
else if (elemProperties.Has(INTERNAL_FRICTION_ANGLE)) 
{
friction_angle = elemProperties[INTERNAL_FRICTION_ANGLE];
cohesion = elemProperties[COHESION];
adaptive_exponent = elemProperties[ADAPTIVE_EXPONENT];
}
else if (elemProperties.Has(STATIC_FRICTION)) 
{
static_friction = elemProperties[STATIC_FRICTION];
dynamic_friction = elemProperties[DYNAMIC_FRICTION];
inertial_number_zero = elemProperties[INERTIAL_NUMBER_ZERO];
grain_diameter = elemProperties[GRAIN_DIAMETER];
grain_density = elemProperties[GRAIN_DENSITY];
regularization_coefficient = elemProperties[REGULARIZATION_COEFFICIENT];
}

Geometry<Node> &rGeom = itElem->GetGeometry();
const SizeType NumNodes = rGeom.PointsNumber();
for (SizeType i = 0; i < NumNodes; ++i)
{

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(PROPERTY_ID))
{
rGeom[i].FastGetSolutionStepValue(PROPERTY_ID) = elem_property_id;
}

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(BULK_MODULUS))
rGeom[i].FastGetSolutionStepValue(BULK_MODULUS) = bulk_modulus;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(DENSITY))
rGeom[i].FastGetSolutionStepValue(DENSITY) = density;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(DYNAMIC_VISCOSITY))
rGeom[i].FastGetSolutionStepValue(DYNAMIC_VISCOSITY) = viscosity;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(YIELD_SHEAR))
rGeom[i].FastGetSolutionStepValue(YIELD_SHEAR) = yield_shear;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(FLOW_INDEX))
rGeom[i].FastGetSolutionStepValue(FLOW_INDEX) = flow_index;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(ADAPTIVE_EXPONENT))
rGeom[i].FastGetSolutionStepValue(ADAPTIVE_EXPONENT) = adaptive_exponent;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(INTERNAL_FRICTION_ANGLE))
rGeom[i].FastGetSolutionStepValue(INTERNAL_FRICTION_ANGLE) = friction_angle;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(COHESION))
rGeom[i].FastGetSolutionStepValue(COHESION) = cohesion;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(ADAPTIVE_EXPONENT))
rGeom[i].FastGetSolutionStepValue(ADAPTIVE_EXPONENT) = adaptive_exponent;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(STATIC_FRICTION))
rGeom[i].FastGetSolutionStepValue(STATIC_FRICTION) = static_friction;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(DYNAMIC_FRICTION))
rGeom[i].FastGetSolutionStepValue(DYNAMIC_FRICTION) = dynamic_friction;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(INERTIAL_NUMBER_ZERO))
rGeom[i].FastGetSolutionStepValue(INERTIAL_NUMBER_ZERO) = inertial_number_zero;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(GRAIN_DIAMETER))
rGeom[i].FastGetSolutionStepValue(GRAIN_DIAMETER) = grain_diameter;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(GRAIN_DENSITY))
rGeom[i].FastGetSolutionStepValue(GRAIN_DENSITY) = grain_density;

if (mrModelPart.GetNodalSolutionStepVariablesList().Has(REGULARIZATION_COEFFICIENT))
rGeom[i].FastGetSolutionStepValue(REGULARIZATION_COEFFICIENT) = regularization_coefficient;
}
}

}

KRATOS_CATCH(" ")
}; 





std::string Info() const override
{
return "SetMaterialPropertiesToFluidNodesProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "SetMaterialPropertiesToFluidNodesProcess";
}

protected:


ModelPart &mrModelPart;




private:







SetMaterialPropertiesToFluidNodesProcess &operator=(SetMaterialPropertiesToFluidNodesProcess const &rOther);


}
; 




inline std::istream &operator>>(std::istream &rIStream,
SetMaterialPropertiesToFluidNodesProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const SetMaterialPropertiesToFluidNodesProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
