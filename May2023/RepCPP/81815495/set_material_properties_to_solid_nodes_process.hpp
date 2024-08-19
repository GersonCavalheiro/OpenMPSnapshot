
#if !defined(KRATOS_SET_MATERIAL_PROPERTIES_TO_SOLID_NODES_PROCESS_H_INCLUDED)
#define KRATOS_SET_MATERIAL_PROPERTIES_TO_SOLID_NODES_PROCESS_H_INCLUDED




#include "spatial_containers/spatial_containers.h"

#include "custom_processes/set_material_properties_to_solid_nodes_process.hpp"
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





class SetMaterialPropertiesToSolidNodesProcess
: public MesherProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SetMaterialPropertiesToSolidNodesProcess);


SetMaterialPropertiesToSolidNodesProcess(ModelPart &rModelPart)
: mrModelPart(rModelPart)
{
}

virtual ~SetMaterialPropertiesToSolidNodesProcess()
{
}

void operator()()
{
Execute();
}


void Execute() override
{
KRATOS_TRY

double density = 0;
double young_modulus = 0;
double poisson_ratio = 0;

#pragma omp parallel
{

ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(mrModelPart.Elements(), ElemBegin, ElemEnd);
for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
ModelPart::PropertiesType &elemProperties = itElem->GetProperties();

density = elemProperties[DENSITY];
young_modulus = elemProperties[YOUNG_MODULUS];
poisson_ratio = elemProperties[POISSON_RATIO];

Geometry<Node> &rGeom = itElem->GetGeometry();
const SizeType NumNodes = rGeom.PointsNumber();
for (SizeType i = 0; i < NumNodes; ++i)
{
rGeom[i].FastGetSolutionStepValue(YOUNG_MODULUS) = young_modulus;
if (rGeom[i].SolutionStepsDataHas(SOLID_DENSITY))
{
rGeom[i].FastGetSolutionStepValue(SOLID_DENSITY) = density;
}
rGeom[i].FastGetSolutionStepValue(DENSITY) = density;
rGeom[i].FastGetSolutionStepValue(POISSON_RATIO) = poisson_ratio;
}
}
}

KRATOS_CATCH(" ")
};





std::string Info() const override
{
return "SetMaterialPropertiesToSolidNodesProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "SetMaterialPropertiesToSolidNodesProcess";
}

protected:


ModelPart &mrModelPart;




private:







SetMaterialPropertiesToSolidNodesProcess &operator=(SetMaterialPropertiesToSolidNodesProcess const &rOther);



}; 




inline std::istream &operator>>(std::istream &rIStream,
SetMaterialPropertiesToSolidNodesProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const SetMaterialPropertiesToSolidNodesProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
