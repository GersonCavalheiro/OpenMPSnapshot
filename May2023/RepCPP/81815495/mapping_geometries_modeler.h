
#pragma once



#include "modeler/modeler.h"
#include "custom_utilities/mapping_intersection_utilities.h"

namespace Kratos
{



class KRATOS_API(MAPPING_APPLICATION) MappingGeometriesModeler
: public Modeler
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MappingGeometriesModeler);

typedef std::size_t SizeType;
typedef std::size_t IndexType;
typedef Node NodeType;
typedef Geometry<NodeType> GeometryType;
typedef typename GeometryType::Pointer GeometryPointerType;


MappingGeometriesModeler()
: Modeler()
{
}

MappingGeometriesModeler(
Model& rModel,
Parameters ModelerParameters = Parameters())
: Modeler(rModel, ModelerParameters)
{
mpModels.resize(1);
mpModels[0] = &rModel;
}

virtual ~MappingGeometriesModeler() = default;

Modeler::Pointer Create(
Model& rModel, const Parameters ModelParameters) const override
{
return Kratos::make_shared<MappingGeometriesModeler>(rModel, ModelParameters);
}

void GenerateNodes(ModelPart& ThisModelPart) override
{
mpModels.push_back(&ThisModelPart.GetModel());
}


void SetupGeometryModel() override;


std::string Info() const override
{
return "MappingGeometriesModeler";
}

void PrintInfo(std::ostream & rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream & rOStream) const override
{
}


private:
std::vector<Model*> mpModels;

void CopySubModelPart(ModelPart& rDestinationMP, ModelPart& rReferenceMP)
{
rDestinationMP.SetNodes(rReferenceMP.pNodes());
rDestinationMP.SetNodalSolutionStepVariablesList(rReferenceMP.pGetNodalSolutionStepVariablesList());
ModelPart& coupling_conditions = rReferenceMP.GetSubModelPart("coupling_conditions");
rDestinationMP.SetConditions(coupling_conditions.pConditions());
}

void CreateInterfaceLineCouplingConditions(ModelPart& rInterfaceModelPart);

void CheckParameters();

}; 


inline std::istream& operator >> (
std::istream& rIStream,
MappingGeometriesModeler& rThis);

inline std::ostream& operator << (
std::ostream& rOStream,
const MappingGeometriesModeler& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
