#pragma once 



#include "modeler/modeler.h"
#include "containers/model.h"
#include "includes/kratos_parameters.h"

namespace Kratos 
{






class KRATOS_API(KRATOS_CORE) CombineModelPartModeler 
: public Modeler {
public:

KRATOS_CLASS_POINTER_DEFINITION(CombineModelPartModeler);


CombineModelPartModeler() : Modeler()
{
}


CombineModelPartModeler(
Model& rModel,
Parameters ModelerParameters
);

virtual ~CombineModelPartModeler() = default;



const Parameters GetDefaultParameters() const override;


Modeler::Pointer Create(
Model& rModel,
const Parameters ModelParameters
) const override;


void SetupModelPart() override;


std::string Info() const override
{
return "CombineModelPartModeler";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
}

protected:






void DuplicateElements(
ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Element& rReferenceElement) const;

void DuplicateConditions(
ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Condition& rReferenceBoundaryCondition) const;

void DuplicateCommunicatorData(
ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart) const;

void DuplicateSubModelParts(
ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart) const;

void CopyCommonData(
ModelPart& rCombinedModelPart) const;

void DuplicateMesh() const;

void CreateSubModelParts();

void CreateCommunicators();

void PopulateCommunicators();

void PopulateLocalMesh(
Communicator& rReferenceComm,
Communicator& rDestinationComm,
ModelPart& rDestinationModelPart
) const;

void ResetModelPart(ModelPart& rCombinedModelPart) const;

void CheckOriginModelPartsAndAssignRoot();




private:

Model* mpModel = nullptr; 

Parameters mParameters;   

ModelPart* mpOriginRootModelPart = nullptr; 





friend class Serializer;

void save(Serializer& rSerializer) const
{
}

void load(Serializer& rSerializer)
{
}



}; 


inline std::istream& operator>>(std::istream& rIStream,
CombineModelPartModeler& rThis)
{
return rIStream;
}

inline std::ostream& operator<<(std::ostream& rOStream, const CombineModelPartModeler& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 

