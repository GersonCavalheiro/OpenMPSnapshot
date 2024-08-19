
#pragma once

#include <unordered_set>


#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"


#include "geometries/prism_3d_6.h"
#include "geometries/hexahedra_3d_8.h"

namespace Kratos
{


typedef std::size_t SizeType;




template<SizeType TNumNodes>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ShellToSolidShellProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ShellToSolidShellProcess);

typedef std::size_t                                     IndexType;

typedef Node                                          NodeType;
typedef Geometry<NodeType>                           GeometryType;

typedef ModelPart::NodesContainerType              NodesArrayType;
typedef ModelPart::ConditionsContainerType    ConditionsArrayType;
typedef ModelPart::ElementsContainerType        ElementsArrayType;



ShellToSolidShellProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

~ShellToSolidShellProcess() override
= default;






void operator()()
{
Execute();
}


void Execute() override;


const Parameters GetDefaultParameters() const override;






std::string Info() const override
{
return "ShellToSolidShellProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ShellToSolidShellProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart;              
Parameters mThisParameters;              




void ReorderAllIds(const bool ReorderAccordingShellConnectivity = false);


void ExecuteExtrusion();


void ExecuteCollapse();


void ReplacePreviousGeometry(
ModelPart& rGeometryModelPart,
ModelPart& rAuxiliaryModelPart
);


void ReassignConstitutiveLaw(
ModelPart& rGeometryModelPart,
std::unordered_set<IndexType>& rSetIdProperties
);


void InitializeElements();


void ExportToMDPA();


void CleanModel();


inline void ComputeNodesMeanNormalModelPartNonHistorical();


inline void CopyVariablesList(
NodeType::Pointer pNodeNew,
NodeType::Pointer pNodeOld
);






ShellToSolidShellProcess& operator=(ShellToSolidShellProcess const& rOther) = delete;




}; 






}
