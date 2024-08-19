
#pragma once

#include <unordered_set>
#include <unordered_map>


#include "processes/process.h"
#include "includes/key_hash.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "custom_utilities/mmg/mmg_utilities.h"
#include "containers/variables_list.h"
#include "meshing_application.h"



namespace Kratos
{


typedef std::size_t                  IndexType;

typedef std::size_t                   SizeType;

typedef std::vector<IndexType> IndexVectorType;





template<MMGLibrary TMMGLibrary>
class KRATOS_API(MESHING_APPLICATION) MmgProcess
: public Process
{
public:


KRATOS_CLASS_POINTER_DEFINITION(MmgProcess);

typedef Node                                                   NodeType;
typedef Geometry<NodeType>                                     GeometryType;

static constexpr SizeType Dimension = (TMMGLibrary == MMGLibrary::MMG2D) ? 2 : 3;

typedef typename std::conditional<Dimension == 2, array_1d<double, 3>, array_1d<double, 6>>::type TensorArrayType;

typedef std::unordered_map<IndexType,IndexType> ColorsMapType;

typedef std::pair<IndexType,IndexType> IndexPairType;






MmgProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

~MmgProcess() override = default;






void operator()();



void Execute() override;


void ExecuteInitialize() override;


void ExecuteBeforeSolutionLoop() override;


void ExecuteInitializeSolutionStep() override;


void ExecuteFinalizeSolutionStep() override;


void ExecuteBeforeOutputStep() override;


void ExecuteAfterOutputStep() override;


void ExecuteFinalize() override;


virtual void OutputMdpa();


void CleanSuperfluousNodes();


void CleanSuperfluousConditions();


std::string GetMmgVersion();


const Parameters GetDefaultParameters() const override;





std::string Info() const override
{
return "MmgProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MmgProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

protected:



ModelPart& mrThisModelPart;                                      
Parameters mThisParameters;                                      
NodeType::DofsContainerType mDofs;                               

std::string mFilename;                                           
IndexType mEchoLevel;                                            

FrameworkEulerLagrange mFramework;                               

DiscretizationOption mDiscretization;                            
bool mRemoveRegions;                                             

std::unordered_map<IndexType,std::vector<std::string>> mColors;  

std::unordered_map<IndexType,Element::Pointer>   mpRefElement;   
std::unordered_map<IndexType,Condition::Pointer> mpRefCondition; 




MmgProcess(
ModelPart* pThisModelPart
);


static inline FrameworkEulerLagrange ConvertFramework(const std::string& rString)
{
if(rString == "Lagrangian" || rString == "LAGRANGIAN")
return FrameworkEulerLagrange::LAGRANGIAN;
else if(rString == "Eulerian" || rString == "EULERIAN")
return FrameworkEulerLagrange::EULERIAN;
else if(rString == "ALE")
return FrameworkEulerLagrange::ALE;
else
return FrameworkEulerLagrange::EULERIAN;
}


static inline DiscretizationOption ConvertDiscretization(const std::string& rString)
{
if(rString == "Lagrangian" || rString == "LAGRANGIAN")
return DiscretizationOption::LAGRANGIAN;
else if(rString == "Standard" || rString == "STANDARD")
return DiscretizationOption::STANDARD;
else if(rString == "Isosurface" || rString == "ISOSURFACE" || rString == "IsoSurface")
return DiscretizationOption::ISOSURFACE;
else
return DiscretizationOption::STANDARD;
}


virtual void InitializeMeshData();


virtual void InitializeSolDataMetric();


virtual void InitializeSolDataDistance();


virtual void InitializeDisplacementData();


virtual void ExecuteRemeshing();


virtual void InitializeElementsAndConditions();


virtual void SaveSolutionToFile(const bool PostOutput);


virtual void FreeMemory();


template<class TContainerType>
void SetToZeroEntityData(
TContainerType& rNewContainer,
const TContainerType& rOldContainer
)
{
std::unordered_set<std::string> list_variables;
const auto it_begin_old = rOldContainer.begin();
auto& data = it_begin_old->GetData();
for(auto i = data.begin() ; i != data.end() ; ++i) {
list_variables.insert((i->first)->Name());
}

for (auto& var_name : list_variables) {
if (KratosComponents<Variable<bool>>::Has(var_name)) {
const Variable<bool>& r_var = KratosComponents<Variable<bool>>::Get(var_name);
VariableUtils().SetNonHistoricalVariable(r_var, false, rNewContainer);
} else if (KratosComponents<Variable<double>>::Has(var_name)) {
const Variable<double>& r_var = KratosComponents<Variable<double>>::Get(var_name);
VariableUtils().SetNonHistoricalVariable(r_var, 0.0, rNewContainer);
} else if (KratosComponents<Variable<array_1d<double, 3>>>::Has(var_name)) {
const Variable<array_1d<double, 3>>& r_var = KratosComponents<Variable<array_1d<double, 3>>>::Get(var_name);
const array_1d<double, 3> aux_value = ZeroVector(3);
VariableUtils().SetNonHistoricalVariable(r_var, aux_value, rNewContainer);
} else if (KratosComponents<Variable<array_1d<double, 4>>>::Has(var_name)) {
const Variable<array_1d<double, 4>>& r_var = KratosComponents<Variable<array_1d<double, 4>>>::Get(var_name);
const array_1d<double, 4> aux_value = ZeroVector(4);
VariableUtils().SetNonHistoricalVariable(r_var, aux_value, rNewContainer);
} else if (KratosComponents<Variable<array_1d<double, 6>>>::Has(var_name)) {
const Variable<array_1d<double, 6>>& r_var = KratosComponents<Variable<array_1d<double, 6>>>::Get(var_name);
const array_1d<double, 6> aux_value = ZeroVector(6);
VariableUtils().SetNonHistoricalVariable(r_var, aux_value, rNewContainer);
} else if (KratosComponents<Variable<array_1d<double, 9>>>::Has(var_name)) {
const Variable<array_1d<double, 9>>& r_var = KratosComponents<Variable<array_1d<double, 9>>>::Get(var_name);
const array_1d<double, 9> aux_value = ZeroVector(9);
VariableUtils().SetNonHistoricalVariable(r_var, aux_value, rNewContainer);
} else if (KratosComponents<Variable<Vector>>::Has(var_name)) {
const Variable<Vector>& r_var = KratosComponents<Variable<Vector>>::Get(var_name);
Vector aux_value = ZeroVector(it_begin_old->GetValue(r_var).size());
VariableUtils().SetNonHistoricalVariable(r_var, aux_value, rNewContainer);
} else if (KratosComponents<Variable<Matrix>>::Has(var_name)) {
const Variable<Matrix>& r_var = KratosComponents<Variable<Matrix>>::Get(var_name);
const Matrix& ref_matrix = it_begin_old->GetValue(r_var);
Matrix aux_value = ZeroMatrix(ref_matrix.size1(), ref_matrix.size2());
VariableUtils().SetNonHistoricalVariable(r_var, aux_value, rNewContainer);
}
}
}


virtual void ClearConditionsDuplicatedGeometries();


virtual void CreateDebugPrePostRemeshOutput(ModelPart& rOldModelPart);


void ApplyLocalParameters();





private:



MmgUtilities<TMMGLibrary> mMmgUtilities;                         




void CollapsePrismsToTriangles();


void ExtrudeTrianglestoPrisms(ModelPart& rOldModelPart);


void MarkConditionsSubmodelParts(ModelPart& rModelPart);






MmgProcess& operator=(MmgProcess const& rOther);

MmgProcess(MmgProcess const& rOther);


};




template<MMGLibrary TMMGLibrary>
inline std::istream& operator >> (std::istream& rIStream,
MmgProcess<TMMGLibrary>& rThis);

template<MMGLibrary TMMGLibrary>
inline std::ostream& operator << (std::ostream& rOStream,
const MmgProcess<TMMGLibrary>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}
