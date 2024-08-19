
#pragma once



#include "includes/model_part.h"
#include "custom_utilities/mapper_local_system.h"

namespace Kratos {

template<class TSparseSpace, class TDenseSpace>
class KRATOS_API(MAPPING_APPLICATION) MappingMatrixUtilities
{
public:
static void InitializeSystemVector(
Kratos::unique_ptr<typename TSparseSpace::VectorType>& rpVector,
const std::size_t VectorSize);

static void BuildMappingMatrix(
Kratos::unique_ptr<typename TSparseSpace::MatrixType>& rpMappingMatrix,
Kratos::unique_ptr<typename TSparseSpace::VectorType>& rpInterfaceVectorOrigin,
Kratos::unique_ptr<typename TSparseSpace::VectorType>& rpInterfaceVectorDestination,
const ModelPart& rModelPartOrigin,
const ModelPart& rModelPartDestination,
std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems,
const int EchoLevel);

static void CheckRowSum(
const typename TSparseSpace::MatrixType& rM,
const std::string& rBaseFileName,
const bool ThrowError = false,
const double Tolerance = 1e-15);

};

}  
