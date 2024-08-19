
#include <unordered_set>


#include "mapping_matrix_utilities.h"
#include "mappers/mapper_define.h"
#include "custom_utilities/mapper_utilities.h"

namespace Kratos {

namespace {

typedef typename MapperDefinitions::SparseSpaceType MappingSparseSpaceType;
typedef typename MapperDefinitions::DenseSpaceType  DenseSpaceType;

typedef MappingMatrixUtilities<MappingSparseSpaceType, DenseSpaceType> MappingMatrixUtilitiesType;

typedef typename MapperLocalSystem::MatrixType MatrixType;
typedef typename MapperLocalSystem::EquationIdVectorType EquationIdVectorType;

typedef std::size_t IndexType;
typedef std::size_t SizeType;





void ConstructMatrixStructure(Kratos::unique_ptr<typename MappingSparseSpaceType::MatrixType>& rpMdo,
std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems,
const SizeType NumNodesOrigin,
const SizeType NumNodesDestination)
{
std::vector<std::unordered_set<IndexType> > indices(NumNodesDestination);

for (IndexType i=0; i<NumNodesDestination; ++i) {
indices[i].reserve(3);
}

EquationIdVectorType origin_ids;
EquationIdVectorType destination_ids;

for (auto& r_local_sys : rMapperLocalSystems) { 
r_local_sys->EquationIdVectors(origin_ids, destination_ids);
for (const auto dest_idx : destination_ids) {
indices[dest_idx].insert(origin_ids.begin(), origin_ids.end());
}
}

SizeType num_non_zero_entries = 0;
for (const auto& r_row_indices : indices) { 
num_non_zero_entries += r_row_indices.size(); 
}

auto p_Mdo = Kratos::make_unique<typename MappingSparseSpaceType::MatrixType>(
NumNodesDestination,
NumNodesOrigin,
num_non_zero_entries);

double* p_matrix_values = p_Mdo->value_data().begin();
IndexType* p_matrix_row_indices = p_Mdo->index1_data().begin();
IndexType* p_matrix_col_indices = p_Mdo->index2_data().begin();

p_matrix_row_indices[0] = 0;
for (IndexType i=0; i<NumNodesDestination; ++i) {
p_matrix_row_indices[i+1] = p_matrix_row_indices[i] + indices[i].size();
}

for (IndexType i=0; i<NumNodesDestination; ++i) {
const IndexType row_begin = p_matrix_row_indices[i];
const IndexType row_end = p_matrix_row_indices[i+1];
IndexType j = row_begin;
for (const auto index : indices[i]) {
p_matrix_col_indices[j] = index;
p_matrix_values[j] = 0.0;
++j;
}

indices[i].clear(); 

std::sort(&p_matrix_col_indices[row_begin], &p_matrix_col_indices[row_end]);
}

p_Mdo->set_filled(indices.size()+1, num_non_zero_entries);

rpMdo.swap(p_Mdo);
}

void BuildMatrix(Kratos::unique_ptr<typename MappingSparseSpaceType::MatrixType>& rpMdo,
std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems)
{
MatrixType local_mapping_matrix;
EquationIdVectorType origin_ids;
EquationIdVectorType destination_ids;

for (auto& r_local_sys : rMapperLocalSystems) { 

r_local_sys->CalculateLocalSystem(local_mapping_matrix, origin_ids, destination_ids);

KRATOS_DEBUG_ERROR_IF(local_mapping_matrix.size1() != destination_ids.size()) << "MappingMatrixAssembly: DestinationID vector size mismatch: LocalMappingMatrix-Size1: " << local_mapping_matrix.size1() << " | DestinationIDs-size: " << destination_ids.size() << std::endl;
KRATOS_DEBUG_ERROR_IF(local_mapping_matrix.size2() != origin_ids.size()) << "MappingMatrixAssembly: OriginID vector size mismatch: LocalMappingMatrix-Size2: " << local_mapping_matrix.size2() << " | OriginIDs-size: " << origin_ids.size() << std::endl;

for (IndexType i=0; i<destination_ids.size(); ++i) {
for (IndexType j=0; j<origin_ids.size(); ++j) {
(*rpMdo)(destination_ids[i], origin_ids[j]) += local_mapping_matrix(i,j);
}
}

r_local_sys->Clear();
}
}

} 

template<>
void MappingMatrixUtilitiesType::CheckRowSum(
const typename MappingSparseSpaceType::MatrixType& rM,
const std::string& rBaseFileName,
const bool ThrowError,
const double Tolerance)
{
typename MappingSparseSpaceType::VectorType unit_vector(MappingSparseSpaceType::Size2(rM));
MappingSparseSpaceType::Set(unit_vector, 1.0);

typename MappingSparseSpaceType::VectorType row_sums_vector(MappingSparseSpaceType::Size1(rM));

MappingSparseSpaceType::Mult(rM, unit_vector, row_sums_vector);

bool write_mm_file = false;
for (std::size_t i = 0; i < MappingSparseSpaceType::Size(row_sums_vector); ++i) {
if (std::abs(row_sums_vector[i] - 1.0) > Tolerance) {
KRATOS_WARNING("MappingMatrixAssembly") << "The row sum in row " << i << " is unequal 1.0: " << row_sums_vector[i] << std::endl;
write_mm_file = true;
}
}

if (write_mm_file) {
MappingSparseSpaceType::WriteMatrixMarketVector(("RowSumVector_" + rBaseFileName).c_str(), row_sums_vector);
KRATOS_ERROR_IF(ThrowError) << "Mapping matrix does not sum to unity. Please check file " << rBaseFileName << " in your project directory for row sums\n";
}
}

template<>
void MappingMatrixUtilitiesType::InitializeSystemVector(
Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType>& rpVector,
const std::size_t VectorSize)
{
if (rpVector == nullptr || rpVector->size() != VectorSize) { 
Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType> p_new_vector = Kratos::make_unique<typename MappingSparseSpaceType::VectorType>(VectorSize);
rpVector.swap(p_new_vector);

}
else {
MappingSparseSpaceType::SetToZero(*rpVector);
}
}

template<>
void MappingMatrixUtilitiesType::BuildMappingMatrix(
Kratos::unique_ptr<typename MappingSparseSpaceType::MatrixType>& rpMappingMatrix,
Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType>& rpInterfaceVectorOrigin,
Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType>& rpInterfaceVectorDestination,
const ModelPart& rModelPartOrigin,
const ModelPart& rModelPartDestination,
std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems,
const int EchoLevel)
{
KRATOS_TRY

static_assert(!MappingSparseSpaceType::IsDistributed(), "Using a distributed Space!");

const SizeType num_nodes_origin = rModelPartOrigin.NumberOfNodes();
const SizeType num_nodes_destination = rModelPartDestination.NumberOfNodes();

ConstructMatrixStructure(rpMappingMatrix, rMapperLocalSystems,
num_nodes_origin, num_nodes_destination);

BuildMatrix(rpMappingMatrix, rMapperLocalSystems);


MappingMatrixUtilitiesType::InitializeSystemVector(rpInterfaceVectorOrigin, num_nodes_origin);
MappingMatrixUtilitiesType::InitializeSystemVector(rpInterfaceVectorDestination, num_nodes_destination);

KRATOS_CATCH("")
}

template class MappingMatrixUtilities< MappingSparseSpaceType, DenseSpaceType >;

}  
