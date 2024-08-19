
#pragma once




#include <Epetra_Import.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_FEVector.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_IntSerialDenseVector.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_SerialDenseVector.h>
#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_VectorIn.h>
#include <EpetraExt_RowMatrixOut.h>
#include <EpetraExt_MultiVectorOut.h>
#include <EpetraExt_MatrixMatrix.h>

#include "includes/ublas_interface.h"
#include "spaces/ublas_space.h"
#include "includes/data_communicator.h"
#include "mpi/includes/mpi_data_communicator.h"
#include "custom_utilities/trilinos_dof_updater.h"

namespace Kratos
{







template<class TMatrixType, class TVectorType>
class TrilinosSpace
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TrilinosSpace);

using DataType = double;

using MatrixType = TMatrixType;

using VectorType = TVectorType;

using IndexType = std::size_t;

using SizeType = std::size_t;

using MatrixPointerType = typename Kratos::shared_ptr<TMatrixType>;
using VectorPointerType = typename Kratos::shared_ptr<TVectorType>;

using DofUpdaterType = TrilinosDofUpdater< TrilinosSpace<TMatrixType,TVectorType>>;
using DofUpdaterPointerType = typename DofUpdater<TrilinosSpace<TMatrixType,TVectorType>>::UniquePointer;


TrilinosSpace()
{
}

virtual ~TrilinosSpace()
{
}




static MatrixPointerType CreateEmptyMatrixPointer()
{
return MatrixPointerType(nullptr);
}


static VectorPointerType CreateEmptyVectorPointer()
{
return VectorPointerType(nullptr);
}


static MatrixPointerType CreateEmptyMatrixPointer(Epetra_MpiComm& rComm)
{
const int global_elems = 0;
Epetra_Map Map(global_elems, 0, rComm);
return MatrixPointerType(new TMatrixType(::Copy, Map, 0));
}


static VectorPointerType CreateEmptyVectorPointer(Epetra_MpiComm& rComm)
{
const int global_elems = 0;
Epetra_Map Map(global_elems, 0, rComm);
return VectorPointerType(new TVectorType(Map));
}


static IndexType Size(const VectorType& rV)
{
const int size = rV.GlobalLength();
return size;
}


static IndexType Size1(MatrixType const& rM)
{
const int size1 = rM.NumGlobalRows();
return size1;
}


static IndexType Size2(MatrixType const& rM)
{
const int size1 = rM.NumGlobalCols();
return size1;
}


static void GetColumn(
const unsigned int j,
const MatrixType& rM,
VectorType& rX
)
{
KRATOS_ERROR << "GetColumn method is not currently implemented" << std::endl;
}


static void Copy(
const MatrixType& rX,
MatrixType& rY
)
{
rY = rX;
}


static void Copy(
const VectorType& rX,
VectorType& rY
)
{
rY = rX;
}


static double Dot(
const VectorType& rX,
const VectorType& rY
)
{
double value;
const int sucess = rY.Dot(rX, &value); 
KRATOS_ERROR_IF_NOT(sucess == 0) << "Error computing dot product" <<  std::endl;
return value;
}


static double Max(const VectorType& rX)
{
double value;
const int sucess = rX.MaxValue(&value); 
KRATOS_ERROR_IF_NOT(sucess == 0) << "Error computing maximum value" <<  std::endl;
return value;
}


static double Min(const VectorType& rX)
{
double value;
const int sucess = rX.MinValue(&value); 
KRATOS_ERROR_IF_NOT(sucess == 0) << "Error computing minimum value" <<  std::endl;
return value;
}


static double TwoNorm(const VectorType& rX)
{
double value;
const int sucess = rX.Norm2(&value); 
KRATOS_ERROR_IF_NOT(sucess == 0) << "Error computing norm" <<  std::endl;
return value;
}


static double TwoNorm(const MatrixType& rA)
{
return rA.NormFrobenius();
}


static void Mult(
const MatrixType& rA,
const VectorType& rX,
VectorType& rY
)
{
constexpr bool transpose_flag = false;
const int ierr = rA.Multiply(transpose_flag, rX, rY);
KRATOS_ERROR_IF(ierr != 0) << "Epetra multiplication failure " << ierr << std::endl;
}


static void Mult(
const MatrixType& rA,
const MatrixType& rB,
MatrixType& rC,
const bool CallFillCompleteOnResult = true,
const bool KeepAllHardZeros = false
)
{
KRATOS_TRY

constexpr bool transpose_flag = false;
const int ierr = EpetraExt::MatrixMatrix::Multiply(rA, transpose_flag, rB, transpose_flag, rC, CallFillCompleteOnResult, KeepAllHardZeros);
KRATOS_ERROR_IF(ierr != 0) << "Epetra multiplication failure. This may result if A or B are not already Filled, or if errors occur in putting values into C, etc. " << std::endl;

KRATOS_CATCH("")
}


static void TransposeMult(
const MatrixType& rA,
const VectorType& rX,
VectorType& rY
)
{
constexpr bool transpose_flag = true;
const int ierr = rA.Multiply(transpose_flag, rX, rY);
KRATOS_ERROR_IF(ierr != 0) << "Epetra multiplication failure " << ierr << std::endl;
}


static void TransposeMult(
const MatrixType& rA,
const MatrixType& rB,
MatrixType& rC,
const std::pair<bool, bool> TransposeFlag = {false, false},
const bool CallFillCompleteOnResult = true,
const bool KeepAllHardZeros = false
)
{
KRATOS_TRY

const int ierr = EpetraExt::MatrixMatrix::Multiply(rA, TransposeFlag.first, rB, TransposeFlag.second, rC, CallFillCompleteOnResult, KeepAllHardZeros);
KRATOS_ERROR_IF(ierr != 0) << "Epetra multiplication failure. This may result if A or B are not already Filled, or if errors occur in putting values into C, etc. " << std::endl;

KRATOS_CATCH("")
}


static void BtDBProductOperation(
MatrixType& rA,
const MatrixType& rD,
const MatrixType& rB,
const bool CallFillCompleteOnResult = true,
const bool KeepAllHardZeros = false,
const bool EnforceInitialGraph = false
)
{
std::vector<int> NumNz;
MatrixType aux_1(::Copy, rA.RowMap(), NumNz.data());

TransposeMult(rB, rD, aux_1, {true, false}, CallFillCompleteOnResult, KeepAllHardZeros);

if (EnforceInitialGraph) {
MatrixType aux_2(::Copy, rA.RowMap(), NumNz.data());
Mult(aux_1, rB, aux_2, CallFillCompleteOnResult, KeepAllHardZeros);

MatrixType* aux_3 =  new MatrixType(::Copy, CombineMatricesGraphs(rA, aux_2));

CopyMatrixValues(*aux_3, aux_2);

std::swap(rA, *aux_3);

delete aux_3;
} else { 
if (rA.NumGlobalNonzeros() > 0) {
MatrixType* aux_2 =  new MatrixType(::Copy, rB.RowMap(), NumNz.data());

Mult(aux_1, rB, *aux_2, CallFillCompleteOnResult, KeepAllHardZeros);

std::swap(rA, *aux_2);

delete aux_2;
} else { 
Mult(aux_1, rB, rA, CallFillCompleteOnResult, KeepAllHardZeros);
}
}
}


static void BDBtProductOperation(
MatrixType& rA,
const MatrixType& rD,
const MatrixType& rB,
const bool CallFillCompleteOnResult = true,
const bool KeepAllHardZeros = false,
const bool EnforceInitialGraph = false
)
{
std::vector<int> NumNz;
MatrixType aux_1(::Copy, rA.RowMap(), NumNz.data());

Mult(rB, rD, aux_1, CallFillCompleteOnResult, KeepAllHardZeros);

if (EnforceInitialGraph) {
MatrixType aux_2(::Copy, rA.RowMap(), NumNz.data());
TransposeMult(aux_1, rB, aux_2, {false, true}, CallFillCompleteOnResult, KeepAllHardZeros);

MatrixType* aux_3 =  new MatrixType(::Copy, CombineMatricesGraphs(rA, aux_2));

CopyMatrixValues(*aux_3, aux_2);

std::swap(rA, *aux_3);

delete aux_3;
} else { 
if (rA.NumGlobalNonzeros() > 0) {
MatrixType* aux_2 =  new MatrixType(::Copy, rA.RowMap(), NumNz.data());

TransposeMult(aux_1, rB, *aux_2, {false, true}, CallFillCompleteOnResult, KeepAllHardZeros);

std::swap(rA, *aux_2);

delete aux_2;
} else { 
TransposeMult(aux_1, rB, rA, {false, true}, CallFillCompleteOnResult, KeepAllHardZeros);
}
}
}


static void InplaceMult(
VectorType& rX,
const double A
)
{
if (A != 1.00) {
const int ierr = rX.Scale(A);
KRATOS_ERROR_IF(ierr != 0) << "Epetra scaling failure " << ierr << std::endl;
}
}


static void Assign(
VectorType& rX,
const double A,
const VectorType& rY
)
{
if (A != 1.00) {
const int ierr = rX.Scale(A, rY); 
KRATOS_ERROR_IF(ierr != 0) << "Epetra assign failure " << ierr << std::endl;
} else {
rX = rY;
}
}


static void UnaliasedAdd(
VectorType& rX,
const double A,
const VectorType& rY
)
{
const int ierr = rX.Update(A, rY, 1.0);
KRATOS_ERROR_IF(ierr != 0) << "Epetra unaliased add failure " << ierr << std::endl;
}


static void ScaleAndAdd(
const double A,
const VectorType& rX,
const double B,
const VectorType& rY,
VectorType& rZ
)
{
const int ierr = rZ.Update(A, rX, B, rY, 0.0);
KRATOS_ERROR_IF(ierr != 0) << "Epetra scale and add failure " << ierr << std::endl;
}


static void ScaleAndAdd(
const double A,
const VectorType& rX,
const double B,
VectorType& rY
)
{
const int ierr = rY.Update(A, rX, B);
KRATOS_ERROR_IF(ierr != 0) << "Epetra scale and add failure " << ierr << std::endl;
}


static void SetValue(
VectorType& rX,
IndexType i,
const double value
)
{
Epetra_IntSerialDenseVector indices(1);
Epetra_SerialDenseVector values(1);
indices[0] = i;
values[0] = value;

int ierr = rX.ReplaceGlobalValues(indices, values);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

ierr = rX.GlobalAssemble(Insert,true); 
KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
}


static void Set(
VectorType& rX,
const DataType A
)
{
const int ierr = rX.PutScalar(A);
KRATOS_ERROR_IF(ierr != 0) << "Epetra set failure " << ierr << std::endl;
}


static void Resize(
MatrixType& rA,
const SizeType m,
const SizeType n
)
{
KRATOS_ERROR << "Resize is not defined for Trilinos Sparse Matrix" << std::endl;
}


static void Resize(
VectorType& rX,
const SizeType n
)
{
KRATOS_ERROR << "Resize is not defined for a reference to Trilinos Vector - need to use the version passing a Pointer" << std::endl;
}


static void Resize(
VectorPointerType& pX,
const SizeType n
)
{
int global_elems = n;
Epetra_Map Map(global_elems, 0, pX->Comm());
VectorPointerType pNewEmptyX = Kratos::make_shared<VectorType>(Map);
pX.swap(pNewEmptyX);
}


static void Clear(MatrixPointerType& pA)
{
if(pA != NULL) {
int global_elems = 0;
Epetra_Map Map(global_elems, 0, pA->Comm());
MatrixPointerType pNewEmptyA = MatrixPointerType(new TMatrixType(::Copy, Map, 0));
pA.swap(pNewEmptyA);
}
}


static void Clear(VectorPointerType& pX)
{
if(pX != NULL) {
int global_elems = 0;
Epetra_Map Map(global_elems, 0, pX->Comm());
VectorPointerType pNewEmptyX = VectorPointerType(new VectorType(Map));
pX.swap(pNewEmptyX);
}
}


inline static void SetToZero(MatrixType& rA)
{
const int ierr = rA.PutScalar(0.0);
KRATOS_ERROR_IF(ierr != 0) << "Epetra set to zero failure " << ierr << std::endl;
}


inline static void SetToZero(VectorType& rX)
{
const int ierr = rX.PutScalar(0.0);
KRATOS_ERROR_IF(ierr != 0) << "Epetra set to zero failure " << ierr << std::endl;
}



inline static void AssembleLHS(
MatrixType& rA,
const Matrix& rLHSContribution,
const std::vector<std::size_t>& rEquationId
)
{
const unsigned int system_size = Size1(rA);

unsigned int active_indices = 0;
for (unsigned int i = 0; i < rEquationId.size(); i++)
if (rEquationId[i] < system_size)
++active_indices;

if (active_indices > 0) {
Epetra_IntSerialDenseVector indices(active_indices);
Epetra_SerialDenseMatrix values(active_indices, active_indices);

unsigned int loc_i = 0;
for (unsigned int i = 0; i < rEquationId.size(); i++) {
if (rEquationId[i] < system_size) {
indices[loc_i] = rEquationId[i];

unsigned int loc_j = 0;
for (unsigned int j = 0; j < rEquationId.size(); j++) {
if (rEquationId[j] < system_size) {
values(loc_i, loc_j) = rLHSContribution(i, j);
++loc_j;
}
}
++loc_i;
}
}

const int ierr = rA.SumIntoGlobalValues(indices, values);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}
}

/
inline static void AssembleRHS(
VectorType& rb,
const Vector& rRHSContribution,
const std::vector<std::size_t>& rEquationId
)
{
const unsigned int system_size = Size(rb);

unsigned int active_indices = 0;
for (unsigned int i = 0; i < rEquationId.size(); i++)
if (rEquationId[i] < system_size)
++active_indices;

if (active_indices > 0) {
Epetra_IntSerialDenseVector indices(active_indices);
Epetra_SerialDenseVector values(active_indices);

unsigned int loc_i = 0;
for (unsigned int i = 0; i < rEquationId.size(); i++) {
if (rEquationId[i] < system_size) {
indices[loc_i] = rEquationId[i];
values[loc_i] = rRHSContribution[i];
++loc_i;
}
}

const int ierr = rb.SumIntoGlobalValues(indices, values);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}
}


inline static constexpr bool IsDistributed()
{
return true;
}


inline static double GetValue(
const VectorType& rX,
const std::size_t I
)
{
KRATOS_ERROR_IF_NOT(rX.Map().MyGID(static_cast<int>(I))) << " non-local id: " << I << "." << std::endl;
return rX[0][rX.Map().LID(static_cast<int>(I))];
}


static void GatherValues(
const VectorType& rX,
const std::vector<int>& IndexArray,
double* pValues
)
{
KRATOS_TRY
double tot_size = IndexArray.size();

Epetra_Map dof_update_map(-1, tot_size, &(*(IndexArray.begin())), 0, rX.Comm());

Epetra_Import importer(dof_update_map, rX.Map());

Epetra_Vector temp(importer.TargetMap());

int ierr = temp.Import(rX, importer, Insert);
if(ierr != 0) KRATOS_THROW_ERROR(std::logic_error,"Epetra failure found","");

temp.ExtractCopy(&pValues);

rX.Comm().Barrier();
KRATOS_CATCH("")
}


MatrixPointerType ReadMatrixMarket(
const std::string FileName,
Epetra_MpiComm& rComm
)
{
KRATOS_TRY

Epetra_CrsMatrix* pp = nullptr;

int error_code = EpetraExt::MatrixMarketFileToCrsMatrix(FileName.c_str(), rComm, pp);

KRATOS_ERROR_IF(error_code != 0) << "Eerror thrown while reading Matrix Market file "<<FileName<< " error code is : " << error_code;

rComm.Barrier();

const Epetra_CrsGraph& rGraph = pp->Graph();
MatrixPointerType paux = Kratos::make_shared<Epetra_FECrsMatrix>( ::Copy, rGraph, false );

IndexType NumMyRows = rGraph.RowMap().NumMyElements();

int* MyGlobalElements = new int[NumMyRows];
rGraph.RowMap().MyGlobalElements(MyGlobalElements);

for(IndexType i = 0; i < NumMyRows; ++i) {
IndexType GlobalRow = MyGlobalElements[i];

int NumEntries;
std::size_t Length = pp->NumGlobalEntries(GlobalRow);  

double* Values = new double[Length];     
int* Indices = new int[Length];          

error_code = pp->ExtractGlobalRowCopy(GlobalRow, Length, NumEntries, Values, Indices);

KRATOS_ERROR_IF(error_code != 0) << "Error thrown in ExtractGlobalRowCopy : " << error_code;

error_code = paux->ReplaceGlobalValues(GlobalRow, Length, Values, Indices);

KRATOS_ERROR_IF(error_code != 0) << "Error thrown in ReplaceGlobalValues : " << error_code;

delete [] Values;
delete [] Indices;
}

paux->GlobalAssemble();

delete [] MyGlobalElements;
delete pp;

return paux;
KRATOS_CATCH("");
}


VectorPointerType ReadMatrixMarketVector(
const std::string& rFileName,
Epetra_MpiComm& rComm,
const int N
)
{
KRATOS_TRY

Epetra_Map my_map(N, 0, rComm);
Epetra_Vector* pv = nullptr;

int error_code = EpetraExt::MatrixMarketFileToVector(rFileName.c_str(), my_map, pv);

KRATOS_ERROR_IF(error_code != 0) << "error thrown while reading Matrix Market Vector file " << rFileName << " error code is: " << error_code;

rComm.Barrier();

IndexType num_my_rows = my_map.NumMyElements();
std::vector<int> gids(num_my_rows);
my_map.MyGlobalElements(gids.data());

std::vector<double> values(num_my_rows);
pv->ExtractCopy(values.data());

VectorPointerType final_vector = Kratos::make_shared<VectorType>(my_map);
int ierr = final_vector->ReplaceGlobalValues(gids.size(),gids.data(), values.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found with code ierr = " << ierr << std::endl;

final_vector->GlobalAssemble();

delete pv;
return final_vector;
KRATOS_CATCH("");
}


static Epetra_CrsGraph CombineMatricesGraphs(
const MatrixType& rA,
const MatrixType& rB
)
{
KRATOS_ERROR_IF_NOT(rA.RowMap().SameAs(rB.RowMap())) << "Row maps are not compatible" << std::endl;

const auto& r_graph_a = rA.Graph();
const auto& r_graph_b = rB.Graph();

KRATOS_ERROR_IF_NOT(r_graph_a.IndicesAreLocal() && r_graph_b.IndicesAreLocal()) << "Graphs indexes must be local" << std::endl;

int i, j, ierr;
int num_entries; 
int* cols;       
std::unordered_set<int> combined_indexes;
const bool same_col_map = rA.ColMap().SameAs(rB.ColMap());
Epetra_CrsGraph graph = same_col_map ? Epetra_CrsGraph(::Copy, rA.RowMap(), rA.ColMap(), 1000) : Epetra_CrsGraph(::Copy, rA.RowMap(), 1000);

if (same_col_map) {
for (i = 0; i < r_graph_a.NumMyRows(); i++) {
ierr = r_graph_a.ExtractMyRowView(i, num_entries, cols);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found extracting indices (I) with code ierr = " << ierr << std::endl;
for (j = 0; j < num_entries; j++) {
combined_indexes.insert(cols[j]);
}
ierr = r_graph_b.ExtractMyRowView(i, num_entries, cols);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found extracting indices (II) with code ierr = " << ierr << std::endl;
for (j = 0; j < num_entries; j++) {
combined_indexes.insert(cols[j]);
}
std::vector<int> combined_indexes_vector(combined_indexes.begin(), combined_indexes.end());
num_entries = combined_indexes_vector.size();
ierr = graph.InsertMyIndices(i, num_entries, combined_indexes_vector.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure inserting indices with code ierr = " << ierr << std::endl;
combined_indexes.clear();
}
} else { 
for (i = 0; i < r_graph_a.NumMyRows(); i++) {
const int global_row_index = r_graph_a.GRID(i);
ierr = r_graph_a.ExtractMyRowView(i, num_entries, cols);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found extracting indices (I) with code ierr = " << ierr << std::endl;
for (j = 0; j < num_entries; j++) {
combined_indexes.insert(r_graph_a.GCID(cols[j]));
}
ierr = r_graph_b.ExtractMyRowView(i, num_entries, cols);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found extracting indices (II) with code ierr = " << ierr << std::endl;
for (j = 0; j < num_entries; j++) {
combined_indexes.insert(r_graph_b.GCID(cols[j]));
}
std::vector<int> combined_indexes_vector(combined_indexes.begin(), combined_indexes.end());
num_entries = combined_indexes_vector.size();
ierr = graph.InsertGlobalIndices(global_row_index, num_entries, combined_indexes_vector.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure inserting indices with code ierr = " << ierr << std::endl;
combined_indexes.clear();
}
}

ierr = graph.FillComplete();
KRATOS_ERROR_IF(ierr < 0) << ": Epetra failure in Epetra_CrsGraph.FillComplete. Error code: " << ierr << std::endl;

return graph;
}


static void CopyMatrixValues(
MatrixType& rA,
const MatrixType& rB
)
{
SetToZero(rA);

int i, ierr;
int num_entries; 
double* vals;    
int* cols;       
for (i = 0; i < rB.NumMyRows(); i++) {
ierr = rB.ExtractMyRowView(i, num_entries, vals, cols);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found extracting values with code ierr = " << ierr << std::endl;
ierr = rA.ReplaceMyValues(i, num_entries, vals, cols);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found replacing values with code ierr = " << ierr << std::endl;
}
}


static double CheckAndCorrectZeroDiagonalValues(
const ProcessInfo& rProcessInfo,
MatrixType& rA,
VectorType& rb,
const SCALING_DIAGONAL ScalingDiagonal = SCALING_DIAGONAL::NO_SCALING
)
{
KRATOS_TRY

const double zero_tolerance = std::numeric_limits<double>::epsilon();

const double scale_factor = GetScaleNorm(rProcessInfo, rA, ScalingDiagonal);

for (int i = 0; i < rA.NumMyRows(); i++) {
int numEntries; 
double* vals;   
int* cols;      
rA.ExtractMyRowView(i, numEntries, vals, cols);
const int row_gid = rA.RowMap().GID(i);
bool empty = true;
int j;
for (j = 0; j < numEntries; j++) {
const int col_gid = rA.ColMap().GID(cols[j]);
if (col_gid == row_gid) {
if(std::abs(vals[j]) > zero_tolerance) {
empty = false;
}
break;
}
}

if (empty) {
vals[j] = scale_factor;
rb[0][i] = 0.0;
}
}

rb.GlobalAssemble();
rA.GlobalAssemble();

return scale_factor;

KRATOS_CATCH("");
}


static double GetScaleNorm(
const ProcessInfo& rProcessInfo,
const MatrixType& rA,
const SCALING_DIAGONAL ScalingDiagonal = SCALING_DIAGONAL::NO_SCALING
)
{
KRATOS_TRY

switch (ScalingDiagonal) {
case SCALING_DIAGONAL::NO_SCALING:
return 1.0;
case SCALING_DIAGONAL::CONSIDER_PRESCRIBED_DIAGONAL: {
KRATOS_ERROR_IF_NOT(rProcessInfo.Has(BUILD_SCALE_FACTOR)) << "Scale factor not defined at process info" << std::endl;
return rProcessInfo.GetValue(BUILD_SCALE_FACTOR);
}
case SCALING_DIAGONAL::CONSIDER_NORM_DIAGONAL:
return GetDiagonalNorm(rA)/static_cast<double>(Size1(rA));
case SCALING_DIAGONAL::CONSIDER_MAX_DIAGONAL:
return GetMaxDiagonal(rA);
default:
return GetMaxDiagonal(rA);
}

KRATOS_CATCH("");
}


static double GetDiagonalNorm(const MatrixType& rA)
{
KRATOS_TRY

Epetra_Vector diagonal(rA.RowMap());
const int ierr = rA.ExtractDiagonalCopy(diagonal);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure extracting diagonal " << ierr << std::endl;

return TrilinosSpace<Epetra_FECrsMatrix, Epetra_Vector>::TwoNorm(diagonal);

KRATOS_CATCH("");
}


static double GetAveragevalueDiagonal(const MatrixType& rA)
{
KRATOS_TRY

return 0.5 * (GetMaxDiagonal(rA) + GetMinDiagonal(rA));

KRATOS_CATCH("");
}


static double GetMaxDiagonal(const MatrixType& rA)
{
KRATOS_TRY

Epetra_Vector diagonal(rA.RowMap());
const int ierr = rA.ExtractDiagonalCopy(diagonal);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure extracting diagonal " << ierr << std::endl;
return TrilinosSpace<Epetra_FECrsMatrix, Epetra_Vector>::Max(diagonal);

KRATOS_CATCH("");
}


static double GetMinDiagonal(const MatrixType& rA)
{
KRATOS_TRY

Epetra_Vector diagonal(rA.RowMap());
const int ierr = rA.ExtractDiagonalCopy(diagonal);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure extracting diagonal " << ierr << std::endl;
return TrilinosSpace<Epetra_FECrsMatrix, Epetra_Vector>::Min(diagonal);

KRATOS_CATCH("");
}





virtual std::string Info() const
{
return "TrilinosSpace";
}


virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "TrilinosSpace";
}


virtual void PrintData(std::ostream& rOStream) const
{
}


template< class TOtherMatrixType >
static bool WriteMatrixMarketMatrix(
const char* pFileName,
const TOtherMatrixType& rM,
const bool Symmetric
)
{
KRATOS_TRY;
return EpetraExt::RowMatrixToMatrixMarketFile(pFileName, rM); 
KRATOS_CATCH("");
}


template< class VectorType >
static bool WriteMatrixMarketVector(
const char* pFileName,
const VectorType& rV
)
{
KRATOS_TRY;
return EpetraExt::MultiVectorToMatrixMarketFile(pFileName, rV);
KRATOS_CATCH("");
}


static DofUpdaterPointerType CreateDofUpdater()
{
DofUpdaterType tmp;
return tmp.Create();
}

private:

TrilinosSpace & operator=(TrilinosSpace const& rOther);

TrilinosSpace(TrilinosSpace const& rOther);

}; 


} 
