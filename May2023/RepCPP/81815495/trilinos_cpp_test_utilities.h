
#pragma once



#include "trilinos_space.h"

namespace Kratos
{



class TrilinosCPPTestUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TrilinosCPPTestUtilities);

using TrilinosSparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
using TrilinosLocalSpaceType = UblasSpace<double, Matrix, Vector>;

using TrilinosSparseMatrixType = TrilinosSparseSpaceType::MatrixType;
using TrilinosVectorType = TrilinosSparseSpaceType::VectorType;

using TrilinosLocalMatrixType = TrilinosLocalSpaceType::MatrixType;
using TrilinosLocalVectorType = TrilinosLocalSpaceType::VectorType;




static TrilinosLocalMatrixType GenerateDummyLocalMatrix(
const int NumGlobalElements = 12,
const double Offset = 0.0,
const bool AddNoDiagonalValues = false
);


static TrilinosSparseMatrixType GenerateDummySparseMatrix(
const DataCommunicator& rDataCommunicator,
const int NumGlobalElements = 12,
const double Offset = 0.0,
const bool AddNoDiagonalValues = false
);


static TrilinosLocalVectorType GenerateDummyLocalVector(
const int NumGlobalElements = 12,
const double Offset = 0.0
);


static TrilinosVectorType GenerateDummySparseVector(
const DataCommunicator& rDataCommunicator,
const int NumGlobalElements = 12,
const double Offset = 0.0
);


static void CheckSparseVectorFromLocalVector(
const TrilinosVectorType& rA,
const TrilinosLocalVectorType& rB,
const double NegligibleValueThreshold = 1e-8
);


static void CheckSparseVector(
const TrilinosVectorType& rb,
const std::vector<int>& rIndexes,
const std::vector<double>& rValues,
const double NegligibleValueThreshold = 1e-8
);


static void CheckSparseMatrixFromLocalMatrix(
const TrilinosSparseMatrixType& rA,
const TrilinosLocalMatrixType& rB,
const double NegligibleValueThreshold = 1e-8
);


static void CheckSparseMatrixFromLocalMatrix(
const TrilinosSparseMatrixType& rA,
const std::vector<int>& rRowIndexes,
const std::vector<int>& rColumnIndexes,
const TrilinosLocalMatrixType& rB,
const double NegligibleValueThreshold = 1e-8
);


static void CheckSparseMatrix(
const TrilinosSparseMatrixType& rA,
const std::vector<int>& rRowIndexes,
const std::vector<int>& rColumnIndexes,
const std::vector<double>& rValues,
const double NegligibleValueThreshold = 1e-8
);


static void GenerateSparseMatrixIndexAndValuesVectors(
const TrilinosSparseSpaceType::MatrixType& rA,
std::vector<int>& rRowIndexes,
std::vector<int>& rColumnIndexes,
std::vector<double>& rValues,
const bool PrintValues = false,
const double ThresholdIncludeHardZeros = -1
);


static TrilinosSparseMatrixType GenerateSparseMatrix(
const DataCommunicator& rDataCommunicator,
const int NumGlobalElements,
const std::vector<int>& rRowIndexes,
const std::vector<int>& rColumnIndexes,
const std::vector<double>& rValues,
const Epetra_Map* pMap =  nullptr
);


}; 

} 