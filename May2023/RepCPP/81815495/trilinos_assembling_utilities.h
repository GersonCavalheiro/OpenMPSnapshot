
#pragma once



#include "trilinos_space.h"

namespace Kratos
{







class TrilinosAssemblingUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TrilinosAssemblingUtilities);

using TrilinosSparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;

using MatrixType = TrilinosSparseSpaceType::MatrixType;

using VectorType = TrilinosSparseSpaceType::VectorType;

using IndexType = std::size_t;

using SizeType = std::size_t;


TrilinosAssemblingUtilities() = delete;




inline static void AssembleRelationMatrixT(
MatrixType& rT,
const Matrix& rTContribution,
const std::vector<std::size_t>& rSlaveEquationId,
const std::vector<std::size_t>& rMasterEquationId
)
{
const unsigned int system_size = TrilinosSparseSpaceType::Size1(rT);

int slave_active_indices = 0;
for (unsigned int i = 0; i < rSlaveEquationId.size(); i++) {
if (rSlaveEquationId[i] < system_size) {
++slave_active_indices;
}
}
int master_active_indices = 0;
for (unsigned int i = 0; i < rMasterEquationId.size(); i++) {
if (rMasterEquationId[i] < system_size) {
++master_active_indices;
}
}

if (slave_active_indices > 0 && master_active_indices > 0) {
std::vector<int> indices(slave_active_indices);
std::vector<double> values(master_active_indices);

unsigned int loc_i = 0;
for (unsigned int i = 0; i < rSlaveEquationId.size(); i++) {
if (rSlaveEquationId[i] < system_size) {
const int current_global_row = rSlaveEquationId[i];

unsigned int loc_j = 0;
for (unsigned int j = 0; j < rMasterEquationId.size(); j++) {
if (rMasterEquationId[j] < system_size) {
indices[loc_j] = rMasterEquationId[j];
values[loc_j] = rTContribution(i, j);
++loc_j;
}
}

const int ierr = rT.SumIntoGlobalValues(current_global_row, master_active_indices, values.data(), indices.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

++loc_i;
}
}
}
}


inline static void AssembleConstantVector(
VectorType& rC,
const Vector& rConstantContribution,
const std::vector<std::size_t>& rSlaveEquationId
)
{
const unsigned int system_size = TrilinosSparseSpaceType::Size(rC);

unsigned int slave_active_indices = 0;
for (unsigned int i = 0; i < rSlaveEquationId.size(); i++)
if (rSlaveEquationId[i] < system_size)
++slave_active_indices;

if (slave_active_indices > 0) {
Epetra_IntSerialDenseVector indices(slave_active_indices);
Epetra_SerialDenseVector values(slave_active_indices);

unsigned int loc_i = 0;
for (unsigned int i = 0; i < rSlaveEquationId.size(); i++) {
if (rSlaveEquationId[i] < system_size) {
indices[loc_i] = rSlaveEquationId[i];
values[loc_i] = rConstantContribution[i];
++loc_i;
}
}

const int ierr = rC.SumIntoGlobalValues(indices, values);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}
}


static inline void SetGlobalValue(
VectorType& rX,
IndexType i,
const double Value
)
{
Epetra_IntSerialDenseVector indices(1);
Epetra_SerialDenseVector values(1);
indices[0] = i;
values[0] = Value;
int ierr = rX.ReplaceGlobalValues(indices, values);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

ierr = rX.GlobalAssemble(Insert,true); 
KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
}


static inline void SetGlobalValueWithoutGlobalAssembly(
VectorType& rX,
IndexType i,
const double Value
)
{
Epetra_IntSerialDenseVector indices(1);
Epetra_SerialDenseVector values(1);
indices[0] = i;
values[0] = Value;
const int ierr = rX.ReplaceGlobalValues(indices, values);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}


static inline void SetLocalValue(
VectorType& rX,
IndexType i,
const double Value
)
{
int ierr = rX.ReplaceMyValue(static_cast<int>(i), 0, Value);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
ierr = rX.GlobalAssemble(Insert,true); 
KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
}


static inline void SetLocalValueWithoutGlobalAssembly(
VectorType& rX,
IndexType i,
const double Value
)
{
const int ierr = rX.ReplaceMyValue(static_cast<int>(i), 0, Value);
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}


static inline void SetGlobalValue(
MatrixType& rA,
IndexType i,
IndexType j,
const double Value
)
{
std::vector<double> values(1, Value);
std::vector<int> indices(1, j);

int ierr = rA.ReplaceGlobalValues(static_cast<int>(i), 1, values.data(), indices.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

ierr = rA.GlobalAssemble();
KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
}


static inline void SetGlobalValueWithoutGlobalAssembly(
MatrixType& rA,
IndexType i,
IndexType j,
const double Value
)
{
std::vector<double> values(1, Value);
std::vector<int> indices(1, j);

const int ierr = rA.ReplaceGlobalValues(static_cast<int>(i), 1, values.data(), indices.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}


static inline void SetLocalValue(
MatrixType& rA,
IndexType i,
IndexType j,
const double Value
)
{
std::vector<double> values(1, Value);
std::vector<int> indices(1, j);

int ierr = rA.ReplaceMyValues(static_cast<int>(i), 1, values.data(), indices.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;

ierr = rA.GlobalAssemble();
KRATOS_ERROR_IF(ierr < 0) << "Epetra failure when attempting to insert value in function SetValue" << std::endl;
}


static inline void SetLocalValueWithoutGlobalAssembly(
MatrixType& rA,
IndexType i,
IndexType j,
const double Value
)
{
std::vector<double> values(1, Value);
std::vector<int> indices(1, j);

const int ierr = rA.ReplaceMyValues(static_cast<int>(i), 1, values.data(), indices.data());
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found" << std::endl;
}





virtual std::string Info() const
{
return "TrilinosAssemblingUtilities";
}


virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "TrilinosAssemblingUtilities";
}


virtual void PrintData(std::ostream& rOStream) const
{
}

private:

TrilinosAssemblingUtilities & operator=(TrilinosAssemblingUtilities const& rOther);

TrilinosAssemblingUtilities(TrilinosAssemblingUtilities const& rOther);

}; 


} 
