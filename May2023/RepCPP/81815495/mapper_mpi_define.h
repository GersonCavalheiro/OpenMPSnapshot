
#pragma once


#include "Epetra_FEVector.h"
#include "Epetra_FECrsMatrix.h"

#include "trilinos_space.h"
#include "spaces/ublas_space.h"

namespace Kratos {

namespace MPIMapperDefinitions {

typedef TUblasDenseSpace<double> DenseSpaceType;

typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> SparseSpaceType;

}  

}  
