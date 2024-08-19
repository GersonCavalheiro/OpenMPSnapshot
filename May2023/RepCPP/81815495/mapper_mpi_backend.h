
#pragma once



#include "custom_searching/interface_communicator_mpi.h"
#include "custom_utilities/mapping_matrix_utilities.h"
#include "custom_utilities/interface_vector_container.h"

namespace Kratos {

template<class TSparseSpace, class TDenseSpace>
struct MapperMPIBackend
{
using InterfaceCommunicatorType = InterfaceCommunicatorMPI;
using MappingMatrixUtilitiesType = MappingMatrixUtilities<TSparseSpace, TDenseSpace>;
using InterfaceVectorContainerType = InterfaceVectorContainer<TSparseSpace, TDenseSpace>;
};

}  
