#pragma once
#include <cusp/transpose.h>
#include "blas2_cusp.h"
#include "tensor_traits.h"
#ifdef MPI_VERSION
#include "blas2_dispatch_mpi.h"
#endif 

namespace dg
{

namespace detail
{
template <class Matrix>
Matrix doTranspose( const Matrix& src, CuspMatrixTag)
{
Matrix out;
cusp::transpose( src, out);
return out;
}
#ifdef MPI_VERSION
template <class LocalMatrix, class Collective>
MPIDistMat<LocalMatrix, Collective> doTranspose( const MPIDistMat<LocalMatrix, Collective>& src, MPIMatrixTag)
{
LocalMatrix tr = doTranspose( src.matrix(), get_tensor_category<LocalMatrix>());
MPIDistMat<LocalMatrix, Collective> out( tr, src.collective());
if( src.get_dist() == dg::row_dist) out.set_dist( dg::col_dist);
if( src.get_dist() == dg::col_dist) out.set_dist( dg::row_dist);
return out;
}
#endif
}


template<class Matrix>
Matrix transpose( const Matrix& src)
{
return detail::doTranspose( src, get_tensor_category<Matrix>());
}

} 
