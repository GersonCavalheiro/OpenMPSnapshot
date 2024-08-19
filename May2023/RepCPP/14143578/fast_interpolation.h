#pragma once

#include <thrust/host_vector.h>
#include "dg/backend/memory.h"
#include "dg/backend/typedefs.h"
#include "dg/enums.h"
#include "dg/blas.h"
#include "grid.h"
#include "interpolation.h"
#include "projection.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#endif 





namespace dg
{


template <class MatrixType, class ContainerType>
struct MultiMatrix
{
using real_type = get_value_type<ContainerType>;
MultiMatrix(){}

MultiMatrix( int dimension): inter_(dimension), temp_(dimension-1 > 0 ? dimension-1 : 0 ){}

template<class OtherMatrix, class OtherContainer, class ... Params>
MultiMatrix( const MultiMatrix<OtherMatrix, OtherContainer>& src, Params&& ... ps){
unsigned dimsM = src.get_matrices().size();
unsigned dimsT = src.get_temp().size();
inter_.resize( dimsM);
temp_.resize(  dimsT);
for( unsigned i=0; i<dimsM; i++)
inter_[i] = src.get_matrices()[i];
for( unsigned i=0; i<dimsT; i++)
dg::assign( src.get_temp()[i].data(), temp_[i].data(), std::forward<Params>(ps)...);

}
template<class ...Params>
void construct( Params&& ...ps){
*this = MultiMatrix( std::forward<Params>(ps)...);
}


template<class ContainerType0, class ContainerType1>
void symv( const ContainerType0& x, ContainerType1& y) const{ symv( 1., x,0,y);}
template<class ContainerType0, class ContainerType1>
void symv(real_type alpha, const ContainerType0& x, real_type beta, ContainerType1& y) const
{
int dims = inter_.size();
if( dims == 1)
{
dg::blas2::symv( alpha, inter_[0], x, beta, y);
return;
}
dg::blas2::symv( inter_[0], x,temp_[0].data());
for( int i=1; i<dims-1; i++)
dg::blas2::symv( inter_[i], temp_[i-1].data(), temp_[i].data());
dg::blas2::symv( alpha, inter_[dims-1], temp_[dims-2].data(), beta, y);
}
std::vector<Buffer<ContainerType> >& get_temp(){ return temp_;}
const std::vector<Buffer<ContainerType> >& get_temp()const{ return temp_;}
std::vector<MatrixType>& get_matrices(){ return inter_;}
const std::vector<MatrixType>& get_matrices()const{ return inter_;}
private:
std::vector<MatrixType > inter_;
std::vector<Buffer<ContainerType> > temp_;
};

template <class M, class V>
struct TensorTraits<MultiMatrix<M, V> >
{
using value_type  = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};

namespace detail
{
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > multiply( const dg::HMatrix_t<real_type>& left, const dg::HMatrix_t<real_type>& right)
{
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > matrix(2);
if( right.total_num_rows() != left.total_num_cols())
throw Error( Message(_ping_)<< "left and right cannot be multiplied due to wrong sizes" << left.total_num_cols() << " vs "<<right.total_num_rows());
matrix.get_matrices()[0] = right;
matrix.get_matrices()[1] = left;
thrust::host_vector<real_type> vec( right.total_num_rows());
matrix.get_temp()[0] = Buffer<dg::HVec_t<real_type>>(vec);
return matrix;
}
template<class real_type>
void set_right_size( dg::HMatrix_t<real_type>& left, const dg::HMatrix_t<real_type>& right)
{
left.set_right_size(right.num_rows*right.n*right.right_size);
}
#ifdef MPI_VERSION
template<class real_type>
MultiMatrix< dg::MHMatrix_t<real_type>, dg::MHVec_t<real_type> > multiply( const dg::MHMatrix_t<real_type>& left, const dg::MHMatrix_t<real_type>& right)
{
MultiMatrix< dg::MHMatrix_t<real_type>, dg::MHVec_t<real_type> > matrix(2);
matrix.get_matrices()[0] = right;
matrix.get_matrices()[1] = left;
thrust::host_vector<real_type> vec( right.inner_matrix().total_num_rows());
matrix.get_temp()[0] = Buffer<dg::MHVec_t<real_type>>({vec, left.collective().communicator()});
return matrix;
}
template<class real_type>
void set_right_size( dg::MHMatrix_t<real_type>& left, const dg::MHMatrix_t<real_type>& right)
{
const HMatrix_t<real_type>& in = right.inner_matrix();
unsigned right_size = in.num_rows*in.n*in.right_size;
left.inner_matrix().set_right_size(right_size);
left.outer_matrix().right_size = right_size;
}
#endif
} 


namespace create
{



template<class real_type>
dg::HMatrix_t<real_type> fast_interpolation( const RealGrid1d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
unsigned n=t.n();
dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
dg::RealGrid1d<real_type> g_new( -1., 1., n*multiplyn, multiplyNx);
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolX = dg::create::interpolation( g_new, g_old);
EllSparseBlockMat<real_type> iX( multiplyn*multiplyNx*t.N(), t.N(), 1, multiplyNx*multiplyn, t.n());
for( unsigned  k=0; k<multiplyNx*multiplyn; k++)
for( unsigned  i=0; i<n; i++)
for( unsigned  j=0; j<n; j++)
iX.data[(k*n+i)*n+j] = interpolX.values[(k*n+i)*n+j];
for( unsigned i=0; i<multiplyNx*multiplyn*t.N(); i++)
{
iX.cols_idx[i] = i/(multiplyNx*multiplyn);
iX.data_idx[i] = i%(multiplyNx*multiplyn);
}
return iX;
}


template<class real_type>
dg::HMatrix_t<real_type> fast_projection( const RealGrid1d<real_type>& t, unsigned dividen, unsigned divideNx)
{
if( t.N()%divideNx != 0) throw Error( Message(_ping_)<< "Nx and divideNx don't match: Nx: " << t.N()<< " divideNx "<< (unsigned)divideNx);
if( t.n()%dividen != 0) throw Error( Message(_ping_)<< "n and dividen don't match: n: " << t.n()<< " dividen "<< (unsigned)dividen);
unsigned n=t.n()/dividen;
dg::RealGrid1d<real_type> g_old( -1., 1., n*dividen, divideNx);
dg::RealGrid1d<real_type> g_new( -1., 1., n, 1);
dg::HVec w1d = dg::create::weights( g_old);
dg::HVec v1d = dg::create::inv_weights( g_new);
cusp::coo_matrix<int, real_type, cusp::host_memory> projectX, tmp;
tmp = dg::create::interpolation( g_old, g_new);
cusp::transpose( tmp, projectX);
EllSparseBlockMat<real_type> pX( t.N()/divideNx, t.N()*dividen, divideNx*dividen, divideNx*dividen, n);
for( unsigned k=0; k<divideNx; k++)
for( unsigned l=0; l<dividen; l++)
for( unsigned i=0; i<n; i++)
for( unsigned j=0; j<n; j++)
{
pX.data[((k*dividen+l)*n+i)*n+j] = projectX.values[((i*divideNx+k)*dividen + l)*n+j];
pX.data[((k*dividen+l)*n+i)*n+j] *= v1d[i]*w1d[l*n+j];
}
for( unsigned i=0; i<t.N()/divideNx; i++)
for( unsigned d=0; d<divideNx*dividen; d++)
{
pX.cols_idx[i*divideNx*dividen+d] = i*divideNx*dividen+d;
pX.data_idx[i*divideNx*dividen+d] = d;
}
return pX;
}


template<class real_type>
dg::HMatrix_t<real_type> fast_transform( dg::Operator<real_type> opx, const RealGrid1d<real_type>& t)
{
EllSparseBlockMat<real_type> A( t.N(), t.N(), 1, 1, t.n());
if( opx.size() != t.n())
throw Error( Message(_ping_)<< "Operator must have same n as grid!");
dg::assign( opx.data(), A.data);
for( unsigned i=0; i<t.N(); i++)
{
A.cols_idx[i] = i;
A.data_idx[i] = 0;
}
return A;
}

template<class real_type>
dg::HMatrix_t<real_type> fast_interpolation( enum coo3d direction, const aRealTopology2d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
if( direction == dg::coo3d::x)
{
auto trafo = dg::create::fast_interpolation( t.gx(), multiplyn,multiplyNx);
trafo.set_left_size ( t.ny()*t.Ny());
return trafo;
}
auto trafo = dg::create::fast_interpolation( t.gy(), multiplyn,multiplyNx);
trafo.set_right_size ( t.nx()*t.Nx());
return trafo;
}

template<class real_type>
dg::HMatrix_t<real_type> fast_projection( enum coo3d direction, const aRealTopology2d<real_type>& t, unsigned dividen, unsigned divideNx)
{
if( direction == dg::coo3d::x)
{
auto trafo = dg::create::fast_projection( t.gx(), dividen,divideNx);
trafo.set_left_size ( t.ny()*t.Ny());
return trafo;
}
auto trafo = dg::create::fast_projection( t.gy(), dividen,divideNx);
trafo.set_right_size ( t.nx()*t.Nx());
return trafo;
}

template<class real_type>
dg::HMatrix_t<real_type> fast_transform( enum coo3d direction, dg::Operator<real_type> opx, const aRealTopology2d<real_type>& t)
{
if( direction == dg::coo3d::x)
{
auto trafo = fast_transform( opx, t.gx());
trafo.set_left_size ( t.ny()*t.Ny());
return trafo;
}
auto trafo = fast_transform( opx, t.gy());
trafo.set_right_size ( t.nx()*t.Nx());
return trafo;
}

template<class real_type>
dg::HMatrix_t<real_type> fast_interpolation( enum coo3d direction, const aRealTopology3d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
if( direction == dg::coo3d::x)
{
auto trafo = fast_interpolation( t.gx(), multiplyn, multiplyNx);
trafo.set_left_size ( t.ny()*t.Ny()*t.nz()*t.Nz());
return trafo;
}
if( direction == dg::coo3d::y)
{
auto trafo = fast_interpolation( t.gy(), multiplyn, multiplyNx);
trafo.set_left_size ( t.nz()*t.Nz());
trafo.set_right_size ( t.nx()*t.Nx());
return trafo;
}
auto trafo = fast_interpolation( t.gz(), multiplyn, multiplyNx);
trafo.set_right_size ( t.nx()*t.Nx()*t.ny()*t.Ny());
return trafo;
}

template<class real_type>
dg::HMatrix_t<real_type> fast_projection( enum coo3d direction, const aRealTopology3d<real_type>& t, unsigned dividen, unsigned divideNx)
{
if( direction == dg::coo3d::x)
{
auto trafo = fast_projection( t.gx(), dividen, divideNx);
trafo.set_left_size ( t.ny()*t.Ny()*t.nz()*t.Nz());
return trafo;
}
if( direction == dg::coo3d::y)
{
auto trafo = fast_projection( t.gy(), dividen, divideNx);
trafo.set_left_size ( t.nz()*t.Nz());
trafo.set_right_size ( t.nx()*t.Nx());
return trafo;
}
auto trafo = fast_projection( t.gz(), dividen, divideNx);
trafo.set_right_size ( t.nx()*t.Nx()*t.ny()*t.Ny());
return trafo;
}

template<class real_type>
dg::HMatrix_t<real_type> fast_transform( enum coo3d direction, dg::Operator<real_type> opx, const aRealTopology3d<real_type>& t)
{
if( direction == dg::coo3d::x)
{
auto trafo = fast_transform( opx, t.gx());
trafo.set_left_size ( t.ny()*t.Ny()*t.nz()*t.Nz());
return trafo;
}
if( direction == dg::coo3d::y)
{
auto trafo = fast_transform( opx, t.gy());
trafo.set_left_size ( t.nz()*t.Nz());
trafo.set_right_size ( t.nx()*t.Nx());
return trafo;
}
auto trafo = fast_transform( opx, t.gz());
trafo.set_right_size ( t.nx()*t.Nx()*t.ny()*t.Ny());
return trafo;
}


#ifdef MPI_VERSION
namespace detail
{
template<class real_type>
MHMatrix_t<real_type> elevate_no_comm( const HMatrix_t<real_type>& local, MPI_Comm comm)
{
return MHMatrix_t<real_type>( local, CooSparseBlockMat<real_type>(), NNCH<real_type>(comm));
}

}

template<class real_type>
dg::MHMatrix_t<real_type> fast_interpolation( enum coo3d direction, const aRealMPITopology2d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
return detail::elevate_no_comm( dg::create::fast_interpolation( direction, t.local(), multiplyn, multiplyNx), t.communicator());
}
template<class real_type>
dg::MHMatrix_t<real_type> fast_projection( enum coo3d direction, const aRealMPITopology2d<real_type>& t, unsigned dividen, unsigned divideNx)
{
return detail::elevate_no_comm( dg::create::fast_projection( direction, t.local(), dividen, divideNx), t.communicator());
}
template<class real_type>
MHMatrix_t<real_type> fast_transform( enum coo3d direction, dg::Operator<real_type> opx, const aRealMPITopology2d<real_type>& t)
{
return detail::elevate_no_comm( dg::create::fast_transform( direction, opx, t.local()), t.communicator());
}

template<class real_type>
dg::MHMatrix_t<real_type> fast_interpolation( enum coo3d direction, const aRealMPITopology3d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
return detail::elevate_no_comm( dg::create::fast_interpolation( direction, t.local(), multiplyn, multiplyNx), t.communicator());
}
template<class real_type>
dg::MHMatrix_t<real_type> fast_projection( enum coo3d direction, const aRealMPITopology3d<real_type>& t, unsigned dividen, unsigned divideNx)
{
return detail::elevate_no_comm( dg::create::fast_projection( direction, t.local(), dividen, divideNx), t.communicator());
}
template<class real_type>
MHMatrix_t<real_type> fast_transform( enum coo3d direction, dg::Operator<real_type> opx, const aRealMPITopology3d<real_type>& t)
{
return detail::elevate_no_comm( dg::create::fast_transform( direction, opx, t.local()), t.communicator());
}
#endif 

template<class Topology>
auto fast_interpolation( const Topology& t, unsigned multiplyn, unsigned multiplyNx, unsigned multiplyNy)
{
auto interX = dg::create::fast_interpolation( dg::coo3d::x, t, multiplyn,multiplyNx);
auto interY = dg::create::fast_interpolation( dg::coo3d::y, t, multiplyn,multiplyNy);
dg::detail::set_right_size( interY, interX);
return dg::detail::multiply( interY, interX);
}

template<class Topology>
auto fast_projection( const Topology& t, unsigned dividen, unsigned divideNx, unsigned divideNy)
{
auto interX = dg::create::fast_projection( dg::coo3d::x, t, dividen, divideNx);
auto interY = dg::create::fast_projection( dg::coo3d::y, t, dividen, divideNy);
dg::detail::set_right_size( interY, interX);
return dg::detail::multiply( interY, interX);
}
template<class Topology>
auto fast_transform( dg::Operator<typename Topology::value_type> opx, dg::Operator<typename Topology::value_type> opy, const Topology& t)
{
auto interX = dg::create::fast_transform( dg::coo3d::x, opx, t);
auto interY = dg::create::fast_transform( dg::coo3d::y, opy, t);
return dg::detail::multiply( interY, interX);
}

}


template<class real_type>
thrust::host_vector<real_type> forward_transform( const thrust::host_vector<real_type>& in, const aRealTopology2d<real_type>& g)
{
thrust::host_vector<real_type> out(in.size(), 0);
auto forward = create::fast_transform( g.dltx().forward(),
g.dlty().forward(), g);
dg::blas2::symv( forward, in, out);
return out;
}

}
