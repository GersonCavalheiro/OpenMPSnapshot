#pragma once
#include "topological_traits.h"
#include "multiply.h"
#include "base_geometry.h"
#include "weights.h"


namespace dg
{

template< class Functor, class real_type>
thrust::host_vector<real_type> pullback( const Functor& f, const aRealGeometry2d<real_type>& g)
{
std::vector<thrust::host_vector<real_type> > map = g.map();
thrust::host_vector<real_type> vec( g.size());
for( unsigned i=0; i<g.size(); i++)
vec[i] = f( map[0][i], map[1][i]);
return vec;
}


template< class Functor, class real_type>
thrust::host_vector<real_type> pullback( const Functor& f, const aRealGeometry3d<real_type>& g)
{
std::vector<thrust::host_vector<real_type> > map = g.map();
thrust::host_vector<real_type> vec( g.size());
for( unsigned i=0; i<g.size(); i++)
vec[i] = f( map[0][i], map[1][i], map[2][i]);
return vec;
}

#ifdef MPI_VERSION

template< class Functor, class real_type>
MPI_Vector<thrust::host_vector<real_type> > pullback( const Functor& f, const aRealMPIGeometry2d<real_type>& g)
{
std::vector<MPI_Vector<thrust::host_vector<real_type> > > map = g.map();
thrust::host_vector<real_type> vec( g.local().size());
for( unsigned i=0; i<g.local().size(); i++)
vec[i] = f( map[0].data()[i], map[1].data()[i]);
return MPI_Vector<thrust::host_vector<real_type> >( vec, g.communicator());
}


template< class Functor, class real_type>
MPI_Vector<thrust::host_vector<real_type> > pullback( const Functor& f, const aRealMPIGeometry3d<real_type>& g)
{
std::vector<MPI_Vector<thrust::host_vector<real_type> > > map = g.map();
thrust::host_vector<real_type> vec( g.local().size());
for( unsigned i=0; i<g.local().size(); i++)
vec[i] = f( map[0].data()[i], map[1].data()[i], map[2].data()[i]);
return MPI_Vector<thrust::host_vector<real_type> >( vec, g.communicator());
}

#endif 


template<class Functor1, class Functor2, class container, class Geometry>
void pushForwardPerp( const Functor1& vR, const Functor2& vZ,
container& vx, container& vy,
const Geometry& g)
{
using host_vec = get_host_vector<Geometry>;
host_vec out1 = pullback( vR, g);
host_vec out2 = pullback( vZ, g);
dg::tensor::multiply2d(g.jacobian(), out1, out2, out1, out2);
dg::assign( out1, vx);
dg::assign( out2, vy);
}


template<class Functor1, class Functor2, class Functor3, class container, class Geometry>
void pushForward( const Functor1& vR, const Functor2& vZ, const Functor3& vPhi,
container& vx, container& vy, container& vz,
const Geometry& g)
{
using host_vec = get_host_vector<Geometry>;
host_vec out1 = pullback( vR, g);
host_vec out2 = pullback( vZ, g);
host_vec out3 = pullback( vPhi, g);
dg::tensor::multiply3d(g.jacobian(), out1, out2, out3, out1, out2, out3);
dg::assign( out1, vx);
dg::assign( out2, vy);
dg::assign( out3, vz);
}


template<class FunctorRR, class FunctorRZ, class FunctorZZ, class container, class Geometry>
void pushForwardPerp( const FunctorRR& chiRR, const FunctorRZ& chiRZ, const FunctorZZ& chiZZ,
SparseTensor<container>& chi,
const Geometry& g)
{
using host_vec = get_host_vector<Geometry>;
host_vec chiRR_ = pullback( chiRR, g);
host_vec chiRZ_ = pullback( chiRZ, g);
host_vec chiZZ_ = pullback( chiZZ, g);

const dg::SparseTensor<container> jac = g.jacobian();
std::vector<container> values( 5);
dg::assign( dg::evaluate( dg::zero,g), values[0]);
dg::assign( dg::evaluate( dg::one, g), values[1]);
dg::assign( chiRR_, values[2]);
dg::assign( chiRZ_, values[3]);
dg::assign( chiZZ_, values[4]);
chi.idx(0,0)=2, chi.idx(0,1)=chi.idx(1,0)=3, chi.idx(1,1)=4;
chi.idx(2,0)=chi.idx(2,1)=chi.idx(0,2)=chi.idx(1,2) = 0;
chi.idx(2,2)=1;
chi.values() = values;
container tmp00(jac.value(0,0)), tmp01(tmp00), tmp10(tmp00), tmp11(tmp00);
dg::tensor::multiply2d( chi, jac.value(0,0), jac.value(0,1), tmp00, tmp10);
dg::tensor::multiply2d( chi, jac.value(1,0), jac.value(1,1), tmp01, tmp11);
dg::tensor::multiply2d( jac, tmp00, tmp10, chi.values()[2], chi.values()[3]);
dg::tensor::multiply2d( jac, tmp01, tmp11, chi.values()[3], chi.values()[4]);
}

namespace create{




template< class Geometry>
get_host_vector<Geometry> volume( const Geometry& g)
{
using host_vector = get_host_vector<Geometry>;
host_vector vol = dg::tensor::volume(g.metric());
host_vector weights = dg::create::weights( g);
dg::blas1::pointwiseDot( weights, vol, vol);
return vol;
}


template< class Geometry>
get_host_vector<Geometry> inv_volume( const Geometry& g)
{
using host_vector = get_host_vector<Geometry>;
using real_type = get_value_type<host_vector>;
host_vector vol = volume(g);
dg::blas1::transform( vol, vol, dg::INVERT<real_type>());
return vol;
}

}

} 
