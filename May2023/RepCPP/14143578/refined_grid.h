#pragma once

#include "cusp/transpose.h"
#include "dg/backend/memory.h"
#include "dg/blas.h"
#include "grid.h"
#include "weights.h"
#include "interpolation.h"

#include "base_geometry.h"


namespace dg
{

namespace detail
{


template<class real_type>
thrust::host_vector<real_type> normalize_weights_and_compute_abscissas( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights)
{
unsigned Nx_new = weights.size()/g.n();
for( unsigned i=0;i<weights.size(); i++)
weights[i] *= (real_type)g.N()/(real_type)Nx_new;

thrust::host_vector<real_type> boundaries(Nx_new+1), abs(g.n()*Nx_new);
boundaries[0] = g.x0();
for( unsigned i=0; i<Nx_new; i++)
{
boundaries[i+1] = boundaries[i] + g.lx()/(real_type)Nx_new/weights[g.n()*i];
for( unsigned j=0; j<g.n(); j++)
{
abs[i*g.n()+j] =  (boundaries[i+1]+boundaries[i])/2. +
(boundaries[i+1]-boundaries[i])/2.*g.dlt().abscissas()[j];
}
}
return abs;
}

}



template<class real_type>
struct aRealRefinement1d
{

void generate( const RealGrid1d<real_type>& g_old, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const
{
weights.resize( N_new(g_old.N(), g_old.bcx()));
abscissas.resize( N_new(g_old.N(), g_old.bcx()));
do_generate(g_old,weights,abscissas);
}

unsigned N_new( unsigned N_old, bc bcx) const
{
return do_N_new(N_old, bcx);
}
virtual aRealRefinement1d* clone()const=0;
virtual ~aRealRefinement1d() = default;
protected:
aRealRefinement1d() = default;
aRealRefinement1d(const aRealRefinement1d& src) = default;
aRealRefinement1d& operator=(const aRealRefinement1d& src) = default;
private:
virtual void do_generate( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const =0;
virtual unsigned do_N_new( unsigned N_old, bc bcx) const =0;
};


template<class real_type>
struct RealIdentityRefinement : public aRealRefinement1d<real_type>
{
virtual RealIdentityRefinement* clone()const{return new RealIdentityRefinement();}
private:
virtual void do_generate( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final{
weights=dg::create::weights(g);
abscissas=dg::create::abscissas(g);
}
virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final{
return N_old;
}
};


template<class real_type>
struct RealLinearRefinement : public aRealRefinement1d<real_type>
{

RealLinearRefinement( unsigned multiple): m_(multiple){
assert( multiple>= 1);
}
virtual RealLinearRefinement* clone()const{return new RealLinearRefinement(*this);}
private:
unsigned m_;
virtual void do_generate( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
{
thrust::host_vector< real_type> left( g.n()*g.N()*m_, 1);
for( unsigned k=0; k<left.size(); k++)
left[k] = (real_type)m_;
weights = left;
abscissas = dg::detail::normalize_weights_and_compute_abscissas( g, weights);
}
virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final{
return N_old*m_;
}
};


template<class real_type>
struct RealFemRefinement : public aRealRefinement1d<real_type>
{

RealFemRefinement( unsigned multiple): m_M(multiple){
assert( multiple>= 1);
}
virtual RealFemRefinement* clone()const{return new RealFemRefinement(*this);}
private:
unsigned m_M;
std::vector<real_type> simpsons_weights( unsigned N, double h) const
{
std::vector<real_type> simpson( N, h);
if( N == 2)
{
simpson[0] = simpson[1] = h/2.;
return simpson;
}
simpson[0]=1./3. * h;
for( unsigned i=0; i<(N-3)/2; i++)
{
simpson[2*i+1] = 4./3. * h;
simpson[2*i+2] = 2./3. * h;
}
if( N%2 != 0)
{
simpson[N-2] = 4./3. * h;
simpson[N-1] = 1./3. * h;
}
else
{
simpson[N-4] = N==4 ? 3./8.*h : (1./3.+3./8.)*h;
simpson[N-3] = (9./8.)*h;
simpson[N-2] = (9./8.)*h;
simpson[N-1] = (3./8.)*h;
}
return simpson;
}
virtual void do_generate( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
{
thrust::host_vector<real_type> old = dg::create::abscissas(g);
if( m_M == 1)
{
abscissas = old;
weights = dg::evaluate( dg::one, g);
return;
}
abscissas.resize( g.size()*m_M);
weights.resize( g.size()*m_M);
dg::blas1::copy( 0., weights);
unsigned NLeft = m_M/2;
double hxleft = (old[0] - g.x0())/(double)(NLeft+1);
abscissas[0] = g.x0()+hxleft;
weights[0] = hxleft;
unsigned idx = 0;
std::vector<real_type> sim = simpsons_weights( NLeft+1, hxleft);
weights[idx] += sim[0];
for( unsigned k=1; k<=NLeft; k++)
{
idx++;
abscissas[idx] = abscissas[idx-1]+hxleft;
weights[idx] += sim[k];
}
unsigned NMiddle = m_M;
for( unsigned i=0; i<old.size()-1; i++)
{
double hxmiddle = (old[i+1] - old[i])/(double)NMiddle;
sim = simpsons_weights( NMiddle+1, hxmiddle);
weights[idx] += sim[0];
for( unsigned k=1; k<=NMiddle; k++)
{
idx++;
abscissas[idx] = abscissas[idx-1]+hxmiddle;
weights[idx] += sim[k];
}
}
unsigned NRight = m_M - 1 - m_M/2;
double hxright = (g.x1() - old[g.size()-1] )/(double)(NRight+1);
if( NRight > 0)
{
sim = simpsons_weights( NRight+1, hxright);
weights[idx] += sim[0];
for( unsigned k=1; k<=NRight; k++)
{
idx++;
abscissas[idx] = abscissas[idx-1]+hxright;
weights[idx] += sim[k];
}
}
weights[idx] += hxright;
RealGrid1d<real_type> nGrid( g.x0(), g.x1(), g.n(), g.N()*m_M);
thrust::host_vector<real_type> wrong_weights = dg::create::weights(nGrid);
dg::blas1::pointwiseDivide( wrong_weights, weights, weights);
}
virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final{
return N_old*m_M;
}
};


template<class real_type>
struct RealEquidistRefinement : public aRealRefinement1d<real_type>
{

RealEquidistRefinement( unsigned add_x, unsigned node, unsigned howmany=1): add_x_(add_x), node_(node), howm_(howmany){ }
virtual RealEquidistRefinement* clone()const{return new RealEquidistRefinement(*this);}
private:
unsigned add_x_, node_, howm_;
virtual void do_generate( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
{
if( add_x_ == 0 || howm_ == 0)
{
thrust::host_vector<real_type> w_( g.size(), 1);
abscissas = dg::create::abscissas(g);
weights = w_;
return;
}
weights = equidist_ref( add_x_, node_, g.n(), g.N(), g.bcx(), howm_);
abscissas = detail::normalize_weights_and_compute_abscissas( g, weights);
}
virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final
{
if( bcx == dg::PER) return N_old + 2*add_x_*howm_;
return N_old + add_x_*howm_;
}
thrust::host_vector<real_type> equidist_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx, unsigned howmany) const
{
assert( howm_ <= N);
assert( node_ <= N);
if( node_ != 0 && node_ != N)
assert( howm_ <= node_ && howm_ <= N-node_);
if( add_x_ == 0 || howm_ == 0)
{
thrust::host_vector<real_type> w_( n*N, 1);
return w_;
}
thrust::host_vector< real_type> left( n*N+n*add_x_*howm_, 1), right(left);
for( unsigned i=0; i<(add_x_+1)*howm_; i++)
for( unsigned k=0; k<n; k++)
left[i*n+k] = add_x_ + 1;
for( unsigned i=0; i<right.size(); i++)
right[i] = left[ (left.size()-1)-i];
thrust::host_vector< real_type> both( n*N+2*n*add_x_*howm_, 1);
for( unsigned i=0; i<left.size(); i++)
both[i] *= left[i];
for( unsigned i=0; i<right.size(); i++)
both[i+n*add_x_*howm_] *= right[i];
if(      node_ == 0     && bcx != dg::PER) { return left; }
else if( node_ == N && bcx != dg::PER) { return right; }
else if((node_ == N || node_ == 0) && bcx == dg::PER) { return both; }
else
{
thrust::host_vector<real_type> w_ = both;
for( unsigned i=0; i<both.size(); i++)
w_[((howm_*add_x_+node_)*n+i)%both.size()] = both[i];
return w_;
}
}

};


template<class real_type>
struct RealExponentialRefinement : public aRealRefinement1d<real_type>
{

RealExponentialRefinement( unsigned add_x, unsigned node): add_x_(add_x), node_(node) {}
virtual RealExponentialRefinement* clone()const{return new RealExponentialRefinement(*this);}
private:
unsigned add_x_, node_;
virtual void do_generate( const RealGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
{
if( add_x_ == 0)
{
thrust::host_vector<real_type> w_( g.size(), 1);
abscissas= dg::create::abscissas(g);
weights = w_;
return;
}
weights = exponential_ref( add_x_, node_, g.n(), g.N(), g.bcx());
abscissas = detail::normalize_weights_and_compute_abscissas( g, weights);
}
virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final
{
if( bcx == dg::PER) return N_old + 2*add_x_;
return N_old + add_x_;
}
thrust::host_vector<real_type> exponential_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx) const
{
if( add_x_ == 0)
{
thrust::host_vector<real_type> w_( n*N, 1);
return w_;
}
assert( node_ <= N);
thrust::host_vector< real_type> left( n*N+n*add_x_, 1), right(left);
for( unsigned k=0; k<n; k++)
left[k] = pow( 2, add_x_);
for( unsigned i=0; i<add_x_; i++)
for( unsigned k=0; k<n; k++)
left[(i+1)*n+k] = pow( 2, add_x_-i);
for( unsigned i=0; i<right.size(); i++)
right[i] = left[ (left.size()-1)-i];
thrust::host_vector< real_type> both( n*N+2*n*add_x_, 1);
for( unsigned i=0; i<left.size(); i++)
both[i] *= left[i];
for( unsigned i=0; i<right.size(); i++)
both[i+n*add_x_] *= right[i];
if(      node_ == 0     && bcx != dg::PER) { return left; }
else if( node_ == N && bcx != dg::PER) { return right; }
else if((node_ == N || node_ == 0) && bcx == dg::PER) { return both; }
else
{
thrust::host_vector<real_type> w_ = both;
for( unsigned i=0; i<both.size(); i++)
w_[((add_x_+node_)*n+i)%both.size()] = both[i];
return w_;
}
}
};

using aRefinement1d         = dg::aRealRefinement1d<double>;
using IdentityRefinement    = dg::RealIdentityRefinement<double>;
using FemRefinement         = dg::RealFemRefinement<double>;
using LinearRefinement      = dg::RealLinearRefinement<double>;
using EquidistRefinement    = dg::RealEquidistRefinement<double>;
using ExponentialRefinement = dg::RealExponentialRefinement<double>;



template<class real_type>
struct RealCartesianRefinedGrid2d : public dg::aRealGeometry2d<real_type>
{
RealCartesianRefinedGrid2d( const aRealRefinement1d<real_type>& refX, const aRealRefinement1d<real_type>& refY, real_type x0, real_type x1, real_type y0, real_type y1,
unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER) : dg::aGeometry2d( {x0, x1, n, refX.N_new(Nx,bcx), bcx}, {y0, y1, n, refY.N_new(Ny,bcy),bcy}), refX_(refX), refY_(refY), w_(2), a_(2)
{
construct_weights_and_abscissas(n,Nx,n,Ny);
}

virtual RealCartesianRefinedGrid2d* clone()const{return new RealCartesianRefinedGrid2d(*this);}
private:
ClonePtr<aRealRefinement1d<real_type>> refX_, refY_;
std::vector<thrust::host_vector<real_type> > w_, a_;
void construct_weights_and_abscissas(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny)
{
RealGrid1d<real_type> gx( this->x0(), this->x1(), nx, Nx, this->bcx());
RealGrid1d<real_type> gy( this->y0(), this->y1(), ny, Ny, this->bcy());
thrust::host_vector<real_type> wx, ax, wy, ay;
refX_->generate( gx, wx, ax);
refY_->generate( gy, wy, ay);
w_[0].resize(this->size()), w_[1].resize(this->size());
a_[0].resize(this->size()), a_[1].resize(this->size());
for( unsigned i=0; i<wy.size(); i++)
for( unsigned j=0; j<wx.size(); j++)
{
w_[0][i*wx.size()+j] = wx[j];
w_[1][i*wx.size()+j] = wy[i];
a_[0][i*wx.size()+j] = ax[j];
a_[1][i*wx.size()+j] = ay[i];
}
}
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny)override final{
aRealTopology2d<real_type>::do_set(nx,refX_->N_new(Nx,this->bcx()),ny,refY_->N_new(Ny, this->bcy()));
construct_weights_and_abscissas(nx,Nx,ny,Ny);
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final{
SparseTensor<thrust::host_vector<real_type> > t(*this);
t.values().push_back( w_[0]);
t.values().push_back( w_[1]);
dg::blas1::pointwiseDot( w_[0], w_[0], t.values()[2]);
dg::blas1::pointwiseDot( w_[1], w_[1], t.values()[3]);
t.idx(0,0)=2, t.idx(1,1)=3;
return t;
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const override final{
SparseTensor<thrust::host_vector<real_type> > t(*this);
t.values().push_back( w_[0]);
t.values().push_back( w_[1]);
t.idx(0,0)=2, t.idx(1,1)=3;
return t;
}
virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{
return a_;
}
};


template< class real_type>
struct RealCartesianRefinedGrid3d : public dg::aRealGeometry3d<real_type>
{
RealCartesianRefinedGrid3d( const aRealRefinement1d<real_type>& refX, const aRealRefinement1d<real_type>& refY, aRealRefinement1d<real_type>& refZ, real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1,
unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = dg::PER, bc bcy = dg::PER, bc bcz=dg::PER) :
dg::aGeometry3d( {x0, x1, n, refX.N_new(Nx,bcx), bcx}, { y0, y1, refY.N_new(Ny,bcy), bcy}, {z0,z1, 1, refZ.N_new(Nz,bcz), bcz}), refX_(refX), refY_(refY), refZ_(refZ), w_(3), a_(3)
{
construct_weights_and_abscissas(n, Nx, n, Ny,1, Nz);
}

virtual RealCartesianRefinedGrid3d* clone()const{return new RealCartesianRefinedGrid3d(*this);}
private:
ClonePtr<aRealRefinement1d<real_type>> refX_, refY_, refZ_;
std::vector<thrust::host_vector<real_type> > w_, a_;
void construct_weights_and_abscissas(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz)
{
RealGrid1d<real_type> gx( this->x0(), this->x1(), nx, Nx, this->bcx());
RealGrid1d<real_type> gy( this->y0(), this->y1(), ny, Ny, this->bcy());
RealGrid1d<real_type> gz( this->y0(), this->y1(), nz, Nz, this->bcz());
thrust::host_vector<real_type> w[3], a[3];
refX_->generate( gx, w[0], a[0]);
refY_->generate( gy, w[1], a[1]);
refZ_->generate( gz, w[2], a[2]);
w_[0].resize(this->size()), w_[1].resize(this->size()), w_[2].resize(this->size());
a_[0].resize(this->size()), a_[1].resize(this->size()), a_[2].resize(this->size());
for( unsigned s=0; s<w[2].size(); s++)
for( unsigned i=0; i<w[1].size(); i++)
for( unsigned j=0; j<w[0].size(); j++)
{
w_[0][(s*w[1].size()+i)*w[0].size()+j] = w[0][j];
w_[1][(s*w[1].size()+i)*w[0].size()+j] = w[1][i];
w_[2][(s*w[1].size()+i)*w[0].size()+j] = w[2][s];
a_[0][(s*w[1].size()+i)*w[0].size()+j] = a[0][j];
a_[1][(s*w[1].size()+i)*w[0].size()+j] = a[1][i];
a_[2][(s*w[1].size()+i)*w[0].size()+j] = a[1][s];
}
}
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) override final{
aRealTopology3d<real_type>::do_set(nx,refX_->N_new(Nx, this->bcx()),ny,refY_->N_new(Ny,this->bcy()), nz, refZ_->N_new(Nz,this->bcz()));
construct_weights_and_abscissas(nx, Nx, ny, Ny, nz, Nz);
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final {
SparseTensor<thrust::host_vector<real_type> > t(*this);
t.values().resize( 4, t.values()[0]);
dg::blas1::pointwiseDot( w_[0], w_[0], t.values()[1]);
dg::blas1::pointwiseDot( w_[1], w_[1], t.values()[2]);
dg::blas1::pointwiseDot( w_[2], w_[2], t.values()[3]);
t.idx(0,0)=1, t.idx(1,1)=2, t.idx(2,2)=3;
return t;
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const override final{
SparseTensor<thrust::host_vector<real_type> > t(*this);
t.values().push_back( w_[0]);
t.values().push_back( w_[1]);
t.values().push_back( w_[2]);
t.idx(0,0)=2, t.idx(1,1)=3, t.idx(2,2)=4;
return t;
}
virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final {
return a_;
}
};

using CartesianRefinedGrid2d = dg::RealCartesianRefinedGrid2d<double>;
using CartesianRefinedGrid3d = dg::RealCartesianRefinedGrid3d<double>;

}
