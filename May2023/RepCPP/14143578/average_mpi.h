#pragma once

#include "mpi.h"
#include "average.h"
#include "mpi_grid.h"
#include "mpi_weights.h"


namespace dg{

template<class container>
void simple_mpi_average( unsigned nx, unsigned ny, const container& in0, const container& in1, container& out, MPI_Comm comm)
{
const double* in0_ptr = thrust::raw_pointer_cast( in0.data());
const double* in1_ptr = thrust::raw_pointer_cast( in1.data());
double* out_ptr = thrust::raw_pointer_cast( out.data());
dg::View<const container> in0_view( in0_ptr, nx), in1_view( in1_ptr, nx);
dg::View<container> out_view( out_ptr, nx);
dg::blas1::pointwiseDot( 1., in0_view, in1_view, 0, out_view);
for( unsigned i=1; i<ny; i++)
{
in0_view.construct( in0_ptr+i*nx, nx);
in1_view.construct( in1_ptr+i*nx, nx);
dg::blas1::pointwiseDot( 1., in0_view, in1_view, 1, out_view);
}
static thrust::host_vector<double> send_buf;
send_buf.resize( nx);
dg::assign( out_view, send_buf);
MPI_Allreduce(MPI_IN_PLACE, send_buf.data(), nx, MPI_DOUBLE, MPI_SUM, comm);
dg::assign( send_buf, out);
}


template< class container>
struct Average<MPI_Vector<container> >
{


Average( const aMPITopology2d& g, enum coo2d direction, std::string mode = "exact") : m_mode( mode)
{
m_nx = g.local().Nx()*g.nx(), m_ny = g.local().Ny()*g.ny();
m_w=dg::construct<MPI_Vector<container>>(dg::create::weights(g, direction));
m_temp = m_w;
int remain_dims[] = {false,false}; 
m_transpose = false;
unsigned size1d = 0;
if( direction == dg::coo2d::x)
{
dg::blas1::scal( m_w, 1./g.lx());
dg::blas1::scal( m_temp, 1./g.lx());
size1d = m_ny;
remain_dims[0] = true;
if( "simple" == mode)
dg::transpose( m_nx, m_ny, m_temp.data(), m_w.data());
}
else
{
m_transpose = true;
remain_dims[1] = true;
dg::blas1::scal( m_w, 1./g.ly());
dg::blas1::scal( m_temp, 1./g.ly());
if( "exact" == mode)
dg::transpose( m_nx, m_ny, m_temp.data(), m_w.data());
size1d = m_nx;
}

MPI_Cart_sub( g.communicator(), remain_dims, &m_comm);
exblas::mpi_reduce_communicator( m_comm, &m_comm_mod, &m_comm_mod_reduce);
for( unsigned i=0; i<2; i++)
remain_dims[i] = !remain_dims[i];
MPI_Comm comm2;
MPI_Cart_sub( g.communicator(), remain_dims, &comm2);
thrust::host_vector<double> t1d( size1d);
m_temp1d = MPI_Vector<container>( dg::construct<container>( t1d), comm2);
if( !("exact"==mode || "simple" == mode))
throw dg::Error( dg::Message( _ping_) << "Mode must either be exact or simple!");
}

Average( const aMPITopology3d& g, enum coo3d direction, std::string mode = "exact") : m_mode( mode)
{
m_w = dg::construct<MPI_Vector<container>>(dg::create::weights(g, direction));
m_temp = m_w;
m_transpose = false;
unsigned nx = g.nx()*g.local().Nx(), ny = g.ny()*g.local().Ny(), nz = g.nz()*g.local().Nz();
int remain_dims[] = {false,false,false};
m_transpose = false;
if( direction == dg::coo3d::x) {
dg::blas1::scal( m_w, 1./g.lx());
dg::blas1::scal( m_temp, 1./g.lx());
m_nx = nx, m_ny = ny*nz;
remain_dims[0] = true;
if( "simple" == mode)
dg::transpose( m_nx, m_ny, m_temp.data(), m_w.data());
}
else if( direction == dg::coo3d::z) {
m_transpose = true;
remain_dims[2] = true;
m_nx = nx*ny, m_ny = nz;
dg::blas1::scal( m_w, 1./g.lz());
dg::blas1::scal( m_temp, 1./g.lz());
if( "exact" == mode)
dg::transpose( m_nx, m_ny, m_temp.data(), m_w.data());
}
else if( direction == dg::coo3d::xy) {
dg::blas1::scal( m_w, 1./g.lx()/g.ly());
dg::blas1::scal( m_temp, 1./g.lx()/g.ly());
m_nx = nx*ny, m_ny = nz;
remain_dims[0] = remain_dims[1] = true;
if( "simple" == mode)
dg::transpose( m_nx, m_ny, m_temp.data(), m_w.data());
}
else if( direction == dg::coo3d::yz) {
m_transpose = true;
m_nx = nx, m_ny = ny*nz;
remain_dims[1] = remain_dims[2] = true;
dg::blas1::scal( m_w, 1./g.ly()/g.lz());
dg::blas1::scal( m_temp, 1./g.ly()/g.lz());
if( "exact" == mode)
dg::transpose( m_nx, m_ny, m_temp.data(), m_w.data());
}
else
std::cerr << "Warning: this direction is not implemented\n";
MPI_Cart_sub( g.communicator(), remain_dims, &m_comm);
exblas::mpi_reduce_communicator( m_comm, &m_comm_mod, &m_comm_mod_reduce);
for( unsigned i=0; i<3; i++)
remain_dims[i] = !remain_dims[i];
MPI_Comm comm2;
MPI_Cart_sub( g.communicator(), remain_dims, &comm2);
thrust::host_vector<double> t1d;
if(!m_transpose)
t1d = thrust::host_vector<double>( m_ny,0.);
else
t1d = thrust::host_vector<double>( m_nx,0.);
m_temp1d = MPI_Vector<container>( dg::construct<container>( t1d), comm2);
if( !("exact"==mode || "simple" == mode))
throw dg::Error( dg::Message( _ping_) << "Mode must either be exact or simple!");
}

void operator() (const MPI_Vector<container>& src, MPI_Vector<container>& res, bool extend = true)
{
if( !m_transpose)
{
if( "exact" == m_mode)
dg::mpi_average( m_nx, m_ny, src.data(), m_w.data(),
m_temp1d.data(), m_comm, m_comm_mod, m_comm_mod_reduce);
else
{
dg::transpose( m_nx, m_ny, src.data(), m_temp.data());
dg::simple_mpi_average( m_ny, m_nx, m_temp.data(), m_w.data(),
m_temp1d.data(), m_comm);
}

if( extend )
dg::extend_column( m_nx, m_ny, m_temp1d.data(), res.data());
else
res = m_temp1d;
}
else
{
if( "exact" == m_mode)
{
dg::transpose( m_nx, m_ny, src.data(), m_temp.data());
dg::mpi_average( m_ny, m_nx, m_temp.data(), m_w.data(),
m_temp1d.data(), m_comm, m_comm_mod, m_comm_mod_reduce);
}
else
dg::simple_mpi_average( m_nx, m_ny, src.data(), m_w.data(),
m_temp1d.data(), m_comm);

if( extend )
dg::extend_line( m_nx, m_ny, m_temp1d.data(), res.data());
else
res = m_temp1d;
}
}
private:
unsigned m_nx, m_ny;
MPI_Vector<container> m_w, m_temp, m_temp1d;
bool m_transpose;
MPI_Comm m_comm, m_comm_mod, m_comm_mod_reduce;
std::string m_mode;
};


}
