#pragma once

#include <cmath>
#include "dg/backend/mpi_vector.h"
#include "dg/enums.h"
#include "grid.h"



namespace dg
{




template<class real_type>
struct RealMPIGrid2d;
template<class real_type>
struct RealMPIGrid3d;



template<class real_type>
struct aRealMPITopology2d
{
using value_type = real_type;
using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
using host_grid = RealMPIGrid2d<real_type>;


real_type x0() const { return g.x0();}

real_type x1() const { return g.x1(); }

real_type y0() const { return g.y0();}

real_type y1() const { return g.y1();}

real_type lx() const {return g.lx();}

real_type ly() const {return g.ly();}

real_type hx() const {return g.hx();}

real_type hy() const {return g.hy();}

unsigned n() const {return g.n();}
unsigned nx() const {return g.nx();}
unsigned ny() const {return g.ny();}

unsigned Nx() const { return g.Nx();}

unsigned Ny() const { return g.Ny();}

bc bcx() const {return g.bcx();}

bc bcy() const {return g.bcy();}

MPI_Comm communicator() const{return comm;}
const DLT<real_type>& dltx() const{return g.dltx();}
const DLT<real_type>& dlty() const{return g.dlty();}

unsigned size() const { return g.size();}

unsigned local_size() const { return l.size();}

void display( std::ostream& os = std::cout) const
{
os << "GLOBAL GRID \n";
g.display();
os << "LOCAL GRID \n";
l.display();
}


int pidOf( real_type x, real_type y) const;

void multiplyCellNumbers( real_type fx, real_type fy){
if( fx != 1 || fy != 1)
do_set(nx(), round(fx*(real_type)Nx()), ny(), round(fy*(real_type)Ny()));
}
void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) {
set( new_n, new_Nx, new_n, new_Ny);
}
void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny) {
check_division( new_Nx, new_Ny, g.bcx(), g.bcy());
if( new_nx == nx() && new_Nx == Nx() && new_ny == ny() && new_Ny == Ny())
return;
do_set( new_nx,new_Nx,new_ny,new_Ny);
}

bool local2globalIdx( int localIdx, int PID, int& globalIdx)const
{
if( localIdx < 0 || localIdx >= (int)size()) return -1;
int coords[2];
if( MPI_Cart_coords( comm, PID, 2, coords) != MPI_SUCCESS)
return false;
int lIdx0 = localIdx %(l.nx()*l.Nx());
int lIdx1 = localIdx /(l.nx()*l.Nx());
int gIdx0 = coords[0]*l.nx()*l.Nx()+lIdx0;
int gIdx1 = coords[1]*l.ny()*l.Ny()+lIdx1;
globalIdx = gIdx1*g.nx()*g.Nx() + gIdx0;
return true;
}

bool global2localIdx( int globalIdx, int& localIdx, int& PID)const
{
if( globalIdx < 0 || globalIdx >= (int)g.size()) return -1;
int coords[2];
int gIdx0 = globalIdx%(g.nx()*g.Nx());
int gIdx1 = globalIdx/(g.nx()*g.Nx());
coords[0] = gIdx0/(l.nx()*l.Nx());
coords[1] = gIdx1/(l.ny()*l.Ny());
int lIdx0 = gIdx0%(l.nx()*l.Nx());
int lIdx1 = gIdx1%(l.ny()*l.Ny());
localIdx = lIdx1*l.nx()*l.Nx() + lIdx0;
if( MPI_Cart_rank( comm, coords, &PID) == MPI_SUCCESS )
return true;
else
{
std::cout<<"Failed "<<PID<<"\n";
return false;
}
}

const RealGrid2d<real_type>& local() const {return l;}

const RealGrid2d<real_type>& global() const {return g;}
protected:
~aRealMPITopology2d() = default;

aRealMPITopology2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, MPI_Comm comm): g( gx, gy), l(gx, gy), comm(comm){
check_division( gx.N(), gy.N(), gx.bcx(), gy.bcx());
update_local();
}
aRealMPITopology2d(const aRealMPITopology2d& src) = default;
aRealMPITopology2d& operator=(const aRealMPITopology2d& src) = default;
virtual void do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny)=0;
private:
void check_division( unsigned Nx, unsigned Ny, bc bcx, bc bcy)
{
int rank, dims[2], periods[2], coords[2];
MPI_Cart_get( comm, 2, dims, periods, coords);
MPI_Comm_rank( comm, &rank);
if( rank == 0)
{
if(Nx%dims[0]!=0)
std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
assert( Nx%dims[0] == 0);
if(Ny%dims[1]!=0)
std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
assert( Ny%dims[1] == 0);
if( bcx == dg::PER) assert( periods[0] == true);
else assert( periods[0] == false);
if( bcy == dg::PER) assert( periods[1] == true);
else assert( periods[1] == false);
}
}
void update_local(){
int dims[2], periods[2], coords[2];
MPI_Cart_get( comm, 2, dims, periods, coords);
real_type x0 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)coords[0];
real_type x1 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)(coords[0]+1);
if( coords[0] == dims[0]-1)
x1 = g.x1();
real_type y0 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)coords[1];
real_type y1 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)(coords[1]+1);
if( coords[1] == dims[1]-1)
y1 = g.y1();
unsigned Nx = g.Nx()/dims[0];
unsigned Ny = g.Ny()/dims[1];
l = RealGrid2d<real_type>(
{ x0, x1, g.nx(), Nx, g.bcx()},
{ y0, y1, g.ny(), Ny, g.bcy()});
}
RealGrid2d<real_type> g, l; 
MPI_Comm comm; 
};



template<class real_type>
struct aRealMPITopology3d
{
using value_type = real_type;
using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
using host_grid = RealMPIGrid3d<real_type>;


real_type x0() const { return g.x0();}

real_type x1() const { return g.x1();}

real_type y0() const { return g.y0();}

real_type y1() const { return g.y1();}

real_type z0() const { return g.z0();}

real_type z1() const { return g.z1();}

real_type lx() const {return g.lx();}

real_type ly() const {return g.ly();}

real_type lz() const {return g.lz();}

real_type hx() const {return g.hx();}

real_type hy() const {return g.hy();}

real_type hz() const {return g.hz();}

unsigned n() const {return g.n();}
unsigned nx() const {return g.nx();}
unsigned ny() const {return g.ny();}
unsigned nz() const {return g.nz();}

unsigned Nx() const { return g.Nx();}

unsigned Ny() const { return g.Ny();}

unsigned Nz() const { return g.Nz();}

bc bcx() const {return g.bcx();}

bc bcy() const {return g.bcy();}

bc bcz() const {return g.bcz();}

MPI_Comm communicator() const{return comm;}

MPI_Comm get_perp_comm() const {return planeComm;}

const DLT<real_type>& dlt() const{return g.dlt();}
const DLT<real_type>& dltx() const{return g.dltx();}
const DLT<real_type>& dlty() const{return g.dlty();}
const DLT<real_type>& dltz() const{return g.dltz();}

unsigned size() const { return g.size();}

unsigned local_size() const { return l.size();}

void display( std::ostream& os = std::cout) const
{
os << "GLOBAL GRID \n";
g.display();
os << "LOCAL GRID \n";
l.display();
}

int pidOf( real_type x, real_type y, real_type z) const;
void multiplyCellNumbers( real_type fx, real_type fy){
if( fx != 1 || fy != 1)
do_set(nx(), round(fx*(real_type)Nx()), ny(),
round(fy*(real_type)Ny()), nz(), Nz());
}
void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) {
set(new_n,new_Nx,new_n,new_Ny,1,new_Nz);
}
void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz) {
check_division( new_Nx,new_Ny,new_Nz,g.bcx(),g.bcy(),g.bcz());
if( new_nx == nx() && new_Nx == Nx() && new_ny == ny() && new_Ny == Ny() && new_nz == nz() && new_Nz == Nz())
return;
do_set(new_nx,new_Nx,new_ny,new_Ny,new_nz,new_Nz);
}
bool local2globalIdx( int localIdx, int PID, int& globalIdx)const
{
if( localIdx < 0 || localIdx >= (int)size()) return false;
int coords[3];
if( MPI_Cart_coords( comm, PID, 3, coords) != MPI_SUCCESS)
return false;
int lIdx0 = localIdx %(l.nx()*l.Nx());
int lIdx1 = (localIdx /(l.nx()*l.Nx())) % (l.ny()*l.Ny());
int lIdx2 = localIdx / (l.nx()*l.ny()*l.Nx()*l.Ny());
int gIdx0 = coords[0]*l.nx()*l.Nx()+lIdx0;
int gIdx1 = coords[1]*l.ny()*l.Ny()+lIdx1;
int gIdx2 = coords[2]*l.nz()*l.Nz()+lIdx2;
globalIdx = (gIdx2*g.ny()*g.Ny() + gIdx1)*g.nx()*g.Nx() + gIdx0;
return true;
}
bool global2localIdx( int globalIdx, int& localIdx, int& PID)const
{
if( globalIdx < 0 || globalIdx >= (int)g.size()) return false;
int coords[3];
int gIdx0 = globalIdx%(g.nx()*g.Nx());
int gIdx1 = (globalIdx/(g.nx()*g.Nx())) % (g.ny()*g.Ny());
int gIdx2 = globalIdx/(g.nx()*g.ny()*g.Nx()*g.Ny());
coords[0] = gIdx0/(l.nx()*l.Nx());
coords[1] = gIdx1/(l.ny()*l.Ny());
coords[2] = gIdx2/(l.nz()*l.Nz());
int lIdx0 = gIdx0%(l.nx()*l.Nx());
int lIdx1 = gIdx1%(l.ny()*l.Ny());
int lIdx2 = gIdx2%(l.nz()*l.Nz());
localIdx = (lIdx2*l.ny()*l.Ny() + lIdx1)*l.nx()*l.Nx() + lIdx0;
if( MPI_Cart_rank( comm, coords, &PID) == MPI_SUCCESS )
return true;
else
return false;
}
const RealGrid3d<real_type>& local() const {return l;}
const RealGrid3d<real_type>& global() const {return g;}
protected:
~aRealMPITopology3d() = default;

aRealMPITopology3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz, MPI_Comm comm): g( gx, gy, gz), l(gx, gy, gz), comm(comm){
check_division( gx.N(), gy.N(), gz.N(), gx.bcx(), gy.bcx(), gz.bcx());
update_local();
int remain_dims[] = {true,true,false}; 
MPI_Cart_sub( comm, remain_dims, &planeComm);
}
aRealMPITopology3d(const aRealMPITopology3d& src) = default;
aRealMPITopology3d& operator=(const aRealMPITopology3d& src) = default;
virtual void do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz)=0;
private:
void check_division( unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz)
{
int rank, dims[3], periods[3], coords[3];
MPI_Cart_get( comm, 3, dims, periods, coords);
MPI_Comm_rank( comm, &rank);
if( rank == 0)
{
if(!(Nx%dims[0]==0))
std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
assert( Nx%dims[0] == 0);
if( !(Ny%dims[1]==0))
std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
assert( Ny%dims[1] == 0);
if( !(Nz%dims[2]==0))
std::cerr << "Nz "<<Nz<<" npz "<<dims[2]<<std::endl;
assert( Nz%dims[2] == 0);
if( bcx == dg::PER) assert( periods[0] == true);
else assert( periods[0] == false);
if( bcy == dg::PER) assert( periods[1] == true);
else assert( periods[1] == false);
if( bcz == dg::PER) assert( periods[2] == true);
else assert( periods[2] == false);
}
}
void update_local(){
int dims[3], periods[3], coords[3];
MPI_Cart_get( comm, 3, dims, periods, coords);
real_type x0 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)coords[0];
real_type x1 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)(coords[0]+1);
if( coords[0] == dims[0]-1)
x1 = g.x1();

real_type y0 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)coords[1];
real_type y1 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)(coords[1]+1);
if( coords[1] == dims[1]-1)
y1 = g.y1();

real_type z0 = g.z0() + g.lz()/(real_type)dims[2]*(real_type)coords[2];
real_type z1 = g.z0() + g.lz()/(real_type)dims[2]*(real_type)(coords[2]+1);
if( coords[2] == dims[2]-1)
z1 = g.z1();
unsigned Nx = g.Nx()/dims[0];
unsigned Ny = g.Ny()/dims[1];
unsigned Nz = g.Nz()/dims[2];

l = RealGrid3d<real_type>(
{ x0, x1, g.nx(), Nx, g.bcx()},
{ y0, y1, g.ny(), Ny, g.bcy()},
{ z0, z1, g.nz(), Nz, g.bcz()});
}
RealGrid3d<real_type> g, l; 
MPI_Comm comm, planeComm; 
};
template<class real_type>
int aRealMPITopology2d<real_type>::pidOf( real_type x, real_type y) const
{
int dims[2], periods[2], coords[2];
MPI_Cart_get( comm, 2, dims, periods, coords);
coords[0] = (unsigned)floor( (x-g.x0())/g.lx()*(real_type)dims[0] );
coords[1] = (unsigned)floor( (y-g.y0())/g.ly()*(real_type)dims[1] );
coords[0]=(coords[0]==dims[0]) ? coords[0]-1 :coords[0];
coords[1]=(coords[1]==dims[1]) ? coords[1]-1 :coords[1];
int rank;
if( MPI_Cart_rank( comm, coords, &rank) == MPI_SUCCESS )
return rank;
else
return -1;
}
template<class real_type>
int aRealMPITopology3d<real_type>::pidOf( real_type x, real_type y, real_type z) const
{
int dims[3], periods[3], coords[3];
MPI_Cart_get( comm, 3, dims, periods, coords);
coords[0] = (unsigned)floor( (x-g.x0())/g.lx()*(real_type)dims[0] );
coords[1] = (unsigned)floor( (y-g.y0())/g.ly()*(real_type)dims[1] );
coords[2] = (unsigned)floor( (z-g.z0())/g.lz()*(real_type)dims[2] );
coords[0]=(coords[0]==dims[0]) ? coords[0]-1 :coords[0];
coords[1]=(coords[1]==dims[1]) ? coords[1]-1 :coords[1];
coords[2]=(coords[2]==dims[2]) ? coords[2]-1 :coords[2];
int rank;
if( MPI_Cart_rank( comm, coords, &rank) == MPI_SUCCESS )
return rank;
else
return -1;
}
template<class real_type>
void aRealMPITopology2d<real_type>::do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) {
g.set(nx,Nx,ny,Ny);
update_local();
}
template<class real_type>
void aRealMPITopology3d<real_type>::do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) {
g.set(nx,Nx,ny,Ny,nz,Nz);
update_local();
}



template<class real_type>
struct RealMPIGrid2d: public aRealMPITopology2d<real_type>
{

RealMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm):
aRealMPITopology2d<real_type>( {x0,x1,n,Nx,dg::PER},
{y0,y1,n,Ny,dg::PER}, comm)
{ }


RealMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
aRealMPITopology2d<real_type>( {x0,x1,n,Nx,bcx}, {y0,y1,n,Ny,bcy},comm)
{ }
RealMPIGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, MPI_Comm comm): aRealMPITopology2d<real_type>(gx,gy,comm){ }
explicit RealMPIGrid2d( const aRealMPITopology2d<real_type>& src): aRealMPITopology2d<real_type>(src){}
private:
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
aRealMPITopology2d<real_type>::do_set(nx,Nx,ny,Ny);
}
};


template<class real_type>
struct RealMPIGrid3d : public aRealMPITopology3d<real_type>
{
RealMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm):
aRealMPITopology3d<real_type>( {x0, x1, n, Nx, dg::PER}, {y0, y1, n, Ny, dg::PER}, {z0, z1, 1, Nz, dg::PER}, comm )
{ }

RealMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
aRealMPITopology3d<real_type>( {x0, x1, n, Nx, bcx}, {y0, y1, n, Ny, bcy}, {z0, z1, 1, Nz, bcz}, comm )
{ }
RealMPIGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz, MPI_Comm comm): aRealMPITopology3d<real_type>(gx,gy,gz,comm){ }
explicit RealMPIGrid3d( const aRealMPITopology3d<real_type>& src): aRealMPITopology3d<real_type>(src){ }
private:
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) override final{
aRealMPITopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
}
};

using MPIGrid2d         = dg::RealMPIGrid2d<double>;
using MPIGrid3d         = dg::RealMPIGrid3d<double>;
using aMPITopology2d    = dg::aRealMPITopology2d<double>;
using aMPITopology3d    = dg::aRealMPITopology3d<double>;
namespace x{
using Grid2d          = MPIGrid2d      ;
using Grid3d          = MPIGrid3d      ;
using aTopology2d     = aMPITopology2d ;
using aTopology3d     = aMPITopology3d ;
}

}
