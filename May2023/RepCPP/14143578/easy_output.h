#pragma once

#include <exception>
#include <netcdf.h>
#include "dg/topology/grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "dg/topology/mpi_grid.h"
#endif 



namespace dg
{
namespace file
{



struct NC_Error : public std::exception
{


NC_Error( int error): error_( error) {}

char const* what() const throw(){
return nc_strerror(error_);}
private:
int error_;
};


struct NC_Error_Handle
{

NC_Error_Handle operator=( int err)
{
NC_Error_Handle h;
return h(err);
}

NC_Error_Handle operator()( int err)
{
if( err)
throw NC_Error( err);
return *this;
}
};


template<class host_vector>
void put_var_double(int ncid, int varid, const dg::aTopology2d& grid,
const host_vector& data, bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[2] = {0,0}, count[2];
count[0] = grid.ny()*grid.Ny();
count[1] = grid.nx()*grid.Nx();
err = nc_put_vara_double( ncid, varid, start, count, data.data());
}


template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
const dg::aTopology2d& grid, const host_vector& data, bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[3] = {slice,0,0}, count[3];
count[0] = 1;
count[1] = grid.ny()*grid.Ny();
count[2] = grid.nx()*grid.Nx();
err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

template<class host_vector>
void put_var_double(int ncid, int varid, const dg::aTopology3d& grid,
const host_vector& data, bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[3] = {0,0,0}, count[3];
count[0] = grid.nz()*grid.Nz();
count[1] = grid.ny()*grid.Ny();
count[2] = grid.nx()*grid.Nx();
err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
const dg::aTopology3d& grid, const host_vector& data, bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[4] = {slice, 0,0,0}, count[4];
count[0] = 1;
count[1] = grid.nz()*grid.Nz();
count[2] = grid.ny()*grid.Ny();
count[3] = grid.nx()*grid.Nx();
err = nc_put_vara_double( ncid, varid, start, count, data.data());
}

#ifdef MPI_VERSION
template<class host_vector>
void put_var_double(int ncid, int varid, const dg::aMPITopology2d& grid,
const dg::MPI_Vector<host_vector>& data, bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[3] = {0,0}, count[2];
count[0] = grid.ny()*grid.local().Ny();
count[1] = grid.nx()*grid.local().Nx();
int rank, size;
MPI_Comm comm = grid.communicator();
MPI_Comm_rank( comm, &rank);
MPI_Comm_size( comm, &size);
if( !parallel)
{
MPI_Status status;
size_t local_size = grid.local().size();
std::vector<int> coords( size*2);
for( int rrank=0; rrank<size; rrank++)
MPI_Cart_coords( comm, rrank, 2, &coords[2*rrank]);
if(rank==0)
{
host_vector receive( data.data());
for( int rrank=0; rrank<size; rrank++)
{
if(rrank!=0)
MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
rrank, rrank, comm, &status);
start[0] = coords[2*rrank+1]*count[0],
start[1] = coords[2*rrank+0]*count[1],
err = nc_put_vara_double( ncid, varid, start, count,
receive.data());
}
}
else
MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
0, rank, comm);
MPI_Barrier( comm);
}
else
{
int coords[2];
MPI_Cart_coords( comm, rank, 2, coords);
start[0] = coords[1]*count[0],
start[1] = coords[0]*count[1],
err = nc_put_vara_double( ncid, varid, start, count,
data.data().data());
}
}

template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
const dg::aMPITopology2d& grid, const dg::MPI_Vector<host_vector>& data,
bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[3] = {slice, 0,0}, count[3];
count[0] = 1;
count[1] = grid.ny()*grid.local().Ny();
count[2] = grid.nx()*grid.local().Nx();
int rank, size;
MPI_Comm comm = grid.communicator();
MPI_Comm_rank( comm, &rank);
MPI_Comm_size( comm, &size);
if( parallel)
{
int coords[2];
MPI_Cart_coords( comm, rank, 2, coords);
start[1] = coords[1]*count[1],
start[2] = coords[0]*count[2],
err = nc_put_vara_double( ncid, varid, start, count,
data.data().data());
}
else
{
MPI_Status status;
size_t local_size = grid.local().size();
std::vector<int> coords( size*2);
for( int rrank=0; rrank<size; rrank++)
MPI_Cart_coords( comm, rrank, 2, &coords[2*rrank]);
if(rank==0)
{
host_vector receive( data.data());
for( int rrank=0; rrank<size; rrank++)
{
if(rrank!=0)
MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
rrank, rrank, comm, &status);
start[1] = coords[2*rrank+1]*count[1],
start[2] = coords[2*rrank+0]*count[2],
err = nc_put_vara_double( ncid, varid, start, count,
receive.data());
}
}
else
MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
0, rank, comm);
MPI_Barrier( comm);
}
}

template<class host_vector>
void put_var_double(int ncid, int varid,
const dg::aMPITopology3d& grid, const dg::MPI_Vector<host_vector>& data,
bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[3] = {0,0,0}, count[3];
count[0] = grid.nz()*grid.local().Nz();
count[1] = grid.ny()*grid.local().Ny();
count[2] = grid.nx()*grid.local().Nx();
int rank, size;
MPI_Comm comm = grid.communicator();
MPI_Comm_rank( comm, &rank);
MPI_Comm_size( comm, &size);
if( !parallel)
{
MPI_Status status;
size_t local_size = grid.local().size();
std::vector<int> coords( size*3);
for( int rrank=0; rrank<size; rrank++)
MPI_Cart_coords( comm, rrank, 3, &coords[3*rrank]);
if(rank==0)
{
host_vector receive( data.data());
for( int rrank=0; rrank<size; rrank++)
{
if(rrank!=0)
MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
rrank, rrank, comm, &status);
start[0] = coords[3*rrank+2]*count[0],
start[1] = coords[3*rrank+1]*count[1],
start[2] = coords[3*rrank+0]*count[2];
err = nc_put_vara_double( ncid, varid, start, count,
receive.data());
}
}
else
MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
0, rank, comm);
MPI_Barrier( comm);
}
else
{
int coords[3];
MPI_Cart_coords( comm, rank, 3, coords);
start[0] = coords[2]*count[0],
start[1] = coords[1]*count[1],
start[2] = coords[0]*count[2];
err = nc_put_vara_double( ncid, varid, start, count,
data.data().data());
}
}

template<class host_vector>
void put_vara_double(int ncid, int varid, unsigned slice,
const dg::aMPITopology3d& grid, const dg::MPI_Vector<host_vector>& data,
bool parallel = false)
{
file::NC_Error_Handle err;
size_t start[4] = {slice, 0,0,0}, count[4];
count[0] = 1;
count[1] = grid.nz()*grid.local().Nz();
count[2] = grid.ny()*grid.local().Ny();
count[3] = grid.nx()*grid.local().Nx();
int rank, size;
MPI_Comm comm = grid.communicator();
MPI_Comm_rank( comm, &rank);
MPI_Comm_size( comm, &size);
if( parallel)
{
int coords[3];
MPI_Cart_coords( comm, rank, 3, coords);
start[1] = coords[2]*count[1],
start[2] = coords[1]*count[2],
start[3] = coords[0]*count[3];
err = nc_put_vara_double( ncid, varid, start, count,
data.data().data());
}
else
{
MPI_Status status;
size_t local_size = grid.local().size();
std::vector<int> coords( size*3);
for( int rrank=0; rrank<size; rrank++)
MPI_Cart_coords( comm, rrank, 3, &coords[3*rrank]);
if(rank==0)
{
host_vector receive( data.data());
for( int rrank=0; rrank<size; rrank++)
{
if(rrank!=0)
MPI_Recv( receive.data(), local_size, MPI_DOUBLE,
rrank, rrank, comm, &status);
start[1] = coords[3*rrank+2]*count[1],
start[2] = coords[3*rrank+1]*count[2],
start[3] = coords[3*rrank+0]*count[3];
err = nc_put_vara_double( ncid, varid, start, count,
receive.data());
}
}
else
MPI_Send( data.data().data(), local_size, MPI_DOUBLE,
0, rank, comm);
MPI_Barrier( comm);
}
}
#endif 

}
}
