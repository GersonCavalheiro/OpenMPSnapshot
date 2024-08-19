#pragma once

#include <iostream>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h> 
#include "../enums.h"



namespace dg
{


static inline void mpi_init( int argc, char* argv[])
{
#ifdef _OPENMP
int provided, error;
error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
assert( error == MPI_SUCCESS && "Threaded MPI lib required!\n");
#else
MPI_Init(&argc, &argv);
#endif
}



static inline void mpi_init2d( dg::bc bcx, dg::bc bcy, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
int rank, size;
MPI_Comm_rank( MPI_COMM_WORLD, &rank);
MPI_Comm_size( MPI_COMM_WORLD, &size);
if(rank==0 && verbose)std::cout << "# MPI v"<<MPI_VERSION<<"."<<MPI_SUBVERSION<<std::endl;
int periods[2] = {false,false};
if( bcx == dg::PER) periods[0] = true;
if( bcy == dg::PER) periods[1] = true;
int np[2];
if( rank == 0)
{
int num_threads = 1;
#ifdef _OPENMP
num_threads = omp_get_max_threads( );
#endif 
if(verbose)std::cout << "# Type npx and npy\n";
is >> np[0] >> np[1];
if(verbose)std::cout << "# Computing with "
<< np[0]<<" x "<<np[1]<<" processes x "
<< num_threads<<" threads = "
<<size*num_threads<<" total"<<std::endl;
if( size != np[0]*np[1])
{
std::cerr << "ERROR: Process partition needs to match total number of processes!"<<std::endl;
MPI_Abort(MPI_COMM_WORLD, -1);
exit(-1);
}
}
MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
int num_devices=0;
cudaGetDeviceCount(&num_devices);
if(num_devices == 0)
{
std::cerr << "# No CUDA capable devices found on rank "<<rank<<std::endl;
MPI_Abort(MPI_COMM_WORLD, -1);
exit(-1);
}
int device = rank % num_devices; 
if(verbose)std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
cudaSetDevice( device);
#endif
}

static inline void mpi_init2d(unsigned& n, unsigned& Nx, unsigned& Ny, MPI_Comm comm, std::istream& is = std::cin, bool verbose = true  )
{
int rank;
MPI_Comm_rank( comm, &rank);
if( rank == 0)
{
if(verbose)std::cout << "# Type n, Nx and Ny\n";
is >> n >> Nx >> Ny;
if(verbose)std::cout<< "# On the grid "<<n <<" x "<<Nx<<" x "<<Ny<<std::endl;
}
MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, comm);
MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, comm);
MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, comm);
}


static inline void mpi_init2d( dg::bc bcx, dg::bc bcy, unsigned& n, unsigned& Nx, unsigned& Ny, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
mpi_init2d( bcx, bcy, comm, is, verbose);
mpi_init2d( n, Nx, Ny, comm, is, verbose);
}



static inline void mpi_init3d( dg::bc bcx, dg::bc bcy, dg::bc bcz, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
int rank, size;
MPI_Comm_rank( MPI_COMM_WORLD, &rank);
MPI_Comm_size( MPI_COMM_WORLD, &size);
int periods[3] = {false,false, false};
if( bcx == dg::PER) periods[0] = true;
if( bcy == dg::PER) periods[1] = true;
if( bcz == dg::PER) periods[2] = true;
int np[3];
if( rank == 0)
{
int num_threads = 1;
#ifdef _OPENMP
num_threads = omp_get_max_threads( );
#endif 
if(verbose) std::cout << "# Type npx and npy and npz\n";
is >> np[0] >> np[1]>>np[2];
if(verbose) std::cout << "# Computing with "
<< np[0]<<" x "<<np[1]<<" x "<<np[2] << " processes x "
<< num_threads<<" threads = "
<<size*num_threads<<" total"<<std::endl;
if( size != np[0]*np[1]*np[2])
{
std::cerr << "ERROR: Process partition needs to match total number of processes!"<<std::endl;
MPI_Abort(MPI_COMM_WORLD, -1);
exit(-1);
}
}
MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
int num_devices=0;
cudaGetDeviceCount(&num_devices);
if(num_devices == 0)
{
std::cerr << "# No CUDA capable devices found on rank "<<rank<<std::endl;
MPI_Abort(MPI_COMM_WORLD, -1);
exit(-1);
}
int device = rank % num_devices; 
if(verbose)std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
cudaSetDevice( device);
#endif
}

static inline void mpi_init3d(unsigned& n, unsigned& Nx, unsigned& Ny, unsigned& Nz, MPI_Comm comm, std::istream& is = std::cin, bool verbose = true  )
{
int rank;
MPI_Comm_rank( comm, &rank);
if( rank == 0)
{
if(verbose)std::cout << "# Type n, Nx and Ny and Nz\n";
is >> n >> Nx >> Ny >> Nz;
if(verbose)std::cout<< "# On the grid "<<n <<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<std::endl;
}
MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
MPI_Bcast( &Nz,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}


static inline void mpi_init3d( dg::bc bcx, dg::bc bcy, dg::bc bcz, unsigned& n, unsigned& Nx, unsigned& Ny, unsigned& Nz, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
mpi_init3d( bcx, bcy, bcz, comm, is, verbose);
mpi_init3d( n, Nx, Ny, Nz, comm, is, verbose);
}
} 
