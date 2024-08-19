#pragma once


#include "dg/file/nc_utilities.h"
#include "parameters.h"

namespace poet
{
std::array<dg::x::DVec,2> init_from_file( std::string file_name, const dg::x::CartesianGrid2d& grid, const Parameters& p, double& time){

#ifdef WITH_MPI
int rank;
MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
std::array<dg::x::DVec,2> y0;

dg::file::NC_Error_Handle errIN;
int ncidIN;
errIN = nc_open( file_name.data(), NC_NOWRITE, &ncidIN);
dg::file::WrappedJsonValue jsIN;
size_t length;
errIN = nc_inq_attlen( ncidIN, NC_GLOBAL, "inputfile", &length);
std::string input(length, 'x');
errIN = nc_get_att_text( ncidIN, NC_GLOBAL, "inputfile", &input[0]);
dg::file::string2Json( input, jsIN.asJson(), dg::file::comments::are_forbidden);
poet::Parameters pIN( jsIN);

DG_RANK0 std::cout << "RESTART from file "<<file_name<< std::endl;
DG_RANK0 std::cout << " file parameters:" << pIN.n <<" x "<<pIN.Nx<<" x "<<pIN.Ny<<std::endl;


dg::x::CartesianGrid2d grid_IN( 0, pIN.lx, 0, pIN.ly, pIN.n, pIN.Nx, pIN.Ny,  pIN.bc_x, pIN.bc_y
#ifdef WITH_MPI
, grid.communicator()
#endif 
);
dg::x::IHMatrix interpolateIN;
dg::x::HVec transferIN;


interpolateIN = dg::create::interpolation( grid, grid_IN);
transferIN = dg::evaluate(dg::zero, grid_IN);


#ifdef WITH_MPI
int dimsIN[2],  coordsIN[2];
int periods[2] = {false, false}; 
if( pIN.bc_x == dg::PER) periods[0] = true;
if( pIN.bc_y == dg::PER) periods[1] = true;
MPI_Cart_get( grid.communicator(), 2, dimsIN, periods, coordsIN);
size_t countIN[2] = {grid_IN.n()*(grid_IN.local().Ny()), grid_IN.n()*(grid_IN.local().Nx())};  
size_t startIN[2] = {coordsIN[1]*countIN[0], coordsIN[0]*countIN[1]};
#else 
size_t countIN[2] = { grid_IN.n()*grid_IN.Ny(), grid_IN.n()*grid_IN.Nx()};
size_t startIN[2] = { 0, 0};   
#endif 
std::vector<dg::x::HVec> transferOUTvec( 2, dg::evaluate( dg::zero, grid));

std::string namesIN[2] = {"restart_electrons", "restart_ions"};

int timeIDIN;
size_t size_time, count_time = 1;
errIN = nc_inq_dimid( ncidIN, "time", &timeIDIN);
errIN = nc_inq_dimlen(ncidIN, timeIDIN, &size_time);
errIN = nc_inq_varid( ncidIN, "time", &timeIDIN);
size_time -= 1;
errIN = nc_get_vara_double( ncidIN, timeIDIN, &size_time, &count_time, &time);
DG_RANK0 std::cout << " Current time = "<< time <<  std::endl;
for( unsigned i=0; i<2; i++)
{
int dataID;
errIN = nc_inq_varid( ncidIN, namesIN[i].data(), &dataID);
errIN = nc_get_vara_double( ncidIN, dataID, startIN, countIN,
#ifdef WITH_MPI
transferIN.data().data()
#else 
transferIN.data()
#endif 
);
dg::blas2::gemv( interpolateIN, transferIN, transferOUTvec[i]);

dg::blas1::plus( transferOUTvec[i], -1.0);
dg::assign( transferOUTvec[i], y0[i]); 
}
errIN = nc_close(ncidIN);
return y0;
}
}
