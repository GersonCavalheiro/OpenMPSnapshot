#pragma once
#ifndef _FILE_INCLUDED_BY_DG_
#pragma message( "The inclusion of file/nc_utilities.h is deprecated. Please use dg/file/nc_utilities.h")
#endif 

#include <netcdf.h>
#include "thrust/host_vector.h"

#include "dg/topology/grid.h"
#include "dg/topology/evaluation.h"
#ifdef MPI_VERSION
#include "dg/topology/mpi_grid.h"
#endif 

#include "easy_output.h"




namespace dg
{

namespace file
{
template<class value_type>
static inline nc_type getNCDataType(){ assert( false && "Type not supported!\n" ); return NC_DOUBLE; }
template<>
inline nc_type getNCDataType<double>(){ return NC_DOUBLE;}
template<>
inline nc_type getNCDataType<float>(){ return NC_FLOAT;}
template<>
inline nc_type getNCDataType<int>(){ return NC_INT;}
template<>
inline nc_type getNCDataType<unsigned>(){ return NC_UINT;}

template<class T>
inline int put_var_T( int ncid, int varID, T* data);
template<>
inline int put_var_T<float>( int ncid, int varID, float* data){
return nc_put_var_float( ncid, varID, data);
}
template<>
inline int put_var_T<double>( int ncid, int varID, double* data){
return nc_put_var_double( ncid, varID, data);
}


template<class T>
inline int define_real_time( int ncid, const char* name, int* dimID, int* tvarID)
{
int retval;
if( (retval = nc_def_dim( ncid, name, NC_UNLIMITED, dimID)) ){ return retval;}
if( (retval = nc_def_var( ncid, name, getNCDataType<T>(), 1, dimID, tvarID))){return retval;}
std::string t = "time since start"; 
if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
return retval;
}

static inline int define_time( int ncid, const char* name, int* dimID, int* tvarID)
{
return define_real_time<double>( ncid, name, dimID, tvarID);
}



static inline int define_limited_time( int ncid, const char* name, int size, int* dimID, int* tvarID)
{
int retval;
if( (retval = nc_def_dim( ncid, name, size, dimID)) ){ return retval;}
if( (retval = nc_def_var( ncid, name, NC_DOUBLE, 1, dimID, tvarID))){return retval;}
std::string t = "time since start"; 
if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
return retval;
}


template<class T>
inline int define_dimension( int ncid, int* dimID, const dg::RealGrid1d<T>& g, std::string name_dim = "x", std::string axis = "X")
{
int retval;
std::string long_name = name_dim+"-coordinate in Computational coordinate system";
thrust::host_vector<T> points = dg::create::abscissas( g);
if( (retval = nc_def_dim( ncid, name_dim.data(), points.size(), dimID)) ) { return retval;}
int varID;
if( (retval = nc_def_var( ncid, name_dim.data(), getNCDataType<T>(), 1, dimID, &varID))){return retval;}
if( (retval = put_var_T<T>( ncid, varID, points.data())) ){ return retval;}
retval = nc_put_att_text( ncid, *dimID, "axis", axis.size(), axis.data());
retval = nc_put_att_text( ncid, *dimID, "long_name", long_name.size(), long_name.data());
return retval;
}


template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::RealGrid1d<T>& g, std::array<std::string,2> name_dims = {"time","x"})
{
int retval;
retval = define_real_time<T>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID);
if(retval)
return retval;
return define_dimension( ncid, &dimsIDs[1], g, name_dims[1], "X");
}

template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, const dg::aRealTopology2d<T>& g, std::array<std::string,2> name_dims = {"y", "x"})
{
dg::RealGrid1d<T> gx( g.x0(), g.x1(), g.nx(), g.Nx());
dg::RealGrid1d<T> gy( g.y0(), g.y1(), g.ny(), g.Ny());
int retval;
retval = define_dimension( ncid, &dimsIDs[0], gy, name_dims[0], "Y");
if(retval)
return retval;
return define_dimension( ncid, &dimsIDs[1], gx, name_dims[1], "X");
}

template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aRealTopology2d<T>& g, std::array<std::string,3> name_dims = {"time", "y", "x"})
{
int retval;
retval = define_real_time<T>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID);
if(retval)
return retval;
return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2]});
}


template<class T>
inline int define_limtime_xy( int ncid, int* dimsIDs, int size, int* tvarID, const dg::aRealTopology2d<T>& g, std::array<std::string, 3> name_dims = {"time", "y", "x"})
{
int retval;
retval = define_limited_time( ncid, name_dims[0].data(), size, &dimsIDs[0], tvarID);
if(retval)
return retval;
return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2]});
}

template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, const dg::aRealTopology3d<T>& g, std::array<std::string, 3> name_dims = {"z", "y", "x"})
{
dg::RealGrid1d<T> gx( g.x0(), g.x1(), g.nx(), g.Nx());
dg::RealGrid1d<T> gy( g.y0(), g.y1(), g.ny(), g.Ny());
dg::RealGrid1d<T> gz( g.z0(), g.z1(), g.nz(), g.Nz());
int retval;
retval = define_dimension( ncid, &dimsIDs[0], gz, name_dims[0], "Z");
if(retval)
return retval;
retval = define_dimension( ncid, &dimsIDs[1], gy, name_dims[1], "Y");
if(retval)
return retval;
return define_dimension( ncid, &dimsIDs[2], gx, name_dims[2], "X");
}


template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aRealTopology3d<T>& g, std::array<std::string, 4> name_dims = {"time", "z", "y", "x"})
{
int retval;
retval = define_real_time<T>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID);
if(retval)
return retval;
return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2], name_dims[3]});
}


#ifdef MPI_VERSION

template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, const dg::aRealMPITopology2d<T>& g, std::array<std::string,2> name_dims = {"y", "x"})
{
return define_dimensions( ncid, dimsIDs, g.global(), name_dims);
}
template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aRealMPITopology2d<T>& g, std::array<std::string,3> name_dims = {"time", "y", "x"})
{
return define_dimensions( ncid, dimsIDs, tvarID, g.global(), name_dims);
}
template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, const dg::aRealMPITopology3d<T>& g, std::array<std::string, 3> name_dims = {"z", "y", "x"})
{
return define_dimensions( ncid, dimsIDs, g.global(), name_dims);
}
template<class T>
inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::aRealMPITopology3d<T>& g, std::array<std::string, 4> name_dims = {"time", "z", "y", "x"})
{
return define_dimensions( ncid, dimsIDs, tvarID, g.global(), name_dims);
}
#endif 

} 
} 
