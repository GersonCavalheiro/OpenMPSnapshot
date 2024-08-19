#pragma once


#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py = pybind11;


#include "datatypes/vector.hpp"
#include "accelerator/accelerator.hpp"


namespace paracabs
{
namespace datatypes
{
template <typename type>
struct Matrix : public Vector<type>
{
size_t nrows = 0;
size_t ncols = 0;


inline Matrix ()
{
Vector<type>::set_dat ();
}

inline Matrix (const Matrix& m)
{
nrows = m.nrows;
ncols = m.ncols;

Vector<type>::ptr            = m.ptr;
Vector<type>::allocated      = false;
Vector<type>::allocated_size = 0;
Vector<type>::set_dat ();
}

inline Matrix (const size_t nr, const size_t nc)
{
resize (nr, nc);
}

inline void resize (const size_t nr, const size_t nc)
{
nrows = nr;
ncols = nc;

Vector<type>::vec.resize (nrows*ncols);
Vector<type>::copy_vec_to_ptr ();
Vector<type>::set_dat ();
}

accel inline size_t nwarp () const
{
return ncols;
}

accel inline size_t index (const size_t id_r, const size_t id_c) const
{
return id_c + nwarp()*id_r;
}

accel inline type  operator() (const size_t id_r, const size_t id_c) const
{
return Vector<type>::dat[index(id_r, id_c)];
}

accel inline type &operator() (const size_t id_r, const size_t id_c)
{
return Vector<type>::dat[index(id_r, id_c)];
}

inline void set_2D_array (py::array_t<type, py::array::c_style | py::array::forcecast> arr)
{
py::buffer_info buf = arr.request();

if (buf.ndim != 2)
{
throw std::runtime_error("Number of dimensions must be 2.");
}

type* buf_ptr = (type*) buf.ptr;

nrows = buf.shape[0];
ncols = buf.shape[1];

Vector<type>::vec.resize (nrows*ncols);

for (size_t i = 0; i < nrows*ncols; i++)
{
Vector<type>::vec[i] = buf_ptr[i];
}

Vector<type>::copy_vec_to_ptr ();
Vector<type>::set_dat         ();
}
};
}
}
