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
template <typename type, typename XThreads>
struct MatrixTP : public Vector<type>, XThreads
{
size_t nrows = 0;
size_t ncols = 0;


inline MatrixTP ()
{
Vector<type>::set_dat ();
}

inline MatrixTP (const MatrixTP& m)
{
nrows    = m.nrows;
ncols    = m.ncols;

Vector<type>::ptr            = m.ptr;
Vector<type>::allocated      = false;
Vector<type>::allocated_size = 0;
Vector<type>::set_dat ();
}

inline MatrixTP (const size_t nr, const size_t nc)
{
MatrixTP<type, XThreads>::resize (nr, nc);
}

inline void resize (const size_t nr, const size_t nc)
{
nrows = nr;
ncols = nc;

Vector<type>::vec.resize (nrows*ncols*XThreads::tot_nthreads());
Vector<type>::copy_vec_to_ptr ();
Vector<type>::set_dat ();
}

accel inline size_t index (const size_t t, const size_t id_r, const size_t id_c) const
{
return id_c + ncols*(id_r + nrows*t);
}

accel inline type  operator() (const size_t id_r, const size_t id_c) const
{
return Vector<type>::dat[index(XThreads::thread_id(), id_r, id_c)];
}

accel inline type &operator() (const size_t id_r, const size_t id_c)
{
return Vector<type>::dat[index(XThreads::thread_id(), id_r, id_c)];
}

accel inline type  operator() (const size_t t, const size_t id_r, const size_t id_c) const
{
return Vector<type>::dat[index(t, id_r, id_c)];
}

accel inline type &operator() (const size_t t, const size_t id_r, const size_t id_c)
{
return Vector<type>::dat[index(t, id_r, id_c)];
}
};
}
}
