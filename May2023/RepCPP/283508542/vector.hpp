#pragma once


#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py = pybind11;


#include <vector>
#include "accelerator/accelerator.hpp"
#include "vector3d.hpp"


namespace paracabs
{
namespace datatypes
{
template <typename type>
struct Vector
{
type*  dat = nullptr;          
type*  ptr = nullptr;          
bool   allocated = false;      
size_t allocated_size = 0;     
std::vector<type> vec;         

inline Vector ()
{
set_dat ();
}

inline void set_dat ()
{
#if PARACABS_USE_ACCELERATOR
if (copyContextAccelerator()) dat = ptr;
else                          dat = vec.data();
#else
dat = vec.data();
#endif
}

inline Vector (const Vector& v)
{
ptr            = v.ptr;
allocated      = false;
allocated_size = 0;
set_dat ();
}

inline Vector (const std::vector<type>& v) : vec(v)
{
copy_vec_to_ptr ();
set_dat ();
}

inline Vector (const size_t s) : vec(s)
{
copy_vec_to_ptr ();
set_dat ();
}

inline Vector (const size_t s, const type i) : vec(s, i)
{
copy_vec_to_ptr ();
set_dat ();
}

accel inline ~Vector () {deallocate();}

inline void allocate (const size_t size)
{
#if PARACABS_USE_ACCELERATOR
if (allocated_size != size)
{
if (allocated) deallocate();
ptr = (type*) paracabs::accelerator::malloc (size*sizeof(type));
allocated = true;
allocated_size = size;
set_dat ();
}
#endif
}

inline void deallocate ()
{
#if PARACABS_USE_ACCELERATOR
if (allocated)
{
paracabs::accelerator::free (ptr);
allocated = false;
allocated_size = 0;
}
#endif
}

inline void copy_vec_to_ptr ()
{
#if PARACABS_USE_ACCELERATOR
allocate (vec.size());
paracabs::accelerator::memcpy_to_accelerator (ptr, vec.data(), vec.size()*sizeof(type));
set_dat ();
#endif
}

inline void copy_ptr_to_vec ()
{
#if PARACABS_USE_ACCELERATOR
vec.resize (allocated_size);
paracabs::accelerator::memcpy_from_accelerator (vec.data(), ptr, vec.size()*sizeof(type));
set_dat ();
#endif
}

inline void resize (const size_t size)
{
vec.resize (size);
copy_vec_to_ptr ();
set_dat ();
}

inline size_t size () const
{
return vec.size();
}

accel inline type  operator[] (const size_t id) const {return dat[id];}
accel inline type &operator[] (const size_t id)       {return dat[id];}

inline void set_1D_array (py::array_t<type,   py::array::c_style | py::array::forcecast> arr);
inline void set_2D_array (py::array_t<float,  py::array::c_style | py::array::forcecast> arr);
inline void set_2D_array (py::array_t<double, py::array::c_style | py::array::forcecast> arr);
};


template<typename type>
inline void Vector<type> :: set_1D_array (py::array_t<type, py::array::c_style | py::array::forcecast> arr)
{
py::buffer_info buf = arr.request();

if (buf.ndim != 1)
{
throw std::runtime_error("Number of dimensions must be 1.");
}

type* buf_ptr = (type*) buf.ptr;

vec.resize (buf.shape[0]);

for (size_t i = 0; i < buf.shape[0]; i++)
{
vec[i] = buf_ptr[i];
}

copy_vec_to_ptr();
set_dat        ();
}


template<>
inline void Vector<Vector3D<float>> :: set_2D_array (py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
py::buffer_info buf = arr.request();

if (buf.ndim != 2)
{
throw std::runtime_error("Number of dimensions must be 2.");
}

if (buf.shape[1] != 3)
{
throw std::runtime_error("shape[1] should be 3.");
}

float* buf_ptr = (float*) buf.ptr;

vec.resize (buf.shape[0]);

for (size_t i = 0; i < buf.shape[0]; i++)
{
vec[i] = Vector3D<float> (buf_ptr[3*i  ],
buf_ptr[3*i+1],
buf_ptr[3*i+2] );
}

copy_vec_to_ptr();
set_dat        ();
}


template<>
inline void Vector<Vector3D<double>> :: set_2D_array (py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
py::buffer_info buf = arr.request();

if (buf.ndim != 2)
{
throw std::runtime_error("Number of dimensions must be 2.");
}

if (buf.shape[1] != 3)
{
throw std::runtime_error("shape[1] should be 3.");
}

double* buf_ptr = (double*) buf.ptr;

vec.resize (buf.shape[0]);

for (size_t i = 0; i < buf.shape[0]; i++)
{
vec[i] = Vector3D<double> (buf_ptr[3*i  ],
buf_ptr[3*i+1],
buf_ptr[3*i+2] );
}

copy_vec_to_ptr();
set_dat        ();
}
}
}
