#pragma once

#include <iostream>
#include <limits>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef USE_DOUBLE_PRECISION_DATA

#define data_type float

#define string_converter std::stof

#ifdef USE_MPI

#define mpi_data_type MPI_FLOAT
#endif

#else

#define data_type double

#define string_converter std::stod

#ifdef USE_MPI

#define mpi_data_type MPI_DOUBLE
#endif

#endif

#ifdef BIG
using array_size = std::size_t;
#else
using array_size = int;
#endif


template <typename T> class KNode {
T *data;  
int dims; 

KNode<T> *left, 
*right;     

bool is_root; 

public:

KNode(data_type *d, int dms, KNode<T> *l, KNode<T> *r, bool root)
: data{d}, dims{dms}, left{l}, right{r}, is_root{root} {}

KNode(KNode<T> &&other)
: data{other.data}, dims{other.dims}, left{other.left},
right{other.right}, is_root{other.is_root} {
other.data = nullptr;
other.left = nullptr;
other.right = nullptr;
other.dims = -1;
}

KNode()
: data{nullptr}, dims{0}, left{nullptr}, right{nullptr}, is_root{false} {}


~KNode() {
if (is_root)
delete[] data;
delete left;
delete right;
}


T get_data(int i) const { return data[i]; }

int get_dims() const { return dims; }


KNode<T> *get_left() const { return left; }

KNode<T> *get_right() const { return right; }
};
