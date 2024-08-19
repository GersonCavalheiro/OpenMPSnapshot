#pragma once

#include <costa/grid2grid/communication_data.hpp>
#include <costa/grid2grid/grid2D.hpp>
#include <costa/grid2grid/grid_layout.hpp>
#include <costa/grid2grid/comm_volume.hpp>

#include <mpi.h>

namespace costa {
template <typename T>
void transform(grid_layout<T> &initial_layout,
grid_layout<T> &final_layout,
MPI_Comm comm);

template <typename T>
void transform(grid_layout<T> &initial_layout,
grid_layout<T> &final_layout,
const char trans,
const T alpha, const T beta,
MPI_Comm comm);

template <typename T>
void transform(std::vector<layout_ref<T>>& initial_layouts,
std::vector<layout_ref<T>>& final_layouts,
MPI_Comm comm);

template <typename T>
void transform(std::vector<layout_ref<T>>& initial_layouts,
std::vector<layout_ref<T>>& final_layouts,
const char* trans,
const T* alpha, const T* beta,
MPI_Comm comm);

comm_volume communication_volume(assigned_grid2D& g_init,
assigned_grid2D& g_final,
char trans);

} 
