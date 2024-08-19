

#pragma once
#ifndef TESTING_GLOBAL_MATRIX_HPP
#define TESTING_GLOBAL_MATRIX_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_global_matrix_bad_args(void)
{
int safe_size = 100;

set_device_rocalution(device);
init_rocalution();

GlobalMatrix<T> mat;
GlobalVector<T> vec;

int* idata = nullptr;
T*   data  = nullptr;

allocate_host(safe_size, &idata);
allocate_host(safe_size, &data);

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.SetDataPtrCSR(
nullptr, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, nullptr, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_col != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &idata, nullptr, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_val != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &idata, &data, nullptr, &idata, &data, "", safe_size, safe_size),
".*Assertion.*ghost_row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &idata, &data, &idata, nullptr, &data, "", safe_size, safe_size),
".*Assertion.*ghost_col != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &idata, &data, &idata, &idata, nullptr, "", safe_size, safe_size),
".*Assertion.*ghost_val != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&null_int, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &null_int, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_col != (NULL|__null)*");
ASSERT_DEATH(
mat.SetDataPtrCSR(
&idata, &idata, &null_data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_val != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &idata, &data, &null_int, &idata, &data, "", safe_size, safe_size),
".*Assertion.*ghost_row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCSR(
&idata, &idata, &data, &idata, &null_int, &data, "", safe_size, safe_size),
".*Assertion.*ghost_col != (NULL|__null)*");
ASSERT_DEATH(
mat.SetDataPtrCSR(
&idata, &idata, &data, &idata, &idata, &null_data, "", safe_size, safe_size),
".*Assertion.*ghost_val != (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.SetDataPtrCOO(
nullptr, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_row != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, nullptr, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_col != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &idata, nullptr, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_val != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &idata, &data, nullptr, &idata, &data, "", safe_size, safe_size),
".*Assertion.*ghost_row != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &idata, &data, &idata, nullptr, &data, "", safe_size, safe_size),
".*Assertion.*ghost_col != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &idata, &data, &idata, &idata, nullptr, "", safe_size, safe_size),
".*Assertion.*ghost_val != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&null_int, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_row != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &null_int, &data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_col != (NULL|__null)*");
ASSERT_DEATH(
mat.SetDataPtrCOO(
&idata, &idata, &null_data, &idata, &idata, &data, "", safe_size, safe_size),
".*Assertion.*local_val != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &idata, &data, &null_int, &idata, &data, "", safe_size, safe_size),
".*Assertion.*ghost_row != (NULL|__null)*");
ASSERT_DEATH(mat.SetDataPtrCOO(
&idata, &idata, &data, &idata, &null_int, &data, "", safe_size, safe_size),
".*Assertion.*ghost_col != (NULL|__null)*");
ASSERT_DEATH(
mat.SetDataPtrCOO(
&idata, &idata, &data, &idata, &idata, &null_data, "", safe_size, safe_size),
".*Assertion.*ghost_val != (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.SetLocalDataPtrCSR(nullptr, &idata, &data, "", safe_size),
".*Assertion.*row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, nullptr, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, &idata, nullptr, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCSR(&null_int, &idata, &data, "", safe_size),
".*Assertion.*row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, &null_int, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, &idata, &null_data, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.SetGhostDataPtrCSR(nullptr, &idata, &data, "", safe_size),
".*Assertion.*row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, nullptr, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, &idata, nullptr, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCSR(&null_int, &idata, &data, "", safe_size),
".*Assertion.*row_offset != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, &null_int, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, &idata, &null_data, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.SetLocalDataPtrCOO(nullptr, &idata, &data, "", safe_size),
".*Assertion.*row != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, nullptr, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, &idata, nullptr, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCOO(&null_int, &idata, &data, "", safe_size),
".*Assertion.*row != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, &null_int, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, &idata, &null_data, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.SetGhostDataPtrCOO(nullptr, &idata, &data, "", safe_size),
".*Assertion.*row != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, nullptr, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, &idata, nullptr, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCOO(&null_int, &idata, &data, "", safe_size),
".*Assertion.*row != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, &null_int, &data, "", safe_size),
".*Assertion.*col != (NULL|__null)*");
ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, &idata, &null_data, "", safe_size),
".*Assertion.*val != (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(
mat.LeaveDataPtrCSR(&idata, &null_int, &null_data, &null_int, &null_int, &null_data),
".*Assertion.*local_row_offset == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCSR(&null_int, &idata, &null_data, &null_int, &null_int, &null_data),
".*Assertion.*local_col == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCSR(&null_int, &null_int, &data, &null_int, &null_int, &null_data),
".*Assertion.*local_val == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCSR(&null_int, &null_int, &null_data, &idata, &null_int, &null_data),
".*Assertion.*ghost_row_offset == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCSR(&null_int, &null_int, &null_data, &null_int, &idata, &null_data),
".*Assertion.*ghost_col == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCSR(&null_int, &null_int, &null_data, &null_int, &null_int, &data),
".*Assertion.*ghost_val == (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(
mat.LeaveDataPtrCOO(&idata, &null_int, &null_data, &null_int, &null_int, &null_data),
".*Assertion.*local_row == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCOO(&null_int, &idata, &null_data, &null_int, &null_int, &null_data),
".*Assertion.*local_col == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCOO(&null_int, &null_int, &data, &null_int, &null_int, &null_data),
".*Assertion.*local_val == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCOO(&null_int, &null_int, &null_data, &idata, &null_int, &null_data),
".*Assertion.*ghost_row == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCOO(&null_int, &null_int, &null_data, &null_int, &idata, &null_data),
".*Assertion.*ghost_col == (NULL|__null)*");
ASSERT_DEATH(
mat.LeaveDataPtrCOO(&null_int, &null_int, &null_data, &null_int, &null_int, &data),
".*Assertion.*ghost_val == (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.LeaveLocalDataPtrCSR(&idata, &null_int, &null_data),
".*Assertion.*row_offset == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveLocalDataPtrCSR(&null_int, &idata, &null_data),
".*Assertion.*col == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveLocalDataPtrCSR(&null_int, &null_int, &data),
".*Assertion.*val == (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.LeaveGhostDataPtrCSR(&idata, &null_int, &null_data),
".*Assertion.*row_offset == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveGhostDataPtrCSR(&null_int, &idata, &null_data),
".*Assertion.*col == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveGhostDataPtrCSR(&null_int, &null_int, &data),
".*Assertion.*val == (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.LeaveLocalDataPtrCOO(&idata, &null_int, &null_data),
".*Assertion.*row == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveLocalDataPtrCOO(&null_int, &idata, &null_data),
".*Assertion.*col == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveLocalDataPtrCOO(&null_int, &null_int, &data),
".*Assertion.*val == (NULL|__null)*");
}

{
int* null_int  = nullptr;
T*   null_data = nullptr;
ASSERT_DEATH(mat.LeaveGhostDataPtrCOO(&idata, &null_int, &null_data),
".*Assertion.*row == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveGhostDataPtrCOO(&null_int, &idata, &null_data),
".*Assertion.*col == (NULL|__null)*");
ASSERT_DEATH(mat.LeaveGhostDataPtrCOO(&null_int, &null_int, &data),
".*Assertion.*val == (NULL|__null)*");
}

{
GlobalVector<T>* null_vec = nullptr;
ASSERT_DEATH(mat.Apply(vec, null_vec), ".*Assertion.*out != (NULL|__null)*");
}

{
GlobalVector<T>* null_vec = nullptr;
ASSERT_DEATH(mat.ApplyAdd(vec, 1.0, null_vec), ".*Assertion.*out != (NULL|__null)*");
}

{
GlobalVector<T>* null_vec = nullptr;
ASSERT_DEATH(mat.ExtractInverseDiagonal(null_vec),
".*Assertion.*vec_inv_diag != (NULL|__null)*");
}

{
int               val;
int*              null_int = nullptr;
LocalVector<int>  lvint;
LocalVector<int>* null_vec = nullptr;
ASSERT_DEATH(mat.InitialPairwiseAggregation(0.1, val, null_vec, val, &null_int, val, 0),
".*Assertion.*G != (NULL|__null)*");
ASSERT_DEATH(mat.InitialPairwiseAggregation(0.1, val, &lvint, val, &idata, val, 0),
".*Assertion.*rG == (NULL|__null)*");
}

{
int               val;
int*              null_int = nullptr;
LocalVector<int>  lvint;
LocalVector<int>* null_vec = nullptr;
ASSERT_DEATH(mat.FurtherPairwiseAggregation(0.1, val, null_vec, val, &idata, val, 0),
".*Assertion.*G != (NULL|__null)*");
ASSERT_DEATH(mat.FurtherPairwiseAggregation(0.1, val, &lvint, val, &null_int, val, 0),
".*Assertion.*rG != (NULL|__null)*");
}

{
GlobalMatrix<T>  Ac;
ParallelManager  pm;
LocalVector<int> lvint;
GlobalMatrix<T>* null_mat = nullptr;
int*             null_int = nullptr;
ASSERT_DEATH(
mat.CoarsenOperator(null_mat, safe_size, safe_size, lvint, safe_size, idata, safe_size),
".*Assertion.*Ac != (NULL|__null)*");
ASSERT_DEATH(
mat.CoarsenOperator(&Ac, safe_size, safe_size, lvint, safe_size, null_int, safe_size),
".*Assertion.*rG != (NULL|__null)*");
}

free_host(&idata);
free_host(&data);

stop_rocalution();
}

#endif 
