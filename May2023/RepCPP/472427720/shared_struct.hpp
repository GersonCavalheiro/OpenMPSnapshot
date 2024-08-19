#pragma once
#include "../include_for_cas766/PCH.hpp"
#include "../include_for_cas766/matrix_struct.hpp"

enum class MAT_TYPE
{
coo,
csc,
csr
};

enum class VEC_TYPE
{
atomic,
general
};

enum class SYNC_TYPE
{
mutex_var,
mutex_arr,
atomic,
none
};