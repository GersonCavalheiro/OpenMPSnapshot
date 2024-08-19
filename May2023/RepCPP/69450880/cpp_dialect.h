

#pragma once

#if   __cplusplus < 201103L
#define HYDRA_THRUST_CPP03
#define HYDRA_THRUST_CPP_DIALECT 2003
#elif __cplusplus < 201402L
#define HYDRA_THRUST_CPP11
#define HYDRA_THRUST_CPP_DIALECT 2011
#elif __cplusplus < 201703L
#define HYDRA_THRUST_CPP14
#define HYDRA_THRUST_CPP_DIALECT 2014
#else
#define HYDRA_THRUST_CPP17
#define HYDRA_THRUST_CPP_DIALECT 2017
#endif

