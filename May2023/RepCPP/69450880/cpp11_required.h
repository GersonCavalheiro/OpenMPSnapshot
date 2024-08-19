

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config/cpp_dialect.h>

#ifndef HYDRA_THRUST_CPP11_REQUIRED_NO_ERROR
#  if HYDRA_THRUST_CPP_DIALECT < 2011 
#    error C++11 is required for this Thrust feature; please upgrade your compiler or pass the appropriate -std=c++XX flag to it.
#  endif
#endif

