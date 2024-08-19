

#pragma once

#ifndef HYDRA_THRUST_DEBUG
#  ifndef NDEBUG
#    if defined(DEBUG) || defined(_DEBUG)
#      define HYDRA_THRUST_DEBUG 1
#    endif 
#  endif 
#endif 

#if HYDRA_THRUST_DEBUG
#  ifndef __HYDRA_THRUST_SYNCHRONOUS
#    define __HYDRA_THRUST_SYNCHRONOUS 1
#  endif 
#endif 

