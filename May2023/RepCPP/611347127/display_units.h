#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _Output_Units_Fixed
double constexpr eV  = 1; char const _eV[] = "Ha"; 
double constexpr Ang = 1; char const _Ang[] = "Bohr"; 
#else
extern double eV;  extern char const *_eV;  
extern double Ang; extern char const *_Ang; 
#endif
double constexpr Kelvin = 315773.244215; char const _Kelvin[] = "Kelvin";

#ifdef __cplusplus
} 
#endif
