#pragma once

#include "quantum_numbers.h" 

int constexpr TRU=0, SMT=1;
int constexpr TRU_AND_SMT=2, TRU_ONLY=1;

template <int Pseudo> 
struct energy_level_t {
double* wave[Pseudo]; 
double* wKin[Pseudo]; 
double energy; 
double kinetic_energy; 
double occupation; 
char tag[8]; 
enn_QN_t nrn[Pseudo]; 
enn_QN_t enn; 
ell_QN_t ell; 
int8_t csv; 
}; 


typedef struct energy_level_t<TRU_AND_SMT> partial_wave_t; 
