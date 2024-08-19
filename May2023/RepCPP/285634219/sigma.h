

#pragma once
#include "lsqt.h"
class Model;
class Hamiltonian;
class Vector;

void find_dos(Model&, Hamiltonian&, Vector&, int);
void find_vac0(Model&, Hamiltonian&, Vector&);
void find_vac(Model&, Hamiltonian&, Vector&);
void find_msd(Model&, Hamiltonian&, Vector&);
void find_spin_polarization(Model&, Hamiltonian&, Vector&);
