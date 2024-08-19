

#pragma once
#include "lsqt.h"
#include <random>

class Charge
{
public:
void add_impurities(
std::mt19937&,
int,
real*,
int*,
std::vector<real>&,
std::vector<real>&,
std::vector<real>&,
real*);
bool has = false;
int Ni;  
real W;  
real xi; 
private:
int Nx, Ny, Nz, Nxyz; 
real rc;              
real rc2;             
std::vector<int> cell_count;
std::vector<int> cell_count_sum;
std::vector<int> cell_contents;
std::vector<int> impurity_indices;
std::vector<real> impurity_strength;
void find_impurity_indices(std::mt19937&, int);
void find_impurity_strength(std::mt19937&);
void find_potentials(
int, real*, int*, std::vector<real>&, std::vector<real>&, std::vector<real>&, real*);
int find_cell_id(real, real, real, real);
void find_cell_id(real, real, real, real, int&, int&, int&, int&);
void find_cell_numbers(int*, real*);
void
find_cell_contents(int, int*, real*, std::vector<real>&, std::vector<real>&, std::vector<real>&);
int find_neighbor_cell(int, int, int, int, int, int, int);
};
