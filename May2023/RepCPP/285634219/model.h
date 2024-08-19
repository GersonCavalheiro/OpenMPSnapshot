

#pragma once
#include "anderson.h"
#include "charge.h"
#include "lsqt.h"
#include <random>
class Vector;

class Model
{
public:
Model(std::string input_dir);
~Model();
void initialize_state(Vector& random_state, int orbital);

bool calculate_vac0 = false;
bool calculate_vac = false;
bool calculate_msd = false;
bool calculate_spin = false;
bool calculate_ldos = false;

int number_of_random_vectors = 1;
int number_of_atoms = 0;
int max_neighbor = 0;
int number_of_pairs = 0;
int number_of_energy_points = 0;
int number_of_moments = 1000;
int number_of_steps_correlation = 0;
int number_of_local_orbitals = 0;
std::string input_dir;
real energy_max = 10;

real* energy;
real* time_step;
std::vector<int> local_orbitals;

int* neighbor_number;
int* neighbor_list;
real* xx;
real* potential;
real* hopping_real;
real* hopping_imag;

real volume;

private:
void print_started_reading(std::string filename);
void print_finished_reading(std::string filename);

void initialize_parameters();
void verify_parameters();
void initialize_energy();
void initialize_time();
void initialize_local_orbitals();

void initialize_neighbor();
void initialize_positions();
void initialize_potential();
void initialize_hopping();
void initialize_model_general();

void initialize_lattice_model();
void add_vacancies();
void create_random_numbers(int, int, int*);
void specify_vacancies(int*, int);
void find_new_atom_index(int*, int*, int);

bool requires_time = false;
bool use_lattice_model = false;

Anderson anderson;
Charge charge;

bool has_vacancy_disorder = false;
int number_of_vacancies;

int pbc[3];
real box_length[3];
std::vector<real> x, y, z;

std::mt19937 generator;
};
