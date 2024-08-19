#include "OpenMPDynamicSearch.h"
#include <omp.h>
int OpenMPDynamicSearch::dynamical_aperture_search() {
double x[2] = {-0.10, +0.10};
double y[2] = {+0.000, +0.003};
unsigned int nr_stable_points = 0;
#pragma omp parallel for num_threads(this->n_threads)
for(unsigned int i = 0; i < N_POINTS_X; ++i) {
for(unsigned int j = 0; j < N_POINTS_Y; ++j) {
double posx = x[0] + i*(x[1] - x[0])/(N_POINTS_X - 1);
double posy = y[0] + j*(y[1] - y[0])/(N_POINTS_Y - 1);
int index = i * N_POINTS_X + j;
this->result_set[index][0] = posx;
this->result_set[index][2] = posy;
this->result_set[index][1] = this->result_set[index][3] = this->result_set[index][4] = this->result_set[index][5] = 0;
for(unsigned int k = 0; k < this->turns; ++k) {
this->performOneTurn(this->result_set[index]);
}
if (this->testSolution(this->result_set[index])) {
#pragma omp atomic
nr_stable_points++;
printf ("%f %f %f %f %f %f (%d / %d) - (%d)\n", this->result_set[index][0], this->result_set[index][1],
this->result_set[index][2], this->result_set[index][3], this->result_set[index][4], this->result_set[index][5],
nr_stable_points , N_POINTS_X * N_POINTS_Y, index);
}
}
}
return nr_stable_points;
}
