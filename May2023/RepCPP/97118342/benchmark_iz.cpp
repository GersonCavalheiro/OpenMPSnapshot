#include <stdlib.h>
#include "benchmark.h"

extern "C" {

#define IZ_CONST_DT 0
#define IZ_CONST_A 1
#define IZ_CONST_B 2
#define IZ_CONST_C 3
#define IZ_CONST_D 4
#define IZ_CONST_I 5
#define IZ_CONST_V_1 6
#define IZ_CONST_V_2 7
#define IZ_CONST_V_3 8
#define IZ_CONST_TH 9

FLOAT calc_iz_c_core(FLOAT *iz_v, FLOAT *iz_u, FLOAT *iz_const) {

FLOAT dv, du;
FLOAT tmp_v = *iz_v;
FLOAT tmp_u = *iz_u;
FLOAT next_v;

dv = (iz_const[IZ_CONST_V_1] * tmp_v * tmp_v
+ iz_const[IZ_CONST_V_2] * tmp_v
+ iz_const[IZ_CONST_V_3]
- tmp_u
+ iz_const[IZ_CONST_I]
) * iz_const[IZ_CONST_DT];
du = (iz_const[IZ_CONST_A] * (iz_const[IZ_CONST_B] * tmp_v - tmp_u)) *
iz_const[IZ_CONST_DT];
next_v = tmp_v + dv;
tmp_u = tmp_u + du;

if (next_v > iz_const[IZ_CONST_TH]) {
next_v = iz_const[IZ_CONST_C];
tmp_u += iz_const[IZ_CONST_D];
}
*iz_u = tmp_u;
return next_v;
}

double benchmark_iz(int max_step, int n_cell) {
int step, i;
FLOAT *u_array;
FLOAT *v_array, *v_array_head;
FLOAT const_table[10];
double start_time, stop_time;

const_table[IZ_CONST_DT] = 0.1;
const_table[IZ_CONST_A] = 0.02;
const_table[IZ_CONST_B] = 0.2;
const_table[IZ_CONST_C] = -65.0;
const_table[IZ_CONST_D] = 8.0;
const_table[IZ_CONST_I] = 10.0;
const_table[IZ_CONST_V_1] = 0.04;
const_table[IZ_CONST_V_2] = 5.0;
const_table[IZ_CONST_V_3] = 140.0;
const_table[IZ_CONST_TH] = 30.0;

v_array = (FLOAT *) malloc(max_step * n_cell * sizeof(FLOAT));
v_array_head = v_array;
init_array(n_cell, v_array, -65.0);
v_array += n_cell;

u_array = (FLOAT *) malloc(n_cell * sizeof(FLOAT));
init_array(n_cell, u_array, -13.0);

start_time = getTime();
for (step = 0; step < (max_step - 1); step++) {

#pragma omp parallel for
for (i = 0; i < n_cell; i++) {
v_array[step * n_cell + i] = calc_iz_c_core(v_array - n_cell, &(u_array[i]), const_table);
}
}
stop_time = getTime();

free(u_array);
free(v_array_head);

return (stop_time - start_time);
}

}