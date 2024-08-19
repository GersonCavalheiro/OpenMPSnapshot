#include <stdlib.h>
#include <math.h>
#include "benchmark.h"

extern "C" {

#define HH_CONST_DT 0
#define HH_CONST_CM_INV 1
#define HH_CONST_GK 2
#define HH_CONST_E_K 3
#define HH_CONST_GNA 4
#define HH_CONST_E_NA 5
#define HH_CONST_GM 6
#define HH_CONST_REST 7
#define HH_CONST_I_INJ 8

FLOAT v_trap(FLOAT x, FLOAT y) {
return ((fabs(x / y) > 1e-6) ? (x / (EXP(x / y) - 1.0)) : (y * (1. - x / y / 2.)));
}
FLOAT calc_alpha_n(FLOAT v) { return (0.01 * v_trap(-(v + 55.), 10.)); }
FLOAT calc_beta_n(FLOAT v) { return (0.125 * EXP(-(v + 65.) / 80.)); }
FLOAT calc_alpha_m(FLOAT v) { return (0.1 * v_trap(-(v + 40.), 10.)); }
FLOAT calc_beta_m(FLOAT v) { return (4. * EXP(-(v + 65) / 18.)); }
FLOAT calc_alpha_h(FLOAT v) { return (0.07 * EXP(-(v + 65) / 20.)); }
FLOAT calc_beta_h(FLOAT v) { return (1. / (EXP(-(v + 35) / 10.) + 1.)); }
FLOAT calc_dm_core(FLOAT v, FLOAT m, FLOAT dt) {
return ((calc_alpha_m(v) * (1.0 - m) - calc_beta_m(v) * (m)) * dt);
}
FLOAT calc_dn_core(FLOAT v, FLOAT n, FLOAT dt) {
return ((calc_alpha_n(v) * (1.0 - n) - calc_beta_n(v) * (n)) * dt);
}
FLOAT calc_dh_core(FLOAT v, FLOAT h, FLOAT dt) {
return ((calc_alpha_h(v) * (1.0 - h) - calc_beta_h(v) * (h)) * dt);
}

FLOAT calc_dv_core(FLOAT v, FLOAT m, FLOAT n, FLOAT h, const FLOAT *const_table) {
FLOAT dv;
dv = const_table[HH_CONST_DT] * const_table[HH_CONST_CM_INV]
* (const_table[HH_CONST_GK] * n * n * n * n * (const_table[HH_CONST_E_K] - v)
+ const_table[HH_CONST_GNA] * m * m * m * h * (const_table[HH_CONST_E_NA] - v)
+ const_table[HH_CONST_GM] * (const_table[HH_CONST_REST] - v)
+ const_table[HH_CONST_I_INJ]);
return (dv);
}

FLOAT calc_hh_c_core(FLOAT *hh_v, FLOAT *hh_m, FLOAT *hh_n, FLOAT *hh_h, const FLOAT *const_table) {
FLOAT m, h, n, v;
m = *hh_m;
h = *hh_h;
n = *hh_n;
v = *hh_v;

m += calc_dm_core(v, m, const_table[HH_CONST_DT]);
n += calc_dn_core(v, m, const_table[HH_CONST_DT]);
h += calc_dh_core(v, m, const_table[HH_CONST_DT]);
v += calc_dv_core(v, m, n, h, const_table);

*hh_m = m;
*hh_n = n;
*hh_h = h;
return (v);
}

double benchmark_hh(int max_step, int n_cell) {
int step, i;
FLOAT *m_array, *n_array, *h_array;
FLOAT *v_array, *v_array_head;
FLOAT const_table[10];
double start_time, stop_time;

const_table[HH_CONST_DT] = 0.025;
const_table[HH_CONST_CM_INV] = 1.0;
const_table[HH_CONST_GK] = 36.0;
const_table[HH_CONST_E_K] = -77.0;
const_table[HH_CONST_GNA] = 120.0;
const_table[HH_CONST_E_NA] = 50.0;
const_table[HH_CONST_GM] = 0.3;
const_table[HH_CONST_REST] = -54.3;
const_table[HH_CONST_I_INJ] = 5;

v_array = (FLOAT *) malloc(max_step * n_cell * sizeof(FLOAT));
v_array_head = v_array;
init_array(n_cell, v_array, -65.0);
v_array += n_cell;

m_array = (FLOAT *) malloc(n_cell * sizeof(FLOAT));
n_array = (FLOAT *) malloc(n_cell * sizeof(FLOAT));
h_array = (FLOAT *) malloc(n_cell * sizeof(FLOAT));
init_array(n_cell, m_array, calc_alpha_m(-65.0) / (calc_alpha_m(-65.0) + calc_beta_m(-65.0)));
init_array(n_cell, n_array, calc_alpha_n(-65.0) / (calc_alpha_n(-65.0) + calc_beta_n(-65.0)));
init_array(n_cell, h_array, calc_alpha_h(-65.0) / (calc_alpha_h(-65.0) + calc_beta_h(-65.0)));

start_time = getTime();
for (step = 0; step < (max_step - 1); step++) {

#pragma omp parallel for
for (i = 0; i < n_cell; i++) {
v_array[step * n_cell + i] = calc_hh_c_core(v_array - n_cell, &(m_array[i]), &(n_array[i]),
&(h_array[i]), const_table);
}
}
stop_time = getTime();

free(m_array);
free(n_array);
free(h_array);
free(v_array_head);

return(stop_time - start_time);

}

}