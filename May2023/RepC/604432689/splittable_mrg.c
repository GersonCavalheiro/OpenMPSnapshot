#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "mod_arith.h"
#include "splittable_mrg.h"
typedef struct mrg_transition_matrix {
uint_fast32_t s, t, u, v, w;
uint_fast32_t a, b, c, d;
} mrg_transition_matrix;
#ifdef DUMP_TRANSITION_TABLE
static void mrg_update_cache(mrg_transition_matrix* restrict p) { 
p->a = mod_add(mod_mul_x(p->s), p->t);
p->b = mod_add(mod_mul_x(p->a), p->u);
p->c = mod_add(mod_mul_x(p->b), p->v);
p->d = mod_add(mod_mul_x(p->c), p->w);
}
static void mrg_make_identity(mrg_transition_matrix* result) {
result->s = result->t = result->u = result->v = 0;
result->w = 1;
mrg_update_cache(result);
}
static void mrg_make_A(mrg_transition_matrix* result) { 
result->s = result->t = result->u = result->w = 0;
result->v = 1;
mrg_update_cache(result);
}
static void mrg_multiply(const mrg_transition_matrix* restrict m, const mrg_transition_matrix* restrict n, mrg_transition_matrix* result) {
uint_least32_t rs = mod_mac(mod_mac(mod_mac(mod_mac(mod_mul(m->s, n->d), m->t, n->c), m->u, n->b), m->v, n->a), m->w, n->s);
uint_least32_t rt = mod_mac(mod_mac(mod_mac(mod_mac(mod_mul_y(mod_mul(m->s, n->s)), m->t, n->w), m->u, n->v), m->v, n->u), m->w, n->t);
uint_least32_t ru = mod_mac(mod_mac(mod_mac(mod_mul_y(mod_mac(mod_mul(m->s, n->a), m->t, n->s)), m->u, n->w), m->v, n->v), m->w, n->u);
uint_least32_t rv = mod_mac(mod_mac(mod_mul_y(mod_mac(mod_mac(mod_mul(m->s, n->b), m->t, n->a), m->u, n->s)), m->v, n->w), m->w, n->v);
uint_least32_t rw = mod_mac(mod_mul_y(mod_mac(mod_mac(mod_mac(mod_mul(m->s, n->c), m->t, n->b), m->u, n->a), m->v, n->s)), m->w, n->w);
result->s = rs;
result->t = rt;
result->u = ru;
result->v = rv;
result->w = rw;
mrg_update_cache(result);
}
static void mrg_power(const mrg_transition_matrix* restrict m, unsigned int exponent, mrg_transition_matrix* restrict result) {
mrg_transition_matrix current_power_of_2 = *m;
mrg_make_identity(result);
while (exponent > 0) {
if (exponent % 2 == 1) mrg_multiply(result, &current_power_of_2, result);
mrg_multiply(&current_power_of_2, &current_power_of_2, &current_power_of_2);
exponent /= 2;
}
}
#endif
#ifdef __MTA__
#pragma mta inline
#endif
static void mrg_apply_transition(const mrg_transition_matrix* restrict mat, const mrg_state* restrict st, mrg_state* r) {
#ifdef __MTA__
uint_fast64_t s = mat->s;
uint_fast64_t t = mat->t;
uint_fast64_t u = mat->u;
uint_fast64_t v = mat->v;
uint_fast64_t w = mat->w;
uint_fast64_t z1 = st->z1;
uint_fast64_t z2 = st->z2;
uint_fast64_t z3 = st->z3;
uint_fast64_t z4 = st->z4;
uint_fast64_t z5 = st->z5;
uint_fast64_t temp = s * z1 + t * z2 + u * z3 + v * z4;
r->z5 = mod_down(mod_down_fast(temp) + w * z5);
uint_fast64_t a = mod_down(107374182 * s + t);
uint_fast64_t sy = mod_down(104480 * s);
r->z4 = mod_down(mod_down_fast(a * z1 + u * z2 + v * z3) + w * z4 + sy * z5);
uint_fast64_t b = mod_down(107374182 * a + u);
uint_fast64_t ay = mod_down(104480 * a);
r->z3 = mod_down(mod_down_fast(b * z1 + v * z2 + w * z3) + sy * z4 + ay * z5);
uint_fast64_t c = mod_down(107374182 * b + v);
uint_fast64_t by = mod_down(104480 * b);
r->z2 = mod_down(mod_down_fast(c * z1 + w * z2 + sy * z3) + ay * z4 + by * z5);
uint_fast64_t d = mod_down(107374182 * c + w);
uint_fast64_t cy = mod_down(104480 * c);
r->z1 = mod_down(mod_down_fast(d * z1 + sy * z2 + ay * z3) + by * z4 + cy * z5);
#else
uint_fast32_t o1 = mod_mac_y(mod_mul(mat->d, st->z1), mod_mac4(0, mat->s, st->z2, mat->a, st->z3, mat->b, st->z4, mat->c, st->z5));
uint_fast32_t o2 = mod_mac_y(mod_mac2(0, mat->c, st->z1, mat->w, st->z2), mod_mac3(0, mat->s, st->z3, mat->a, st->z4, mat->b, st->z5));
uint_fast32_t o3 = mod_mac_y(mod_mac3(0, mat->b, st->z1, mat->v, st->z2, mat->w, st->z3), mod_mac2(0, mat->s, st->z4, mat->a, st->z5));
uint_fast32_t o4 = mod_mac_y(mod_mac4(0, mat->a, st->z1, mat->u, st->z2, mat->v, st->z3, mat->w, st->z4), mod_mul(mat->s, st->z5));
uint_fast32_t o5 = mod_mac2(mod_mac3(0, mat->s, st->z1, mat->t, st->z2, mat->u, st->z3), mat->v, st->z4, mat->w, st->z5);
r->z1 = o1;
r->z2 = o2;
r->z3 = o3;
r->z4 = o4;
r->z5 = o5;
#endif
}
#ifdef __MTA__
#pragma mta inline
#endif
static void mrg_step(const mrg_transition_matrix* mat, mrg_state* state) {
mrg_apply_transition(mat, state, state);
}
#ifdef __MTA__
#pragma mta inline
#endif
static void mrg_orig_step(mrg_state* state) { 
uint_fast32_t new_elt = mod_mac_y(mod_mul_x(state->z1), state->z5);
state->z5 = state->z4;
state->z4 = state->z3;
state->z3 = state->z2;
state->z2 = state->z1;
state->z1 = new_elt;
}
#ifndef DUMP_TRANSITION_TABLE
#include "mrg_transitions.c"
#endif
void mrg_skip(mrg_state* state, uint_least64_t exponent_high, uint_least64_t exponent_middle, uint_least64_t exponent_low) {
int byte_index;
for (byte_index = 0; exponent_low; ++byte_index, exponent_low >>= 8) {
uint_least8_t val = (uint_least8_t)(exponent_low & 0xFF);
if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
}
for (byte_index = 8; exponent_middle; ++byte_index, exponent_middle >>= 8) {
uint_least8_t val = (uint_least8_t)(exponent_middle & 0xFF);
if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
}
for (byte_index = 16; exponent_high; ++byte_index, exponent_high >>= 8) {
uint_least8_t val = (uint_least8_t)(exponent_high & 0xFF);
if (val != 0) mrg_step(&mrg_skip_matrices[byte_index][val], state);
}
}
#ifdef DUMP_TRANSITION_TABLE
const mrg_transition_matrix mrg_skip_matrices[][256] = {}; 
void dump_mrg(FILE* out, const mrg_transition_matrix* m) {
fprintf(out, "{%" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 ", %" PRIuFAST32 "}\n", m->s, m->t, m->u, m->v, m->w, m->a, m->b, m->c, m->d);
}
void dump_mrg_powers(void) {
int i, j;
mrg_transition_matrix transitions[192 / 8];
FILE* out = fopen("mrg_transitions.c", "w");
if (!out) {
fprintf(stderr, "dump_mrg_powers: could not open mrg_transitions.c for output\n");
exit (1);
}
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n");
fprintf(out, "\n\n");
fprintf(out, "#include \"splittable_mrg.h\"\n");
fprintf(out, "const mrg_transition_matrix mrg_skip_matrices[][256] = {\n");
for (i = 0; i < 192 / 8; ++i) {
if (i != 0) fprintf(out, ",");
fprintf(out, " {\n", i);
mrg_transition_matrix m;
mrg_make_identity(&m);
dump_mrg(out, &m);
if (i == 0) {
mrg_make_A(&transitions[i]);
} else {
mrg_power(&transitions[i - 1], 256, &transitions[i]);
}
fprintf(out, ",");
dump_mrg(out, &transitions[i]);
for (j = 2; j < 256; ++j) {
fprintf(out, ",");
mrg_power(&transitions[i], j, &m);
dump_mrg(out, &m);
}
fprintf(out, "} \n", i);
}
fprintf(out, "};\n");
fclose(out);
}
int main(int argc, char** argv) {
dump_mrg_powers();
return 0;
}
#endif
uint_fast32_t mrg_get_uint_orig(mrg_state* state) {
mrg_orig_step(state);
return state->z1;
}
double mrg_get_double_orig(mrg_state* state) {
return (double)mrg_get_uint_orig(state) * .000000000465661287524579692  +
(double)mrg_get_uint_orig(state) * .0000000000000000002168404346990492787 
;
}
void mrg_seed(mrg_state* st, const uint_fast32_t seed[5]) {
st->z1 = seed[0];
st->z2 = seed[1];
st->z3 = seed[2];
st->z4 = seed[3];
st->z5 = seed[4];
}
