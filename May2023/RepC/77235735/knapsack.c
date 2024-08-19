#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "app-desc.h"
#include "bots.h"
int best_so_far;
int number_of_tasks;
#pragma omp threadprivate(number_of_tasks)
int compare(struct item *a, struct item *b)
{
double c = ((double) a->value / a->weight) -
((double) b->value / b->weight);
if (c > 0) return -1;
if (c < 0) return 1;
return 0;
}
int read_input(const char *filename, struct item *items, int *capacity, int *n)
{
int i;
FILE *f;
if (filename == NULL) filename = "\0";
f = fopen(filename, "r");
if (f == NULL) {
fprintf(stderr, "open_input(\"%s\") failed\n", filename);
return -1;
}
fscanf(f, "%d", n);
fscanf(f, "%d", capacity);
for (i = 0; i < *n; ++i)
fscanf(f, "%d %d", &items[i].value, &items[i].weight);
fclose(f);
qsort(items, *n, sizeof(struct item), (int (*)(const void *, const void *)) compare);
return 0;
}
#if defined(IF_CUTOFF)
void knapsack_par(struct item *e, int c, int n, int v, int *sol, int l)
{
int with, without, best;
double ub;
number_of_tasks++;
if (c < 0)
{
*sol = INT_MIN;
return;
}
if (n == 0 || c == 0)
{
*sol = v;
return;
}
ub = (double) v + c * e->value / e->weight;
if (ub < best_so_far) {
*sol = INT_MIN;
return;
}
#pragma omp task untied firstprivate(e,c,n,v,l) shared(without) if (l < bots_cutoff_value)
knapsack_par(e + 1, c, n - 1, v, &without,l+1);
#pragma omp task untied firstprivate(e,c,n,v,l) shared(with)  if (l < bots_cutoff_value)
knapsack_par(e + 1, c - e->weight, n - 1, v + e->value, &with,l+1);
#pragma omp taskwait
best = with > without ? with : without;
if (best > best_so_far) best_so_far = best;
*sol = best;
}
#elif defined (MANUAL_CUTOFF)
void knapsack_par(struct item *e, int c, int n, int v, int *sol, int l)
{
int with, without, best;
double ub;
number_of_tasks++;
if (c < 0)
{
*sol = INT_MIN;
return;
}
if (n == 0 || c == 0)
{
*sol = v;
return;
}
ub = (double) v + c * e->value / e->weight;
if (ub < best_so_far) {
*sol = INT_MIN;
return;
}
if (l < bots_cutoff_value)
{
#pragma omp task untied firstprivate(e,c,n,v,l) shared(without)
knapsack_par(e + 1, c, n - 1, v, &without,l+1);
#pragma omp task untied firstprivate(e,c,n,v,l) shared(with)
knapsack_par(e + 1, c - e->weight, n - 1, v + e->value, &with,l+1);
#pragma omp taskwait
}
else
{
knapsack_seq(e + 1, c, n - 1, v, &without);
knapsack_seq(e + 1, c - e->weight, n - 1, v + e->value, &with);
}
best = with > without ? with : without;
if (best > best_so_far) best_so_far = best;
*sol = best;
}
#else
void knapsack_par(struct item *e, int c, int n, int v, int *sol, int l)
{
int with, without, best;
double ub;
number_of_tasks++;
if (c < 0)
{
*sol = INT_MIN;
return;
}
if (n == 0 || c == 0)
{
*sol = v;
return;
}
ub = (double) v + c * e->value / e->weight;
if (ub < best_so_far) {
*sol = INT_MIN;
return;
}
#pragma omp task untied firstprivate(e,c,n,v,l) shared(without)
knapsack_par(e + 1, c, n - 1, v, &without,l+1);
#pragma omp task untied firstprivate(e,c,n,v,l) shared(with)
knapsack_par(e + 1, c - e->weight, n - 1, v + e->value, &with,l+1);
#pragma omp taskwait
best = with > without ? with : without;
if (best > best_so_far) best_so_far = best;
*sol = best;
}
#endif
void knapsack_seq(struct item *e, int c, int n, int v, int *sol)
{
int with, without, best;
double ub;
number_of_tasks++;
if (c < 0)
{
*sol = INT_MIN;
return;
}
if (n == 0 || c == 0)
{
*sol = v;
return;
}
ub = (double) v + c * e->value / e->weight;
if (ub < best_so_far) {
*sol = INT_MIN;
return;
}
knapsack_seq(e + 1, c, n - 1, v, &without);
knapsack_seq(e + 1, c - e->weight, n - 1, v + e->value, &with);
best = with > without ? with : without;
if (best > best_so_far) best_so_far = best;
*sol = best;
}
void knapsack_main_par (struct item *e, int c, int n, int *sol)
{
best_so_far = INT_MIN;
#pragma omp parallel
{
number_of_tasks = 0;
#pragma omp single
#pragma omp task untied
{
knapsack_par(e, c, n, 0, sol, 0);
}
#pragma omp critical
bots_number_of_tasks += number_of_tasks;
}
if (bots_verbose_mode) printf("Best value for parallel execution is %d\n\n", *sol);
}
void knapsack_main_seq (struct item *e, int c, int n, int *sol)
{
best_so_far = INT_MIN;
number_of_tasks = 0;
knapsack_seq(e, c, n, 0, sol);
if (bots_verbose_mode) printf("Best value for sequential execution is %d\n\n", *sol);
}
int  knapsack_check (int sol_seq, int sol_par)
{
if (sol_seq == sol_par) return BOTS_RESULT_SUCCESSFUL;
else return BOTS_RESULT_UNSUCCESSFUL;
}
