#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "ompdist/vector.h"
#include "ompdist/graph.h"
#include "ompdist/graph_gen.h"
#include "ompdist/utils.h"
#include "ompdist/msr.h"
#include "config.h"
#define WHITE 0
#define GRAY  1
#define BLACK 2
typedef struct {
int color;
vector W;
int w;
int w_tilde;
int w_hat;
int active;
int s;
int s_hat;
int candidate;
int c;
int joined;
vector n2;
} payload;
int ceil_power_of_2(int x) {
if ((x & (x-1)) == 0)
return x;
int r = 1;
int p = 1;
while (r < x) {
p = r;
r <<= 1;
}
return p;
}
void initialize_graph(graph* g) {
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* data = malloc(sizeof(payload));
data->color = WHITE;
data->joined = 0;
initialize_vector(&data->W, sizeof(node*));
int* visited = malloc(g->N * sizeof(int));
memset(visited, 0, g->N * sizeof(int));
initialize_vector(&data->n2, sizeof(node*));
for (int j = 0; j < v->degree; j++) {
node* u = *((node**) elem_at(&v->neighbors, j));
if (visited[u->label])
continue;
visited[u->label] = 1;
append_to_vector(&data->n2, &u);
for (int k = 0; k < u->degree; k++) {
node* w = *((node**) elem_at(&u->neighbors, k));
if (w == v)
continue;
if (visited[w->label])
continue;
visited[w->label] = 1;
append_to_vector(&data->n2, &w);
}
}
append_to_vector(&data->n2, &v);
free(visited);
v->data = data;
}
}
int unjoined_nodes_exist(graph* g) {
int result = 0;
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* data = v->data;
if (data->color == WHITE) {
DEBUG("%d->color = WHITE\n", v->label);
result = 1;
}
}
return result;
}
void compute_w(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
v_data->W.used = 0;
if (v_data->color == WHITE)
append_to_vector(&v_data->W, &v);
for (int j = 0; j < v->degree; j++) {
node* u = *((node**) elem_at(&v->neighbors, j));
payload* u_data = u->data;
if (u_data->color == WHITE)
append_to_vector(&v_data->W, &u);
}
v_data->w = v_data->W.used;
DEBUG("%d->w = %d\n", v->label, v_data->w);
}
}
void compute_w_tilde(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
v_data->w_tilde = ceil_power_of_2(v_data->w);
DEBUG("%d->w_tilde = %d\n", v->label, v_data->w_tilde);
}
}
void compute_w_hat(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
int w_hat = 0;
for (int j = 0; j < v_data->n2.used; j++) {
node* u = *((node**) elem_at(&v_data->n2, j));
payload* u_data = u->data;
if (u_data->w_tilde > w_hat)
w_hat = u_data->w_tilde;
}
v_data->w_hat = w_hat;
DEBUG("%d->w_hat = %d\n", v->label, w_hat);
}
}
void compute_active(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
if (v_data->w_hat == v_data->w_tilde)
v_data->active = 1;
else
v_data->active = 0;
DEBUG("%d->active = %d\n", v->label, v_data->active);
}
}
void compute_s(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
int support = v_data->active;
for (int j = 0; j < v->degree; j++) {
node* u = *((node**) elem_at(&v->neighbors, j));
payload* u_data = u->data;
if (u_data->active)
support++;
}
DEBUG("%d->s = %d\n", v->label, support);
v_data->s = support;
}
}
void compute_s_hat(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
int s_hat = 0;
for (int j = 0; j < v_data->W.used; j++) {
node* u = *((node**) elem_at(&v_data->W, j));
payload* u_data = u->data;
if (u_data->s > s_hat)
s_hat = u_data->s;
}
DEBUG("%d->s_hat = %d\n", v->label, s_hat);
v_data->s_hat = s_hat;
}
}
void compute_candidacy(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
v_data->candidate = 0;
if (v_data->active) {
int r = rand() % (v_data->s_hat);
if (r == 0)
v_data->candidate = 1;
}
DEBUG("%d->candidate = %d\n", v->label, v_data->candidate);
}
}
void compute_c(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
v_data->c = 0;
for (int j = 0; j < v_data->W.used; j++) {
node* u = *((node**) elem_at(&v_data->W, j));
payload* u_data = u->data;
if (u_data->candidate)
v_data->c++;
}
DEBUG("%d->c = %d\n", v->label, v_data->c);
}
}
void compute_join(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->W.used == 0)
continue;
int sigma_c = 0;
for (int j = 0; j < v_data->W.used; j++) {
node* u = *((node**) elem_at(&v_data->W, j));
payload* u_data = u->data;
sigma_c += u_data->c;
}
if (v_data->candidate && sigma_c <= 3*v_data->w) {
DEBUG("%d joining\n", v->label);
v_data->color = BLACK;
v_data->joined = 1;
}
}
}
void colorize(graph* g) {
DEBUG("starting\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (v_data->color != WHITE)
continue;
for (int j = 0; j < v->degree; j++) {
node* u = *((node**) elem_at(&v->neighbors, j));
payload* u_data = u->data;
if (u_data->color == BLACK) {
v_data->color = GRAY;
break;
}
}
}
}
int verify_and_print_solution(graph* g) {
INFO("Vertices in the dominating set: ");
for (int i = 0; i < g->N; i++) {
node* v = elem_at(&g->vertices, i);
payload* v_data = v->data;
if (!v_data->joined)
continue;
INFO("%d ", v->label);
}
INFO("\n");
INFO("Solution verified to be correct because the program halted\n");
return 0;
}
int main(int argc, char* argv[]) {
int N;
int M;
graph* g;
int iterate;
int iterations = 1;
if ((iterate = input_through_argv(argc, argv))) {
FILE* in = fopen(argv[2], "r");
fscanf(in, "%d\n", &N);
g = new_graph(N, 0);
g->M = M = read_graph(g, in);
fclose(in);
sscanf(argv[3], "%d", &iterations);
}
else {
N = 16;
M = 64;
if (argc > 1) {
sscanf(argv[1], "%d", &N);
sscanf(argv[2], "%d", &M);
}
g = generate_new_connected_graph(N, M);
}
long long duration = 0;
double total_energy = 0;
int verification;
for (int i = 0; i < iterations; i++) {
begin_timer();
init_energy_measure();
initialize_graph(g);
while (unjoined_nodes_exist(g)) {
compute_w(g);
compute_w_tilde(g);
compute_w_hat(g);
compute_active(g);
compute_s(g);
compute_s_hat(g);
compute_candidacy(g);
compute_c(g);
compute_join(g);
colorize(g);
}
total_energy += total_energy_used();
duration += time_elapsed();
verification = verify_and_print_solution(g);
}
if (iterate)
printf("%.2lf %.2lf\n", ((double) duration) / iterations, total_energy / iterations);
return verification;
}
