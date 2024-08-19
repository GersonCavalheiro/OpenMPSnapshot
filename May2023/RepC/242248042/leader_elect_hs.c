#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "ompdist/election.h"
#include "ompdist/vector.h"
#include "ompdist/utils.h"
#include "ompdist/queues.h"
#include "ompdist/msr.h"
#include "config.h"
typedef struct {
int starter_label;
int hops_left;
int direction;
int direction_changed;
int stop_initiating;
} message;
void generate_send_messages(process* processes,
int l,
int N,
queuelist* send_ql) {
DEBUG("Generating 2*%d messages for %d processes\n", N, N);
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < N; i++) {
process* p = processes+i;
if (p->status == -1)
continue;
message to_right = {i, 1 << l,  1, 0, 0};
message to_left  = {i, 1 << l, -1, 0, 0};
enqueue(send_ql, i, &to_right);
enqueue(send_ql, i, &to_left);
}
}
void propagate_messages(process* processes,
int l,
int N,
queuelist* send_ql,
queuelist* recv_ql) {
DEBUG("propagating messages on phase %d\n", l);
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < N; i++) {
DEBUG("i = %d\n", i);
process* p = processes+i;
while (!is_ql_queue_empty(send_ql, i)) {
message* m = dequeue(send_ql, i);
DEBUG("m->starter_label = %d\n", m->starter_label);
if (m->starter_label == i && m->hops_left != (1 << l)) {
if (m->stop_initiating)
p->status = -1;
else {
if (m->direction_changed)
p->status++;
else {
p->status = 3;
break;
}
}
continue;
}
if (m->hops_left == 0) {
DEBUG("zero hops left\n");
m->hops_left = 1 << l;
m->direction *= -1;
m->direction_changed = 1;
}
if (m->starter_label < i) {
m->hops_left = (1 << l) - m->hops_left;
m->direction *= -1;
m->direction_changed = 1;
m->stop_initiating = 1;
continue;
}
else {
m->hops_left--;
p->status = -1;
}
int next_label = (N + i + m->direction) % N;
enqueue(recv_ql, next_label, m);
}
}
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < N; i++) {
process* p = processes+i;
while (!is_ql_queue_empty(recv_ql, i)) {
enqueue(send_ql, i, dequeue(recv_ql, i));
}
}
}
int check_statuses(process* processes,
int N,
queuelist* send_ql) {
for (int i = 0; i < N; i++) {
process* p = processes+i;
if (p->status == 3)
return -i;
}
for (int i = 0; i < N; i++) {
if (!is_ql_queue_empty(send_ql, i))
return 1;
}
return 2;
}
void debug_display_queuelist(queuelist* ql) {
DEBUG("displaying the queuelist\n");
for (int i = 0; i < ql->N; i++) {
vector* v = ql->queues[i];
for (int j = ql->front[i]; j < v->used; j++) {
message* m = elem_at(v, j);
DEBUG("%d: {%d %d %2d %d %d}\n", i,
m->starter_label,
m->hops_left,
m->direction,
m->direction_changed,
m->stop_initiating);
}
}
}
int main(int argc, char* argv[]) {
int N;
process* processes;
int iterate;
int iterations = 1;
if ((iterate = input_through_argv(argc, argv))) {
FILE* in = fopen(argv[2], "r");
fscanf(in, "%d", &N);
processes = generate_nodes(N);
for (int i = 0; i < N; i++) {
int x;
fscanf(in, "%d", &x);
processes[i].id = processes[i].leader = processes[i].send = x;
}
sscanf(argv[3], "%d", &iterations);
}
else {
N = 16;
if (argc > 1)
sscanf(argv[1], "%d", &N);
processes = generate_nodes(N);
}
long long duration = 0;
double total_energy = 0;
int verification;
for (int i = 0; i < iterations; i++) {
process* ps = generate_nodes(N);
memcpy(ps, processes, sizeof(process)*N);
queuelist* recv_ql = new_queuelist(N, sizeof(message));
queuelist* send_ql = new_queuelist(N, sizeof(message));
begin_timer();
init_energy_measure();
int chosen_id = -1;
int l = 0;
int finished = 0;
while (!finished) {
l += 1;
DEBUG("starting phase %d\n", l);
generate_send_messages(ps, l, N, send_ql);
while (1) {
propagate_messages(ps, l, N, send_ql, recv_ql);
int status = check_statuses(ps, N, send_ql);
DEBUG("status = %d\n", status);
if (status == 1)
continue;
if (status == 2)
break;
if (status <= 0) {
chosen_id = -status;
set_leader(ps, N, chosen_id);
finished = 1;
break;
}
}
}
total_energy += total_energy_used();
duration += time_elapsed();
INFO("chosen leader: %d\n", chosen_id);
INFO("number of phases: %d\n", l);
free_queuelist(send_ql);
free_queuelist(recv_ql);
free(ps);
}
if (iterate)
printf("%.2lf %.2lf\n", ((double) duration) / iterations, total_energy / iterations);
return 0;
}
