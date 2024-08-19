#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompdist/queues.h"
#include "ompdist/utils.h"
#include "config.h"
#define L ((5*N)/8 + 1)
#define H ((3*N)/4 + 1)
#define G ((7*N)/8)
typedef struct {
int b;
int vote;
int good;
int d;
int decided;
} processor;
typedef struct {
int from;
int vote;
} vote;
processor* new_processors(int N) {
DEBUG("allocating %d processors\n", N);
processor* processors = malloc(N * sizeof(processor));
DEBUG("initializing %d processors\n", N);
for (int i = 0; i < N; i++) {
processor* p = processors+i;
p->good = rand()%100 > 10; 
p->b = rand()%2;
p->vote = p->b;
p->d = 0;
p->decided = 0;
}
return processors;
}
int any_good_undecided_processor(processor* processors, int N) {
int ret = 0;
DEBUG("checking if there are any undecided processors\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < N; i++) {
processor* p = processors+i;
if (p->decided == 0)
ret = 1;
}
DEBUG("ret = %d\n", ret);
return ret;
}
void broadcast_vote(processor* processors, int N, queuelist* vote_ql) {
DEBUG("broadcasting votes\n");
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < N; i++) {
processor* p = processors+i;
for (int j = 0; j < N; j++) {
if (i == j)
continue;
vote v = {i, p->vote};
enqueue(vote_ql, j, &v);
}
}
}
void receive_votes(processor* processors, int N, queuelist* vote_ql) {
#pragma omp parallel for schedule(SCHEDULING_METHOD)
for (int i = 0; i < N; i++) {
processor* p = processors+i;
int yes = 0;
int no = 0;
while (!is_ql_queue_empty(vote_ql, i)) {
vote* v = dequeue(vote_ql, i);
if (v->vote)
yes++;
else
no++;
}
int maj = 1;
int tally = yes;
if (no > yes) {
maj = 0;
tally = no;
}
int threshold;
if (rand() % 2 == 0)
threshold = L;
else
threshold = H;
if (tally > threshold)
p->vote = maj;
else
p->vote = 0;
if (tally >= G) {
p->decided = 1;
p->d = maj;
}
}
}
int verify_and_print_solution(processor* processors, int N) {
int yes = 0;
int no = 0;
int good = 0;
for (int i = 0; i < N; i++) {
processor* p = processors+i;
if (p->good)
good++;
if (p->decided && p->good) {
if (p->d == 0)
no++;
else
yes++;
}
}
if (yes+no != good) {
WARN("incorrect: some processors haven't decided yet\n");
return 1;
}
if (yes != 0 && no != 0) {
WARN("incorrect: there's no consensus\n");
return 1;
}
INFO("correct: there's consensus: of the %d good processors, yes=%d, no=%d\n", good, yes, no);
return 0;
}
int main(int argc, char* argv[]) {
int N = 16;
if (argc > 1) {
sscanf(argv[1], "%d", &N);
}
srand(0);
processor* processors = new_processors(N);
queuelist* vote_ql = new_queuelist(N, sizeof(vote));
while (1) {
if (!any_good_undecided_processor(processors, N))
break;
broadcast_vote(processors, N, vote_ql);
receive_votes(processors, N, vote_ql);
}
return verify_and_print_solution(processors, N);
}
