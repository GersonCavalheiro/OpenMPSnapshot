
#if defined(_OPENMP)

#include <omp.h>

#endif

#include "min_examples.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

using namespace std;

void parallel() {
}

void single() {
#pragma omp parallel
{
printf("Th %i: A - Run this in parallel.\n", omp_get_thread_num());
#pragma omp single
{
printf("--- Th %i: Region run only by one thread, the other are waiting before starting B\n",
omp_get_thread_num());
}
printf("Th %i: B - This is parallel again.\n", omp_get_thread_num());
#pragma omp single nowait
{
printf("--- Th %i: Another region run only by one thread (also not necessarily master)\n",
omp_get_thread_num());
}
printf("Th %i: C - This is parallel again, without the other threads waiting for the single region.\n",
omp_get_thread_num());
}
}

void master() {
#pragma omp parallel
{
printf("Th %i: A - Run this in parallel.\n", omp_get_thread_num());
#pragma omp master
{ printf("--- Th %i: Region run only by the master thread\n", omp_get_thread_num()); }
printf("Th %i: B - This is also parallel.\n", omp_get_thread_num());
#pragma omp master
{
printf("--- Th %i: Another region run only by one thread, because of barrier the others will wait to do C\n",
omp_get_thread_num());
}
#pragma omp barrier
printf("Th %i: C - This is parallel again, after the barrier.\n", omp_get_thread_num());
}
}

void runtime_functions() {
#ifdef _OPENMP
int cores = omp_get_num_procs();
int max_threads = omp_get_max_threads();
int master_id = omp_get_thread_num();
int team_id = omp_get_team_num();
printf("Available cores: %i\n", cores);
printf("Max available threads: %i\n", max_threads);
printf("Master thread: %i\n", master_id);

printf("\n----------\n");
printf("Parallel region!\n");
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
int team_id = omp_get_team_num();
int num_threads = omp_get_num_threads();

printf("Thread %i of %i, in team %i\n", thread_id, num_threads, team_id);
if (thread_id == master_id) {
int num_teams = omp_get_num_teams();
int team_size = omp_get_team_size(omp_get_level());

printf("Per default all available threads are used. Threads in this section: %i\n", num_threads);
printf("Threads are organised in teams, here: %i team(s) à %i threads\n", num_teams, team_size);
}
}

printf("\n----------\n");
printf("Parallel region with more threads!\n");
#pragma omp parallel num_threads(5)
{
int thread_id = omp_get_thread_num();
int num_threads = omp_get_num_threads();
printf("Thread %i of %i\n", thread_id, num_threads);
}

printf("Parallel region with normal threads!\n");
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
int num_threads = omp_get_num_threads();
printf("Thread %i of %i\n", thread_id, num_threads);
}

printf("\n----------\n");
printf("Parallel region with less threads!\n");
omp_set_num_threads(2);
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
int num_threads = omp_get_num_threads();
printf("Thread %i of %i\n", thread_id, num_threads);
}

printf("\n----------\n");
printf("Sequential region!\n");
int thread_id = omp_get_thread_num();
int num_threads = omp_get_num_threads();
printf("Only master thread remains: Thread %i of %i, in team %i\n", thread_id, num_threads, team_id);
omp_set_num_threads(4);
#else
printf("No OpenMP supported");
#endif
}

void parallel_for() {
#pragma omp parallel for
for (int i = 0; i < 6; i++)
printf("Th %i: iteration %i\n", omp_get_thread_num(), i);
printf("\n");
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < 6; i++)
printf("Th %i: iteration %i\n", omp_get_thread_num(), i);
}
}

void parallel_nested() {
omp_set_nested(2);
#pragma omp parallel num_threads(2)
{
int thread_id = omp_get_thread_num();
int team_id = omp_get_team_num();
int level = omp_get_level();
printf("Lvl %i: Thread %i in team %i\n", level, thread_id, team_id);
#pragma omp parallel num_threads(3)
{
int thread_id_nst = omp_get_thread_num();
int team_id_nst = omp_get_team_num();
int level_nst = omp_get_level();
printf("-- Lvl %i: Thread %i in team %i\n", level_nst, thread_id_nst, team_id_nst);
}
thread_id = omp_get_thread_num();
team_id = omp_get_team_num();
level = omp_get_level();
printf("Lvl %i: And back to thread %i in team %i\n", level, thread_id, team_id);
}
}

void nowait() {
int i;
printf("count upwards - rather randomly:\n");
#pragma omp parallel
{
#pragma omp for nowait
for (i = 10; i < 15; i++)
printf("%i ", i);
#pragma omp for nowait
for (i = 20; i < 25; i++)
printf("%i ", i);
}
printf("\n");
}

void memory() {
int x = 5;
int y = 20;
int s = 1;
printf("s = %i, x = %i, y = %i\n", s, x, y);
#pragma omp parallel private(x) firstprivate(y) shared(s)
{
x = 10;
int z = x + y;
s++;
if (omp_get_thread_num() != 0) {
y += 3;}
printf("\tTh %i: s=%i, x=%i, y=%i\n", omp_get_thread_num(), s, x, y);
}
printf("s = %i, x = %i, y = %i\n", s, x, y);
}

void parallel_for_collapse() {
#pragma omp parallel for collapse(2)
for (int i = 1; i < 5; i++) {
for (int j = 1; j < 5; j++) {
printf("Th %i: %i - %i\n", omp_get_thread_num(), i, j);
}
}
printf("\n");
}

void parallel_for_reduction() {
int m = 0;
#pragma omp parallel for reduction(max:m)
for (int i = 0; i <= 6; i++) {
m = max(m, (rand() % 10));
printf("Th %i: m = %i\n", omp_get_thread_num(), m);
}
printf("Max of all m = %i\n", m);
}

void parallel_for_scheduling() {
printf("static\n");
#pragma omp parallel for schedule(static)
for (int i = 0; i < 12; i++) {
printf("%i: Th %i sleeping\n", i, omp_get_thread_num());
sleep(i);
}

printf("\ndynamic\n");
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < 12; i++) {
printf("%i: Th %i sleeping\n", i, omp_get_thread_num());
sleep(i);
}

printf("\nguided\n");
#pragma omp parallel for schedule(guided)
for (int i = 0; i < 12; i++) {
printf("%i: Th %i sleeping\n", i, omp_get_thread_num());
sleep(i);
}

printf("\nauto\n");
#pragma omp parallel for schedule(auto)
for (int i = 0; i < 12; i++) {
printf("%i: Th %i sleeping\n", i, omp_get_thread_num());
sleep(i);
}
}

void sections(bool addCont) {
#pragma omp parallel
if (addCont) omp_set_num_threads(2);
{
#pragma omp sections
{
{ printf("A "); }
#pragma omp section
{ printf("race "); } 
#pragma omp section
{ printf("car "); } 
}
#pragma omp single
{ printf("is fun to watch.\n"); }
if (addCont) {
printf("Parallel after: Thread %i\n", omp_get_thread_num());
omp_set_num_threads(4);
}
}
}

void tasks(bool addCont) { 
if (addCont) omp_set_num_threads(2);
#pragma omp parallel
{
if (addCont) printf("Parallel before: Thread %i\n", omp_get_thread_num());
#pragma omp single
{
printf("A ");
#pragma omp task
{ printf("race "); } 
#pragma omp task
{ printf("car "); } 
#pragma omp taskwait
printf("is fun to watch.\n");
}
if (addCont) {
printf("Parallel after: Thread %i\n", omp_get_thread_num());
omp_set_num_threads(4);
}
}
}

void syncing() {
int x;
#pragma omp parallel
{
#pragma omp atomic
x += 1;
}

omp_lock_t my_lock;
(void) omp_init_lock(&my_lock);
#pragma omp parallel
{
#pragma omp master
{
(void) omp_set_lock(&my_lock);
(void) omp_unset_lock(&my_lock);
} 
} 
(void) omp_destroy_lock(&my_lock);
}


void say_what(string s) {
printf("\n\n%s\n-------\n", s.c_str());
}


void minimal_examples(bool schedule) {


printf("--------------------\n"
"| Minimal Examples |\n"
"--------------------\n\n");

say_what("Memory");
memory();
}
