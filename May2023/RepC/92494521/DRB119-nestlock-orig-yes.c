#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct {
int a, b;
omp_nest_lock_t lck;
} pair;
void incr_a(pair *p){
p->a += 1;
}
void incr_b(pair *p){
p->b += 1;
}
int main(int argc, char* argv[])
{
pair p[1];
p->a = 0;
p->b = 0;
omp_init_nest_lock(&p->lck);
#pragma omp parallel sections
{
#pragma omp section
{
omp_set_nest_lock(&p->lck);
incr_b(p);
incr_a(p);
omp_unset_nest_lock(&p->lck);
}
#pragma omp section
incr_b(p);
}
omp_destroy_nest_lock(&p->lck);
printf("%d\n",p->b);
return 0;
}
