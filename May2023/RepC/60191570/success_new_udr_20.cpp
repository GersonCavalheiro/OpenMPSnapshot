#include <cstdio>
#include <cstdlib>
struct my_data_t
{
int my_data;
};
void reducer(my_data_t* out, my_data_t* in)
{
out->my_data += in->my_data;
}
void init(my_data_t* priv)
{
priv->my_data = 0;
}
#pragma omp declare reduction (plus: my_data_t : reducer(&omp_out,&omp_in)) initializer(init(&omp_priv))
my_data_t foo(my_data_t* v, int n)
{
my_data_t sum;
sum.my_data = 0;
#pragma omp parallel for reduction(plus : sum)
for (int i = 0; i < n; i++)
{
reducer(&sum, &v[i]);
}
return sum;
}
const int NUM_ITEMS = 1000;
int main(int, char**)
{
my_data_t m[NUM_ITEMS];
int sum = 0;
for (int i = 0; i <  NUM_ITEMS; i++)
{
m[i].my_data = i;
sum += i;
}
my_data_t s = foo(m, NUM_ITEMS);
if (s.my_data != sum)
{
fprintf(stderr, "%d != %d\n", s.my_data, sum);
abort();
}
return 0;
}
