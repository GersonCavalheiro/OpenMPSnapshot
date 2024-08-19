#include "route_iterator.h"
#include "route_cost.h"
#include "tsp_utils.h"

#include "SimpleTimer.h"

#ifdef _OPENMP
#include <omp.h>
#endif


#include <cstdio> 
#include <cstdlib> 
#include <iostream>


void print_factorials(int N)
{
for (int k=0; k<N; ++k)
printf("factorial(%d)=%" PRId64 "\n",k,factorial(k));
}

template<int N>
void test_permutation()
{

for (int k=0; k<factorial(N); ++k)
{

route_iterator<N> rit(k);

rit.print();

}

} 

void test_city_distance()
{
auto cityIndex = makeCityMap();
auto size = cityIndex.size();

auto distances = init_distance_matrix();

for (size_t from=0; from<size; ++from)
{
for (size_t to=0; to<size; ++to)
{
printf("%05d ",distances[to+size*from]);
}
printf("\n");
}

int small_n = 5;
auto distances_small = init_distance_matrix_small(small_n);

for (int from=0; from<small_n; ++from)
{
for (int to=0; to<small_n; ++to)
{
printf("%05d ",distances_small[to+small_n*from]);
}
printf("\n");
}

} 

template<int N>
route_cost find_best_route(int const* distances)
{

int64_t num_routes = factorial(N);

#pragma omp declare reduction                                           \
(route_min : route_cost : omp_out = route_cost::min(omp_out, omp_in)) \
initializer(omp_priv = route_cost())

route_cost best_route;

#pragma omp target teams distribute parallel for\
map(to:distances[0:N*N]) reduction(route_min : best_route)
for (int64_t i = 0; i < num_routes; ++i)
{
int cost = 0;

route_iterator<N> it(i);

int from = it.first();

while (!it.done())
{
int to = it.next();
cost += distances[to + N*from];
from = to;
}


best_route = route_cost::min(best_route, route_cost(i, cost));
}

return best_route;

} 

template <int N>
void solve_traveling_salesman(int nbRepeat = 1)
{
auto distances_small = init_distance_matrix_small(N);

route_cost best_route;

SimpleTimer timer;
timer.start();
for (int i = 0; i<nbRepeat; ++i)
best_route = find_best_route<N>(distances_small.data());
timer.stop();

double time = timer.elapsedSeconds()/nbRepeat;

printf("Trav Salesman Prob N=%d, best route cost is: %d, average time is %f seconds\n",N, best_route.cost, time);

printf("Solution route is ");
route_iterator<N> rit(best_route.route);
rit.print();

} 

int main(int argc, char* argv[])
{







int n = 8;
if (argc>1)
n = atoi(argv[1]);

if (n==10)
solve_traveling_salesman<10>();
else if (n==11)
solve_traveling_salesman<11>();
else if (n==12)
solve_traveling_salesman<12>();
else if (n==13)
solve_traveling_salesman<13>();
else if (n==14)
solve_traveling_salesman<14>();

return EXIT_SUCCESS;

} 
