#include <vector>
int
main ()
{
std::vector<double> vec(10);
#pragma omp parallel
__builtin_exit (0);
}
