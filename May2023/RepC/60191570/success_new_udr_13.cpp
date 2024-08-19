#include <algorithm>
#include <functional>
#include <vector>
#pragma omp declare reduction( + : std::vector<int> : std::transform(omp_in.begin( ), omp_in.end( ), omp_out.begin( ), omp_out.begin ( ), std::plus<int >() ) )
#pragma omp declare reduction( merge: std::vector<int>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end() ) )
int main (int argc, char* argv[])
{
std::vector<int> v1(5);
std::vector<int> v2(5);
#pragma omp parallel for reduction(merge : v2)
for (int i=0; i<5; i++)
{
v1 = v2;
}
return 0;
}
