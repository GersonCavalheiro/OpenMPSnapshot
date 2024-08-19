#include <bits/stdc++.h>
#include <omp.h>
#define MAX_NUMBER (20000)
#define SIEVE_UPPER_LIMIT (MAX_NUMBER * 2)
std::vector<int> primes_g;
std::vector<bool> is_prime_g;
void sieve(int N = SIEVE_UPPER_LIMIT) {
std::vector<int> primes;
std::vector<bool> in(N + 1, true);
in[0] = in[1] = false;
for(int i = 2; i <= N; ++i) {
if(in[i]) {
primes.push_back(i);
for(int j = i * i; j <= N; j += i)
in[j] = false;
}
}
primes_g = primes;
is_prime_g = in;
}
inline bool is_prime(long long int x) {
if (x <= SIEVE_UPPER_LIMIT)
return is_prime_g[x];
else {
for (auto& p : primes_g) {
if(!(x % p))
return false;
}
int last_prime = primes_g.back();
if(last_prime == 2)
++last_prime;
for(int i = last_prime + 2; i * i <= x; i += 2) {
if(!(x % i))
return false;
}
return true;
}
}
inline int goldbach(int n) {
int ans = 0;
for(std::vector<int>::iterator itp1 = primes_g.begin(); itp1 != primes_g.end() && *itp1 <= n / 2; ++itp1) {
if(is_prime(n - *itp1))
++ans;
}
return ans;
}
#ifdef DEBUG
void test_goldbach() {
assert(goldbach(2) == 0);
assert(goldbach(802) == 16);
assert(goldbach(1602) == 53);
assert(goldbach(2402) == 37);
assert(goldbach(6402) == 156);
assert(goldbach(7202) == 78);
}
#endif
int main() {
sieve();
#ifdef DEBUG
test_goldbach();
#endif
std::vector<int> numpairs(MAX_NUMBER + 1, 0);
#pragma omp parallel
{
#pragma omp single
{
std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
omp_sched_t s;
int t;
omp_get_schedule(&s, &t);
std::cout << "Schedule: (" << s << ", " << t << ")" << std::endl;
}
}
auto t1 = omp_get_wtime();
#pragma omp parallel for schedule(runtime)
for(int i = 2; i <= MAX_NUMBER; ++i) {
numpairs[i] = goldbach(2 * i);
}
auto t2 = omp_get_wtime();
std::cout << "Elapsed time: " << 1000 * (t2 - t1) << " ms" << std::endl;
assert(numpairs[1] == 0);
assert(numpairs[401] == 16);
assert(numpairs[801] == 53);
assert(numpairs[1201] == 37);
assert(numpairs[3201] == 156);
assert(numpairs[3601] == 78);
return 0;
}
