#pragma omp declare target
template <typename T>
struct S { T a; };
template <typename T>
struct U { T a; };
template <typename T>
struct V { T a; };
template <typename T>
struct W { T a; };
S<int> d;
U<long> e[10];
extern V<char> f[5];
extern W<short> g[];		
#pragma omp end declare target
