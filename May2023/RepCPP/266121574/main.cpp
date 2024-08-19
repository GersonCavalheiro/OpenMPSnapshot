#pragma GCC optimize("03", "unroll-loops", "omit-frame-pointer","inline")
#pragma GCC option("arch=native" ,"tune-native", "n0-zero-upper")
#pragma GCC target("avx")
#include<bits/stdc++.h>
#include<vector>

using namespace std;
using namespace std::chrono;

template<typename type>
void runLoop(type a, type b, type c, type result)
{

int N = 200000;
int loop = 100000;
auto time = high_resolution_clock:: now();

for (int i = 0; i < loop; ++i)
for (int j = 0; j < N; ++j)
{
result[j] = a[j]+b[j]-c[j]+3*(float)i;
}
auto end_time = duration_cast<duration<double>>(high_resolution_clock::now() - time).count();
assert( result[2] == ( 2.0f + 0.1335f)+( 1.50f*2.0f + 0.9383f)-(0.33f*2.0f+0.1172f)+3*(float)(loop-1));
cout << "Time used: "<< end_time << "s, N * noTests="<<(float(N)*float(loop))<< endl;
}

int main()
{
int N = 200000;
int loop = 100000;
float a[N],b[N],c[N],result[N];
auto time = high_resolution_clock:: now();
for (int i = 0; i < N; ++i)
{
a[i] =       ((float)i)+ 0.1335f;
b[i] = 1.50f*((float)i)+ 0.9383f;
c[i] = 0.33f*((float)i)+ 0.1172f;
}

for (int i = 0; i < loop; ++i)
for (int j = 0; j < N; ++j)
{
result[j] = a[j]+b[j]-c[j]+3*(float)i;
}
auto end_time = duration_cast<duration<double>>(high_resolution_clock::now() - time).count();
assert( result[2] == ( 2.0f + 0.1335f)+( 1.50f*2.0f + 0.9383f)-(0.33f*2.0f+0.1172f)+3*(float)(loop-1));
cout << "Time used: "<< end_time << "s, N * noTests="<<(float(N)*float(loop))<< endl;

return 0;
}
