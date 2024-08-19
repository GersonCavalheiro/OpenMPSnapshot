#include <iostream>
#include <string>
#include <cstdlib>
#include <math.h>
#include <omp.h>
using namespace std;
unsigned isprime(unsigned n) {
if (n<2 || ((n != 2) && (n&1) == 0) )
return 0;
else 
if (n == 2 || n==3 || n==5 || n==7 || n==11)
return 1;
for (unsigned i=3; i<=sqrt(n); i+=2) {
if (n%i == 0) {
return 0;
}
}
return 1;
}
int main() {
string input;
unsigned cases, i, j, n;
char *st;
getline (cin, input);
cases = strtol(input.c_str(), NULL, 0);
for (unsigned c=0; c<cases; c++)
{
int s = 0; 
getline (cin, input);
i = strtol(input.c_str(), &st, 0);
j = strtol(st, NULL, 0);
if (j>0)
{
#pragma omp parallel for reduction(+:s)
for (n=i; n<=j; n++)
if (isprime(n))
s++;
}
cout << "Primes found: " << s << endl;
}
return 0;
}
