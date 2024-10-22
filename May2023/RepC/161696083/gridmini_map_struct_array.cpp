#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include "ompvv.h"
#include "omp.h"
using namespace std;
struct vec {
float v[2];
};
inline  vec mult(vec x, vec y){
vec out;
out.v[0] = x.v[0]*y.v[0];
out.v[1] = x.v[1]*y.v[1];
return out;
}
int main(int argc, char* argv[]){
OMPVV_TEST_OFFLOADING;
int errors = 0;
int N = 10;
float x = (float)rand()/(float)(RAND_MAX/10.0);
float y = (float)rand()/(float)(RAND_MAX/5.0);
vec in1,in2;
in1.v[0] = x;
in1.v[1] = x;
in2.v[0] = y;
in2.v[1] = y;
vec out[N];
vec expected = mult(in1,in2);
#pragma omp target teams distribute parallel for map(to:in1,in2) map(from:out[0:N])
for(int n = 0; n < N; n++) {
out[n] = mult(in1,in2);
}
for(int n = 0; n < N; n++) {
OMPVV_TEST_AND_SET(errors,out[n].v[0] != expected.v[0]);
OMPVV_TEST_AND_SET(errors,out[n].v[0] != expected.v[1]);
}
OMPVV_REPORT_AND_RETURN(errors);
}
