#include <stdio.h>
#include <omp.h>
int main(){
int i,j,k,m;
double tmp1;
double a[12][12][12];
m = 3.0;
#pragma omp parallel for private(j,k,tmp1)   
for (i = 0; i < 12; i++) {
for (j = 0; j < 12; j++) {
for (k = 0; k < 12; k++) {
tmp1 = 6.0/m;
a[i][j][k] = tmp1+4;
}
}
}
return 0;
}
