#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
double funcion(double x){
return x*exp(cos(x+7)) + sin(x);
}
double integrar(double (*f)(double),double inicio, double final, int div){
double ResLocal = 0;
double step = (final - inicio) / div;
double x;
x = inicio;
ResLocal = f(inicio) + f (final);
ResLocal /= 2; 
#pragma omp parallel for default(none) shared(step, div, st, f) private(x) reduction(+: ResLocal)
for (int i = 1; i < div; i++){
x = inicio + i * step;
ResLocal += f (x);
}
ResLocal *= step;
return ResLocal;
}
int main(int argc,char *argv[]){
double IntVal = integrar(funcion,0,4,10000);
printf("La itegral de f de 0 a 4 es: %f\n",IntVal); 
return 0;
}
