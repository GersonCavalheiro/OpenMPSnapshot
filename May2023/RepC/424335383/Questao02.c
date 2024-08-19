#include <stdio.h>
double calcularPi();
int main(void){
int precisao;
scanf("%d", &precisao);
double pi = calcularPi(precisao);
printf("O valor de PI é: %f\n", pi);
}
double calcularPi(int precisao) {
double x = 1, operacao = 0.0,resultado = 4.0;
#pragma omp parallel for private(operacao)
for(int i=0; i < precisao; i++){
#pragma omp critical 
x += 2;	
if ( i % 2 == 0 ) {
operacao -= 4/x;
}
else {
operacao += 4/x;
}
#pragma omp critical
resultado += operacao;
}
return resultado;	  
}
