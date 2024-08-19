#include <stdio.h>
#include <stdlib.h>  
#include <unistd.h>  
#include <omp.h>
#include <math.h>
#include <time.h>
#define SQUARE 2
#define N 5 
static long num_steps = 10000; 
double step;
void generatorPI();
double generatorPIr(int n);
double generatorPInra(int n);
double generatorPInrc(int n);
void generatorPIr2(int n);
double generatorPIrD(int n);
double getAverage(double*nums);
double getStdDeviation(double*nums, double avg);
double seleccionador (int num, int i);
void pi();
void menu_nivel_1();
int menu_nivel_2();
int pedirStep();
int pedirTamMatrix();
int main(){
menu_nivel_1();
return 0;
}
void menu_nivel_1(){
int opcion ;    
do {
system("clear"); 
printf("\n");
printf("Elija una opcion del menu:\n");
printf("\n");
printf("    1- pi\n");
printf("    2- matrix\n");
printf("    3- hardware\n");
printf("    4- Salir\n");
printf("\n");
printf("Ingrese una opción:  ");
scanf("%d", &opcion); 
printf("\n");
switch (opcion) {
case 1:  
int opcion_1;
do
{
opcion_1 = menu_nivel_2();
switch (opcion_1)
{
case 1:
case 2:
int numStep1 = pedirStep();
pi();
break;
}
} while (opcion_1 != 3);
break;
case 2:
int opcion_2;
do
{
opcion_2 = menu_nivel_2();
switch (opcion_2)
{
case 1:
case 2:
int n = pedirTamMatrix();
break;
}
} while (opcion_2 != 3);
break;
case 3:
printf("Opcion 3\n");
printf("\n");  
printf("Aca se describe el harware de la compu:\n"); 
printf("\n");  
sleep(1);
break;
case 4:
printf("Fin de la ejecución\n");       
break;
}  
printf("\n");
sleep(2); 
}
while (opcion!=4);    
}
int menu_nivel_2(){
int opcion;
system("clear"); 
printf("\n");
printf("Que tipo de paralelismo desea usar:\n");
printf("\n");
printf("    1- sin reduccion\n");
printf("    2- con reduccion\n");
printf("    3- atras\n");
printf("\n");
printf("Ingrese una opción:  ");
scanf("%d", &opcion); 
printf("\n");
return opcion;
}
int pedirStep(){
int step;
printf("\n");
printf("Ingresa numero de step:\n");
printf("\n");
scanf("%d", &step);   
return step;
}
int pedirTamMatrix(){
int n;
printf("\n");
printf("Ingresa tamaño de matriz:\n");
printf("\n");
scanf("%d", &n);   
return n;
}
void pi(){
double*nums = (double*)malloc(N*sizeof(double)); 
printf("\ncon un num_step de %d \n" , num_steps);
int opcion ;
printf("\n");
printf("Elija una opcion del menu:\n");
printf("\n");
printf("    1- critical\n");
printf("    2- atomic\n");
printf("    3- redoccuion\n");
printf("    4- reduccion dinamic\n");
printf("\n");
printf("Ingrese una opción:  ");
scanf("%d", &opcion); 
printf("\n");
for(int i = 0 ; i < omp_get_num_procs() ; i++){
sleep(4);  
printf("reduccion con %d procesadores\n\n", (i+1));
for(int j = 0 ; j < N ; j++){
nums[j] = seleccionador(opcion,i+1);
printf("\n");
}
double avg = getAverage(nums); 
printf("El promedio de los tiempos es de: %lf\n\n", avg);
}
}
double seleccionador(int num, int i){
switch (num)
{
case 1:
return generatorPInrc(i);
break;
case 2:
return generatorPInra(i);
break;
case 3:
return generatorPIr(i);
break;
case 4:
return generatorPIrD(i);
break;
default:
return 1.00;
}
return 1.00;
}
double getAverage(double*nums) {
double adder = 0; 
for (int i = 0 ; i < N ; i++)
adder+= nums[i]; 
printf("\n");
return adder / N ; 
}
double generatorPIr(int n){
double tiempo_inicial= omp_get_wtime();
double x, pi , sum = 0.0;
step = 1.0 / (double) num_steps;
omp_set_num_threads(n); 
#pragma omp parallel for reduction(+:sum) private(x)
for(int i = 0 ; i < num_steps ; i++){
x = (i+0.5)*step;
sum += 4.0/(1.0+x*x);
}
pi = step * sum;
double tiempo_final= omp_get_wtime();
printf(" reduccion %lf \n",tiempo_final-tiempo_inicial);
return tiempo_final-tiempo_inicial;
}
double generatorPInrc(int n){
float tiempo_inicial= omp_get_wtime();
double x, pi , sum= 0.0;
step = 1.0 / (double) num_steps;
omp_set_num_threads(n); 
#pragma omp parallel for
for(int i = 0 ; i < num_steps ; i++){
double x = 4.0/(1.0+(((i+0.5)*step)*((i+0.5)*step)));
#pragma omp critical
sum = sum + x;
}
pi = step * sum;
float tiempo_final= omp_get_wtime();
printf("no reduccion %f critical \n",tiempo_final-tiempo_inicial);
return tiempo_final-tiempo_inicial;
}
double generatorPInra(int n){
float tiempo_inicial= omp_get_wtime();
double pi , sum = 0.0;
step = 1.0 / (double) num_steps;
omp_set_num_threads(n); 
#pragma omp parallel for
for(int i = 0 ; i < num_steps ; i++){
double x = 4.0/(1.0+(((i+0.5)*step)*((i+0.5)*step)));
#pragma omp atomic
sum = sum + x;
}
pi = step * sum;
float tiempo_final= omp_get_wtime();
printf("no reduccion %f atomic     \n",tiempo_final-tiempo_inicial);
return tiempo_final-tiempo_inicial;
}
void generatorPI(){
float tiempo_inicial= omp_get_wtime();
double x, pi , sum = 0.0;
step = 1.0 / (double) num_steps;
for(int i = 0 ; i < num_steps ; i++){
x = (i+0.5)*step;
sum += 4.0/(1.0+x*x);
}
pi = step * sum;
float tiempo_final= omp_get_wtime();
printf("comun-seq %f  The pi's valor : %lf\n",tiempo_final-tiempo_inicial,pi);
}
double generatorPIrD(int n){
float tiempo_inicial= omp_get_wtime();
double x, pi , sum = 0.0;
step = 1.0 / (double) num_steps;
omp_set_num_threads(n); 
#pragma omp parallel for schedule(guided , 200000) reduction(+:sum) private(x) 
for(int i = 0 ; i < num_steps ; i++){
x = (i+0.5)*step;
sum += 4.0/(1.0+x*x);
}
pi = step * sum;
float tiempo_final= omp_get_wtime();
printf(" reduccion %f  \n",tiempo_final-tiempo_inicial );
return tiempo_final-tiempo_inicial;
}