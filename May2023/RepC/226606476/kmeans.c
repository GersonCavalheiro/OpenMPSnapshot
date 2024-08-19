#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define NUM_ATRIBUTOS 4
#define NUM_ID 9
#define ROOT 0
typedef enum { false, true } bool;
struct Cliente
{
char id[NUM_ID];
float diasMora;
float lineaCredito;
float anioCastigo;
float anioUltimoPago;
int grupo;
};
struct Centroide
{
int numElementos;
float diasMora;
float lineaCredito;
float anioCastigo;
float anioUltimoPago;
};
void validarParametros(int argc, char **argv, int *numCentroides, char **ruta, int *iterMax);
void obtenerNumeroObservaciones(char *rutaArchivo, int *observaciones);
void leerArchivo(char *rutaArchivo, int *observaciones);
void leerClientes(struct Cliente* clientes, int observaciones, char *rutaArchivo);
void mostrarClientes(struct Cliente clientes[], int observaciones);
void estandarizarDatos(struct Cliente *clientes, int observaciones);
void inicializarCentroides(struct Cliente *clientes, int observaciones, int numCentroides);
void calcularCentroides(struct Cliente *clientes,
int inicio_hilo,
int num_obs_hilo,
struct Centroide *centroides, 
int numCentroides);
void mostrarCentroides(struct Centroide *centroides, int numCentroides);
float calcularDistancia(struct Cliente cliente, struct Centroide centroide);
int main(int argc, char **argv) {
int i = 0, j, k;
int numCentroides, numObservaciones;
float *distancias; 
char *rutaArchivo;
struct Cliente *clientes;
struct Centroide *centroides;
bool termina = false;
bool debug = false;
int iterMax;
int my_rank, num_hilos, num_obs_hilo, inicio_hilo;
validarParametros(argc, argv, &numCentroides, &rutaArchivo, &iterMax);
obtenerNumeroObservaciones(rutaArchivo, &numObservaciones);
clientes = (struct Cliente *)malloc(sizeof(struct Cliente) * numObservaciones);
leerClientes(clientes, numObservaciones, rutaArchivo);
estandarizarDatos(clientes, numObservaciones);
centroides = (struct Centroide *)malloc(sizeof(struct Centroide) * numCentroides);
inicializarCentroides(clientes, numObservaciones, numCentroides);
calcularCentroides(clientes, ROOT, numObservaciones, centroides, numCentroides);
printf("\n ------------ CLIENTES: INICIO\n");
mostrarClientes(clientes, numObservaciones);
printf("\n ------------ CENTROIDES: INICIO\n");
mostrarCentroides(centroides, numCentroides);
distancias = (float *)malloc(sizeof(float) * numCentroides);
while (!termina) {
if (i == iterMax) {
printf("\n----------> FIN ITERACION\n", i+1);
} else {
printf("\n----------> ITERACION #%d\n", i+1);
}
#pragma openmp parallel default(shared) private(my_rank, num_hilos, num_obs_hilo, inicio_hilo)
{
my_rank = omp_get_thread_num();
num_hilos =  omp_get_num_threads();
num_obs_hilo = numObservaciones / num_hilos;
inicio_hilo = num_obs_hilo * my_rank;
if (my_rank == num_hilos-1) {
num_obs_hilo = numObservaciones - inicio_hilo;
}
for (j=0; j < num_obs_hilo; j++) {
for (k=0; k < numCentroides; k++) {
distancias[k] = calcularDistancia(clientes[inicio_hilo+j], centroides[k]);
if (debug) {
printf("\nDistancia Cliente #%d\tCentroide #%d: %f\n", inicio_hilo+j, k, distancias[k]);
}
}
int menor = 0;
for (k=1; k < numCentroides; k++) {                
if (distancias[k] < distancias[menor]) {
menor = k;
}
}
clientes[inicio_hilo+j].grupo = menor + 1;
}
}
i++;
if (i > iterMax) {
termina = true;
} else {
#pragma openmp parallel default(shared) private(my_rank, num_hilos, num_obs_hilo, inicio_hilo)
{
my_rank = omp_get_thread_num();
num_hilos =  omp_get_num_threads();
num_obs_hilo = numObservaciones / num_hilos;
inicio_hilo = num_obs_hilo * my_rank;
if (my_rank == num_hilos-1) {
num_obs_hilo = numObservaciones - inicio_hilo;
}
calcularCentroides(clientes, inicio_hilo, num_obs_hilo, centroides, numCentroides);                
}
#pragma omp barrier
if (debug) {
mostrarCentroides(centroides, numCentroides);
}
}
}
printf("\n ------------ CLIENTES: RESULTADO\n");
mostrarClientes(clientes, numObservaciones);
printf("\n ------------ CENTROIDES: RESULTADO\n");
mostrarCentroides(centroides, numCentroides);
exit(0);
}
void validarParametros(int argc, char **argv, int *numCentroides, char **ruta, int *iterMax) {
if (argc == 1) {
*numCentroides = 4;
} else if (argc == 7) {
*numCentroides = atoi(argv[2]);
*iterMax = atoi(argv[4]);
*ruta = argv[6];
printf("\n\nPrograma kmeans con k=%d centroides\n", *numCentroides);
printf("\nProcesado el archivo %s\n\n\n",*ruta);
} else {
fprintf(stderr, "\n\t** Uso incorrecto del programa**\n");
fprintf(stderr, "\t<programa> -c <numero_de_centroides> -i <numero_iteraciones_max> -a <nombre_archivo>\n\n");
exit(1);
}
}
void obtenerNumeroObservaciones(char *rutaArchivo, int *observaciones) {
int ch = 0;
*observaciones = 0;
FILE *archivo = fopen (rutaArchivo, "r");
if (archivo == NULL) { 
fprintf(stderr, "\n\t** Error al abrir el archivo **\n");
fprintf(stderr, "\n\t** Valide su existencia en la ruta indicada **\n\n");
exit(1);
}
while((ch=fgetc(archivo))!=EOF) {
if(ch == '\n') {
*observaciones +=1;
}
}
rewind(archivo);
fclose(archivo);
printf("\n\nNúmero de registros ob=%d\n", *observaciones);
}
void leerClientes(struct Cliente* clientes, int observaciones, char *rutaArchivo) {
int ch = 0;
int observacionActual = 0;
FILE *archivo = fopen (rutaArchivo, "r");
if (archivo == NULL) { 
fprintf(stderr, "\n\t** Error al abrir el archivo **\n");
fprintf(stderr, "\n\t** Valide su existencia en la ruta indicada **\n\n");
exit(1);
}
while (observacionActual < observaciones) {
printf("\nLeyendo observación #%d\n", observacionActual);
fscanf(archivo, "%s", clientes[observacionActual].id);
fscanf(archivo, "%f", &clientes[observacionActual].diasMora);
fscanf(archivo, "%f", &clientes[observacionActual].lineaCredito);
fscanf(archivo, "%f", &clientes[observacionActual].anioCastigo);
fscanf(archivo, "%f\n", &clientes[observacionActual].anioUltimoPago);
observacionActual++;
}
fclose(archivo);
}
void mostrarClientes(struct Cliente clientes[], int observaciones) {
int i;
printf("\n");
for (i = 0; i < observaciones; i++) {
printf("\nObservacion #%d\n", i+1);
printf("clientes[%d].id -> %s\n", i+1, clientes[i].id);
printf("clientes[%d].diasMora -> %f\n", i+1, clientes[i].diasMora);
printf("clientes[%d].lineaCredito -> %f\n", i+1, clientes[i].lineaCredito);
printf("clientes[%d].anioCastigo -> %f\n", i+1, clientes[i].anioCastigo);
printf("clientes[%d].anioUltimoPago -> %f\n", i+1, clientes[i].anioUltimoPago);
printf("clientes[%d].grupo -> %d\n", i+1, clientes[i].grupo);
}
printf("\n");
}
void estandarizarDatos(struct Cliente *clientes, int observaciones) {
int i;
float medias[NUM_ATRIBUTOS];
float desviaciones[NUM_ATRIBUTOS];
#pragma omp parallel for
for (i = 0; i < NUM_ATRIBUTOS; i++) {
medias[i] = 0;
desviaciones[i] = 0;
}
for (i = 0; i < observaciones; i++) {
medias[0] += clientes[i].diasMora;
medias[1] += clientes[i].lineaCredito;
medias[2] += clientes[i].anioCastigo;
medias[3] += clientes[i].anioUltimoPago;
}
for (i = 0; i < NUM_ATRIBUTOS; i++) {
medias[i] /= (float)observaciones;
}
#pragma omp parallel for
for (i = 0; i < observaciones; i++) {
desviaciones[0] += pow(clientes[i].diasMora - medias[0], 2);
desviaciones[1] += pow(clientes[i].lineaCredito - medias[1], 2);
desviaciones[2] += pow(clientes[i].anioCastigo - medias[2], 2);
desviaciones[3] += pow(clientes[i].anioUltimoPago - medias[3], 2);
}
for (i = 0; i < NUM_ATRIBUTOS; i++) {
desviaciones[i] = sqrt(desviaciones[i]/(float)observaciones);
}
#pragma omp parallel for
for (i = 0; i < observaciones; i++) {
clientes[i].diasMora = (clientes[i].diasMora - medias[0]) / desviaciones[0];
clientes[i].lineaCredito = (clientes[i].lineaCredito - medias[1]) / desviaciones[1];
clientes[i].anioCastigo = (clientes[i].anioCastigo - medias[2]) / desviaciones[2];
clientes[i].anioUltimoPago = (clientes[i].anioUltimoPago - medias[3]) / desviaciones[3];
}
for (i = 0; i < NUM_ATRIBUTOS; i++) {
printf("media[%d] -> %f\n", i+1, medias[i]);
printf("desviaciones[%d] -> %f\n", i+1, desviaciones[i]);
}
}
void inicializarCentroides(struct Cliente *clientes, int observaciones, int numCentroides) {
int i;
srand(time(NULL)); 
#pragma omp parallel for
for (i = 0; i < observaciones; i++) {
clientes[i].grupo = (rand() % numCentroides) + 1;
}
}
void calcularCentroides(struct Cliente *clientes,
int inicio_hilo,
int num_obs_hilo,
struct Centroide *centroides, 
int numCentroides) {
int i;
for (i = 0; i < numCentroides; i++) {
centroides[i].numElementos = 0;
centroides[i].diasMora = 0;
centroides[i].lineaCredito = 0;
centroides[i].anioCastigo = 0;
centroides[i].anioUltimoPago = 0;
}
for (i = 0; i < num_obs_hilo; i++) {
centroides[clientes[inicio_hilo+i].grupo - 1].diasMora = centroides[clientes[inicio_hilo+i].grupo - 1].diasMora +  clientes[inicio_hilo+i].diasMora;
centroides[clientes[inicio_hilo+i].grupo - 1].lineaCredito = centroides[clientes[inicio_hilo+i].grupo - 1].lineaCredito + clientes[inicio_hilo+i].lineaCredito;
centroides[clientes[inicio_hilo+i].grupo - 1].anioCastigo = centroides[clientes[inicio_hilo+i].grupo - 1].anioCastigo + clientes[inicio_hilo+i].anioCastigo;
centroides[clientes[inicio_hilo+i].grupo - 1].anioUltimoPago = centroides[clientes[inicio_hilo+i].grupo - 1].anioUltimoPago + clientes[inicio_hilo+i].anioUltimoPago;
centroides[clientes[inicio_hilo+i].grupo - 1].numElementos = centroides[clientes[inicio_hilo+i].grupo - 1].numElementos + 1;
}
for (i = 0; i < numCentroides; i++) {
centroides[i].diasMora = centroides[i].diasMora / centroides[i].numElementos;
centroides[i].lineaCredito = centroides[i].lineaCredito / centroides[i].numElementos;
centroides[i].anioCastigo = centroides[i].anioCastigo / centroides[i].numElementos;
centroides[i].anioUltimoPago = centroides[i].anioUltimoPago / centroides[i].numElementos;
}
}
void mostrarCentroides(struct Centroide *centroides, int numCentroides) {
int i;
printf("\n");
for (i = 0; i < numCentroides; i++) {
printf("\nCentroide #%d\n", i+1);
printf("centroides[%d].numElementos -> %d\n", i+1, centroides[i].numElementos);
printf("centroides[%d].diasMora -> %f\n", i+1, centroides[i].diasMora);
printf("centroides[%d].lineaCredito -> %f\n", i+1, centroides[i].lineaCredito);
printf("centroides[%d].anioCastigo -> %f\n", i+1, centroides[i].anioCastigo);
printf("centroides[%d].anioUltimoPago -> %f\n", i+1, centroides[i].anioUltimoPago);
}
printf("\n");
}
float calcularDistancia(struct Cliente cliente, struct Centroide centroide) {
float distancia = 0;
distancia += pow(cliente.diasMora - centroide.diasMora, 2);
distancia += pow(cliente.lineaCredito - centroide.lineaCredito, 2);
distancia += pow(cliente.anioCastigo - centroide.anioCastigo, 2);
distancia += pow(cliente.anioUltimoPago - centroide.anioUltimoPago, 2);
distancia = sqrt(distancia);
return (float)distancia;
}