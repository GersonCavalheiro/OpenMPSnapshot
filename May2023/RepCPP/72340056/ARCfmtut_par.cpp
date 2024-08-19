#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <omp.h>
#include <array>
#include <math.h>
#include <sys/stat.h>

using namespace std;

int ALTURA = 0;
int ANCHURA = 0;

struct pixel {
unsigned char r;
unsigned char g;
unsigned char b;
};



pixel **generarMatrizPixeles(char *rutaEntrada) {
ifstream archivo(rutaEntrada, ios::binary);
int alturaMatrizActual = 0;
int anchuraMatrizActual = 0;
pixel **matrizPixeles;
if (archivo.is_open()) {
archivo.read((char *) &alturaMatrizActual, 4);
archivo.read((char *) &anchuraMatrizActual, 4);

if (ALTURA != 0 && ANCHURA != 0 && (ALTURA != alturaMatrizActual || ANCHURA != anchuraMatrizActual)) {
cerr << "Las dimensiones de la máscara no coinciden con las dimensiones de la imagen original." << endl
<< "Es posible que la ruta del fichero de máscara no sea correcta." << endl;
exit(-1);
} else {
ALTURA = alturaMatrizActual;
ANCHURA = anchuraMatrizActual;
}

struct stat st;
stat(rutaEntrada, &st);
int fileSize = (int) st.st_size;

if (fileSize != ALTURA * ANCHURA * 3 + 8) {
cerr << "El fichero de entrada no tiene un formato correcto." << endl;
exit(-1);
}

matrizPixeles = new pixel *[ALTURA];
for (int i = 0; i < ALTURA; ++i) {
matrizPixeles[i] = new pixel[ANCHURA];
}

char *buffer = new char[fileSize - 8];

archivo.seekg(8, ios::beg);
archivo.read(buffer, fileSize - 8);


#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
matrizPixeles[i][j].r = (unsigned char) buffer[i * ANCHURA + j];
}
}
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
matrizPixeles[i][j].g = (unsigned char) buffer[i * ANCHURA + j + ANCHURA * ALTURA];

}
}
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
matrizPixeles[i][j].b = (unsigned char) buffer[i * ANCHURA + j + ANCHURA * ALTURA * 2];
}
}
}
archivo.close();
} else {
cerr << "El fichero de entrada no existe en la ruta especificada." << endl;
exit(-1);
}
return matrizPixeles;
}


void escribirSalida(pixel **matrizPixeles, char *rutaSalida) {
ofstream archivo(rutaSalida, ios::binary);
if (archivo.is_open()) {

unsigned char *buffer = new unsigned char[ALTURA * ANCHURA * 3];

archivo.write((char *) &ALTURA, 4);
archivo.write((char *) &ANCHURA, 4);

#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
buffer[i * ANCHURA + j] = matrizPixeles[i][j].r;
}
}
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
buffer[i * ANCHURA + j + ALTURA * ANCHURA] = matrizPixeles[i][j].g;

}
}
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
buffer[i * ANCHURA + j + ALTURA * ANCHURA * 2] = matrizPixeles[i][j].b;
}
}
}

archivo.write((const char *) buffer, ALTURA * ANCHURA * 3);
archivo.close();
} else {
cerr << "El fichero de salida no se ha creado correctamente."
"Es posible que la ruta de salida no sea correcta." << endl;
exit(-1);
}
}




void calcularMaximosYMinimos(pixel **matrizPixeles, char *rutaSalida) {
array<int, 6> maximosYMinimos = {0, 0, 0, 0, 0, 0};
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
if (matrizPixeles[i][j].r > maximosYMinimos[0]) {
maximosYMinimos[0] = matrizPixeles[i][j].r;
}
if (matrizPixeles[i][j].r < maximosYMinimos[3]) {
maximosYMinimos[3] = matrizPixeles[i][j].r;
}
}
}
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
if (matrizPixeles[i][j].g > maximosYMinimos[1]) {
maximosYMinimos[1] = matrizPixeles[i][j].g;
}
if (matrizPixeles[i][j].g < maximosYMinimos[4]) {
maximosYMinimos[4] = matrizPixeles[i][j].g;
}
}
}
#pragma omp for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
if (matrizPixeles[i][j].b > maximosYMinimos[2]) {
maximosYMinimos[2] = matrizPixeles[i][j].b;
}
if (matrizPixeles[i][j].b < maximosYMinimos[5]) {
maximosYMinimos[5] = matrizPixeles[i][j].b;
}
}
}
}

ofstream outputFile(rutaSalida);
if (!outputFile.is_open()) {
cerr << "El fichero de salida no se ha creado correctamente."
"Es posible que la ruta de salida no sea correcta." << endl;
exit(-1);
}
outputFile << maximosYMinimos[0] << " " << maximosYMinimos[3] << " "
<< maximosYMinimos[1] << " " << maximosYMinimos[4] << " "
<< maximosYMinimos[2] << " " << maximosYMinimos[5];
outputFile.close();
delete[] matrizPixeles;
}


void histograma(pixel **matrizPixeles, char *rutaSalida, int tramos) {
if (tramos <= 0) {
cerr << "El número de tramos debe ser mayor que 0.";
exit(-1);
}
vector<int> histograma(tramos);
double grises = 0;
double rangoIntervalo = 256 / (double) tramos;

#pragma omp parallel for ordered schedule(auto)
for (int i = 0; i < ALTURA; i++) {
for (int j = 0; j < ANCHURA; j++) {
grises = matrizPixeles[i][j].r * 0.3 + matrizPixeles[i][j].g * 0.59 + matrizPixeles[i][j].b * 0.11;
int intervalo = (int) floor(grises / rangoIntervalo);
#pragma omp ordered
histograma[intervalo] += 1;
}
}

ofstream outputFile(rutaSalida);
if (!outputFile.is_open()) {
cerr << "El fichero de salida no se ha creado correctamente."
"Es posible que la ruta de salida no sea correcta." << endl;
exit(-1);
}

for (int i = 0; i < tramos; ++i) {
outputFile << histograma[i];
if (i != tramos - 1) {
outputFile << " ";
}
}
outputFile.close();
delete[] matrizPixeles;
}


void filtroBN(pixel **matrizPixeles, double radio, char *rutaSalida) {
int centroX = ANCHURA / 2;
int centroY = ALTURA / 2;

#pragma omp parallel for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ALTURA; ++j) {
double suma = pow(i - centroY, 2) + pow(j - centroX, 2);
if (suma > pow(radio, 2)) {
matrizPixeles[i][j].r *= 0.3;
matrizPixeles[i][j].g *= 0.59;
matrizPixeles[i][j].b *= 0.11;
}
}
}
escribirSalida(matrizPixeles, rutaSalida);
}


void mascara(pixel **matrizPixeles, pixel **matrizMascara, char *rutaSalida) {
#pragma omp parallel for
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
matrizPixeles[i][j].r *= matrizMascara[i][j].r;
matrizPixeles[i][j].g *= matrizMascara[i][j].g;
matrizPixeles[i][j].b *= matrizMascara[i][j].b;
}
}
escribirSalida(matrizPixeles, rutaSalida);
delete[] matrizMascara;
}



void rotacion(pixel **matrizPixeles, double grados, char *rutaSalida) {
double filaCentro = ALTURA / 2;
double colCentro = ANCHURA / 2;
int coorXrotada = 0;
int coorYrotada = 0;
double radianes = grados * M_PI / 180;

pixel **rotada = new pixel *[ALTURA]();
for (int i = 0; i < ALTURA; i++) {
rotada[i] = new pixel[ANCHURA]();
}

#pragma omp parallel for private(coorXrotada, coorYrotada) schedule(dynamic)
for (int i = 0; i < ALTURA; ++i) {
for (int j = 0; j < ANCHURA; ++j) {
coorXrotada = (int) ceil((cos(radianes) * (j - colCentro) - sin(radianes) * (i - filaCentro)) + colCentro);
coorYrotada = (int) ceil((sin(radianes) * (j - colCentro) + cos(radianes) * (i - filaCentro)) + filaCentro);

if (coorXrotada >= 0 && coorXrotada <= ANCHURA - 1 && coorYrotada >= 0 && coorYrotada <= ALTURA - 1) {
rotada[coorYrotada][coorXrotada] = matrizPixeles[i][j];
}
}
}
escribirSalida(rotada, rutaSalida);
delete[] matrizPixeles;
}

int main(int argv, char **argc) {
char *rutaEntrada = NULL;
char *rutaSalida = NULL;
char *parametroExtra = NULL;
char *parametroLetra = NULL;
int ejecucion = -1;
for (int i = 1; i < argv; ++i) {
if (strcmp(argc[i], "-u") == 0) {
try {
ejecucion = stoi(argc[i + 1]);
i++;
continue;
} catch (const std::invalid_argument) {
cerr << "No se ha especificado correctamente la acción a realizar. "
"Inserte el parámetro -u seguido de la acción a realizar, "
"utilizando valores entre 0 - 4" << endl
<< "0: Histograma blanco y negro." << endl
<< "1: Máximos y Mínimos." << endl
<< "2: Aplicación de máscara." << endl
<< "3: Rotación de imagen." << endl
<< "4: Filtro blanco y negro." << endl;
exit(-1);
}
}
if (strcmp(argc[i], "-i") == 0) {
rutaEntrada = argc[i + 1];
i++;
continue;
}
if (strcmp(argc[i], "-o") == 0) {
rutaSalida = argc[i + 1];
i++;
continue;
}
if (strcmp(argc[i], "-t") == 0 || strcmp(argc[i], "-r") == 0 || strcmp(argc[i], "-a") == 0 ||
strcmp(argc[i], "-f") == 0) {
parametroLetra = argc[i];
parametroExtra = argc[i + 1];
i++;
continue;
}
}

if (rutaEntrada == NULL) {
cerr << "No se ha especificado el fichero de entrada. Inserte el parámetro -i seguido de la ruta." << endl;
exit(-1);
}
if (rutaSalida == NULL) {
cerr << "No se ha especificado el fichero de salida. Inserte el parámetro -o seguido de la ruta." << endl;
exit(-1);
}
if (ejecucion == -1) {
cerr << "No se ha especificado la acción a realizar. Inserte el parámetro -u seguido de la acción (0-4)."
<< endl;
exit(-1);
}

switch (ejecucion) {
case 0: {
if (parametroExtra == NULL || strcmp(parametroLetra, "-t") != 0) {
cerr << "No se ha especificado el número de tramos. "
"Inserte el parámetro -t seguido del número de tramos."
<< endl;
exit(-1);
}

try {
histograma(generarMatrizPixeles(rutaEntrada), rutaSalida, stoi(parametroExtra));
} catch (const std::invalid_argument) {
cerr << "El parámetro indicado por -t tiene que ser un NÚMERO ENTERO." << endl;
exit(-1);
}
break;
}
case 1: {
calcularMaximosYMinimos(generarMatrizPixeles(rutaEntrada), rutaSalida);
break;
}
case 2: {
if (parametroExtra == NULL || strcmp(parametroLetra, "-f") != 0) {
cerr << "No se ha especificado la dirección del fichero de máscara. "
"Inserte el parámetro -f seguido del ángulo de la rotación de la imagen."
<< endl;
exit(-1);
}
mascara(generarMatrizPixeles(rutaEntrada), generarMatrizPixeles(parametroExtra), rutaSalida);
break;
}
case 3: {
if (parametroExtra == NULL || strcmp(parametroLetra, "-a") != 0) {
cerr << "No se ha especificado el ángulo de rotación. "
"Inserte el parámetro -a seguido del ángulo de la rotación de la imagen."
<< endl;
exit(-1);
}
try {
rotacion(generarMatrizPixeles(rutaEntrada), stod(parametroExtra), rutaSalida);
} catch (const std::invalid_argument) {
cerr << "El parámetro indicado por -a tiene que ser un NÚMERO DECIMAL." << endl;
exit(-1);
}
break;
}
case 4: {
if (parametroExtra == NULL || strcmp(parametroLetra, "-r") != 0) {
cerr << "No se ha especificado el radio del filtro B/N. "
"Inserte el parámetro -r seguido del radio del filtro."
<< endl;
exit(-1);
}
try {
if (stod(parametroExtra) < 0) {
cerr << "El parámetro indicado por -r tiene que ser un número decimal POSITIVO." << endl;
exit(-1);
}
filtroBN(generarMatrizPixeles(rutaEntrada), stod(parametroExtra), rutaSalida);
} catch (const std::invalid_argument) {
cerr << "El parámetro indicado por -r tiene que ser un NÚMERO DECIMAL positivo." << endl;
exit(-1);
}
break;
}
default:
cerr << "No se ha especificado la acción a realizar. "
"Inserte el parámetro -u seguido de la acción a realizar, "
"utilizando VALORES ENTRE 0 - 4:" << endl
<< "0: Histograma blanco y negro." << endl
<< "1: Máximos y Mínimos." << endl
<< "2: Aplicación de máscara." << endl
<< "3: Rotación de imagen." << endl
<< "4: Filtro blanco y negro." << endl;
exit(-1);
}
return 0;
}