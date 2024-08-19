#include <cstdio>
#include <getopt.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <omp.h>

#include "adf.h"


using namespace std;

typedef char mychar;

bool  print_results;
char *input_file_name;
int   num_threads;

#define OPTIMAL

#define BUFFER_SIZE 1024

#define MAX(A,B) A>B ? A : B

typedef set<unsigned int> SetOfVertex;

typedef map<unsigned int, SetOfVertex *> Neighborhood;



class Settings
{
public:
char* dwarfName;
char* inputFile;
char* resultFile;

Settings(char *input_file);
~Settings();
void PrintStringSettings();

private:
static const int maxArraySize = 5000;
};


Settings::Settings(char *input_file)
{
dwarfName = new char[maxArraySize];
inputFile = new char[maxArraySize];
resultFile = new char[maxArraySize];

dwarfName[0] = inputFile[0] = resultFile[0] = '\0';

sprintf(dwarfName, "%s", "UnstructuredGrid");
sprintf(inputFile, "%s", input_file);
sprintf(resultFile, "result_omp.txt");

}

Settings::~Settings()
{
delete[] dwarfName;
delete[] inputFile;
delete[] resultFile;
}

void Settings::PrintStringSettings()
{
char* stringSettings = new char[maxArraySize];
stringSettings[0] = '\0';

sprintf(stringSettings, "%s%s", stringSettings, "Kernel settings summary: ");
sprintf(stringSettings, "%s%s", stringSettings, "\nDwarf name        : ");
sprintf(stringSettings, "%s%s", stringSettings, dwarfName);
sprintf(stringSettings, "%s%s", stringSettings, "\nInputFile         : ");
sprintf(stringSettings, "%s%s", stringSettings, inputFile);
sprintf(stringSettings, "%s%s", stringSettings, "\nResultFile        : ");
sprintf(stringSettings, "%s%s", stringSettings, resultFile);

printf("%s", stringSettings);

delete[] stringSettings;
}




class Configurator
{
public:
Configurator(char *input_file) { settings = new Settings(input_file); }
~Configurator() { delete settings; }

static void getNextLine(char* str, FILE* file);
void GetContent(unsigned int *numVertex, unsigned int *numTriangles, double *epsilon,
double **value, double **tempValue, unsigned int **numNeighbors,
unsigned int ***neighborhood, unsigned int ***triangles);
void WriteSettings() { settings -> PrintStringSettings(); }
void Close(long iterations, unsigned int numVertex, double *resultValue);

private:
Settings* settings;
};


void Configurator :: getNextLine(char* str, FILE* file)
{
short count = 0;
char ch = ' ';

while (!feof(file) && (ch == ' ' || ch == '\n' || ch == '\r')) {
ch = (char)getc(file);
}

str[count] = ch;
while (!feof(file) && ch != '\r' && ch != '\n' && ch != ' ') {
str[++count] = (ch = (char)getc(file));
}

str[count] = '\0';
}


void Configurator :: GetContent(unsigned int *numVertex, unsigned int *numTriangles, double *epsilon,
double **value, double **tempValue, unsigned int **numNeighbors,
unsigned int ***neighborhood, unsigned int ***triangles)
{
FILE *file = fopen(settings -> inputFile, "rb");		
char *str = new char[BUFFER_SIZE];

getNextLine(str, file);
*epsilon = atof(str);

getNextLine(str, file);
*numVertex = atoi(str);

getNextLine(str, file);
*numTriangles = atoi(str);

unsigned int i,j;

*value = new double[*numVertex];
*tempValue = new double[*numVertex];

(*triangles)[0] = new unsigned int[*numTriangles];
(*triangles)[1] = new unsigned int[*numTriangles];
(*triangles)[2] = new unsigned int[*numTriangles];

*neighborhood = new unsigned int*[*numVertex];
*numNeighbors = new unsigned int[*numVertex];

i = j = 0;
double val;

while(!feof(file) && j < 1) {
getNextLine(str, file);
val = atof(str);
(*value)[i] = val;
(*tempValue)[i] = val;

i ++;
if (i >= *numVertex) {
i = 0;
j ++;
}
}

i = j = 0;

while(!feof(file) && j < 3) {
getNextLine(str, file);
(*triangles)[j][i] = atoi(str);

i ++;
if (i >= *numTriangles) {
i = 0;
j ++;
}
}

delete str;


fclose(file);
}

void Configurator :: Close(long iterations, unsigned int numVertex, double *resultValue)
{
if (print_results == true) {
FILE *result = fopen(settings -> resultFile, "wb");
fprintf(result, "#Iterations: %ld\n", iterations);
fprintf(result, "#Result vector:\r\n");

for (unsigned int j = 0; j < numVertex; j ++) {
fprintf(result, "%0.8f ", resultValue[j]);
}
fprintf(result, "\r\n");
fclose(result);
}
}





class Solver
{
public:
Solver(Configurator* configurator);
~Solver();
void Solve();
void PrintSummery();
void Finish();

long           iterations;			
unsigned int   numVertex;     		
unsigned int   numTriangles;  		

double         epsilon;             

double        *value;              	
double        *tempValue;          	
double        *resultValue;        	

unsigned int  *numNeighbors; 		
unsigned int **neighborhood;		
unsigned int **triangles;   		

private:
Configurator	 *dwarfConfigurator;

static inline void compute(unsigned int *vertices, unsigned int num, double *input, double *output);
void constructNeighborhood();
};


Solver :: Solver(Configurator* configurator)
{
iterations = 0;
resultValue = 0;
neighborhood = 0;
numNeighbors = 0;
value = 0;
tempValue = 0;
triangles = new unsigned int*[3];

numVertex = 0;
numTriangles = 0;

dwarfConfigurator = configurator;					
dwarfConfigurator -> WriteSettings();
dwarfConfigurator -> GetContent(&numVertex, &numTriangles, &epsilon, &value, &tempValue,
&numNeighbors, &neighborhood, &triangles);

constructNeighborhood();
}

Solver :: ~Solver()
{
delete triangles[0];
delete triangles[1];
delete triangles[2];

for( unsigned int i = 0; i < numVertex; i ++)
{
delete neighborhood[i];
}

delete numNeighbors;
delete neighborhood;
delete value;
delete tempValue;
delete triangles;
}


void Solver::constructNeighborhood()
{
Neighborhood *nhood = new Neighborhood();
for (unsigned int i = 0; i < numVertex; i ++) {
nhood->insert(make_pair( i, new SetOfVertex() ));
}

SetOfVertex* sov;
for (unsigned int j = 0; j < numTriangles; j ++) {
sov = (SetOfVertex*) (nhood->find(triangles[0][j])->second);
sov->insert(triangles[1][j]);
sov->insert(triangles[2][j]);

sov = (SetOfVertex*) (nhood->find(triangles[1][j])->second);
sov->insert(triangles[0][j]);
sov->insert(triangles[2][j]);

sov = (SetOfVertex*) (nhood->find(triangles[2][j])->second);
sov->insert(triangles[0][j]);
sov->insert(triangles[1][j]);
}

for (unsigned int i = 0; i < numVertex; i ++) {
SetOfVertex *vSet = (SetOfVertex*)(nhood->find(i)->second);

numNeighbors[i] = (unsigned int) vSet->size();
neighborhood[i] = new unsigned int[numNeighbors[i]];

int j = 0;
SetOfVertex::iterator it;
for(  it = vSet->begin(); it != vSet->end(); it++ ) {
neighborhood[i][j++] = *it;
}
}

Neighborhood::iterator iter;
for( iter = nhood->begin(); iter != nhood->end(); iter++ ) {
delete iter->second;
}
}


void Solver::compute(unsigned int *vertices, unsigned int num, double *input, double *output)
{
*output = 0;
for(unsigned int i = 0; i < num; i ++) {
*output += input[vertices[i]];
}

(num != 0) ? *output /= num : *output = 0;
}



void Solver::Solve()
{
static double  gerror = 0.0;
bool converged = false;

printf("\nnumVertex = %u\n", numVertex);

while(!converged) {

#pragma omp parallel shared(converged, gerror)
{
double lerror = 0.0;

#pragma omp for
for (unsigned int i = 0; i < numVertex; i ++) {
compute(neighborhood[i], numNeighbors[i], value ,  tempValue + i);
lerror = MAX (lerror, fabs(tempValue[i] - value[i]));
}

#ifndef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
gerror = MAX(gerror, lerror);
#ifndef OPTIMAL
TRANSACTION_END
#endif
} 

if(gerror <= epsilon)
converged = true;
gerror = 0.0;
iterations++;

double * t = value;
value = tempValue;
tempValue = t;
}

resultValue = value;
}


void Solver::PrintSummery()
{
printf("\n\nIterations: %ld\n", iterations);
}


void Solver::Finish()
{
dwarfConfigurator -> Close(iterations, numVertex, resultValue);
}




void ParseCommandLine(int argc, char **argv)
{
char c;

input_file_name = (char*) "largecontent.txt";
num_threads = 1;
print_results = false;

while ((c = getopt (argc, argv, "hf:n:r")) != -1)
switch (c) {
case 'h':
printf("\nUnstructured Grid - ADF benchmark application\n"
"\n"
"Usage:\n"
"   unstructured_grid_seq [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -f <filename>\n"
"        Input file name.\n"
"   -n <long>\n"
"        Number of worker threads. (default 1)\n"
"   -r \n"
"        print results to a result file\n"
"\n"
);
exit (0);
case 'f':
input_file_name = optarg;
break;
case 'n':
num_threads = strtol(optarg, NULL, 10);
break;
case 'r':
print_results = true;
break;
case '?':
if (optopt == 'c')
fprintf (stderr, "Option -%c requires an argument.\n", optopt);
else if (isprint (optopt))
fprintf (stderr, "Unknown option `-%c'.\n", optopt);
else
fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
exit(1);
default:
exit(1);
}

if(input_file_name == NULL) {
printf("missing input file!\n");
exit(1);
}

printf ("\nStarting openmp Finite Unstructured Grid.\n");
printf ("Running with %d threads. Input file %s\n", num_threads, input_file_name);
printf ("=====================================================\n\n");
}








int main(int argc, char** argv)
{
ParseCommandLine(argc, argv);

Configurator dwarfConfigurator(input_file_name);
Solver solver(&dwarfConfigurator);					

NonADF_init(num_threads);

omp_set_num_threads(num_threads);
omp_set_nested(1);

solver.Solve();										

solver.PrintSummery();

NonADF_terminate();

solver.Finish();									

return 0;
}



