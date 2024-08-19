#include <cstdio>
#include <getopt.h>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "adf.h"


using namespace std;

typedef char mychar;

char *input_file_name;
int   num_threads;

int num_blocks = 25;
int block_size = 50;



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

sprintf(dwarfName, "%s", "SparseAlgebra");
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

printf("%s\n", stringSettings);

delete[] stringSettings;
}




class Configurator
{
public:
Configurator(char *input_file) { settings = new Settings(input_file); }
~Configurator() { delete settings; }
void GetContent(float ***matrix);
void WriteSettings() { settings -> PrintStringSettings(); }
void Close();

private:
Settings* settings;
};



void genmat (float *M[])
{
int  init_val, i, j, ii, jj;
bool null_entry;
float *p;

init_val = 1325;


for (ii=0; ii < num_blocks; ii++)
{
for (jj=0; jj < num_blocks; jj++)
{

null_entry=false;
if ((ii<jj) && (ii%3 !=0)) null_entry = true;
if ((ii>jj) && (jj%3 !=0)) null_entry = true;
if (ii%2==1) null_entry = true;
if (jj%2==1) null_entry = true;
if (ii==jj) null_entry = false;
if (ii==jj-1) null_entry = false;
if (ii-1 == jj) null_entry = false;

if (null_entry == false){
M[ii*num_blocks+jj] =  new float[block_size*block_size];
if ((M[ii*num_blocks+jj] == NULL))
{
printf("Error: Out of memory\n");
exit(1);
}

p = M[ii*num_blocks+jj];
for (i = 0; i < block_size; i++)
{
for (j = 0; j < block_size; j++)
{
init_val = (3125 * init_val) % 65536;
(*p) = (float)((init_val - 32768.0) / 16384.0);
p++;
}
}
}
else
{
M[ii*num_blocks+jj] = NULL;
}
}
}
}




void Configurator :: GetContent(float ***matrix)
{
*matrix = new float*[num_blocks*num_blocks];
genmat(*matrix);
}

void Configurator :: Close()
{
}









class Solver
{
public:
Solver(Configurator* configurator);
~Solver();
void Solve();
void Finish();

private:
Configurator	 *dwarfConfigurator;
float 			**matrix;

float *allocate_clean_block();
void lu0(float *diag);
void bdiv(float *diag, float *row);
void bmod(float *row, float *col, float *inner);
void fwd(float *diag, float *col);

};

Solver::Solver(Configurator* configurator)
{
matrix = NULL;

dwarfConfigurator = configurator;							
dwarfConfigurator -> WriteSettings();
dwarfConfigurator -> GetContent(&matrix);
}

Solver::~Solver()
{
delete [] matrix;
}


float * Solver::allocate_clean_block()
{
int i,j;
float *p, *q;

p = new float [block_size*block_size];
q=p;
if (p!=NULL){
for (i = 0; i < block_size; i++)
for (j = 0; j < block_size; j++){(*p)=0.0; p++;}

}
else {
printf("Error: Out of memory\n");
}
return (q);
}

void Solver::lu0(float *diag)
{
int i, j, k;

for (k=0; k<block_size; k++)
for (i=k+1; i<block_size; i++)
{
diag[i*block_size+k] = diag[i*block_size+k] / diag[k*block_size+k];
for (j=k+1; j<block_size; j++)
diag[i*block_size+j] = diag[i*block_size+j] - diag[i*block_size+k] * diag[k*block_size+j];
}
}


void Solver::bdiv(float *diag, float *row)
{
int i, j, k;
for (i=0; i<block_size; i++)
for (k=0; k<block_size; k++)
{
row[i*block_size+k] = row[i*block_size+k] / diag[k*block_size+k];
for (j=k+1; j<block_size; j++)
row[i*block_size+j] = row[i*block_size+j] - row[i*block_size+k]*diag[k*block_size+j];
}
}

void Solver::bmod(float *row, float *col, float *inner)
{
int i, j, k;
for (i=0; i<block_size; i++)
for (j=0; j<block_size; j++)
for (k=0; k<block_size; k++)
inner[i*block_size+j] = inner[i*block_size+j] - row[i*block_size+k]*col[k*block_size+j];
}

void Solver::fwd(float *diag, float *col)
{
int i, j, k;
for (j=0; j<block_size; j++)
for (k=0; k<block_size; k++)
for (i=k+1; i<block_size; i++)
col[i*block_size+j] = col[i*block_size+j] - diag[i*block_size+k]*col[k*block_size+j];
}


void Solver::Solve()
{
int ii, jj, kk;

printf("\nComputing SparseLU Factorization (%dx%d matrix with %dx%d blocks) \n",
num_blocks,num_blocks,block_size,block_size);


#pragma omp parallel private(kk)
{
for (kk=0; kk<num_blocks; kk++)
{
#pragma omp single
{
lu0(matrix[kk*num_blocks+kk]);

for (jj=kk+1; jj<num_blocks; jj++)
if (matrix[kk*num_blocks+jj] != NULL)
{
#pragma omp task untied firstprivate(kk, jj)
fwd(matrix[kk*num_blocks+kk], matrix[kk*num_blocks+jj]);
}

for (ii=kk+1; ii<num_blocks; ii++)
if (matrix[ii*num_blocks+kk] != NULL)
{
#pragma omp task untied firstprivate(kk, ii)
bdiv (matrix[kk*num_blocks+kk], matrix[ii*num_blocks+kk]);
}
} 


#pragma omp single
{
for (ii=kk+1; ii<num_blocks; ii++)
if (matrix[ii*num_blocks+kk] != NULL)
for (jj=kk+1; jj<num_blocks; jj++)
if (matrix[kk*num_blocks+jj] != NULL)
{
#pragma omp task untied firstprivate(kk, jj, ii)
{
if (matrix[ii*num_blocks+jj]==NULL)
matrix[ii*num_blocks+jj] = allocate_clean_block();
bmod(matrix[ii*num_blocks+kk], matrix[kk*num_blocks+jj], matrix[ii*num_blocks+jj]);
}
}
} 
} 
} 
}


void Solver::Finish()
{
dwarfConfigurator -> Close();
}





void ParseCommandLine(int argc, char **argv)
{
char c;

input_file_name = (char*) "no file - input generated";
num_threads = 1;

while ((c = getopt (argc, argv, "hf:b:s:n:")) != -1)
switch (c) {
case 'h':
printf("\nSparse Algebra - ADF benchmark application\n"
"\n"
"Usage:\n"
"   sparse_algebra_omp [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -f <filename>\n"
"        Input file name.\n"
"   -b <num_blocks>\n"
"        Number of blocks in the row.\n"
"   -s <block_size>\n"
"        Size of the block.\n"
"   -n <long>\n"
"        Number of worker threads. (default 1)\n"
"\n"
);
exit (0);
case 'f':
input_file_name = optarg;
break;
case 'b':
num_blocks = atoi(optarg);
break;
case 's':
block_size = atoi(optarg);
break;
case 'n':
num_threads = strtol(optarg, NULL, 10);
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

if(num_blocks < 5)
num_blocks = 5;

if(block_size < 25)
block_size = 25;

printf ("\nStarting openmp Sparse Algebra.\n");
printf ("Input file %s\n", input_file_name);
printf ("Matrix with %d x %d blocks of size %d\n", num_blocks, num_blocks, block_size);
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

NonADF_terminate();

solver.Finish();									

return 0;
}



