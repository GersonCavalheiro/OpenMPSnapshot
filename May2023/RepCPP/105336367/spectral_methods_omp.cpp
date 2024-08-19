#include <cstdio>
#include <getopt.h>
#include <cstring>
#include <iostream>
#include <sfftw.h>
#include <omp.h>

#include "adf.h"


using namespace std;

typedef char mychar;

int   num_threads;
int   num_elements = 256;

#define GET_I(y,z)     ( ((z)*(num_elements)) + (y) )

#define OPTIMAL



class Settings
{
public:
char* dwarfName;

Settings();
~Settings();
void PrintStringSettings();

private:
static const int maxArraySize = 5000;
};


Settings::Settings()
{
dwarfName = new char[maxArraySize];
dwarfName[0] = '\0';
sprintf(dwarfName, "%s", "SpectralMethods");
}

Settings::~Settings()
{
delete[] dwarfName;
}

void Settings::PrintStringSettings()
{
char* stringSettings = new char[maxArraySize];
stringSettings[0] = '\0';

sprintf(stringSettings, "%s%s", stringSettings, "Kernel settings summary: ");
sprintf(stringSettings, "%s%s", stringSettings, "\nDwarf name        : ");
sprintf(stringSettings, "%s%s", stringSettings, dwarfName);

printf("%s", stringSettings);

delete[] stringSettings;
}




class Configurator
{
public:
Configurator() { settings = new Settings(); }
~Configurator() { delete settings; }
void GetContent(fftw_complex*** data, fftw_plan *xPlan);
void WriteSettings() { settings -> PrintStringSettings(); }
void Close(fftw_complex** data) {};

private:
Settings* settings;
};



void Configurator :: GetContent(fftw_complex*** data, fftw_plan *xPlan)
{
srand(0);

*data = new fftw_complex*[num_elements * num_elements];
for (int i = 0; i < (num_elements * num_elements); i++) {
(*data)[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * num_elements);
for (int j = 0; j < num_elements; j++) {
(*data)[i][j].re = ((j % num_elements == 0) ? (1.0f) : (0.0f));  
(*data)[i][j].im = 0.0f;  
}
}

*xPlan = fftw_create_plan(num_elements, FFTW_FORWARD, FFTW_IN_PLACE | FFTW_ESTIMATE);
}






class Solver
{
public:
Solver(Configurator* configurator);
~Solver();
void Solve();
void Finish();

fftw_complex  **data;		
fftw_plan       xPlan;		

private:
Configurator	 *dwarfConfigurator;
};



Solver :: Solver(Configurator* configurator)
{
data = NULL;

dwarfConfigurator = configurator;					
dwarfConfigurator -> WriteSettings();
dwarfConfigurator -> GetContent(&data, &xPlan);
}


Solver :: ~Solver()
{
fftw_destroy_plan(xPlan);

for (int i = 0; i < (num_elements * num_elements); i++)
if (data[i] != NULL) { fftw_free(data[i]); data[i] = NULL; }
delete [] data;
}


void Solver::Solve()
{
#pragma omp parallel for
for (int i = 0; i < num_elements * num_elements; i++)
fftw_one(xPlan, data[i], NULL);  


#pragma omp parallel for
for (int z = 0; z < num_elements; z++)
for (int y = 0; y < num_elements; y++)
for (int x = y + 1; x < num_elements; x++) {
fftw_complex tmp;
tmp.re = data[GET_I(y,z)][x].re;
tmp.im = data[GET_I(y,z)][x].im;
data[GET_I(y,z)][x].re = data[GET_I(x,z)][y].re;
data[GET_I(y,z)][x].im = data[GET_I(x,z)][y].im;
data[GET_I(x,z)][y].re = tmp.re;
data[GET_I(x,z)][y].im = tmp.im;
}

#pragma omp parallel for
for (int i = 0; i < num_elements * num_elements; i++)
fftw_one(xPlan, data[i], NULL);  


#pragma omp parallel for
for (int y = 0; y < num_elements; y++)
for (int z = 0; z < num_elements; z++)
for (int x = z + 1; x < num_elements; x++) {
fftw_complex tmp;
tmp.re = data[GET_I(y,z)][x].re;
tmp.im = data[GET_I(y,z)][x].im;
data[GET_I(y,z)][x].re = data[GET_I(y,x)][z].re;
data[GET_I(y,z)][x].im = data[GET_I(y,x)][z].im;
data[GET_I(y,x)][z].re = tmp.re;
data[GET_I(y,x)][z].im = tmp.im;
}


#pragma omp parallel for
for (int i = 0; i < num_elements * num_elements; i++)
fftw_one(xPlan, data[i], NULL);  



}


void Solver::Finish()
{
dwarfConfigurator -> Close(data);
}




void ParseCommandLine(int argc, char **argv)
{
char c;

num_threads = 1;

while ((c = getopt (argc, argv, "he:n:")) != -1)
switch (c) {
case 'h':
printf("\nSpectral Methods - ADF benchmark application\n"
"\n"
"Usage:\n"
"   spectral_methods_omp [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -e <int>\n"
"        Number of matrix elements (default 256)\n"
"   -n <long>\n"
"        Number of worker threads. (default 1)\n"
"\n"
);
exit (0);
case 'e':
num_elements = atoi(optarg);
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

if (num_elements < 100)
num_elements = 100;

printf ("\nStarting openmp Spectral Methods.\n");
printf ("Matrix dimension is %d.\n", num_elements);
printf ("Running with %d threads.\n", num_threads);
printf ("=====================================================\n\n");
}








int main(int argc, char** argv)
{
ParseCommandLine(argc, argv);

Configurator dwarfConfigurator;
Solver solver(&dwarfConfigurator);					

NonADF_init(num_threads);

omp_set_num_threads(num_threads);
omp_set_nested(1);

solver.Solve();										

NonADF_terminate();

solver.Finish();									

return 0;
}

