#include <cstdio>
#include <getopt.h>
#include <cmath>
#include <cstring>
#include <iostream>

#include "adf.h"
#include <omp.h>

using namespace std;

typedef char mychar;

bool  print_results;
char *input_file_name;
int   num_threads;

#define OPTIMAL

#define MAX_BLOCK_SIZE 64 




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
static const int maxArraySize = 1024;
};


Settings::Settings(char *input_file)
{
dwarfName = new char[maxArraySize];
inputFile = new char[maxArraySize];
resultFile = new char[maxArraySize];

dwarfName[0] = inputFile[0] = resultFile[0] = '\0';

sprintf(dwarfName, "%s", "CombinatorialLogic");
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
void Close(unsigned long length, unsigned long resultCount);

Settings* settings;
private:

};


void Configurator::Close(unsigned long length, unsigned long resultCount)
{
if (print_results == true) {
FILE *result = fopen(settings -> resultFile, "wb");

fprintf(result, "File length (byte) : %lu\n", length);
fprintf(result, "Result count: %lu\n", resultCount);

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

private:
Configurator	 *dwarfConfigurator;

unsigned long length;       
unsigned long resultCount;  
FILE 		  *input;
};

Solver::Solver(Configurator* configurator) {
dwarfConfigurator = configurator;
input = fopen(configurator->settings->inputFile, "rb");		

fseek(input, 0, SEEK_END );
length = ftell(input);
rewind(input);

resultCount = 0;
}

Solver::~Solver() {
if (input)
fclose(input);
}


void Solver::Solve()
{
unsigned int len;                               			

#pragma omp parallel
{
#pragma omp single
{
while (!feof(input)) {

uint64_t *buffer = new uint64_t[MAX_BLOCK_SIZE];    	
len = (unsigned int) fread((void*) buffer, 8, MAX_BLOCK_SIZE, input);

if (len < MAX_BLOCK_SIZE) {
*(buffer + len) = *(buffer + len) << 8 * (8 - length % 8);
len ++;
}

#pragma omp task firstprivate(len, buffer)
{
STAT_COUNT_EVENT(task_exec, omp_get_thread_num());
#ifndef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
uint64_t count = 0;
while (len --) {

for (int k = 0; k < 1000000; k ++) {
count = *(buffer + len);
count -= (count >> 1) & 0x5555555555555555;
count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
count = (count + (count >> 4)) & 0x0f0f0f0f0f0f0f0f;
}

#ifdef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
resultCount += count * 0x0101010101010101 >> 56;

TRANSACTION_END

}

delete[] buffer;
} 

} 
} 
} 

}


void Solver::PrintSummery()
{
printf("File length (byte) : %lu\n", length);
printf("Result count: %lu\n", resultCount);
}


void Solver::Finish()		
{
dwarfConfigurator -> Close(length, resultCount);
}






void ParseCommandLine(int argc, char **argv)
{
char c;

input_file_name = (char*) "smallcontent.txt";
num_threads = 1;
print_results = false;

while ((c = getopt (argc, argv, "hf:n:r")) != -1)
switch (c) {
case 'h':
printf("\nCombinatorial Logic - ADF benchmark application\n"
"\n"
"Usage:\n"
"   combinatorial_logic_omp [options ...]\n"
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

printf ("\nStarting openmp Combinatorial Logic.\n");
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

