#include <cstdio>
#include <getopt.h>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "adf.h"
#include "list.h"


using namespace std;

typedef char mychar;

bool  print_results;
char *input_file_name;
int   num_threads;

#define OPTIMAL



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

sprintf(dwarfName, "%s", "FiniteStateMachine");
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
sprintf(stringSettings, "%s%s", stringSettings, "\nProfileFile       : ");


printf("%s", stringSettings);

delete[] stringSettings;
}




class Configurator
{
public:
Configurator(char *input_file);
~Configurator();
void GetContent(mychar **mainContent, long *contentSize, mychar **patternContent, int *stateDimensionCoincidence);
void WriteSettings();
void Close(list<long> & positions);

private:
Settings* settings;
};



Configurator::Configurator(char *input_file)
{
settings = new Settings(input_file);
}

Configurator::~Configurator()
{
delete settings;
}

void Configurator :: WriteSettings()
{
settings -> PrintStringSettings();
}






void Configurator :: GetContent(mychar **mainContent, long *contentSize, mychar **patternContent, int *stateDimensionCoincidence)
{
int size = 0;
size_t ret;
FILE *file = fopen(settings -> inputFile, "rb");		
int currentChar = getc(file);
while (currentChar != 1)								
{
currentChar = getc(file);
size++;
}

*stateDimensionCoincidence = size;
rewind (file);											

*patternContent = new mychar[*stateDimensionCoincidence];
ret = fread (*patternContent, sizeof(char), *stateDimensionCoincidence, file);		

fseek (file, 0 , SEEK_END);								
*contentSize = ftell (file) - size;		     		    

rewind (file);											

currentChar = getc(file);								
while (currentChar != 1) {								
currentChar = getc(file);
}

*mainContent = new mychar[*contentSize];
ret = fread (*mainContent, sizeof(char), *contentSize, file);			

fclose(file);
}

void Configurator :: Close(list<long> & positions)
{
if (print_results == true) {
FILE *result = fopen(settings -> resultFile, "wb");
if (! positions.empty()) {
while(! positions.empty()){		
fprintf(result, "Position: %ld\r\n", positions.front());
positions.pop_front();		
}
}
else {
fprintf(result, "%s\r", "Pattern not found");
}
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
static const int stateDimensionChars = 65534;				

Configurator	 *dwarfConfigurator;
list<long> 		  positions;								

int 			**states;									
int 			  stateDimensionCoincidence;				
mychar 			 *patternContent;							
long 			  contentSize;								
mychar 			 *mainContent;								

void ConstructStates();										
void StatesProcessing(long myBeginPoint, long myEndPoint);
__attribute__ ((noinline)) void PushBack(long i);
};



Solver :: Solver(Configurator* configurator)
{
states = NULL;
patternContent = NULL;
mainContent = NULL;

dwarfConfigurator = configurator;							
dwarfConfigurator -> WriteSettings();
dwarfConfigurator -> GetContent(&mainContent, &contentSize, &patternContent, &stateDimensionCoincidence);
}

Solver :: ~Solver() {
delete[] mainContent;
for (int i = 0; i < stateDimensionChars; i++)
delete[] states[i];
delete[] states;
}

void Solver::ConstructStates()
{
states = new int*[stateDimensionChars];

for (int i = 0; i < stateDimensionChars; i++) {
states[i] = new int[stateDimensionCoincidence + 1];
}

for (int i = 0; i < stateDimensionCoincidence + 1; i++) {
for (int j = 0; j < stateDimensionChars; j++) {
states[j][i] = 0;
}
}

for (int i = 0; i < stateDimensionCoincidence; i++) {
states[(int) patternContent[0]][i] = 1;						
}

for (int i = 1; i < stateDimensionCoincidence; i++) {
int currentState = states[(int) patternContent[i]] [i];
if (currentState > 0) {										
states[(int) patternContent[currentState]] [i + 1] = currentState + 1;    
}
states[(int) patternContent[i]] [i] = i + 1;				
}

delete[] patternContent;										
}


__attribute__ ((noinline)) void Solver::PushBack(long i) {
#ifdef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
positions.push_back(i);							
#ifdef OPTIMAL
TRANSACTION_END
#endif
}

void Solver :: StatesProcessing(long myBeginPoint, long myEndPoint)
{
int state = 0;

for (long i = myBeginPoint; i < myEndPoint; i++)
{
if (mainContent[i] > 0) {
state = states[(int) mainContent[i]][state];	
}

if (state == stateDimensionCoincidence) {			
PushBack(i - stateDimensionCoincidence);
state = 0;
} 
} 
}



void Solver::Solve()
{
long partSize = max( contentSize / (num_threads * 100 ), (long) 2 * stateDimensionCoincidence);

long beginPoint = 0, endPoint = partSize;

ConstructStates();													

#pragma omp parallel
{
#pragma omp single
{
while (beginPoint < contentSize) {

#pragma omp task firstprivate(beginPoint, endPoint)
{
STAT_COUNT_EVENT(task_exec, omp_get_thread_num());
#ifndef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif

long myEndPoint = endPoint + stateDimensionCoincidence;
if (myEndPoint > contentSize) myEndPoint = contentSize;

StatesProcessing(beginPoint, myEndPoint);

#ifndef OPTIMAL
TRANSACTION_END
#endif
} 

beginPoint += partSize;
endPoint += partSize;

} 
} 
}

}

void Solver::PrintSummery()
{
if (! positions.empty()) {
printf("\n\nTotal count of pattern entries: %d \n", (int) positions.size());
}
else {
printf("\n\nPattern not found\n");
}
}



void Solver::Finish()
{
dwarfConfigurator -> Close(positions);
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
printf("\nFinite State Machine - ADF benchmark application\n"
"\n"
"Usage:\n"
"   finite_state_machine_omp [options ...]\n"
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

printf ("\nStarting openmp Finite State Machine.\n");
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
