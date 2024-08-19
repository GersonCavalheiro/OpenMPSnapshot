#include <cstdio>
#include <getopt.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <omp.h>

#include "adf.h"


using std::string;
using std::map;


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

sprintf(dwarfName, "%s", "MapReduce");
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
void GetContent(char **mainContent, int *contentSize);
void WriteSettings() { settings -> PrintStringSettings(); }
void Close(map<string, int> & stringTotal);

private:
Settings* settings;
};

void Configurator :: GetContent(char **mainContent, int *contentSize)
{
int ret;
FILE *file = fopen(settings -> inputFile, "rb");				
fseek (file, 0 , SEEK_END);										
*contentSize = ftell (file);									

rewind (file);													
*mainContent = new char[*contentSize];							
ret = fread (*mainContent, sizeof(char), *contentSize, file);	

fclose(file);
}

void Configurator::Close(map<string, int> & stringTotal)
{
if (print_results == true) {
FILE *result = fopen(settings -> resultFile, "wb");
for (map<string, int>::const_iterator it = stringTotal.begin(); it != stringTotal.end(); ++it)
{												
string key = it -> first;					
int value = it -> second;					
fprintf(result, "%s", key.c_str());			
fprintf(result, " %i\r\n", value);
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
Configurator	 *dwarfConfigurator;

map<string, int>  stringTotal;			 	
std::list<string>      stringEntries;	    
int               contentSize;				
char              *mainContent;				

string TransformCharFragment(int begin, int end);
void GetContent();
void Map(std::list<string> &partialStringEntries, int start, int end);
void Reduce(map<string, int> &partialStringTotal, std::list<string> &partialStringEntries);
void SumThreadResult(map<string, int> &partialStringTotal);
};

Solver::Solver(Configurator* configurator)
{
mainContent = NULL;
contentSize = 0;

dwarfConfigurator = configurator;					
dwarfConfigurator -> WriteSettings();
dwarfConfigurator -> GetContent(&mainContent, &contentSize);
}

Solver::~Solver()
{
delete[] mainContent;
}


string Solver::TransformCharFragment(int begin, int end)
{
int length = end - begin;
char* currentWord = new char[length];									
for (int j = 0; j < length; j++) {
currentWord[j] = mainContent[begin + j];							
}
string transform(reinterpret_cast<const char*>(currentWord), length);	
delete[] currentWord;													
return transform;
}


void Solver::Map(std::list<string> &partialStringEntries, int start, int end)
{
int i = start;									
int count = 0;									
while (i < end)
{
char c = mainContent[i];
if ((c > 47 && c < 58)  ||
(c > 96 && c < 123) ||
(c > 64 && c < 91)  ||
c == 95			||
c == 45			||
c == 39)
{
count++;															
}
else
{
if (count > 0)
{
string stringFragment = TransformCharFragment(i - count, i);
partialStringEntries.push_back(stringFragment);					
}
string stringFragment = TransformCharFragment(i, i + 1);
partialStringEntries.push_back(stringFragment);						
count = 0;															
}
i++;
}
}


void Solver::Reduce(map<string, int> &partialStringTotal, std::list<string> &partialStringEntries)
{
partialStringEntries.sort();
int count = 1;										
string strCur;
string strPre = partialStringEntries.front();
partialStringEntries.pop_front();
while (!partialStringEntries.empty())				
{
strCur = partialStringEntries.front();
partialStringEntries.pop_front();
if (strCur == strPre)							
{
count++;                                    
}
else
{
partialStringTotal[strPre] = count;			
count = 1;									
strPre = strCur;
}
}
partialStringTotal[strPre] = count;					
partialStringEntries.clear();						
}


void Solver::SumThreadResult(map<string, int> &partialStringTotal)
{
for (map<string, int>::const_iterator it = partialStringTotal.begin(); it != partialStringTotal.end(); it++) {
string partialKey = it -> first;
int partialValue = it -> second;
map<string, int>::iterator ittotal;

#ifdef OPTIMAL
__transaction_relaxed {
#endif
ittotal =  stringTotal.find(partialKey);
if (ittotal != stringTotal.end()) {
stringTotal[partialKey] += partialValue;			
}
else {
stringTotal[partialKey] = partialValue;				
}
#ifdef OPTIMAL
TRANSACTION_END
#endif

}
partialStringTotal.clear();					
}


void Solver::Solve()
{
int partSize = 256*1024; 

int start = 0;
int end = partSize;

printf("\ncontentSize = %d\n", contentSize);

#pragma omp parallel
{
#pragma omp single
{
while (start < contentSize) {
if (end > contentSize) {
end = contentSize;
}
else {
while (mainContent[end] != ' ')  {						
end++;
}
end++;
}

#pragma omp task firstprivate(start, end)
{
STAT_COUNT_EVENT(task_exec, omp_get_thread_num());
#ifndef OPTIMAL
TRANSACTION_BEGIN(1,RW)
#endif
map<string, int> partialStringTotal;
std::list<string> partialStringEntries;
Map(partialStringEntries, start, end);					
Reduce(partialStringTotal, partialStringEntries);		
SumThreadResult(partialStringTotal);
#ifndef OPTIMAL
TRANSACTION_END
#endif
} 
start = end;
end += partSize;
} 
} 
} 

}


void Solver::PrintSummery()
{
printf("\nNumber of pairs : %ld\n", (long) stringTotal.size());
}


void Solver::Finish()	
{
dwarfConfigurator->Close(stringTotal);
stringTotal.clear();			
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
printf("\nMap-Reduce - ADF benchmark application\n"
"\n"
"Usage:\n"
"   map_reduce_omp [options ...]\n"
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

printf ("\nStarting openmp Map-Reduce.\n");
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


