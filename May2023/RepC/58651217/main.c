#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "collatz.h"
void printHelp();
void writeIter(int* result, int startNumber, int endNumber, char* filename);
void writeBench(double* timings, int maxThreads, int startNumber, int endNumber, char* filename);
void benchmark(int startNumber, int endNumber, int maxThreads, double* timings);
int main(int argc, char* argv[]) {
int i; 
int append = 0; 
int bench = 0; 
int nThreads = 1; 
int inputRequired = 1; 
int arg; 
char flag; 
char *flags = "abhov";
char *flagRequireInputs = "ot"; 
char *flagInputs = "11"; 
char *flagIndex; 
char outputFileName[200] = "results.txt";  
int startNumber = 1; 
int endNumber = 10000; 
if (argc == 1)
printHelp();
argv[argc] = "";
for (arg = 1; argv[arg][0] == '-';) {
flag = argv[arg++][1];
if (flagIndex = strchr(flagRequireInputs, flag))
for (i = 0; i < flagInputs[flagIndex - flagRequireInputs] - '0'; i++)
if (!isalnum(argv[arg + i][0])) {
fprintf(stderr, ("Non-alphanumerical argument to -%c\n"), flag);
return -2;
}
switch (flag) {
case 'a':
append = 1;
break;
case 'b':
bench = 1;
break;
case 'o':
strcpy(outputFileName, argv[arg++]);
break;
case 'h':
printHelp();
inputRequired = 0;
break;
case 't':
nThreads = argv[arg++];
break;
case 'v':
printf("version 1.0\n");
inputRequired = 0;
break;
default:
fprintf(stderr, ("Unknown option \"-%c\".\n"), flag);
return -3;
}
}
if (arg == argc) {
if (inputRequired);	
else
return 0;
}
else if (arg + 1 == argc) 
endNumber = atol(argv[arg++]);
else if (arg + 2 == argc) { 
startNumber = atol(argv[arg++]);
endNumber = atol(argv[arg++]);
}
else
printf("");
double* timings;
if (bench == 1) {
int maxThreads;
#pragma omp parallel
maxThreads = omp_get_num_threads();
double* timings = malloc(maxThreads * sizeof(double));
benchmark(startNumber, endNumber, maxThreads, timings);
writeBench(timings, maxThreads, startNumber, endNumber, outputFileName);
free(timings);
}
else {
int nNumbers = endNumber - startNumber + 1;
int* iter = malloc(nNumbers*sizeof(int));
omp_set_num_threads(nThreads);
collatz(startNumber, endNumber, iter, nThreads);
writeIter(iter, startNumber, endNumber, outputFileName);
free(iter);
}
}
void printHelp() {
printf("\nUsage:   Collatz [OPTIONS] startNumber endNumber\n\n");
printf("\t OPTIONS\n");
printf("\t -a: append the output file instead of overwriting it\n");
printf("\t -b: write the benchmark results to file, not the iteration values\n");
printf("\t -h: print this help message\n");
printf("\t -o OUTPUT: save results to the OUTPUT location\n");
printf("\t -v: program version\n\n");
}