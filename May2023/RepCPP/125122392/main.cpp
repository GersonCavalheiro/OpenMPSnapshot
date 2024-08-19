

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "omp.h"

#include "satellite.hpp"
#include "common.hpp"

#define FREQ_PERCENT_PRINT 0.1 
#define DELTA_TIME 1  

void update_satellite(satellite_t *sat, double delta_time);
void logStep(FILE *f, int &simTime, satellite_t *sat, int &satItter, double &x_eci, double &y_eci, double &z_eci);

int main(int argc, char **argv) {
FILE *fOut;

fOut = fopen ("output.txt", "w");
if (fOut == NULL) {
fOut = fopen("output.txt", "wb");
}

char buf[65536]; 
setvbuf(fOut, buf, _IOFBF, sizeof(buf));

char *inputFileName = (char *)"input.txt";

int numThreads = 0; 
int totalItter = (12 * 60 * 60) / DELTA_TIME; 
int secondBetweenOutputLog = 60; 

const char * helpString = \
"Usage: ./main [-time t (mins)] [-logfreq lf (seconds)] [-in i (input file)] [-threads n]";

for (int i = 1; i < argc; i++) {
if (strcmp(argv[i], "-time") == 0 || strcmp(argv[i], "-t") == 0) {
char* totalTimeStr = argv[i + 1];
totalItter = (atoi(totalTimeStr) * 60) / DELTA_TIME;
}
else if (strcmp(argv[i], "-logfreq") == 0 || strcmp(argv[i], "-lf") == 0) { 
char* str = argv[i + 1];
secondBetweenOutputLog = atoi(str);
}
else if (strcmp(argv[i], "-in") == 0 || strcmp(argv[i], "-i") == 0) { 
inputFileName = argv[i + 1];
}
else if (strcmp(argv[i], "-threads") == 0 || strcmp(argv[i], "-n") == 0) {
char* numThreadsStr = argv[i + 1];
numThreads = atoi(numThreadsStr);
printf("numThreads = %d\n", numThreads);
}
else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { 
printf("\n%s\n\n",helpString);
return 0;
}
}



int numberSats = 0;

printf("\n** Initing Sats ** %f\n", read_timer());

printf("Reading: %s\n", inputFileName);
satellite_t *satellites = loadCSVConfig(inputFileName, &numberSats);
if (satellites == NULL) {
printf("File `%s` could not be read (probably does not exist).\n", inputFileName);
return 0;
}


float total_lineOfSightSum = 0; 
int lineOfSightSum = 0; 

int total_ClosePasses = 0; 
int closePassesSum = 0; 



printf("** Starting Simulation (of %.0f min) with %i satellites ** %f\n", (totalItter*DELTA_TIME)/60.0, numberSats, read_timer());

const int freqPercentCount = (float)totalItter * FREQ_PERCENT_PRINT;
const int freqOutputLogComparator = secondBetweenOutputLog / DELTA_TIME; 

int curItter = 0;
int curSimTime = 0; 
for (int curItter = 0; curItter<totalItter; curItter++) {
curSimTime = curItter * DELTA_TIME;



lineOfSightSum = 0;
closePassesSum = 0;

double x_eci, y_eci, z_eci;
#pragma omp parallel for num_threads(numThreads) private (x_eci, y_eci, z_eci) reduction(+:lineOfSightSum) reduction(+:closePassesSum)
for(int i=0; i<numberSats; i++) {

double distance = -1;
satellites[i].getECI_XYZ(x_eci, y_eci, z_eci);

for(int j = i+1; j<numberSats; j++) { 
lineOfSightSum += satellitesHaveLineOfSight(x_eci, y_eci, z_eci, &satellites[j], distance);

if (distance < 100 && distance >= 0) { 
closePassesSum += 1;
}
}

}

#pragma omp parallel for num_threads(numThreads)
for(int i=0; i<numberSats; i++) {
update_satellite(&satellites[i], DELTA_TIME);
}

#pragma omp master
if (curItter % freqPercentCount == 0) {
printf("| %i%%\n", (int)(curItter/(float)totalItter * 100));
}

#pragma omp master
if (curItter % freqOutputLogComparator == 0) {
for(int i=0; i<numberSats; i++) {
satellites[i].getECI_XYZ(x_eci, y_eci, z_eci);
logStep(fOut, curSimTime, &satellites[i], i, x_eci, y_eci, z_eci);
}
fflush(fOut);
}

#pragma omp master
{
total_ClosePasses += closePassesSum;
closePassesSum = 0;
total_lineOfSightSum += lineOfSightSum/(float)numberSats;
}

} 


printf("** End Simulation ** took %f sec.\n", read_timer());

printf("Avg. Number lines of sight (per sat). %f\n", total_lineOfSightSum/(double)totalItter);
printf("Total Close Passes %i\n", total_ClosePasses);



free(satellites);

fflush(fOut);
fclose(fOut);
}

void update_satellite(satellite_t *sat, double delta_time) {
double mean_anoml = sat->true_to_mean_anoml(); 

mean_anoml += sat->calc_mean_motion() * delta_time;

sat->trueAnomaly = sat->mean_to_true_anoml(mean_anoml); 
}


void logStep(FILE *f, int &simTime, satellite_t *sat, int &satItter, double &x_eci, double &y_eci, double &z_eci) {
static int logItter = 0;
static int lastSimTime;

if (lastSimTime != simTime) {
lastSimTime = simTime;
logItter += 1; 
}



fprintf(f, "%d,%d,%d,%0.2f,%0.1f,%0.1f,%0.1f\n", logItter, simTime, satItter, radToDegPos(sat->trueAnomaly), x_eci, y_eci, z_eci);
}
