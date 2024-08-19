#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <papi.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>


using namespace std;

#define SYSTEMTIME struct timespec

ofstream logFile;
float papi_real_time, papi_proc_time, papi_mflops;
long long papi_flpins;


void papiHandleError (std::string msg, int ret)
{
cout << msg << endl;
printf("PAPI error %d: %s\n", ret, PAPI_strerror(ret));
exit(1);
}





int papiSubscribeEvents(){
int ret, eventSet = PAPI_NULL;


ret = PAPI_library_init(PAPI_VER_CURRENT);
if (ret != PAPI_VER_CURRENT && ret < 0) {
printf("PAPI library version mismatch!\n");
exit(1);
}
if (ret < 0) papiHandleError("PAPI_library_init", ret);


std::cout << "PAPI Version Number: MAJOR: " << PAPI_VERSION_MAJOR(ret)
<< " MINOR: " << PAPI_VERSION_MINOR(ret)
<< " REVISION: " << PAPI_VERSION_REVISION(ret) << "\n";

ret = PAPI_create_eventset(&eventSet);
if (ret != PAPI_OK) cout << "ERRO: create eventset" << endl;


ret = PAPI_add_event(eventSet,PAPI_L1_DCM);
if (ret != PAPI_OK) papiHandleError("PAPI_add_event PAPI_L1_DCM", ret);


ret = PAPI_add_event(eventSet,PAPI_L2_DCM);
if (ret != PAPI_OK) papiHandleError("PAPI_add_event PAPI_L2_DCM", ret);






return eventSet;
}

void papiUnsubscribeEvents(int eventSet){
int ret;
ret = PAPI_remove_event( eventSet, PAPI_L1_DCM );
if ( ret != PAPI_OK ) cout << "FAIL remove event" << endl;

ret = PAPI_remove_event( eventSet, PAPI_L2_DCM );
if ( ret != PAPI_OK ) cout << "FAIL remove event" << endl;

}


void papiStartCounter(int eventSet){
int ret;
ret = PAPI_start(eventSet);
if (ret != PAPI_OK) cout << "ERRO: Start PAPI" << endl;

}

void papiResetCounter(int eventSet){
int ret;

ret = PAPI_reset( eventSet );
if ( ret != PAPI_OK ) cout << "FAIL reset" << endl;

}





void initMatrices(double ** pha, double ** phb, double ** phc, int m_ar, int m_br){
long int i,j;


*pha = (double *)malloc((m_ar * m_br) * sizeof(double));
*phb = (double *)malloc((m_ar * m_br) * sizeof(double));
*phc = (double *)malloc((m_ar * m_ar) * sizeof(double));


for(i=0; i< m_ar; i++)
for(j=0; j< m_br; j++) {
(*pha)[i*m_br + j] = (double)1.0;
}


for(i=0; i< m_br; i++)
for(j=0; j< m_ar; j++) {
(*phb)[i*m_ar + j] = (double)(i+1);
}

for(i=0; i< m_ar; i++)
for(j=0; j< m_ar; j++) {
(*phc)[i*m_ar + j] = 0;
}

}                


void printResultMatrix(double ** phc, int m_br) {
int i,j;

cout << "Result matrix: " << endl;
for(i=0; i<1; i++)
{
for(j=0; j<min(10,m_br); j++)
cout << (*phc)[j] << " ";
}
cout << endl << endl;
}

void printLogFileHeaders(){
if(logFile.is_open()) {
logFile << "Function ID" << ";";
logFile << "Rows" << ";";
logFile << "Cols" << ";";
logFile << "Number of threads" << ";";
logFile << "Duration (seconds)" << ";";
logFile << "Performance (MFLOPS)" << ";";
logFile << "Theoretical performance (MFLOPS)" << ";";
logFile << "L1 DCM" << ";";
logFile << "L1 DCM" << ";";
logFile << "\n";

}
}

void printResults(SYSTEMTIME time1, int op, int m_ar, int m_br, int nthreads, int eventSet){

int ret;
long long values[3];
SYSTEMTIME time2;

if(clock_gettime(CLOCK_MONOTONIC,&time2) < 0) {
perror("Failure getting process time (2)");
exit(EXIT_FAILURE);
}



ret = PAPI_stop(eventSet, values);
if (ret != PAPI_OK) cout << "ERRO: Stop PAPI" << endl;




double delta = (double)(time2.tv_sec - time1.tv_sec) + ((double)(time2.tv_nsec - time1.tv_nsec) / 1000000000.0);

long long calcNrInsOp = ((long long)3*m_ar*m_br*m_br);
double tPerformance = calcNrInsOp/(double)(delta*1000000.0);


cout << "PAPI results:" << endl;
printf("L1 DCM: %lld \n",values[0]);
printf("L2 DCM: %lld \n",values[1]);


printf("T OPS: %lld \n", calcNrInsOp);


printf("Time  : %3.3f seconds\n", delta);
printf("T.Prfr: %3.3f MFLOPS\n", tPerformance);




if(logFile.is_open()) {
logFile << op << ";";
logFile << m_ar << ";";
logFile << m_br << ";";
logFile << nthreads << ";";
logFile << delta << ";";
logFile << tPerformance << ";";
logFile << values[0] << ";";
logFile << values[1] << ";\n";
}
}


void freeMatrices(double * pha, double * phb, double * phc){
free(pha);
free(phb);
free(phc);
}


void multMatrices(double * pha, double * phb, double * phc, int m_ar, int m_br);
void multMatricesLineByLine(double * pha, double * phb, double * phc, int m_ar, int m_br);
void multMatricesParallel(double * pha, double * phb, double * phc, int m_ar, int m_br, int nthreads);
void multMatricesParallelLineByLine(double * pha, double * phb, double * phc, int m_ar, int m_br, int nthreads);


void multiplyMatrices(int eventSet, int m_ar, int m_br, int op, int nthreads)
{
double *pha, *phb, *phc;
SYSTEMTIME time1;

initMatrices(&pha,&phb,&phc, m_ar,m_br);


if(clock_gettime(CLOCK_MONOTONIC,&time1) < 0) {
perror("Failure getting process time (1)");
exit(EXIT_FAILURE);
}
papiStartCounter(eventSet);


switch(op) {
case 1:
multMatrices(pha,phb,phc,m_ar,m_br);
break;
case 2:
multMatricesLineByLine(pha,phb,phc,m_ar,m_br);
break;
case 3:
multMatricesParallel(pha,phb,phc,m_ar,m_br,nthreads);
break;
case 4:
multMatricesParallelLineByLine(pha,phb,phc,m_ar,m_br,nthreads);
break;

}
printResults(time1,op, m_ar,m_br,nthreads,  eventSet);
printResultMatrix(&phc,m_br);

freeMatrices(pha,phb,phc);
}

bool showMenu(int & lin, int & col, int & op, int & nthreads){

cout << endl;
cout << endl;
cout << "======================================" << endl;
cout << "1. Multiplication" << endl;
cout << "2. Line Multiplication" << endl;
cout << "3. Parallel Multiplication" << endl;
cout << "4. Line Parallel Multiplication" << endl;
cout << "0. End" << endl << endl;
cout << "Selection?: ";
cin >> op;

if( op < 0 || op > 4) {
cout << "Bad option. " << endl;
return true;
} else if (op == 0) { 
return false;
}

printf("Dimensions: lins cols ? ");
cin >> lin >> col;

nthreads = 1;
if(op == 3 || op == 4) {
printf("Number of threads ? ");
cin >> nthreads;
cout << endl;
}

cout << endl;

cout << "Calculating ..." << endl;

return true;
}

void multMatrices(double * pha, double * phb, double * phc, int m_ar, int m_br)
{
double temp;
int i, j, k;

for(i=0; i<m_ar; i++)
{
for( j=0; j<m_ar; j++)
{ temp = 0;
for( k=0; k<m_br; k++)
{
temp += pha[i*m_ar+k] * phb[k*m_br+j];
}
phc[i*m_ar+j]=temp; }
}

}

void multMatricesLineByLine(double * pha, double * phb, double * phc, int m_ar, int m_br)
{
int i, j, k;


for(i=0; i<m_ar; i++)
{
for( k=0; k<m_br; k++)
{
for( j=0; j<m_ar; j++)
{
phc[i*m_ar+j] += (pha[i*m_ar+k] * phb[k*m_br+j]);

}

}
}


}

void multMatricesParallel(double * pha, double * phb, double * phc, int m_ar, int m_br, int nthreads)
{
double temp;
int i, j, k;

#pragma omp parallel for num_threads(nthreads) private(k) private(j) private(temp)
for(i=0; i<m_ar; i++)
{
for( j=0; j<m_ar; j++)
{ temp = 0;
for( k=0; k<m_br; k++)
{
temp += pha[i*m_ar+k] * phb[k*m_br+j];
}
phc[i*m_ar+j]=temp; }
}
}

void multMatricesParallelLineByLine(double * pha, double * phb, double * phc, int m_ar, int m_br, int nthreads)
{
int i, j, k;

#pragma omp parallel for num_threads(nthreads) private(k) private(j)
for(i=0; i<m_ar; i++)
{
for( k=0; k<m_br; k++)
{
for( j=0; j<m_ar; j++)
{
phc[i*m_ar+j] += (pha[i*m_ar+k] * phb[k*m_br+j]);

}

}
}


}


void mult(int min, int max, int inc, int op, int eventSet, int nthreads){
int n = (max - min) / inc;

for(int i = 0; i <= n; i++) {
multiplyMatrices(eventSet, min + inc * i, min + inc * i, op, nthreads);
}

papiResetCounter(eventSet);
}

void automaticTests(int eventSet, int test_number){

switch (test_number) {
case 1:
logFile.open("log_auto1.txt");
printLogFileHeaders();
cout << "================" << endl << "Automatic test 1" << endl << "================" << endl << endl;
mult(600, 3000, 400, 1, eventSet, 1);
break;
case 2:
logFile.open("log_auto2.txt");
printLogFileHeaders();
cout << "================" << endl << "Automatic test 2" << endl << "================" << endl << endl;
mult(600, 3000, 400, 2, eventSet, 1);
break;
case 25:
logFile.open("log_auto2-5.txt");
printLogFileHeaders();
cout << "================" << endl << "Automatic test 2.5" << endl << "================" << endl << endl;
for(int i = 0; i < 4; i++) {
mult(4000, 10000, 2000, 2, eventSet, 1);
}
case 3:
logFile.open("log_auto3.txt");
printLogFileHeaders();
cout << "================" << endl << "Automatic test 3" << endl << "================" << endl << endl;
for(int i = 0; i < 4; i++) {
mult(600, 3000, 400, 3, eventSet, i + 1);
}
case 4:
logFile.open("log_auto4.txt");
cout << "================" << endl << "Automatic test 4 " << endl << "================" << endl << endl;
for(int i = 0; i < 4; i++) {
mult(600, 3000, 400, 4, eventSet, i + 1);
}
break;
default:
cout << "There is no test with this number. Please choose 1, 2 or 3." << endl;
return;
}

papiUnsubscribeEvents(eventSet);

logFile.close();
}

int main (int argc, char *argv[])
{
bool cntinue = false;
int eventSet = papiSubscribeEvents();
int lin,col,op,nthreads;


if(argc > 1 && strcmp(argv[1],"-a") == 0) {
logFile.open("log.txt", ios::app | ios::ate);
}
else if(argc > 1 && strcmp(argv[1],"-t") == 0) {
if(argc == 2) {
cout << "Running default test. You can choose the test by using 1, 2 or 3 after the '-t'" << endl;
automaticTests(eventSet, 1);
} else
automaticTests(eventSet, atoi(argv[2]));
return 0;
}
else {
cout << endl << "Next time, you may want to use: «matrixprod -a» to append results on the log file." << endl;
cout << "You can also run automated tests by using «matrixprod -t 'test_number'»." << endl;
logFile.open("log.txt");
printLogFileHeaders();
}

do {
cntinue = showMenu(lin,col,op,nthreads);
if(cntinue) multiplyMatrices(eventSet,lin,col,op,nthreads);
papiResetCounter(eventSet);
} while (cntinue);

papiUnsubscribeEvents(eventSet);

cout << endl << "Bye!" << endl;
logFile.close();
}
