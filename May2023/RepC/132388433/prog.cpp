#include "stdio.h"
#include <omp.h>
#include <ctime>
#include <math.h>
#include <stdlib.h>
int	NowYear;		
int	NowMonth;		
double	NowPrecip;		
double	NowTemp;		
double	NowHeight;		
int	    NowNumDeer;		
const double GRAIN_GROWS_PER_MONTH =		8.0;
const double ONE_DEER_EATS_PER_MONTH =		0.5;
const double AVG_PRECIP_PER_MONTH =		6.0;	
const double AMP_PRECIP_PER_MONTH =		6.0;	
const double RANDOM_PRECIP =			2.0;	
const double AVG_TEMP =				50.0;	
const double AMP_TEMP =				20.0;	
const double RANDOM_TEMP =			10.0;	
const double MIDTEMP =				40.0;
const double MIDPRECIP =				10.0;
unsigned int seed = 0;  
double tempFactor = 0.0;
double precipFactor = 0.0;
double SQR( double x )
{
return x*x;
}
double Ranf( unsigned int *seedp,  double low, double high ){
double r = (double) rand_r( seedp );              
return(   low  +  r * ( high - low ) / (double)RAND_MAX   );
}
int Ranf( unsigned int *seedp, int ilow, int ihigh ){
double low = (double)ilow;
double high = (double)ihigh + 0.9999f;
return (int)(  Ranf(seedp, low,high) );
}
void getWeather(){
printf("Hello from Get Weather\n");
double ang = (  30.*(double)NowMonth + 15.  ) * ( M_PI / 180. );
double temp = AVG_TEMP - AMP_TEMP * cos( ang );
unsigned int seed = 0;
NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP );
double precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
if( NowPrecip < 0. )
NowPrecip = 0.;
tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
}
void Grain(){
printf("Hello from Grain\n");
while( NowYear < 2023 ){
double curHeight = NowHeight;
curHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
curHeight -= (double)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
if(curHeight < 0){
curHeight = 0;
}
#pragma omp barrier
printf("Done computing grain\n");
NowHeight = curHeight;
#pragma omp barrier
#pragma omp barrier
}
}
void GrainDeer(){
printf("Hello from Graindeer\n");
while(NowYear < 2023 ){
double curDeerPop = NowNumDeer;
if(curDeerPop > NowHeight){
curDeerPop--;
}
else if (curDeerPop < NowHeight){
curDeerPop ++;
}
else{
curDeerPop == curDeerPop;
}
if(curDeerPop <0){
curDeerPop = 1;
}
#pragma omp barrier
printf("Done computing deer\n");
NowNumDeer = curDeerPop;
#pragma omp barrier
#pragma omp barrier
}
}
void printState(){
FILE *fp;
fp = fopen("graindeer.txt", "a");
printf("Month: %d\n",NowMonth);
printf("Year: %d\n",NowYear);
fprintf (fp, "%d\t", NowMonth);
fprintf (fp, "%d\t", NowYear);
printf("Current height of the grain: %f\n", NowHeight);
printf("Current deer population: %d\n", NowNumDeer);
fprintf (fp, "%f\t", NowHeight);
fprintf (fp, "%d\t", NowNumDeer);
printf("Current temp: %f\n", NowTemp);
printf("Current precip: %f\n", NowPrecip);
fprintf (fp, "%f\t", NowTemp);
fprintf (fp, "%f\t", NowPrecip);
fprintf (fp, "\n");
printf("***********\n");
fclose(fp);
}
void Watcher(){
printf("Hello from Watcher\n");
while( NowYear < 2023 ){   
#pragma omp barrier
#pragma omp barrier
NowMonth++;
if(NowMonth >11){
NowYear++;
NowMonth = 0;
}
getWeather();
printState();
#pragma omp barrier
}
}
int main( int argc, char *argv[ ] ){
printf("Hello from main!\n");
NowMonth =    0;
NowYear  = 2017;
NowNumDeer = 1;
NowHeight =  1.;
getWeather();
printState();
omp_set_num_threads(3);	
#pragma omp parallel sections
{
#pragma omp section
{
printf("About to call GrainDeer()\n");
GrainDeer();
}
#pragma omp section
{
Grain();
}
#pragma omp section
{
Watcher();
}
}    
printf("We're done!\n");
return 0;
}