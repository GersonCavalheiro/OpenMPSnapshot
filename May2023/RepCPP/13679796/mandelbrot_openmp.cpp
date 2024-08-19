#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

int drawFractal(double positiveImaginary,double negativeImaginary,double positiveReal, double negativeReal,int threadsNumber,bool drawToConsole)
{
double realCoord, imagCoord;
double realTemp, imagTemp, realTemp2, arg;
double imaginaryStep = 0.05;
double realStep = 0.03;
int iterations,columns=0,lines=0,i=0,j=0;
columns = ceil((abs(positiveReal)+abs(negativeReal))/realStep);
lines = ceil((abs(positiveImaginary)+abs(negativeImaginary))/imaginaryStep);
int imageSize = columns*lines;
int operations;
char *image = new char[imageSize];
if (drawToConsole)
{
cout << "Calculated columns: "<<columns<<"\n";
cout << "Calculated lines: "<<lines<<"\n";
cout << "Total symbols: "<<lines*columns<<"\n";
}
imagCoord=positiveImaginary;
#if defined(_OPENMP)
omp_set_num_threads(threadsNumber);
#pragma omp parallel for private(i,j,realCoord,imagCoord,realTemp,imagTemp,realTemp2,arg,iterations) 
#endif
for (i=0;i<lines;i++)
{
realCoord = positiveReal;
for (j=0; j<columns;j++ )
{
iterations = 0;
realTemp = realCoord;
imagTemp = imagCoord;
arg = (realCoord * realCoord) + (imagCoord * imagCoord);
while ((arg < 4) && (iterations < 40))
{
realTemp2 = (realTemp * realTemp) - (imagTemp * imagTemp) - realCoord;
imagTemp = (2 * realTemp * imagTemp) - imagCoord;
realTemp = realTemp2;
arg = (realTemp * realTemp) + (imagTemp * imagTemp);
iterations += 1;
}
if (drawToConsole)
{
switch (iterations % 4)
{
case 0:
image[i*columns+j]='.';
break;
case 1:
image[i*columns+j]='o';
break;
case 2:
image[i*columns+j]='0';
break;
case 3:
image[i*columns+j]='@';
break;
}
}
realCoord = positiveReal -(j+1)*realStep;
}
imagCoord = positiveImaginary - (i+1)*imaginaryStep;
}
if (drawToConsole)
{
for (int i=0;i<lines;i++)
{
for (int j = 0; j < columns; j++)
cout<<image[i*columns+j];
cout<<"\n";
}
}
return lines*columns; 
}

int main()
{ 
int userChoice;
cout << "@@@ The program draws the Mandelbrot set in console using OpenMP\n";
cout << "Choose an option:\n 1. Draw Mandelbrot set\n 2. Benchmark and write results to the output file\n 3. Exit\n";
cin >> userChoice;
int threadsNumber=1;
bool openMPEnabled = false;
#if defined(_OPENMP)
openMPEnabled = true;
cout << "Enter number of threads:\n";
cin >> threadsNumber;
#endif
switch (userChoice)
{
case 1:
{
drawFractal(2,-2.2,2,-2,threadsNumber,true);
}
break;
case 2:
{
double dif;
double positiveImaginary, negativeImaginary, positiveReal, negativeReal;
ifstream dataFile("input.txt");
ofstream outputFile("output_openmp.txt");
while (!dataFile.eof())
{
dataFile >> positiveImaginary >> negativeImaginary>>positiveReal>>negativeReal;
double start,end;
#if defined(_OPENMP)
start = omp_get_wtime( );
omp_set_dynamic(0);
#endif
int symbols = drawFractal(positiveImaginary,negativeImaginary,positiveReal,negativeReal,threadsNumber,false);
#if defined(_OPENMP)
end = omp_get_wtime();
#endif
dif = end - start;
cout <<dif<<"\n";
outputFile <<  positiveImaginary <<" "<< negativeImaginary<<" "<<positiveReal<<" "<<negativeReal<<" Symbols: "<<symbols;
outputFile << " Time: "<<dif<<"\n";
}
}
break;
default:
cout << "Your choice is wrong!";
}
return 0;
}