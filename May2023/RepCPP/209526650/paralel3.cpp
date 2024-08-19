#include<fstream> 
#include<iostream> 
#include <string> 
#include <omp.h> 
#include<sys/time.h> 
#define windowSize 3 
#define tempSize 9 
using namespace std;
int findMedian(int[]); 
int main(int argc, char* argv[])
{
if (argc != 2) 
{
cout << "you must provide one argument to this program.\n";
cout << "usage ./a.out inputFileName.\n";
return 1;
}
ifstream infile(argv[1]);
if (!infile) 
{
cout << "ERROR please check the spelling of the file.\n";
return 1;
}
struct timeval currentTime;
double startTime,endTime,elapsed;
int rows, cols, median,nThreads;
int tempArray[tempSize];
infile >> rows >> cols;
int **anaMatrixPtr = new int*[rows];
int **sonucMatrixPtr = new int*[rows];
for (int i = 0; i<rows; i++)
{
anaMatrixPtr[i] = new int[cols];
}
for (int i = 0; i<rows; i++)
{
sonucMatrixPtr[i] = new int[cols];
}
for (int i = 0; i<rows; i++)
for (int j = 0; j<cols; j++)
infile >> anaMatrixPtr[i][j];
string outputFileName(argv[1]);
string str = "_filtered.txt";
outputFileName = outputFileName.substr(0, outputFileName.length() - 4);
outputFileName = outputFileName.append(str);
remove(outputFileName.c_str()); 
ofstream ofile(outputFileName.c_str(), ios::app);
gettimeofday(&currentTime,NULL);
startTime=currentTime.tv_sec+(currentTime.tv_usec/1000000.0);
#pragma omp parallel private(median,tempArray)
{
nThreads=omp_get_num_threads();
#pragma omp for collapse(2)
for (int i = 0; i<rows; i++)
{
for (int j = 0; j<cols; j++)
{
if ((i != 0) && (i != (rows - 1)) && (j != 0) && (j != (cols - 1)))
{
for (int a = 0, k = i - 1; a<3 && k <= i + 1; a++, k++)
{
for (int b = 0, l = j - 1; b<3 && l <= j + 1; b++, l++)
{ 
tempArray[a*windowSize + b] = anaMatrixPtr[k][l];
}
}
median = findMedian(tempArray);
sonucMatrixPtr[i][j] = median;
}
else
sonucMatrixPtr[i][j] = anaMatrixPtr[i][j];
}
}
}
gettimeofday(&currentTime,NULL); 
endTime=currentTime.tv_sec+(currentTime.tv_usec/1000000.0);
elapsed = endTime-startTime; 
printf("\nThe filtering proccess with %d thread took : %.7f ms\n",nThreads,elapsed*1000);	
for (int i = 0; i<rows; i++)
{
for (int j = 0; j<cols; j++)
{
if (sonucMatrixPtr[i][j] / 100 >= 1)
ofile << sonucMatrixPtr[i][j] << " ";
else if (sonucMatrixPtr[i][j] / 10 >= 1)
ofile << sonucMatrixPtr[i][j] << "  ";
else
ofile << sonucMatrixPtr[i][j] << "   ";
}
ofile << endl;
}


infile.close();
ofile.close();
for (int i = 0; i<rows; i++)
delete[] anaMatrixPtr[i];
delete[] anaMatrixPtr;
for (int i = 0; i<rows; i++)
delete[] sonucMatrixPtr[i];
delete[] sonucMatrixPtr;
return 0;
}
int findMedian(int arr[])
{ 
for (int i = 0; i<tempSize; i++)
{
for (int j = 0; j<tempSize - 1; j++)
{
if (arr[j]>arr[j + 1])
{
int temp = arr[j + 1];
arr[j + 1] = arr[j];
arr[j] = temp;
}
}
}
return 	arr[tempSize / 2];
}
