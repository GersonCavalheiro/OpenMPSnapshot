#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
using namespace std;

int rows, cols, nonzeros;
int *row_indexes;
int *col_indexes;
float *values;
int num_threads;

void readFile(string file_path) {
ifstream fin(file_path);
fin >> rows >> cols >> nonzeros;
row_indexes = new int[nonzeros];
col_indexes = new int[nonzeros];
values = new float[nonzeros];
int line_no = 0;
int row_index, col_index, val;
while (fin >> row_index >> col_index >> val) {
row_indexes[line_no] = row_index;
col_indexes[line_no] = col_index;
values[line_no] = val;
line_no++;
}

}

float multiply(float *B, float *C) {

float **temp = new float*[num_threads];
for(int i = 0; i < num_threads; ++i)
temp[i] = new float[cols];

#pragma omp parallel for
for (int i=0; i<nonzeros; i++) {
int row_index = row_indexes[i];
int col_index = col_indexes[i];	
float val = values[i];
int tid = omp_get_thread_num();
temp[tid][row_index] += B[col_index] * val;			
}

#pragma omp parallel for
for (int i=0; i<cols; i++){
float sum = 0;
for (int j=0; j<num_threads; j++){
sum += temp[j][i];
}
C[i] = sum;
}	

}

int main()
{

string file_path = "higgs-twitter.mtx";

cout<<"Reading File."<<endl;
readFile(file_path);
cout << "File has been read." << endl;
float *B = new float[cols];
float *C = new float[cols];

for (int i=0; i<cols; i++) {
B[i] = 1.0f;
}

cout<<"For 1 Thread:"<<endl;
auto t1 = chrono::high_resolution_clock::now();
omp_set_dynamic(0);
num_threads = 1;
omp_set_num_threads(num_threads);
multiply(B, C);
auto t2 = chrono::high_resolution_clock::now();
auto duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
cout << "C[100]: " << C[40] << endl;
cout <<"Execution time (microseconds): " << duration_1 <<endl<<endl;

cout<<"For 2 Threads:"<<endl;
t1 = chrono::high_resolution_clock::now();
omp_set_dynamic(0);
num_threads = 2;
omp_set_num_threads(num_threads);
multiply(B, C);
t2 = chrono::high_resolution_clock::now();
duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
cout << "C[100]: " << C[40] << endl;
cout <<"Execution time (microseconds): " << duration_1 <<endl<<endl;

cout<<"For 4 Threads:"<<endl;
t1 = chrono::high_resolution_clock::now();
omp_set_dynamic(0);
num_threads = 4;
omp_set_num_threads(num_threads);
multiply(B, C);
t2 = chrono::high_resolution_clock::now();
duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
cout << "C[100]: " << C[40] << endl;
cout <<"Execution time (microseconds): " << duration_1 <<endl<<endl;

cout<<"For 8 Threads:"<<endl;
t1 = chrono::high_resolution_clock::now();
omp_set_dynamic(0);
num_threads = 8;
omp_set_num_threads(num_threads); 
multiply(B, C);
t2 = chrono::high_resolution_clock::now();
duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
cout << "C[100]: " << C[40] << endl;
cout <<"Execution time with (microseconds): " << duration_1 <<endl<<endl;

return 0;

}

