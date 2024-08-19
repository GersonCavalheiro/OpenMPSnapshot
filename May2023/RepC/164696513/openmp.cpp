#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
using namespace std;
double limit = 1000;
int thread_count = 64;
struct DataPacket{
double value;
int idx;
};
vector<double*> initialize(int n){
int i,j;
vector<double*> matrix;
for(i = 0;i<n;i++){
double* address = (double*) malloc(n*sizeof(double));
matrix.push_back(address);
for(j = 0;j<n;j++){
*(matrix[i] + j) = drand48() * limit;
}
}
return matrix;
}
vector<double*> read_data(string file_path,int n){
int i,j;
vector<double*> matrix;
ifstream file;
file.open(file_path);
if(file.is_open()){
for(i = 0;i<n;i++){
double* address = (double*) malloc(n*sizeof(double));
matrix.push_back(address);
for(j = 0;j<n;j++){
file >> *(matrix[i]+j);
}
}
}
file.close();
return matrix;
}
void write_data(vector<double*> matrix,string name){
int n = matrix.size();
ofstream file;
file.open("./" + name);
if(file.is_open()){
for(int i = 0;i<n;i++){
for(int j = 0;j<n;j++){
file << *(matrix[i] + j);
if(j==n-1){
if(i!=n-1){
file << "\n";
}
}else{
file << " ";
}
}
}
}
file.close();
}
vector<double*> matrix_copy(vector<double*> m1){
int i,j,n;
n = m1.size();
vector<double*> matrix;
for(i = 0;i<n;i++){
double* address = (double*) malloc(n*sizeof(double));
matrix.push_back(address);
for(j = 0;j<n;j++){
*(matrix[i] + j) = *(m1[i] + j);
}
}
return matrix;
}
void display(vector<double*> vec){
int i,j,n;
n = vec.size();
for(i = 0;i<n;i++){
for(j = 0;j<n;j++){
cout << *(vec[i] + j) << " ";
}
cout << endl;
}
cout << "**************THE END**************" << endl;
}
double checker(vector<int>& perm, vector<double*> mat, vector<double*> lower, vector<double*> upper){
double v1,v2,temp,ans;
int i,j,k,n;
n = mat.size();
ans = 0.0;
for(j = 0;j<n;j++){
temp = 0.0;
for(i = 0;i<n;i++){
v1 = *(mat[perm[i]] + j);
for(k = 0;k<n;k++){
v1-=((*(lower[i] + k)) * (*(upper[k] + j)));
}
temp+=(v1*v1);
}
ans+=sqrt(temp);
}
return ans;
}
void LUdecomp(vector<double*> mat, vector<int>& perm, vector<double*> lower, vector<double*> upper){
int idx,j;
int n = mat.size();
DataPacket maxima;
maxima.value = 0.0;
maxima.idx = 0;
#pragma omp parallel num_threads(thread_count) 
{
for(int col = 0;col<n;col++){
#pragma omp master
{
for(int i = col;i<n;i++){
if(fabs(*(mat[i] + col))>fabs(maxima.value)){
maxima.value = *(mat[i] + col);
maxima.idx = i;
}
}
idx = maxima.idx;
if(maxima.value==0.0){
cout << "Singular Matrix" << endl;
}
int temp1 = perm[col];
perm[col] = perm[idx];
perm[idx] = temp1;
double* add = mat[col];
mat[col] = mat[idx];
mat[idx] = add;
*(upper[col] + col) = *(mat[col] + col);
maxima.value = 0.0;
}
#pragma omp barrier
#pragma omp for schedule(static)
for(int i = 0;i<col;i++){
double temp2 = *(lower[col] + i);
*(lower[col] + i) = *(lower[idx] + i);
*(lower[idx] + i) = temp2;
}
#pragma omp for schedule(static)
for(int i = col+1;i<n;i++){
*(lower[i] + col) = (*(mat[i] + col))/(*(upper[col] + col));
*(upper[col] + i) = *(mat[col] + i);
}
#pragma omp for schedule(static)
for(int i = col+1;i<n;i++){
for(int j = col+1;j<n;j++){
*(mat[i] + j) = *(mat[i] + j) - (*(lower[i] + col))*(*(upper[col] + j));
}
}
}
}
}
int main(int argc, char const *argv[]){
vector<int> perm;
vector<double*> lower;
vector<double*> upper;
vector<double*> perm_final;
struct timeval start, end;
double time_taken,time_taken2;
int n = atoi(argv[1]);
thread_count = atoi(argv[2]);
int check = atoi(argv[3]);
string file_path = argv[4];
int write = atoi(argv[5]);
vector<double*> matrix = read_data(file_path,n);
vector<double*> matrix_cp = matrix_copy(matrix);
int i,j,k;
for(i = 0;i<n;i++){
perm.push_back(i);
}
for(i = 0;i<n;i++){
double* add1 = (double*)malloc(n*sizeof(double));
double* add2 = (double*)malloc(n*sizeof(double));
double* add3 = (double*)malloc(n*sizeof(double));
lower.push_back(add1);
upper.push_back(add2);
perm_final.push_back(add3);
}
for(i = 0;i<n;i++){
for(j = 0;j<n;j++){
*(upper[i] + j) = 0.0;
*(lower[i] + j) = 0.0;
*(perm_final[i] + j) = 0.0;
}
*(lower[i] + i) = 1.0;
}
gettimeofday(&start, NULL);
LUdecomp(matrix_cp,perm,lower,upper);
gettimeofday(&end, NULL);
time_taken = (end.tv_sec - start.tv_sec) * 1e6;
time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
cout << "LU Decomposition in " << time_taken << " sec" << endl; 
if(check==1){
gettimeofday(&start, NULL);
double ans = checker(perm,matrix,lower,upper);
gettimeofday(&end, NULL);
time_taken2 = (end.tv_sec - start.tv_sec) * 1e6;
time_taken2 = (time_taken2 + (end.tv_usec - start.tv_usec)) * 1e-6;
cout << "L2,1 norm = " << ans << endl;
cout << "Time taken for checking : " << time_taken2 << " sec" << endl;
}
if(write==1){
for(int i = 0;i<n;i++){
*(perm_final[i] + perm[i]) = 1.0;
}
string P_name = "P_" + to_string(n) + "_" + to_string(thread_count) + ".txt";
string L_name = "L_" + to_string(n) + "_" + to_string(thread_count) + ".txt";
string U_name = "U_" + to_string(n) + "_" + to_string(thread_count) + ".txt";
write_data(perm_final,P_name);
write_data(lower,L_name);
write_data(upper,U_name);
}
}