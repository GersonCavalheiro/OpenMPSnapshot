#ifndef _SPARSEREPRESENTATION_HPP_
#define _SPARSEREPRESENTATION_HPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include "Eigen/Dense"

#ifdef _OPENMP

#if defined(__clang__)
#include <omp.h>
#elif defined(__GNUG__) || defined(__GNUC__)
#include <omp.h>
#endif

#endif

using namespace std;
using namespace Eigen;

template<class DataType>
vector<DataType> operator*(vector<DataType> vec, DataType data) {

int len = vec.size();
if(!len) {
cerr << "operator*:eeror." << endl;
throw runtime_error("operator*:error");
}
vector<DataType> result(len, 0.0);
for(int i = 0; i < len; i++) {
result[i] = (vec[i] * data);
}
return result;
}

template<class DataType>
vector<DataType> operator-(vector<DataType> vec1, vector<DataType> vec2) {

if (!vec1.size() || !vec2.size() || vec1.size() != vec2.size()){
cerr << "operator-:error." << endl;
throw runtime_error("operator-:error");
}
int len = vec1.size();
vector<DataType> result(len, 0.0);
for (int i = 0; i < len; i++) {
result[i] = (vec1[i] - vec2[i]);
}

return result;
}

template<class DataType>
vector<DataType> operator+(vector<DataType> vec1, vector<DataType> vec2) {

if(!vec1.size() || !vec2.size() || vec1.size() != vec2.size()){
cerr << "operator+:error." << endl;
throw runtime_error("operator+:error");
}
int len = vec1.size();
vector<DataType> result(len, 0.0);
for(int i = 0; i < len; i++) {
result[i] = (vec1[i] + vec2[i]);
}

return result;
}

template<class DataType>
vector<DataType> operator*(vector<vector<DataType>> mat, vector<DataType> vec) {

int row = mat.size();
if (!row || row != vec.size()) {
cerr << "operator*:eeror." << endl;
throw runtime_error("operator*:error");
}
int col = mat[0].size();
if (!col) {
cerr << "operator*:eeror." << endl;
throw runtime_error("operator*:error");
}

vector<DataType> result(col, 0.0);
DataType tmp;
for(int i = 0; i < col; i++) {
tmp = 0.0;
for(int j = 0; j < row; j++) {
tmp += mat[j][i] * vec[j];
}
result[i] = (tmp);
}
return result;
}



template<class DataType>
class LSparseRepresentation {
public:
LSparseRepresentation(vector<vector<vector<DataType>>> dicts);
~LSparseRepresentation();
int SRClassify(vector<DataType>& y, DataType min_residual, int sparsity);
bool SRClassify(vector<vector<DataType>>& y, DataType min_residual, int sparsity, vector<int> &srres);

private:
vector<vector<DataType>> dic;		
vector<int> dicclassnum;			
int	classnum;						

private:
DataType Dot(vector<DataType>& vec1, vector<DataType>& vec2);

DataType Norm(vector<DataType> vec);

bool solve(vector<vector<DataType>>& phi, vector<DataType>& y, vector<DataType>& x);

bool OrthMatchPursuit(vector<DataType>& y, DataType min_residual, int sparsity, vector<DataType>& x, vector<int>& patch_indices);
};

template<class DataType>
LSparseRepresentation<DataType>::LSparseRepresentation(vector<vector<vector<DataType>>> dicts) {
classnum = 0;
dicclassnum.clear();
dic.clear();

this->classnum = dicts.size();
for(auto matrix:dicts) {
this->dicclassnum.push_back(matrix.size());
for(auto item:matrix) {
this->dic.push_back(item);
}
}
}

template<class DataType>
int LSparseRepresentation<DataType>::SRClassify(vector<DataType>& y, DataType min_residual, int sparsity) {

int fsize = y.size(); 
if (!fsize) return -1; 
int dcol = dic.size();      
if (!dcol) return -1; 
int drow = dic[0].size();   
if (!drow || drow != fsize) return -1;
int i, j;
vector<int> patch_indices;	
vector<DataType> coefficient;

if(!OrthMatchPursuit(y, min_residual, sparsity, coefficient, patch_indices)) {
vector<int>().swap(patch_indices);
vector<DataType>().swap(coefficient);
return -1;
}

vector<DataType> x(dcol, 0.0); 
for (i = 0; i < patch_indices.size(); ++i) {
x[patch_indices[i]] = coefficient[i];
}


int result = 0;

int start = 0;
DataType mindist = 100000000;
for(i = 0; i < classnum; ++i) {
vector<DataType> tmp(fsize, 0.0);
for(j = start; j < start + dicclassnum[i]; ++j) {
if (x[j] != 0.0) {
tmp = tmp + dic[j] * x[j];		
}
}
DataType dist = Norm(y - tmp);


if (mindist > dist) {
mindist = dist;
result = i;
}

start += dicclassnum[i];
}

return result; 

}

template<class DataType>
bool LSparseRepresentation<DataType>::SRClassify(vector<vector<DataType>>& y,
DataType min_residual, int sparsity, 
vector<int> &srres){

int i,j;
int size = y.size();
vector<int> result(this->classnum, 0);

vector<int> results(size,0);

#pragma omp parallel for
for (i = 0; i < size; ++i) {
results[i] = SRClassify(y[i], min_residual, sparsity);
}

for(i = 0; i < size; ++i) {
if (results[i] < 0) {
return false;
}
result[results[i]]++;
}

srres.resize(this->classnum);
for(i = 0; i < this->classnum; ++i) {
srres[i] = i;
}

for(i = this->classnum-1; i >= 0; --i) { 
if(result[i] == 0) {
result.erase(result.begin() + i);
srres.erase(srres.begin() + i);
}
}

int resnum = result.size(); 
for(i = 0; i < resnum - 1; ++i) {
int index = i;
for(j = i + 1; j < resnum; ++j){
if(result[index] < result[j]){
index = j;
}
}
swap(result[index], result[i]);
swap(srres[index], srres[i]);
}
return true;
}

template<class DataType>
LSparseRepresentation<DataType>::~LSparseRepresentation() {
int size = dic.size();
for(int i = 0; i < size; ++i)  {
vector<DataType>().swap(dic[i]);
}
vector<vector<DataType>>().swap(dic);
vector<int>().swap(dicclassnum);
}

template<class DataType>
DataType LSparseRepresentation<DataType>::Dot(vector<DataType>& vec1, vector<DataType>& vec2) {

if(!vec1.size() || !vec2.size() || vec1.size() != vec2.size()){
cerr << "Dot:error." << endl;
throw runtime_error("Dot:error");
}
DataType sum = 0;
int len = vec1.size();
for(int i = 0; i < len; ++i) {
sum += vec1[i] * vec2[i];
}

return sum;
}

template<class DataType>
DataType LSparseRepresentation<DataType>::Norm(vector<DataType> vec) {

if (!vec.size()) {
cerr << "Norm:error." << endl;
throw runtime_error("Norm:error");
}
DataType norm = 0;
int len = vec.size();
for(int i = 0; i < len; ++i) {
norm += vec[i] * vec[i];
}

return sqrt(norm);
}

template<class DataType>
bool LSparseRepresentation<DataType>::solve(vector<vector<DataType>>& phi, 
vector<DataType>& y, vector<DataType>& x) {

int col = phi.size();
if (col <= 0) return false;
int row = phi[0].size();
if (row != y.size() || col != x.size()) return false;
MatrixXd A(row, col);
VectorXd b(row);
for(int i = 0; i < row; ++i) {
b(i) = y[i];
for(int j = 0; j < col; ++j)
A(i, j) = phi[j][i];
}

VectorXd result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
for(int i = 0; i < col; ++i) {
x[i] = result(i);
}

return true;
}

template<class DataType>
bool LSparseRepresentation<DataType>::OrthMatchPursuit(vector<DataType>& y, 
DataType min_residual, int sparsity, 
vector<DataType>& x, vector<int>& patch_indices) {

int fsize = y.size(); 
if (!fsize) return false;
int dcol = dic.size();      
if (!dcol) return false;

int drow = dic[0].size();   
if (!drow || drow != fsize) return false;

vector<DataType> residual(y); 
vector<vector<DataType>> phi; 
x.clear();

DataType max_coefficient;
unsigned int patch_index;
vector<DataType> coefficient(dcol,0);

for(;;) {
max_coefficient = 0;
#pragma omp parallel for shared(coefficient)
for (int i = 0; i < dcol; ++i) {
coefficient[i] = (DataType)Dot(dic[i], residual);
}

for (int i = 0; i < dcol; ++i) {
if (fabs(coefficient[i]) > fabs(max_coefficient)) {
max_coefficient = coefficient[i];
patch_index = i;
}
}

patch_indices.push_back(patch_index);
phi.push_back(dic[patch_index]);
x.push_back(0.0);

if( !solve(phi, y, x) ) {
vector<DataType>().swap(residual);
for (int i = 0; i < phi.size(); ++i) vector<DataType>().swap(phi[i]);
vector<vector<DataType>>().swap(phi);
return false;
}

residual = y - phi *x;
DataType res_norm = (DataType)Norm(residual);

if (x.size() >= sparsity)
break;
}

vector<DataType>().swap(residual);
for(int i = 0; i < phi.size(); ++i) vector<DataType>().swap(phi[i]);
vector<vector<DataType>>().swap(phi);

return true;
}

#endif
