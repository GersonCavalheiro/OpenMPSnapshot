#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cctype>		
#include <vector>
#include <limits>		
#include <iterator>
#include <numeric>		

#include "Matrix.h"

template<typename T>
class TXTParser{
public:
TXTParser(std::string path) : 
_path{ path }{

_rows = getRows();
_cols = getCols();
}

size_t getRows(){
size_t rows = 0;
std::ifstream read(_path);
for(std::string line; std::getline(read, line); ++rows){ }
read.close();
return rows;
}

size_t getCols(){
size_t cols = 0;
std::ifstream read(_path);
std::string line;
std::getline(read, line);
std::replace_if(std::begin(line), std::end(line), 
[](unsigned char c){ return std::ispunct(c); },
' ');
std::stringstream ss(line);	
T val;
while(ss){ ss >> val; ++cols; }	
read.close();
return cols-1;
}

void getData(T *container){
std::ifstream read(_path);

for(std::string line; std::getline(read, line);){
std::replace_if(std::begin(line), std::end(line), 
[](unsigned char c){ return std::ispunct(c); },
' ');
std::stringstream ss(line);

for(size_t j = 0; j < _cols; ++j){
ss >> container[j];
std::cout << container[j] << std::endl;
}
}
read.close();
}

std::fstream& gotoRow(std::fstream& file, size_t row){
file.seekg(std::ios::beg);
for(size_t i = 0; i < row; ++i){
file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}
return file;
}


template<typename Iterator>
void getDataIt(Iterator Begin, Iterator End, size_t row = 0){
std::fstream file(_path);
gotoRow(file, row);
for(std::string line; std::getline(file, line), Begin != End; ){
std::replace_if(std::begin(line), std::end(line), 
[](unsigned char c){ return std::ispunct(c); },
' ');
std::stringstream ss(line);
for(size_t j{ 0 }; j < _cols; ++j, ++Begin){
ss >> *Begin;
}
}
file.close();
}

template<typename Iterator>
void putData(Iterator Begin, Iterator End, int rows, int cols, const std::string& to_path, bool append=true, const char separator=','){
std::ofstream file;
if(append) file.open(to_path, std::ios::app);
else file.open(to_path);

for(int i = 0; i < rows; ++i){
for(int j = 0; j < cols-1; ++j, ++Begin){
file << *Begin << separator;
}
file << *Begin << '\n';
++Begin;
}
file.close();
}

private:
std::string _path;
size_t _rows = 0;
size_t _cols = 0;
};


class CSVRow{
public:
CSVRow(int size) : 
_size{ size }{

_data = std::make_unique<std::string[]>(size);
}

inline const std::string& operator[](std::size_t index) const{
return _data[index];
}

inline std::size_t size() const{
return _size;
}

void readNextRow(std::istream& str){
std::string line;
std::getline(str, line);
std::stringstream lineStream(line);
std::string cell;

int i = 0;
while(std::getline(lineStream, cell, ',')){
_data[i] = cell; ++i;
}
if(!lineStream && cell.empty()){
_data[_size-1] = "";
}
}
private:
std::unique_ptr<std::string[]> _data;
int _size;
};

std::istream& operator>>(std::istream& str, CSVRow& data){
data.readNextRow(str);
return str;
}

template<typename T>
class CSVParser{
public:
CSVParser(std::string path) : 
_path{ path }{

_rows = getRows();
_cols = getCols();
}

int getRows(){
std::ifstream file(_path);
int rows = 0;
while(skipRow(file)) ++rows;
file.close();
return rows;
}

int getCols(){
std::ifstream file(_path);
std::string line;
std::getline(file, line);
std::stringstream lineStream(line);
std::string cell;
int cols = 0;
while(std::getline(lineStream, cell, ',')) ++cols;
file.close();
return cols;
}

inline std::istream& skipRow(std::istream& str){
std::string line;
std::getline(str, line);
std::stringstream lineStream(line);
std::string cell;
while(std::getline(lineStream, cell, ',')){}
return str;
}

Matrix<T> parseChunk(int from_idx,int max_size){
assert(from_idx+max_size <= _rows);

std::ifstream file(_path);
CSVRow row(_cols);
int skipped_rows = 0;
while(skipped_rows < from_idx && skipRow(file)){ ++skipped_rows; }

file >> row;
Matrix<T> CHUNK(max_size, row.size());
for(size_t j = 0; j < row.size(); ++j){
CHUNK(0, j) = static_cast<T>(atof(row[j].c_str()));
}
int curr_line = 1;
while(file >> row && curr_line < max_size){
for(size_t j = 0; j < row.size(); ++j){
CHUNK(curr_line, j) = static_cast<T>(atof(row[j].c_str()));
}
++curr_line;
}
file.close();
return CHUNK;
}

Matrix<T>& parseRows(Matrix<T>& DATA, const std::vector<int>& indices_vect){
std::vector<int> idx_of_idx(indices_vect.size());
std::iota(idx_of_idx.begin(), idx_of_idx.end(), 0);

std::stable_sort(idx_of_idx.begin(), idx_of_idx.end(),
[&indices_vect](int i1, int i2){
return indices_vect[i1] < indices_vect[i2];
}
);
std::ifstream file(_path);
CSVRow row(_cols);
int curr_row = 0;
for(size_t i = 0; i < indices_vect.size(); ++i){
while(curr_row < indices_vect[idx_of_idx[i]] && skipRow(file)) ++curr_row;
file >> row; ++curr_row;
for(int j = 0; j < _cols; ++j){
DATA(idx_of_idx[i], j) = static_cast<T>(atof(row[j].c_str()));
}
}
file.close();
return DATA;
}
private:
std::string _path;
int _rows;
int _cols;
};