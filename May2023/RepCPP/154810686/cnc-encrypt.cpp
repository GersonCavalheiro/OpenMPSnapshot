#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <omp.h>
#include <chrono>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcpu.hpp"

using namespace std;

void CNC(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers){

int n;                      
byte expandedKey[176];      

for(int i = 0; i < uData.size(); i++) {

n = uLens[i];
byte *cipher = new byte[n];

KeyExpansion(uKeys[i], expandedKey);

omp_set_num_threads(4);
#pragma omp parallel for 
for(int curr_index = 0 ; curr_index<uLens[i] ; curr_index+=16){

AddRoundKey(uData[i] + curr_index , expandedKey);
for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
}

cipher = uData[i];
ciphers.push_back(move(cipher));
}
}

long long get_data(opts vars, vector<byte*> &msgs, vector<int> &lens, vector<byte*> &keys, int i, int j) {

if(i < vars.n_files_start || i > vars.n_files_end || j < 0 || j >= vars.m_batches ) {
cout << "Invalid getdata params";
return -1;
}

string msg_path, key_path;
ifstream f_msg, f_key;

int k, n;
long long sum = 0;
for(k = 0; k < i; k++) {
msg_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k);
key_path = msg_path+"_key";

f_msg.open(msg_path, ios::binary);
f_key.open(key_path, ios::binary);

if(f_msg && f_key) {

f_msg.seekg(0, f_msg.end);
n = f_msg.tellg();
f_msg.seekg(0, f_msg.beg);
sum += n;
byte *message = new byte[n];
byte *key = new byte[16];

f_msg.read( reinterpret_cast<char *> (message), n);
f_key.read( reinterpret_cast<char *> (key), 16);

msgs.push_back(move(message));
lens.push_back(n);
keys.push_back(move(key));

f_msg.close();
f_key.close();
}
else {
cout << "read failed";
}
}
return sum;
}

int main() {
opts vars = get_defaults();
ofstream data_dump;
data_dump.open(vars.datadump, fstream::app);

int i, j;
for(i = vars.n_files_start; i <= vars.n_files_end; i += vars.step) {
for(j = 0; j < vars.m_batches; j++) {

vector<double> batchtimes;
double sum = 0;

vector<byte*> uData;
vector<int> uLens;
vector<byte*> uKeys;

long long len = get_data(vars, uData, uLens, uKeys, i, j);
vector<byte*> ciphers;
ciphers.reserve(i);

auto start = chrono::high_resolution_clock::now();
CNC(uData, uLens, uKeys, ciphers);
auto end = chrono::high_resolution_clock::now();


string out_path;
ofstream fout;
for(int k = 0; k < i; k++) {
out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_cnc";
fout.open(out_path, ios::binary);
fout.write(reinterpret_cast<char *> (ciphers[k]), uLens[k]);
fout.close();
delete[] uData[k];
delete[] uKeys[k];
}

auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);
printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10ld ms", i, j, _time.count());
data_dump << vars.path << ",CNC," << i << "," << j << "," << _time.count() << "," << len << endl;
}
cout << endl;
}
return 0;
}










