#include <iostream>
#include <fstream>
#include <cmath>
#include <bits/stdc++.h>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

int k;
struct Compare
{
bool operator()(const pair<int, double> &a, const pair<int, double> &b)
{
return a.second < b.second;
}
};

double cosine_dist(vector<double> &x, vector<double> &y)
{
double a = 0.0, b = 0.0, c = 0.0;
for (int i = 0; i < x.size(); i++)
{
a += x[i] * y[i];
b += x[i] * x[i];
c += y[i] * y[i];
}
double pl = sqrt(b), ql = sqrt(c);
return 1-(double)(a / (pl * ql));
}

priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> SearchLayer(int k, vector<double> &q, priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> &candidates, vector<int> &indptr, vector<int> &index, vector<int> &level_offset, int lc, unordered_set<int> &visited, vector<vector<double>> &vect)
{
priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> top_k(candidates);
while (candidates.size() > 0)
{
int ep = candidates.top().first;
candidates.pop();
int start = indptr[ep] + level_offset[lc];
int end = indptr[ep] + level_offset[lc + 1];
for (int as = start; as < end; as++)
{
int px = index[as];
if (px == -1 || visited.find(px) != visited.end())
{
continue;
}
visited.insert(px);
double _dist = cosine_dist(q, vect[px]);
if (_dist >= top_k.top().second && top_k.size() >= k)
{
continue;
}
pair<int, double> inst(px, _dist);
top_k.push(inst);
while (top_k.size() > k)
{
top_k.pop();
}
candidates.push(inst);
}
}
return top_k;
}

priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> QueryHNSW(int k, vector<double> &q, priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> &topk, int ep, vector<int> &indptr, vector<int> &index, vector<int> &level_offset, int max_level, vector<vector<double>> &vect)
{
priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> top_k(topk);
pair<int, double> adr(ep, cosine_dist(q, vect[ep]));
top_k.push(adr);
unordered_set<int> visited;
visited.insert(ep);
int L = max_level;
for (int level = L-1; level >= 0; level--)
{
top_k = SearchLayer(k, q, top_k, indptr, index, level_offset, level, visited, vect);
}
return top_k;
}

int main(int argc, char *argv[])
{
MPI_Init(NULL, NULL);
double starting_time_1 = MPI_Wtime();
int minus_one = -1;
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

assert(argc > 4);
string out_path = argv[1];
k = stoi(argv[2]);
string user_file = argv[3];
string user_output_file = argv[4];
int buffer_arr[15];


vector<vector<double>> user;
int ep;
int max_level;
vector<int> index;
vector<int> indptr;
vector<int> level_offset;
vector<int> level;
vector<vector<double>> vect;

if (rank != 0)
{
MPI_Bcast(&buffer_arr, 15, MPI_INT, 0, MPI_COMM_WORLD);
max_level = buffer_arr[0];
ep = buffer_arr[1];
int new_buffer[buffer_arr[2] + buffer_arr[3] + buffer_arr[4] + buffer_arr[5]];
double new_buffer2[buffer_arr[6] * buffer_arr[7] + buffer_arr[8] * buffer_arr[9]];
int top = 0;
int alpha =  buffer_arr[2] + buffer_arr[3] + buffer_arr[4] + buffer_arr[5];
MPI_Bcast(&new_buffer, alpha , MPI_INT, 0, MPI_COMM_WORLD);
int beta = buffer_arr[6] * buffer_arr[7] + buffer_arr[8] * buffer_arr[9];
MPI_Bcast(&new_buffer2, beta , MPI_DOUBLE, 0, MPI_COMM_WORLD);
int sum = buffer_arr[2] + buffer_arr[3] + buffer_arr[4] + buffer_arr[5];

for (int i = 0; i < sum; i++)
{
if (i < buffer_arr[2])
{
level.push_back(new_buffer[i]);
}
else if (i < buffer_arr[2] + buffer_arr[3])
{
index.push_back(new_buffer[i]);
}
else if (i < buffer_arr[2] + buffer_arr[3] + buffer_arr[4])
{
indptr.push_back(new_buffer[i]);
}
else
{
level_offset.push_back(new_buffer[i]);
}
}
for (int i = 0; i < buffer_arr[6]; i++)
{
vector<double> nv;
for (int j = 0; j < buffer_arr[7]; j++)
{
nv.push_back(new_buffer2[top]);
top++;
}
vect.push_back(nv);
}
for (int i = 0; i < buffer_arr[8]; i++)
{
vector<double> nd;
for (int j = 0; j < buffer_arr[9]; j++)
{
nd.push_back(new_buffer2[top]);
top++;
}
user.push_back(nd);
}
}
if (rank == 0)
{
string new_string_max_level = out_path + "/max_level.dat";
FILE *reader_max_level;
reader_max_level = fopen((const char *)new_string_max_level.c_str(), "rb");
unsigned char b1[4];
fread(&b1, 4, 1, reader_max_level);
max_level = (int)b1[0] | ((int)b1[1] << 8) | ((int)b1[2] << 16) | ((int)b1[3] << 24);
fclose(reader_max_level);

string new_string_ep = out_path + "/ep.dat";
FILE *reader_ep;
reader_ep = fopen((const char *)new_string_ep.c_str(), "rb");
unsigned char b2[4];
fread(&b2, 4, 1, reader_ep);
ep = (int)b2[0]|((int)b2[1] << 8) | ((int)b2[2] << 16) | ((int)b2[3] << 24);
fclose(reader_ep);

string new_string_level = out_path + "/level.dat";
FILE *reader_level;
reader_level = fopen((const char *)new_string_level.c_str(), "rb");
unsigned char b3[4];
while (feof(reader_level) == 0)
{
fread(&b3, 4, 1, reader_level);
level.push_back(((int)b3[0] |((int)b3[1] << 8) |((int)b3[2] << 16) |((int)b3[3] << 24)));
}
fclose(reader_level);
level.pop_back();

string new_string_indptr = out_path + "/indptr.dat";
FILE *reader_indptr;
reader_indptr = fopen((const char *)new_string_indptr.c_str(), "rb");
unsigned char b4[4];
while (feof(reader_indptr) == 0)
{
fread(&b4, 4, 1, reader_indptr);
indptr.push_back(((int)b4[0] | ((int)b4[1] << 8) | ((int)b4[2] << 16) | ((int)b4[3] << 24)));
}
fclose(reader_indptr);
indptr.pop_back();


string new_string_level_offset = out_path + "/level_offset.dat";
FILE *reader_level_offset;
reader_level_offset = fopen((const char *)new_string_level_offset.c_str(), "rb");
unsigned char b5[4];
while (feof(reader_level_offset) == 0)
{
fread(&b5, 4, 1, reader_level_offset);
level_offset.push_back(((int)b5[0] | ((int)b5[1] << 8) | ((int)b5[2] << 16) | ((int)b5[3] << 24)));
}
fclose(reader_level_offset);
level_offset.pop_back();

string new_string_index = out_path + "/index.dat";
FILE *reader_index;
reader_index = fopen((const char *)new_string_index.c_str(), "rb");
unsigned char b6[4];
while (feof(reader_index) == 0)
{
fread(&b6, 4, 1, reader_index);
index.push_back(((int)b6[0] | ((int)b6[1] << 8) | ((int)b6[2] << 16) | ((int)b6[3] << 24)));
}
fclose(reader_index);
index.pop_back();

string new_string_vect = out_path + "/vect.dat";
FILE *reader_vect;
unsigned char b10[4];
unsigned char b11[4];
reader_vect = fopen((const char *)new_string_vect.c_str(), "rb");
fread(&b10, 4, 1, reader_vect);
int sz = ((int)b10[0] | ((int)b10[1] << 8) | ((int)b10[2] << 16) | ((int)b10[3] << 24));
fread(&b11, 4, 1, reader_vect);
int sz0 = ((int)b11[0] | ((int)b11[1] << 8) | ((int)b11[2] << 16) | ((int)b11[3] << 24));
for (int i = 0; i < sz; i++)
{
vector<double> ngu;
for (int j = 0; j < sz0; j++)
{
double db;
fread(&db, sizeof(double), 1, reader_vect);
ngu.push_back(db);
}
vect.push_back(ngu);
}
fclose(reader_vect);

ifstream input_user_file;
input_user_file.open(user_file);
string linez ;
while (getline(input_user_file, linez)){
stringstream srtm(linez);
string letterz;
vector<double> one_linez;
while(srtm>>letterz)
{
one_linez.push_back(stod(letterz));
}
user.push_back(one_linez);
}
input_user_file.close();

int top = 0;
buffer_arr[0] = max_level;
buffer_arr[1] = ep;
buffer_arr[2] = level.size();
buffer_arr[3] = index.size();
buffer_arr[4] = indptr.size();
buffer_arr[5] = level_offset.size();
buffer_arr[6] = vect.size();
buffer_arr[7] = vect[0].size();
buffer_arr[8] = user.size();
buffer_arr[9] = user[0].size();
buffer_arr[10] = k;
buffer_arr[11] = top;


MPI_Bcast(&buffer_arr, 15, MPI_INT, 0, MPI_COMM_WORLD);

int summa = buffer_arr[2] + buffer_arr[3] + buffer_arr[4] + buffer_arr[5];
int buta = buffer_arr[6] * buffer_arr[7] + buffer_arr[8] * buffer_arr[9];
int temp_arr[summa];
double tempo_arr[buta];
for (int i = 0; i < summa; i++)
{
if (i < buffer_arr[2])
{
temp_arr[i] = level[i];
}
else if (i < buffer_arr[2] + buffer_arr[3])
{
temp_arr[i] = index[i - buffer_arr[2]];
}
else if (i < buffer_arr[2] + buffer_arr[3] + buffer_arr[4])
{
temp_arr[i] = indptr[i - buffer_arr[2] - buffer_arr[3]];
}
else
{
temp_arr[i] = level_offset[i - buffer_arr[2] - buffer_arr[3] - buffer_arr[4]];
}
}
MPI_Bcast(&temp_arr, summa, MPI_INT, 0, MPI_COMM_WORLD);
for (int i = 0; i < buffer_arr[6]; i++)
{
for (int j = 0; j < buffer_arr[7]; j++)
{
tempo_arr[top] = vect[i][j];
top++;
}
}

for (int i = 0; i < buffer_arr[8]; i++)
{
for (int j = 0; j < buffer_arr[9]; j++)
{
tempo_arr[top] = user[i][j];
top++;
}
}
MPI_Bcast(&tempo_arr, (buta), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

vector<int> startthread;
int l = user.size();
for (int nt = 0; nt < size; nt++)
{
startthread.push_back(nt * (int)(l / size));
}
vector<int> endthread;
for (int nt = 0; nt < size; nt++)
{
if (nt != size - 1)
{
endthread.push_back(((nt + 1) * (int)(l / size)) - 1);
}
else
{
endthread.push_back(l - 1);
}
}
int user_matrix[l][k];
for (int i = 0; i < l; i++)
{
for (int j = 0; j < k; j++)
{
user_matrix[i][j] = minus_one;
}
}
#pragma omp parallel for
for (int it = startthread[rank]; it < endthread[rank] + 1; it++)
{
priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> pq;
priority_queue<pair<int, double>, vector<pair<int, double>>, Compare> topk;



pq = QueryHNSW(k, user[it], pq, ep, indptr, index, level_offset, max_level, vect);
for (int p = k-1; p >=0; p--)
{
if (pq.size() == 0)
{
break;
}
else
{
user_matrix[it][p] = pq.top().first;
pq.pop();
}
}
}

if (rank == 0)
{
for (int i_thread = 1; i_thread < size; i_thread++)
{
for (int m = startthread[i_thread]; m < endthread[i_thread] + 1; m++)
{
int buffer1[k];
MPI_Recv(&user_matrix[m], k, MPI_INT, i_thread, i_thread, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
}
ofstream out_file;
out_file.open(user_output_file);
int sd = l;
for (int i = 0; i < sd; i++)
{
for (int j = 0; j < k; j++)
{
out_file << to_string(user_matrix[i][j]) << " ";
}
out_file << std::endl;
}
out_file.close();
}
if (rank != 0)
{
for (int x = startthread[rank]; x < endthread[rank] + 1; x++)
{
MPI_Send(&user_matrix[x], k, MPI_INT, 0, rank, MPI_COMM_WORLD);
}
}
auto ftime = MPI_Wtime() - starting_time_1;
MPI_Finalize();
return 0;
}