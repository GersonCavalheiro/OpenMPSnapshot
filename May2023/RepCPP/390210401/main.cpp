


#include <iostream>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#define _NUM_THREADS 6

using namespace std;

void printMatrix(vector<vector<float>> countMatrix)
{
for (int i = 0; i < countMatrix.size(); i++)
{
cout << countMatrix[i][0] << " " << countMatrix[i][1] << " " << countMatrix[i][2] << " " << countMatrix[i][3] << endl;
}
}

void readInMatrix(vector<vector<string>> &fullMatrix, int &rowCount, int &colCount)
{
ifstream myFile, rowsFile, colsFile;
rowsFile.open("data/rows.txt");
colsFile.open("data/cols.txt");
myFile.open("data/matrix.csv");

rowsFile >> rowCount;
rowsFile.close();
colsFile >> colCount;
colsFile.close();

while (myFile.good())
{
vector<string> row;
string line;
getline(myFile, line);
stringstream iss(line);

for (int cols = 0; cols < colCount + 1; cols++)
{
string val;
getline(iss, val, ',');
row.push_back(val);
}
fullMatrix.push_back(row);
}
myFile.close();
}

void loadWordCountMatrix(vector<vector<float>> &countMatrix, vector<vector<string>> fullMatrix, vector<string> &bookList, int rowCount, int colCount)
{
for (int j = 2; j < colCount + 1; j++)
{
bookList.push_back(fullMatrix[0][j]);
for (int i = 1; i < rowCount + 1; i++)
{
int val = stoi(fullMatrix[i][j]);
countMatrix[i - 1].push_back(val);
}
}
}

void cosineSimilarity(vector<vector<float>> countMatrix, vector<float> &cosList, vector<string> bookList, int rank, int querybook)
{
int i, j;
for (int j = 0; j < countMatrix[1].size(); j++)
{
float den1 = 0, den2 = 0, num = 0;
#pragma omp parallel for reduction(+ \
: num, den1, den2)

for (int i = 0; i < countMatrix.size(); i++)
{
num += (countMatrix[i][rank] * countMatrix[i][countMatrix[1].size() - 1]); 
den1 += pow(countMatrix[i][rank], 2);                                      
den2 += pow(countMatrix[i][countMatrix[1].size() - 1], 2);                 
}
cosList.push_back(num / (sqrt(den1 * den2))); 
}
if (rank != querybook) 
cout << bookList[rank] << " cos similarity to " << bookList[querybook]
<< ": " << cosList[rank] << endl;
}


typedef struct
{
int comm_rank;
int label;
float number; 
} CommRankNumber;

void *gather_numbers_to_root(void *number, MPI_Datatype datatype, MPI_Comm comm)
{
int comm_rank, comm_size;
MPI_Comm_rank(comm, &comm_rank);
MPI_Comm_size(comm, &comm_size);

int datatype_size;
MPI_Type_size(datatype, &datatype_size);
void *gathered_numbers;
if (comm_rank == 0)
{
gathered_numbers = new int[datatype_size * comm_size];
}

MPI_Gather(number, 1, datatype, gathered_numbers, 1, datatype, 0, comm);

return gathered_numbers;
}

int compare_float_comm_rank_number(const void *a, const void *b)
{
CommRankNumber *comm_rank_number_a = (CommRankNumber *)a;
CommRankNumber *comm_rank_number_b = (CommRankNumber *)b;
if (comm_rank_number_a->number > comm_rank_number_b->number)
{
return -1;
}
else if (comm_rank_number_a->number < comm_rank_number_b->number)
{
return 1;
}
else
{
return 0;
}
}

CommRankNumber *get_sorted(void *gathered_numbers, int gathered_number_count, MPI_Datatype datatype)
{
int datatype_size;
MPI_Type_size(datatype, &datatype_size);

CommRankNumber *comm_rank_numbers = new CommRankNumber[gathered_number_count * sizeof(CommRankNumber)];
int i;
for (i = 0; i < gathered_number_count; i++)
{
comm_rank_numbers[i].comm_rank = i;
memcpy(&(comm_rank_numbers[i].number), gathered_numbers + (i * datatype_size), datatype_size);
}

qsort(comm_rank_numbers, gathered_number_count, sizeof(CommRankNumber), &compare_float_comm_rank_number);

int *nearest_neighbor = (int *)malloc(sizeof(int) * gathered_number_count);

return comm_rank_numbers;
}

int TMPI_Rank(void *send_data, void *recv_data, MPI_Datatype datatype, MPI_Comm comm, int k, vector<string> bookList, int querybook, int *class_labels, int num_classes)
{
if (datatype != MPI_FLOAT)
{
return MPI_ERR_TYPE;
}

int comm_size, comm_rank;
MPI_Comm_size(comm, &comm_size);
MPI_Comm_rank(comm, &comm_rank);

float next, current; 
string name;         
int next_loc;        
int label;           

void *gathered_numbers = gather_numbers_to_root(send_data, datatype, comm);

CommRankNumber *nearest_neighbor = NULL;
if (comm_rank == 0)
{
nearest_neighbor = get_sorted(gathered_numbers, comm_size, datatype);
current = nearest_neighbor[0].number; 

int max_count = 0; 
int max;           

cout << "\nShowing only the nearest k = " << k << " nearest neighbors...\n\n";

int class_count[num_classes];
memset(class_count, 0, num_classes * sizeof(int));

for (int i = 1; i < k + 1; i++)
{
name = bookList[nearest_neighbor[i].comm_rank];
next_loc = nearest_neighbor[i].comm_rank;
next = nearest_neighbor[i].number;
label = class_labels[nearest_neighbor[i].comm_rank];
cout << "nearest neighbor " << i << "\n\t| Rank " << next_loc << " = " << next;
cout << "\n\t| Book (" << name << ") has label " << label << endl
<< endl; 

class_count[label]++;
}
for (int i = 0; i < num_classes; i++)
{
if (max_count < class_count[i])
{
max = i;
max_count = class_count[i];
}
}

cout << "RESULT: using k = " << k << ",\n\tPredicted label class is " << max << " -- if no majority, picked one" << endl;
}

if (comm_rank == 0)
{
free(gathered_numbers);
free(nearest_neighbor);
}

return 0;
}

int main(int argc, char *argv[])
{

enum author
{
fitzgerald = 1,
melville = 2,
shakespeare = 3,
unknown = 0
}; 

int class_labels[4] = {shakespeare, fitzgerald, melville, shakespeare}; 

int k;
k = 3;                             
int querybook = 3;                 
class_labels[querybook] = unknown; 

int rank;
int size;

omp_set_num_threads(_NUM_THREADS); 

MPI_Init(&argc, &argv); 
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

vector<vector<string>> fullMatrix;
vector<vector<float>> countMatrix;
vector<string> bookList; 
vector<float> cosList;   
int rowCount, colCount;

readInMatrix(fullMatrix, rowCount, colCount);
countMatrix.resize(rowCount);
loadWordCountMatrix(countMatrix, fullMatrix, bookList, rowCount, colCount);

cosineSimilarity(countMatrix, cosList, bookList, rank, querybook);

TMPI_Rank(&cosList[rank], &rank, MPI_FLOAT, MPI_COMM_WORLD, k, bookList, querybook, class_labels, sizeof(author)); 

MPI_Finalize();

return 0;
}