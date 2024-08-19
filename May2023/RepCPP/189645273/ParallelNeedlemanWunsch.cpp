
#include <iostream>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <unistd.h>


using namespace std;

struct report
{
int s1;
int s2;
double score;
};

report scores[20];
double minScore = 10000;
int addedScore = 0;

template <typename T>
double NeedlemanWunsch(char *data, int first, int second, int baseSize, T match, T dismatch, T gap);
void addReport(int s1, int s2, float score);
void bubbleSort();



int main() {
int sequenceSize = 15000;
const int baseSize= 200;
char *data = new char[sequenceSize*baseSize];

ifstream file("15K_Sequence.fasta");

char *pointerOfData = data;
while(!file.eof())
{
char temp;
file.read(&temp, 1);
if(temp=='C' || temp=='T' || temp=='A' || temp=='G')
{
*pointerOfData = temp;
file.read(++pointerOfData, 199);
pointerOfData += 199;
}
}

auto t_before = chrono::steady_clock::now();

double gap = -1.832482334, match = 3.621354295, mismatch = -2.451795405;
int ii, i = 0;
#pragma omp parallel num_threads(4) shared(i) private(ii)
{
while(i<15000)
{
#pragma omp critical
{
ii = i++;
}
int border = ii + 1;
for(;ii<border; ii++)
{
for(int j=ii+1;j<15000; j++)
{
double score = NeedlemanWunsch(data, ii, j, baseSize, match, mismatch, gap);
#pragma omp critical
{
addReport(ii, j, score);
}
}
}
}
}
auto t_after = chrono::steady_clock::now();
auto t_diff = chrono::duration_cast<chrono::seconds>(t_after - t_before).count();
cout << "Zaman: " << (double)t_diff/60.0 << endl;

bubbleSort();
for (int i=0; i<20; i++)
{
cout << i+1 << ". " << scores[i].s1 << " " << scores[i].s2 << " " << scores[i].score << endl;
}

return 0;
}


template <typename T>
double NeedlemanWunsch(char *data, int first, int second, int baseSize, T match, T dismatch, T gap)
{
T matrix[baseSize+1][baseSize];
#pragma omp parallel for
for(int i=0; i<baseSize+1; i++)
{
matrix[0][i] = i*gap;
matrix[i][0] = i*gap;
}

#pragma omp parallel for num_threads(4) collapse(2)
for(int i=1; i<baseSize+1; i++)
{
for(int j=1; j<baseSize+1; j++)
{
double temp[3];
temp[0] = matrix[i-1][j] + gap;
temp[1] = matrix[i][j-1] + gap;

if(data[first*baseSize+i] == data[second*baseSize+j])
temp[2] = matrix[i-1][j-1] + match;
else
temp[2] = matrix[i-1][j-1] + dismatch;


matrix[i][j] = max(max(temp[0], temp[1]), temp[2]);
}
}

return matrix[baseSize][baseSize];
}


void addReport(int s1, int s2, float score)
{
if (addedScore > 19)
{
if (score<minScore) return;
else
{
for (int i=0; i<20; i++)
{
if(scores[i].score == minScore)
{
scores[i].s1 = s1;
scores[i].s2 = s2;
scores[i].score = score;
double tempMin = 10000;
for (int j=0; j<20; j++)
{
if (scores[j].score < tempMin) tempMin = scores[j].score;
else continue;
}
minScore = tempMin;
return;
}
}
}
}
else
{
scores[addedScore].s1 = s1;
scores[addedScore].s2 = s2;
scores[addedScore++].score = score;
if(score<minScore) minScore = score;
}
}


void bubbleSort()
{
for (int i=20; i>1; --i)
{
for (int j=0; j<i; j++)
{
if(scores[j].score < scores[j+1].score)
{
report temp;
temp.s1 = scores[j].s1;
temp.s2 = scores[j].s2;
temp.score = scores[j].score;
scores[j].s1 = scores[j+1].s1;
scores[j].s2 = scores[j+1].s2;
scores[j].score = scores[j+1].score;
scores[j+1].s1 = temp.s1;
scores[j+1].s2 = temp.s2;
scores[j+1].score = temp.score;
}
}
}
}
