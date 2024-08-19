#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define mio 1000*1000
#define uint64 unsigned long long

long long global_NumUpdates;
int global_NumCores = 2;

auto global_t0 = steady_clock::now();
auto global_t1 = steady_clock::now();
chrono::duration<float> global_dur;

class TopN {
public:
vector<pair<int, int>> TableN;
int NumOfEntries;
public:
TopN(int Num) {
NumOfEntries = Num;
global_NumUpdates = 0;
for (int i = 0; i <= NumOfEntries + 2; i++)
TableN.push_back(make_pair(0, 0));
TableN[0] = make_pair(10000000, 10000000);
};

~TopN() {
};

void TestChange() {
for (int i = 0; i <= NumOfEntries + 1; i++)
TableN[i] = make_pair(i, 2 * i);
}

static bool TopNCmp(pair<int, int> n1, pair<int, int> n2) {
bool erg;
if (n1.second > n2.second || ((n1.second == n2.second) && (n1.first < n2.first)) )
return true;
else
return false;
}

void TSort() {
global_NumUpdates++;
sort(&TableN[0], &TableN[NumOfEntries+2], TopNCmp);	 
}

void TOut() {
for (int i = 1; i <= NumOfEntries; i++) {
printf("%5d", i);
cout << " " << TableN[i].first << " -> " << TableN[i].second << endl;
}
}
};

struct CPair64 {
uint64 StartWert;
uint64 Len;
};

uint64 CollatzLen(uint64 Num) {
uint64 count = 0;

while (Num >= 1) {
if (Num == 1) return count;
if (Num % 2 == 0) {
Num = Num / 2;
count++;
}
else {
Num = 3 * Num + 1;
count++;
}
}
return count;
}

float CLIntervall(uint64 Start, uint64 End, int NN) {

TopN TopTen = TopN(NN);

global_t0 = steady_clock::now();

omp_set_num_threads(global_NumCores);
#pragma omp parallel
{
int NumT = omp_get_num_threads();
int TN = omp_get_thread_num();
int ILen = (End-Start) / NumT;
int LBegin = TN * ILen + Start;
int LEnd = (TN+1) * ILen + Start;
if (TN == (NumT-1)) LEnd = End;

printf("ThreadNr %d/%d is running.\n", TN, NumT);

for (uint64 i = LEnd; i >= LBegin; i--) {
uint64 temp = CollatzLen(i);
if (temp > TopTen.TableN[NN].second) {
#pragma omp critical
{
TopTen.TableN[NN + 1] = make_pair(i, temp);
TopTen.TSort();
}
}
}
}

global_t1 = steady_clock::now();
global_dur = global_t1 - global_t0;
float dur = global_dur.count();

printf("\n");

TopTen.TOut();
return dur;
};

int main() {

uint64 Limit=1; 
int ChartLen;

while (true) {

cout << "Limit (0=EXIT): ";
cin >> Limit;

if (Limit == 0) return 0;

cout << "ChartLength: ";
cin >> ChartLen;
cout << "Anzahl Kerne (negativ=LOOP): ";
cin >> global_NumCores;
cout << endl;

if (global_NumCores > 0) {
float Zeit = CLIntervall(1, Limit, ChartLen);
printf("\nDauer der Berechnung: %.4g s", Zeit);
printf("\nAnzahl Updates der Hitlist: %d", global_NumUpdates);
printf("\n\n\n");
}
if (global_NumCores < 0) {
int Max_Cores = global_NumCores * (-1);
for (global_NumCores = 1; global_NumCores <= Max_Cores; global_NumCores++) {
float Zeit = CLIntervall(1, Limit, ChartLen);
printf("\nDauer der Berechnung: %.6g s", Zeit);
printf("\nAnzahl Updates der Hitlist: %llu", global_NumUpdates);
printf("\n\n\n");
}
}
if (global_NumCores == 0)
printf("Anzahl Kerne ungltig\n");

char ch = _getch();
}
return 0;
}