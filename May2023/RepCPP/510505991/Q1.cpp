#include<map>
#include<cmath>
#include<chrono>
#include<iostream>
#include<bits/stdc++.h>
using namespace std;
using namespace std::chrono;

bool Algorithm_1_Prime(long int number)
{
bool Prime = true;
for(long int i=2; i<=number/2; i++)
{
if(number%i==0){
Prime = false;
break;
}
}
return Prime;
}

bool Algorithm_2_Prime(long int number)
{
bool prime[number+1];
memset(prime, true, sizeof(prime));

for(long int itr=2; itr*itr<=number; itr++){
if(prime[itr]==true){
for(long int i=itr*2; i<=number; i+=itr){
prime[i] = false;
}
}
}

if(prime[number])
{
return true;
}
return false;	
}

bool Algorithm_3_Prime(int number)
{
int A[number+1] = {0};
for(int i=2; i<number+1; i++){
A[i] = i;
}

for(int j=2; j<sqrt(number+1); j++){
if(A[j]!=0){
int t = j*j;
while(t<number+1){
A[t] = 0;
t = t+j;
}
}
}

int i = 0;
int L[number+1] = {0};
int count = 0;
for(int j=2; j<number+1; j++){
if(A[j]!=0){
L[i] = A[j];
i = i+1;
count++;
}
}

bool isPrime = false;
for(int itr=0; itr<count; itr++){
if(L[itr]==number){
isPrime = true;
}
}
return isPrime;
}


long int Calculate_Static(int algo, long int number, int thread, int limit)
{
long int Prime = 0;
#pragma omp parallel for num_threads(thread) schedule(static, limit) 
for(long int itr=0; itr<number; itr++)
{
if(algo==1){
if(Algorithm_1_Prime(itr))
{
#pragma omp critical 
{
if(Prime<itr){
Prime = itr;
}
}
}
}else if(algo==2){
if(Algorithm_2_Prime(itr))
{
#pragma omp critical 
{
if(Prime<itr){
Prime = itr;
}
}
}
}else if(algo==3){
if(Algorithm_3_Prime(itr))
{
#pragma omp critical 
{
if(Prime<itr){
Prime = itr;
}
}
}
}
}
return Prime;
}

long int Calculate_Dynamic(int algo, long int number, int thread, int limit)
{
long int Prime = 0;
#pragma omp parallel for num_threads(thread) schedule(dynamic, limit)
for(long int itr=0; itr<number; itr++)
{
if(algo==1){
if(Algorithm_1_Prime(itr))
{
#pragma omp critical 
{
if(Prime<itr){
Prime = itr;
}
}
}
}else if(algo==2){
if(Algorithm_2_Prime(itr))
{
#pragma omp critical 
{
if(Prime<itr){
Prime = itr;
}
}
}
}else if(algo==3){
if(Algorithm_3_Prime(itr))
{
#pragma omp critical 
{
if(Prime<itr){
Prime = itr;
}
}
}
}
}
return Prime;
}

long int Calculate_Guided(int algo, long int number, int thread, int limit)
{
long int Prime = 0;
#pragma omp parallel for num_threads(thread) schedule(guided, limit) 
for(long int itr=0; itr<number; itr++)
{
if(algo==1){
if(Algorithm_1_Prime(itr))
{
if(Prime<itr){
Prime = itr;
}
}
}else if(algo==2){
if(Algorithm_2_Prime(itr))
{
if(Prime<itr){
Prime = itr;
}
}
}else if(algo==3){
if(Algorithm_3_Prime(itr))
{
if(Prime<itr){
Prime = itr;
}
}
}
}
return Prime;
}

void Report(int id, long int Algo_1_Prime, map<int, int> Static_1, map<int, int> Dynamic_1, map<int, int> Guided_1)
{
cout<<"----------------------------------------------"<<endl;
cout<<"Algo "<<id<<" Prime: "<<Algo_1_Prime<<endl<<endl;

cout<<endl<<"Algo "<<id<<" Static"<<endl;
map<int, int>::iterator itr;
for(itr=Static_1.begin(); itr!=Static_1.end(); itr++) 
{
cout<<itr->first<<'\t'<<itr->second<<endl;
}
cout<<endl;

cout<<endl<<"Algo "<<id<<" Dynamic"<<endl;
map<int, int>::iterator itr1;
for(itr1=Dynamic_1.begin(); itr1!=Dynamic_1.end(); itr1++) 
{
cout<<itr1->first<<'\t'<<itr1->second<<endl;
}
cout<<endl;

cout<<endl<<"Algo "<<id<<" Guided"<<endl;
map<int, int>::iterator itr2;
for(itr2=Guided_1.begin(); itr2!=Guided_1.end(); itr2++) 
{
cout<<itr2->first<<'\t'<<itr2->second<<endl;
}
cout<<endl;

cout<<"----------------------------------------------"<<endl<<endl;
}

int main()
{
long int Iterations = 100000;
map<int, int> Static_1;
map<int, int> Dynamic_1;
map<int, int> Guided_1;

map<int, int> Static_2;
map<int, int> Dynamic_2;
map<int, int> Guided_2;

map<int, int> Static_3;
map<int, int> Dynamic_3;
map<int, int> Guided_3;

long int Algo_1_Prime = 0;
for(int i=0; i<5; i++)
{
long int temp = 0;
auto start = high_resolution_clock::now();
temp = Calculate_Static(1, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_1_Prime = temp;
}

auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
Static_1.insert(pair<int, int>(7+i, duration.count()));

temp = 0;
auto start_1 = high_resolution_clock::now();
temp = Calculate_Dynamic(1, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_1_Prime = temp;
}

auto stop_1 = high_resolution_clock::now();
auto duration_1 = duration_cast<microseconds>(stop_1 - start_1);
Dynamic_1.insert(pair<int, int>(7+i, duration_1.count()));

temp = 0;
auto start_2 = high_resolution_clock::now();
temp = Calculate_Guided(1, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_1_Prime = temp;
}

auto stop_2 = high_resolution_clock::now();
auto duration_2 = duration_cast<microseconds>(stop_2 - start_2);
Guided_1.insert(pair<int, int>(7+i, duration_2.count()));
}
Report(1, Algo_1_Prime, Static_1, Dynamic_1, Guided_1);

long int Algo_2_Prime = 0;
for(int i=0; i<5; i++)
{
long int temp = 0;
auto start = high_resolution_clock::now();
temp = Calculate_Static(2, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_2_Prime = temp;
}

auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
Static_2.insert(pair<int, int>(7+i, duration.count()));

temp = 0;
auto start_1 = high_resolution_clock::now();
temp = Calculate_Dynamic(2, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_2_Prime = temp;
}

auto stop_1 = high_resolution_clock::now();
auto duration_1 = duration_cast<microseconds>(stop_1 - start_1);
Dynamic_2.insert(pair<int, int>(7+i, duration_1.count()));

temp = 0;
auto start_2 = high_resolution_clock::now();
temp = Calculate_Guided(2, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_2_Prime = temp;
}

auto stop_2 = high_resolution_clock::now();
auto duration_2 = duration_cast<microseconds>(stop_2 - start_2);
Guided_2.insert(pair<int, int>(7+i, duration_2.count()));
}
Report(1, Algo_2_Prime, Static_2, Dynamic_2, Guided_2);

long int Algo_3_Prime = 0;
for(int i=0; i<5; i++)
{
long int temp = 0;
auto start = high_resolution_clock::now();
temp = Calculate_Static(2, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_3_Prime = temp;
}

auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
Static_3.insert(pair<int, int>(7+i, duration.count()));

temp = 0;
auto start_1 = high_resolution_clock::now();
temp = Calculate_Dynamic(2, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_3_Prime = temp;
}

auto stop_1 = high_resolution_clock::now();
auto duration_1 = duration_cast<microseconds>(stop_1 - start_1);
Dynamic_3.insert(pair<int, int>(7+i, duration_1.count()));

temp = 0;
auto start_2 = high_resolution_clock::now();
temp = Calculate_Guided(2, Iterations, 7+9, 7+i);
if(temp!=0){
Algo_3_Prime = temp;
}

auto stop_2 = high_resolution_clock::now();
auto duration_2 = duration_cast<microseconds>(stop_2 - start_2);
Guided_3.insert(pair<int, int>(7+i, duration_2.count()));

}

Report(3, Algo_3_Prime, Static_3, Dynamic_3, Guided_3);

return 0;
}