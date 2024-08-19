#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <omp.h>
#include <memory>
#include <cstring>
#include <array>


void memory_bound(){
const long int size = 512 * 1024 * 1024;
long int* arr = (long int*)aligned_alloc(64, size * sizeof(long int));
memset(arr, 0, size * sizeof(long int));

Timer timer1;
for (long int i = 0; i < size; i++){
arr[i] *= 3;
}
timer1.stop();
const long int jumpSize = 64/sizeof(long int); 
Timer timer8;
for (long int i = 0; i < size; i+=jumpSize){
arr[i] *= 3;
}
timer8.stop();
free(arr);

std::cout << ">> memory_bound 1 : " << timer1.getElapsed() << std::endl;
std::cout << ">> memory_bound 8 : " << timer8.getElapsed() << " (jump of size " << jumpSize << ")" << std::endl;
}

void memory_bound_k(){
const long int size = 1024 * 1024 * 1024;
std::unique_ptr<long int[]> arr(new long int[size]());

const long int MaxK = 1024;
std::unique_ptr<double[]> times(new double[ffsl(MaxK)+1]());

for(long int k = 1 ; k <= MaxK ; k *= 2){
Timer timer;
for (long int i = 0; i < size; i += k){
arr[i] *= 3;
}
timer.stop();
times[k] = timer.getElapsed();
}

for(long int k = 1 ; k <= MaxK ; k *= 2){    
std::cout << ">> memory_bound_k " << k << " : " << times[k] << std::endl;
}
}


void cache_k(){
const long int nbRepeat = 5;
const long int size = 1L * 1024 * 1024 * 1024;
long int* arr = (long int*)aligned_alloc(64, size * sizeof(long int));
memset(arr, 0, size * sizeof(long int));

const long int MaxK = ffsl(size);
std::unique_ptr<double[]> times(new double[MaxK+1]());

for(long int k = 0 ; k < MaxK ; ++k){
const long int limit = (1 << k);
Timer timer;
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i += 16){
arr[i%limit] *= 3;
}
}
timer.stop();
times[k] = timer.getElapsed();
}

free(arr);

for(long int k = 0 ; k < MaxK ; ++k){    
std::cout << ">> cache_k " << k << " : " << times[k]  << " (limit " << (1 << k)/1024 << "KB)" << std::endl;
}
}

void ilp(){
const long int nbRepeat = 1024*1024*1024;
long int arr[2] = {0, 0};

const long int coef = 3;

Timer timerii;
for (int i = 0; i < nbRepeat; i++){
arr[0] *= coef;
arr[0] += coef;
}

timerii.stop();

Timer timeriip1;
for (int i = 0; i < nbRepeat; i++){
arr[0] *= coef;
arr[1] += coef;
}
timeriip1.stop();

volatile long int avoidOverOptim = arr[0] + arr[1];
(void)avoidOverOptim;

std::cout << ">> ilp ii : " << timerii.getElapsed() << std::endl;
std::cout << ">> ilp ii+1 : " << timeriip1.getElapsed() << std::endl;
}

void aliasing4k(){
const long int incrementNoConflict = (4096-128)/sizeof(float);
const long int incrementConflict = (4096-4)/sizeof(float);
const long int size = 1024;
float* arr = (float*)aligned_alloc(64, (size + std::max(incrementNoConflict, incrementConflict)) * sizeof(float));
memset(arr, 0, (size + std::max(incrementNoConflict, incrementConflict)) * sizeof(float));

const long int nbRepeat = 10*1024*1024;

Timer timernoconflict;
{
float* a = arr;
float* b = arr + incrementNoConflict;
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i++){
a[i] += b[i];
}
}
}
timernoconflict.stop();

Timer timer4kconflict;
{
float* a = arr;
float* b = arr + incrementConflict;
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i++){
a[i] += b[i];
}
}
}
timer4kconflict.stop();

free(arr);

std::cout << ">> no conflict : " << timernoconflict.getElapsed() << std::endl;
std::cout << ">> conflict : " << timer4kconflict.getElapsed() << std::endl;
}

void misprediction(){
const long int size = 1024;
std::unique_ptr<long int[]> arr(new long int[size]());

const long int nbRepeat = 2*1024*1024;

Timer timer1024;
{
long int a = 0;
long int b = 0;
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i++){

if (i < size-1)
a += arr[i];
else
b += arr[i];
}
}
volatile const long int avoidOverOptim = a + b;
(void)avoidOverOptim;
}
timer1024.stop();

Timer timer3;
{
long int a = 0;
long int b = 0;
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i++){

if (i%3 < 2)
a += arr[i];
else
b += arr[i];

}
}
volatile const long int avoidOverOptim = a + b;
(void)avoidOverOptim;
}
timer3.stop();

std::cout << ">> if i % 1024 : " << timer1024.getElapsed() << std::endl;
std::cout << ">> if i % 3 : " << timer3.getElapsed() << std::endl;
}

#include <omp.h>
#include <cstdlib>

void falsesharing(){
const long nbThreads = omp_get_max_threads();
const long maxIncrement = (64/sizeof(long int));
const long int size = nbThreads*maxIncrement;
long int* arr = (long int*)aligned_alloc(64, size * sizeof(long int));
memset(arr, 0, size * sizeof(long int));

const long int nbRepeat = 512*1024*1024;

Timer timernofalse;
#pragma omp parallel num_threads(nbThreads)
{
#pragma omp master
timernofalse.start();

#pragma omp barrier

volatile long int& val = arr[omp_get_thread_num()*maxIncrement]; 
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
val += 1;
val -= 1;
val *= 10;
}

#pragma omp barrier

#pragma omp master
timernofalse.stop();
}
timernofalse.stop();


const long minIncrement = (sizeof(long int)/sizeof(long int));

Timer timerfalsesharing;
#pragma omp parallel num_threads(nbThreads)
{
#pragma omp master
timerfalsesharing.start();

#pragma omp barrier

volatile long int& val = arr[omp_get_thread_num()*minIncrement];

for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
val += 1;
val -= 1;
val *= 10;
}

#pragma omp barrier

#pragma omp master
timerfalsesharing.stop();
}
timerfalsesharing.stop();

free(arr);

std::cout << ">> nb threads : " << nbThreads << std::endl;
std::cout << ">> No false sharing : " << timernofalse.getElapsed() << std::endl;
std::cout << ">> False sharing : " << timerfalsesharing.getElapsed() << std::endl;
}



void prefetching(){
const long int nbElements = 64/sizeof(long int);
const long int size = 1024*1024;
long int* arr = (long int*)aligned_alloc(64, size * nbElements * sizeof(long int));
memset(arr, 0, size * nbElements * sizeof(long int));

const long int nbRepeat = 2*1024;
const long int inc = 1;

Timer timerregular;
{
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i++){
long int* elements = &arr[(i*inc)%size * nbElements];
for(long int idxElement = 0 ; idxElement < nbElements ; ++idxElement){
elements[idxElement] += idxElement;
}
}
}
}
timerregular.stop();

Timer timernotregular;
{
for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < size; i++){
const long int shuffle = (i&3);
long int* elements = &arr[((i+shuffle)*inc)%size * nbElements];
for(long int idxElement = 0 ; idxElement < nbElements ; ++idxElement){
elements[idxElement] += idxElement;
}
}
}
}
timernotregular.stop();

free(arr);

std::cout << ">> regular access : " << timerregular.getElapsed() << std::endl;
std::cout << ">> irregular access : " << timernotregular.getElapsed() << std::endl;
}



void cache_effect(){
const long int nbLongIntInCacheLine = 64/sizeof(long int);  
const long int cache_size = 32*1024;
const long int nb_set = cache_size/64;
const long int chunk_jump = (nb_set*64);
const long int chunk_jump_long_int = chunk_jump/sizeof(long int);
const long int kway = 8;
const long int size = chunk_jump_long_int*kway*2;
long int* arr = (long int*)aligned_alloc(64, size * sizeof(long int));
memset(arr, 0, size * sizeof(long int));

const long int nbRepeat = 1024*1024*1024;


Timer allsamewaytwice;

for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < kway; i++){
for(long int idxElement = 0 ; idxElement < nbLongIntInCacheLine ; ++idxElement)
arr[i * chunk_jump_long_int + idxElement] += 3;
}
for (long int i = 0; i < kway; i++){
for(long int idxElement = 0 ; idxElement < nbLongIntInCacheLine ; ++idxElement)
arr[i * chunk_jump_long_int + idxElement] += 3;
}
}

allsamewaytwice.stop();

Timer alldifferentway;

for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < 2*kway; i++){
for(long int idxElement = 0 ; idxElement < nbLongIntInCacheLine ; ++idxElement)
arr[i * nbLongIntInCacheLine + idxElement] += 3;
}
}

alldifferentway.stop();

Timer allsamewayover;

for(long int repeat = 0 ; repeat < nbRepeat ; ++repeat){
for (long int i = 0; i < 2*kway; i++){
for(long int idxElement = 0 ; idxElement < nbLongIntInCacheLine ; ++idxElement)
arr[i * chunk_jump_long_int + idxElement] += 3;
}
}

allsamewayover.stop();

free(arr);

std::cout << ">> all in the same way twice : " << allsamewaytwice.getElapsed() << std::endl;
std::cout << ">> all in different way : " << alldifferentway.getElapsed() << std::endl;
std::cout << ">> all in the same way conflict : " << allsamewayover.getElapsed() << std::endl;
}

int main(int argc, char** argv){
if(argc != 2){
std::cout << "You must pass a parameter" << std::endl;
std::cout << "Possible values are:" << std::endl;
std::cout << "memory_bound memory_bound_k cache_k ilp aliasing4k misprediction falsesharing prefetching cache_effect" << std::endl;
return -1;
}

const std::string choice = argv[1];

if(choice == "memory_bound") memory_bound();
else if(choice == "memory_bound_k") memory_bound_k();
else if(choice == "cache_k") cache_k();
else if(choice == "ilp") ilp();
else if(choice == "aliasing4k") aliasing4k();
else if(choice == "misprediction") misprediction();
else if(choice == "falsesharing") falsesharing();
else if(choice == "prefetching") prefetching();
else if(choice == "cache_effect") cache_effect();
else{
std::cout << "Invalid parameter, should be one of those" << std::endl;
std::cout << "memory_bound memory_bound_k cache_k ilp aliasing4k misprediction falsesharing prefetching cache_effect" << std::endl;        
}

return 0;
}
