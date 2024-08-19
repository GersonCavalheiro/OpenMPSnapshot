#include <iostream>
#include <limits>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <string>
#include <string.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <bits/stdc++.h>
#include <utility>
#include <cctype>
#include <tuple>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;

struct city
{
double x_coordinate;
double y_coordinate;
int city_id;

city(int id, double x, double y){
x_coordinate = x;
y_coordinate = y;
city_id = id;
}

int getXcoordinate(){
return x_coordinate ;
}

int getYcoordinate(){
return y_coordinate ;
}

};

struct chromosome
{
vector<city> gene;
double distance;
};

std::vector<city> cities ;
std::vector<std::vector<city>> population ;
int num_cities;
std::vector<chromosome> generation;
int num_parents;
int threads_generation;

void swap(int l, int i)
{
city temp(cities.at(l).city_id, cities.at(l).x_coordinate, cities.at(l).y_coordinate);
cities.at(l).city_id = cities.at(i).city_id;
cities.at(l).x_coordinate = cities.at(i).x_coordinate;
cities.at(l).y_coordinate = cities.at(i).y_coordinate;

cities.at(i).city_id = temp.city_id;
cities.at(i).x_coordinate = temp.x_coordinate;
cities.at(i).y_coordinate = temp.y_coordinate;
}

std::vector<city> permute(int l, int r, int num)
{
int i;
if (l == r){
std::vector<city> v;
for(int j=0 ; j <num_cities; j++){
city temp(cities[j].city_id, cities[j].x_coordinate, cities[j].y_coordinate);
v.push_back(temp);
}
population.push_back(v);
return v;
}
else{
for (i = l; i <= r; i++){
swap(l, i);
permute(l+1, r, num);
if (population.size()>num){
i = r;
}
swap(l,i); 
}
return cities;
}

}

unsigned long long factorial(int n){
unsigned long long factorial = 1;
for(int i = 1; i<=n; ++i)
{
factorial *= i;
}
return factorial;
}



void createChromosomes(){

num_parents = rand()%6500 + 500;
if(factorial(num_cities) < 500){
num_parents = (int) factorial(num_cities);
}
population.reserve(num_parents);
std::vector<city> ret = permute(0, num_cities-1, num_parents);

for(int i=0;i<num_parents;i++){
chromosome temp_chromo;
temp_chromo.gene = population[i] ;
temp_chromo.distance = -1;
generation.push_back(temp_chromo);
}

}

double calculate_distance (double a,double b, double c, double d)
{
double result;
result=sqrt(pow((a-b),2) + pow((c-d),2));
return result;
}

void print_gen(vector<chromosome> a, int size,int num)
{
for(int d=0;d<size;d++)
{
cout<<"Chromosome no. "<<d<<" with gene permutation as ";
for(int j=0;j<num;j++)
{
cout<<a[d].gene[j].city_id;
}
cout<<" with distance "<<a[d].distance<<endl;
}
}

void sort(int num_cities,int generation_size)
{
cout<<"##################              SORTING           ###################"<<endl;
int i,j;
int n = generation_size;
int t= n-1;
float temp_d;
int gen_1, gen_2;
#pragma omp parallel for private(j, temp_d) shared(n)
for(i=0;i<generation_size;i++)
{
temp_d=0;
for(j=1;j<num_cities+1;j++)
{
temp_d=temp_d+calculate_distance(generation[i].gene[j%num_cities].x_coordinate,generation[i].gene[(j-1)%num_cities].x_coordinate,generation[i].gene[j%num_cities].y_coordinate,generation[i].gene[(j-1)%num_cities].y_coordinate);
}
generation[i].distance=temp_d;    
}
chromosome tmp;
int phase;
#pragma omp parallel num_threads(threads_generation) shared(n,t) private(i, tmp, phase, gen_1, gen_2)
for(phase=0;phase<n;phase++){
if (phase%2 == 0)
{
#pragma omp for 
for(i=1;i<n;i+=2)  {
gen_1 = generation[i-1].distance;
gen_2 = generation[i].distance;
if (gen_1>gen_2)
{
tmp = generation.at(i-1);
generation.at(i-1) = generation.at(i);
generation.at(i) = tmp;
}
}
}
else
{
#pragma omp for
for(i=1;i<t;i+=2) {
if (generation[i].distance > generation[i+1].distance) {
tmp = generation.at(i+1);
generation.at(i+1) = generation.at(i);
generation.at(i) = tmp;
}
}
}
}
}

void pmx(chromosome &a,chromosome &b)
{
int start=rand()%(num_cities-1);
int remaining= num_cities-start-1;
int last = rand()%remaining + (start+1);
for(int i= start; i < last+1; i++){
city temp(a.gene[i].city_id, a.gene[i].x_coordinate, a.gene[i].y_coordinate);
a.gene[i] = b.gene[i];
b.gene[i] = temp;
}
for(int i=0; i<num_cities; i++){
if(i < start || i > last){
for(int j=start; j<last+1; j++){
if(a.gene[i].city_id == a.gene[j].city_id){
a.gene[i] = b.gene[j] ;
j = start-1;
}	
}

for(int j=start; j<last+1; j++){
if(b.gene[i].city_id == b.gene[j].city_id){
b.gene[i] = a.gene[j] ;
j = start-1;
}
}	
}
}
}
int find(chromosome &a, city c)
{
for(int i=0;i<a.gene.size();i++)
{
if(c.city_id==(a.gene[i].city_id))
{
return i;
}
}
return -1;
}
city compare_add(city a,city b,city source)
{
if(calculate_distance(source.x_coordinate,a.x_coordinate,source.y_coordinate,a.y_coordinate)<calculate_distance(source.x_coordinate,b.x_coordinate,source.y_coordinate,b.y_coordinate))
{
return a;
}
return b;
}

void gx(chromosome &a, chromosome &b){

chromosome child;
child.distance=-1;
int pa_c, pb_c;
child.gene.push_back(a.gene[0]);

for(int i=0; i<num_cities-1;i++){

pa_c = find(a, child.gene[i]);
pb_c = find(b, child.gene[i]);

if(find(child, a.gene[((pa_c+1)%num_cities)]) >=0){
if(find(child, b.gene[((pb_c+1)%num_cities)]) >=0){
while(1){
int rand_pos=rand() % num_cities;
if(find(child,a.gene[rand_pos])<0)
{
child.gene.push_back(a.gene[rand_pos]);
break;
}
}
}
else{
child.gene.push_back(b.gene[(pb_c+1)%num_cities]);
}
}
else{
if(find(child, b.gene[((pb_c+1)%num_cities)]) >=0){
child.gene.push_back(a.gene[(pa_c+1)%num_cities]);
}
else{
city shorter(-1,-1,-1);
if(calculate_distance(child.gene[i].x_coordinate,a.gene[(pa_c+1)%num_cities].x_coordinate,child.gene[i].y_coordinate,a.gene[(pa_c+1)%num_cities].y_coordinate)<calculate_distance(child.gene[i].x_coordinate,b.gene[(pb_c+1)%num_cities].x_coordinate,child.gene[i].y_coordinate,b.gene[(pb_c+1)%num_cities].y_coordinate))
{
shorter=a.gene[(pa_c+1)%num_cities];				
}
else
{	
shorter=b.gene[(pb_c+1)%num_cities];
}
child.gene.push_back(shorter);
}
}

}
for(int i=0;i<num_cities;i++)
{
a.gene[i]=child.gene[i];
b.gene[num_cities-i-1]=child.gene[i];
}
}

double evaluateFitness(chromosome c)
{

return 1/(c.distance);
}

void mutate(chromosome &offspring){
double random = rand() % 10;
random = random/10;
if(random > 0.1)
{
return;
}
int random1 = rand() % num_cities;
int random2 = rand() % num_cities;

city tmp(offspring.gene[random1].city_id, offspring.gene[random1].x_coordinate, offspring.gene[random1].y_coordinate);
offspring.gene[random1] = offspring.gene[random2];
offspring.gene[random2] = tmp;
}

void crossover(chromosome &parentA, chromosome &parentB)
{


double random = rand() % 10;
random = random/10;


if(random > 0.8)
{
return;
}
else{
pmx(parentA, parentB);
gx(parentA, parentB);
mutate(parentA);
mutate(parentB);
}
}

void write_output()
{
std::ofstream outfile;
outfile.open( "outfile.txt" );
outfile << "DIMENSION : " ;
outfile << num_cities<<endl;
outfile << "TOUR_LENGTH : ";
outfile<<generation[0].distance<<endl;
for(int i=0; i<num_cities; i++){
outfile<<generation[0].gene[i].city_id;
outfile << "\n" ;
}
outfile<<"-1";

outfile.close();
}


int main(int argc, const char *argv[])
{
if(argc == 2){
threads_generation = atoi(argv[1]) ;
}
else{
printf("%s\n", "Too many arguments passed");
}

srand(time(0));
std::ifstream in("input.txt");
string line;
int line_num = -1;

int city_counter=0;
while (getline(in, line))
{

if(!(line.substr(0,3) == "EOF")) {
if(line_num == 0){
num_cities = std::stoi(line.substr(11, line.length()-11) );

}
if(line_num > 1){
string str = "";
int first_num = 0 ;
int city_id;
double city_x, city_y;
for(int i=0 ; i < line.length(); i++){  			
char c = line.at(i) ;
if(c == ' '){
if(first_num == 0){
int n = std::stoi(str);
city_id = n;
}
if(first_num == 1){
double p = atof(str.c_str());
city_x = p;	
}	
str = "" ;	
first_num++;    			
}
else{
str += c ;
}
}
double p = atof(str.c_str());
city_y = p;
city c(city_id, city_x, city_y );
cities.push_back(c) ;
}
line_num++ ;
}
}

createChromosomes();
if(threads_generation > num_parents){
threads_generation = num_parents;
}
int generationsWithoutImprovement=0;

while(1)   	
{   	
sort(num_cities,num_parents);
int i, threads;
if (threads_generation == num_parents){
threads = threads_generation/2 ;
}
else{
threads = threads_generation;
}
#pragma omp parallel for  num_threads(threads) private(i)
for(i=1 ; i < num_parents; i+=2)
{
cout<<"##################              Generation Number "<<generationsWithoutImprovement+1<<"         ###################"<<endl;
cout<<"##################  Crossing over for chromosome "<<i-1<<" and "<<i<<"  ###################"<<endl;
crossover(generation[i-1], generation[i]);
}
generationsWithoutImprovement++;
if(generationsWithoutImprovement==500)
break;
}

sort(num_cities,num_parents);
cout<<"AFTER SORT"<<endl;
print_gen(generation, 30, num_cities);
write_output();
return 0;
}