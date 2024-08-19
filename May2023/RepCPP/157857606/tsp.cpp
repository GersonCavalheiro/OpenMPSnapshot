#include <iostream>
#include <algorithm> 
#include <fstream>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unordered_map>
#include <climits>
#include <vector>
#include <assert.h>
using namespace std;

unordered_map<int,string> idToname;
unordered_map<int,float> idToXcord;
unordered_map<int,float> idToYcord;
float** distTable;

int n_popl = 3000;
int m_popl = 1000;
int num_iter = 1000;

int numCities;
string initChromosome;
pair<float,string> minCycle = make_pair(INT_MAX,"");



char getChar(int i){		
if (i<26)
return char(97+i);
else
return char(22+i);
}

int getInt(char c){
if (c>='a' && c<='z')
return c-'a';
else
return c-'0'+26;
}

void input(char filename[]){
string line;
ifstream in;
in.open(filename);
if (!in.is_open()){
cout << "File not found.\n";
return;
}
else{
string input_str = "";
in >> input_str;
in >> input_str;

string str = "";
in >> str;
in >> str;
numCities = stoi(str);
initChromosome = "";

in >> str;
for (int i=0; i<numCities; i++){
float c,d;
string a = "";
in >> a >> c >> d;
idToname[i] = a;
idToXcord[i] = c;
idToYcord[i] = d;
initChromosome += getChar(i);
}
return;
}
}

void fillDistances(){
distTable = new float*[numCities];
for (int i=0; i<numCities; i++){
distTable[i] = new float[numCities];
for (int k=0; k<numCities; k++){
int j = 0;
if (i==k)
distTable[i][k] = 0.0;
else{
float n = sqrt(pow(idToXcord[i]-idToXcord[k],2) + pow(idToYcord[i]-idToYcord[k],2));
distTable[i][k] = n;
}
}
}
}

vector<string> generateInitialPopulation(){
vector<string> v;
string s = initChromosome;
for (int i =0; i<n_popl; i++){
random_shuffle(s.begin(),s.end());
v.push_back(s);
}
return v;
}

float getHamiltonianCycle(string s){
float eucD = 0.0;
assert(s.size()==numCities);
for (int i=0; i<numCities; i++){
int j = (i+1)%numCities;
eucD += distTable[getInt(s[i])][getInt(s[j])];
}
return eucD;
}

bool compare(string s1,string s2){
return (getHamiltonianCycle(s1)<getHamiltonianCycle(s2));
}

float evalFitnessPopl(vector<string> &popl){
sort(popl.begin(),popl.end(),compare);
return getHamiltonianCycle(popl[0]);
}

string performPMX(string p1, string p2){
int c = rand()%100;
if (c < 80){
int l1 = rand()%p1.size();
int l2 = rand()%p1.size();
int low = l1 <= l2 ? l1 : l2;
int high = l1 >= l2 ? l1 : l2;
while (low<=high){
char c = p2[low];
for (int i = 0; i<p1.size(); i++){
if (p1[i] == c){
p1[i] = p1[low];
p1[low] = p2[low];
break;
}
}
low++;
}
}
return p1;
}

string performGX(string p1, string p2){
int c = 10;
if (c < 80){
string str = "";
str += p2[0];
int i1 = p1.find(str[0]);
int i2 = 0;
for (int i = 1; i<numCities; i++){
int j1 = (i1+1)%numCities;
int j2 = (i2+1)%numCities;
if (distTable[getInt(p1[i1])][getInt(p1[j1])] <= distTable[getInt(p2[i2])][getInt(p2[j2])]){
if (str.find(p1[j1]) == std::string::npos){
str += p1[j1];
i1 = j1;
i2 = p2.find(str[i]);
}
else if (str.find(p2[j2]) == std::string::npos){
str += p2[j2];
i1 = p1.find(str[i]);
i2 = j2;
}
else{
while (str.find(p2[j2]) != std::string::npos){
j2 = (j2+1)%numCities;
}
str += p2[j2];
i1 = p1.find(str[i]);
i2 = j2;
}
}
else{
if (str.find(p2[j2]) == std::string::npos){
str += p2[j2];
i1 = p1.find(str[i]);
i2 = j2;
}
else if (str.find(p1[j1]) == std::string::npos){
str += p1[j1];
i1 = j1;
i2 = p2.find(str[i]);
} 
else{
while (str.find(p1[j1]) != std::string::npos){
j1 = (j1+1)%numCities;
}
str += p1[j1];
i1 = j1;
i2 = p2.find(str[i]);
}
}
}
return str;
}
return p2;
}

void performMutation(string &c1){
int c = rand()%100;
if (c < 10){
int l1 = rand()%c1.size();
int l2 = rand()%c1.size();
while (l2==l1){
l2 = rand() % c1.size();
}
char c = c1[l1];
c1[l1] = c1[l2];
c1[l2] = c;	
}
}

void solve(int numThreads){
vector<string> initPopl = generateInitialPopulation();
#pragma omp parallel for firstprivate(initPopl) num_threads(numThreads)
for (int i = 0; i < num_iter; i++){

float localMin = evalFitnessPopl(initPopl);
#pragma omp critical
minCycle = min(make_pair(localMin,initPopl[0]),minCycle);

vector<string> newPopulation;
for (int i = 0; i<n_popl/2; i++){
int l1 = rand() % m_popl;
int l2 = rand() % m_popl;
while (l2==l1){
l2 = rand() % m_popl;
}
string p1 = initPopl[l1];
string p2 = initPopl[l2];

string c1 = performPMX(p1,p2);
performMutation(c1);

string c2 = performGX(p1,p2);
performMutation(c2);

newPopulation.push_back(c1);
newPopulation.push_back(c2);
}
initPopl.clear();
initPopl = newPopulation;
}
}

void output(string filename, float cost, string path){
ofstream out(filename, ios::out);
out << "DIMENSION : " << numCities << endl;
out << "TOUR_LENGTH : " << to_string(cost) << endl;
out << "TOUR_SECTION : " << path << endl;
out << "-1\nEOF";
}	


int main(int argc, char * argv[])
{
char* infile = argv[1];
char* outfile = argv[2];
int numThreads = atoi(argv[3]);

srand(time(NULL));

input(infile);

fillDistances();

double start = omp_get_wtime();
solve(numThreads);

cout << "Time taken : " << omp_get_wtime() - start << " s" << endl;
cout << "Cost: " << minCycle.first << endl;
string str = "";
for (int i=0; i < (minCycle.second).size(); i++){
str += idToname[getInt(minCycle.second[i])];
str += " ";
}
cout << "Path : " << str << endl;

output(outfile, minCycle.first, str);
return 0;
}
