
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <unordered_map>
#include <unordered_set>
#include "Graph.h"
#include "Chromosome.h"

#define startingChar 48
#define P 256
#define maxIter 1000
using namespace std;

int totalFitness;
vector<Chromosome> updatedPopulation;
int numCities;
string baseStr;
Graph gGlobal;
int InitPopulation = 1000;


int getIDFromChar(char ch)
{
return (ch - startingChar);
}

struct descSort
{
inline bool operator()(const Chromosome &chrome1, const Chromosome &chrome2)
{
return (chrome1.fitVal > chrome2.fitVal);
}
};

int randInt(int n)
{
int limit = RAND_MAX - RAND_MAX % n;
int random;
do
{
random = rand();
} while (random >= limit);
return random % n;
}

void shuffle(string &str, int n)
{
char temp;
for (int i = n - 1; i > 0; i--)
{
int j = randInt(i + 1);
temp = str[j];
str[j] = str[i];
str[i] = temp;
}
}


bool check(string s)
{
if (s.length() != numCities)
return false;
int arr[numCities];
fill_n(arr, numCities, 0);
for (int i = 0; i < numCities; i++)
{
if (arr[getIDFromChar(s[i]) - 1] == 0)
arr[getIDFromChar(s[i]) - 1] = 1;
else
return false;
}

return true;
}
void printCities(Graph &g)
{
for (int i = 0; i < g.NumCities; i++)
{
printf("CityIds: %d Lat:%f Long:%f\n", g.cities[i].first, g.cities[i].second.first, g.cities[i].second.second);
}
}
void printDistances(Graph &g)
{
printf("Adjacency Matrix\n");
for (int i = 0; i < g.NumCities; i++)
{
printf("Row[%d]: ", i);
for (int j = 0; j < g.NumCities; j++)
printf("%f ", g.distances[i][j]);
printf("\n");
}
}
void printArray(void *start, int length, int args)
{
switch (args)
{
case -1:;
double *ptr0;
ptr0 = (double *)start;
for (int i = 0; i < length; i++)
printf("%f\n", *(ptr0 + i));
printf("\n");
break;
case 0:;
char *ptr1;
ptr1 = (char *)start;
for (int i = 0; i < length; i++)
printf("%c", *(ptr1 + i));
printf("\n");
break;
case 1:;
int *ptr2;
ptr2 = (int *)start;
for (int i = 0; i < length; i++)
printf("%d\n", *(ptr2 + i));
printf("\n");
break;
default:;
char **ptr3;
ptr3 = (char **)start;
for (int i = 0; i < length; i++)
{
printf("Chromosomes[%d]: ", i);
for (int j = 0; j < args; j++)
printf("%c", ptr3[i][j]);
printf("\n");
}
}
}

string getRandomSequence(Graph &g)
{
string ret = "";





ret = baseStr;
shuffle(ret, g.NumCities);
return ret;
}
vector<Chromosome> makePopulation(Graph &g, int population)
{
vector<Chromosome> ret;
unordered_set<string> set;
for (int i = 0; ret.size() != population; i++)
{
string s = getRandomSequence(g);
if (set.find(s) == set.end())
{
Chromosome chromosome;
chromosome.id = i;
chromosome.sequence = s;
chromosome.isFitnessCalculated = false;
ret.push_back(chromosome);
set.insert(s);
}
}
cout << "SET SIZE " << set.size() << endl;
return ret;
}

void calculateFitness(vector<Chromosome> &chromosomes, Graph &g)
{
totalFitness = 0;
for (int i = 0; i < chromosomes.size(); i++)
{
assert(check(chromosomes[i].sequence));
if (!chromosomes[i].isFitnessCalculated)
{
double sum = 0;
string s = chromosomes[i].sequence;
int id1, id2;
for (int j = 0; j < g.NumCities - 1; j++)
{
id1 = getIDFromChar(s[j]);
id2 = getIDFromChar(s[j + 1]);
sum = sum + g.distances[id1 - 1][id2 - 1];
}
sum = sum + g.distances[getIDFromChar(s[0]) - 1][getIDFromChar(s[g.NumCities - 1]) - 1];
chromosomes[i].dist = sum;
sum = ((1.0) / sum);
totalFitness += sum;
chromosomes[i].fitVal = sum;
chromosomes[i].isFitnessCalculated = true;
}
}
}
pair<Chromosome, Chromosome> cxCrossover(Chromosome p1, Chromosome p2)
{
srand(int(time(NULL)) ^ omp_get_thread_num());
vector<int> indices;
int firstRandomIndex = rand() % numCities;
indices.push_back(firstRandomIndex);
char ch = p2.sequence[firstRandomIndex];
int i;
for (i = 0; i < p1.sequence.length(); i++)
if (p1.sequence[i] == ch)
break;
firstRandomIndex = i;
while (firstRandomIndex != indices[0])
{
indices.push_back(firstRandomIndex);
ch = p2.sequence[firstRandomIndex];
int j = 0;
for (j = 0; j < p1.sequence.length(); j++)
if (p1.sequence[j] == ch)
break;
firstRandomIndex = j;
}

Chromosome c1, c2;
c1.sequence = p1.sequence;
c2.sequence = p2.sequence;
c1.isFitnessCalculated = false;
c2.isFitnessCalculated = false;

for (int counter = 0; counter < indices.size(); counter++)
{
char tempCh = c1.sequence[counter];
c1.sequence[counter] = c2.sequence[counter];
c2.sequence[counter] = tempCh;
}

return make_pair(c1,c2);
}
pair<Chromosome, Chromosome> gxCrossover(Chromosome p1, Chromosome p2)
{
Chromosome c1, c2;
c1.sequence = p1.sequence;
c2.sequence = p2.sequence;
c1.isFitnessCalculated = false;
c2.isFitnessCalculated = false;

unordered_map<char, char> umap1;
unordered_map<char, char> umap2;

for (int i = 0; i < c1.sequence.length() - 1; i++)
{
umap1.insert({c1.sequence[i], c1.sequence[i + 1]});
umap2.insert({c2.sequence[i], c2.sequence[i + 1]});
}
umap1.insert({c1.sequence[c1.sequence.length() - 1], c1.sequence[c1.sequence.length() - 2]});
umap2.insert({c2.sequence[c2.sequence.length() - 1], c2.sequence[c2.sequence.length() - 2]});

unordered_set<char> alreadyContains;
alreadyContains.insert(c1.sequence[0]);
int i;
for (i = 0; i < c1.sequence.length() - 1; i++)
{
char ch1 = umap1[c1.sequence[i]];
char ch2 = umap2[c1.sequence[i]];
if (alreadyContains.find(ch1) != alreadyContains.end())
{
if (alreadyContains.find(ch2) != alreadyContains.end())
{
int rIndex = rand() % gGlobal.NumCities;
while (alreadyContains.find(p1.sequence[rIndex]) != alreadyContains.end())
rIndex = rand() % gGlobal.NumCities;
c1.sequence[i + 1] = p1.sequence[rIndex];
alreadyContains.insert(p1.sequence[rIndex]);
}
else
{
c1.sequence[i + 1] = ch2;
alreadyContains.insert(ch2);
}
}
else if (alreadyContains.find(ch2) != alreadyContains.end())
{
c1.sequence[i + 1] = ch1;
alreadyContains.insert(ch1);
}
else
{
int id = getIDFromChar(c1.sequence[i]) - 1;
int id1 = getIDFromChar(ch1) - 1;
int id2 = getIDFromChar(ch2) - 1;
double distance1 = gGlobal.distances[id][id1];
double distance2 = gGlobal.distances[id][id2];
if (distance1 > distance2)
{
c1.sequence[i + 1] = ch2;
alreadyContains.insert(ch2);
}
else
{
c1.sequence[i + 1] = ch1;
alreadyContains.insert(ch1);
}
}
}
string tempStr = c1.sequence;
reverse(tempStr.begin(), tempStr.end());
c2.sequence = tempStr;
return make_pair(c1, c2);
}
pair<Chromosome, Chromosome> pmxCrossover(Chromosome p1, Chromosome p2)
{
int i = rand() % (p1.sequence.length() / 2);
int j = i + rand() % (p1.sequence.length() / 2) + 1;
int start = i;
int end = j;
string mapStr1 = p1.sequence.substr(i, j - i + 1);
string mapStr2 = p2.sequence.substr(i, j - i + 1);
unordered_map<char, char> umap1;
unordered_map<char, char> umap2;
Chromosome c1;
c1.sequence = p1.sequence;
c1.isFitnessCalculated = false;
Chromosome c2;
c2.sequence = p2.sequence;
c1.isFitnessCalculated = false;

int count = 0;
for (i = start; i <= end; i++)
{
c1.sequence[i] = mapStr2[count];
c2.sequence[i] = mapStr1[count];
umap1.insert({mapStr1[count], mapStr2[count]});
umap2.insert({mapStr2[count], mapStr1[count]});
count++;
}

for (i = 0; i < c2.sequence.length(); i++)
{
if (i >= start and i <= end)
continue;
char ch = c2.sequence[i];
while (umap1.find(ch) != umap1.end())
{
ch = umap1[ch];
}
c2.sequence[i] = ch;
}
for (i = 0; i < c1.sequence.length(); i++)
{
if (i >= start and i <= end)
continue;
char ch = c1.sequence[i];
while (umap2.find(ch) != umap2.end())
{
ch = umap2[ch];
}
c1.sequence[i] = ch;
}

return make_pair(c1, c2);
}
vector<Chromosome> crossover(vector<Chromosome> &population, int numParents)
{
vector<Chromosome> children;


#pragma omp parallel
{
srand(int(time(NULL)) ^ omp_get_thread_num());
vector<Chromosome> localChildren;
#pragma omp for private(localChildren)
for (int counter = 0; counter < (numParents / 6); counter++)
{
int i = rand() % numParents;
int j = rand() % numParents;
while (j == i)
j = rand() % numParents;
Chromosome p1 = population[i];
Chromosome p2 = population[j];
if (rand() % 10 <= 7)
{
pair<Chromosome, Chromosome> childrenFromPMX = pmxCrossover(p1, p2);
pair<Chromosome, Chromosome> childrenFromGX = gxCrossover(p1, p2);
pair<Chromosome, Chromosome> childrenFromCX = cxCrossover(p1, p2); 
localChildren.push_back(childrenFromPMX.first);
localChildren.push_back(childrenFromPMX.second);
localChildren.push_back(childrenFromGX.first);
localChildren.push_back(childrenFromGX.second);
}
#pragma omp critical
{
if (localChildren.size() != 0)
children.insert(children.end(), localChildren.begin(), localChildren.end());
}
}
}
return children;
}
vector<Chromosome> slice(vector<Chromosome> &v, int start, int end)
{
int newlen = end - start + 1;
vector<Chromosome> nv(newlen);
for (int i = 0; i < newlen; i++)
{
nv[i] = v[start + i];
}
return nv;
}
void mutate(vector<Chromosome> &children)
{
srand(time(NULL));
#pragma omp parallel
{
srand(int(time(NULL)) ^ omp_get_thread_num());
#pragma omp for
for (int index = 0; index < children.size(); index++)
{
int count = 0;
while (count < numCities)
{
if (rand() % 10 == 1)
{
int i = rand() % numCities;
int j = rand() % numCities;
while (j == i)
j = rand() % numCities;
char temp = children[index].sequence[i];
children[index].sequence[i] = children[index].sequence[j];
children[index].sequence[j] = temp;
}
count++;
}
}
}
}
void mutateAndAddToPopulation(vector<Chromosome> &children, vector<Chromosome> &chromosomes)
{
mutate(children); 
chromosomes = slice(chromosomes, 0, P - 1);
chromosomes.insert(chromosomes.end(), children.begin(), children.end());
InitPopulation = chromosomes.size();
}

pair<Chromosome, double> travellingSalesman(Graph &g)
{
vector<Chromosome> chromosomes;
chromosomes = makePopulation(g, InitPopulation);
srand(time(NULL));
calculateFitness(chromosomes, g);
sort(chromosomes.begin(), chromosomes.end(), descSort());
Chromosome ret = chromosomes[0];
cout << "At iter 0"
<< " "
<< ": " << ret.dist << endl;
cout << "Population: " << chromosomes.size() << endl;

int i = 0;
double start = omp_get_wtime();
while (i < maxIter)
{
cout << "Population before crossover: " << chromosomes.size() << endl;
vector<Chromosome> children = crossover(chromosomes, P); 
cout << "Population before mutation: " << (chromosomes.size() + children.size()) << endl;
mutateAndAddToPopulation(children, chromosomes);
cout << "Population: " << chromosomes.size() << endl;
calculateFitness(chromosomes, g);
sort(chromosomes.begin(), chromosomes.end(), descSort());
ret = chromosomes[0];
i++;
cout << "At iter " << i << " "
<< ": " << ret.dist << endl;
}
double end = omp_get_wtime();
cout << "Time taken: " << end - start << endl;
return make_pair(ret, end - start);
}

string numberToStr(int a, double d, bool flag)
{
string ret = "";
if (flag)
{
int temp = a;
while (temp >= 1)
{
int dig = temp % 10;
ret += (char)(dig + '0');
temp /= 10;
}
reverse(ret.begin(), ret.end());
}
else
{
double temp = d;
string part1 = numberToStr(floor(d), 0, true);
int count = 0;
ret.append(part1);
ret += ".";
temp = temp - floor(temp);
while (count < 6)
{
temp = temp * 10;
int tt = floor(temp);
int dig = tt % 10;
ret += (char)(dig + '0');
temp = temp - floor(temp);
count++;
}
}

return ret;
}
void dumpToFile(string output, double time)
{
ofstream file;
output.append("Time Taken: ");
output.append(numberToStr(0, time, false));
file.open("output.txt");
file << output;
file.close();
return;
}
string getResultAndShow(Chromosome result)
{
string path = "";
for (int i = 0; i < result.sequence.length(); i++)
{
cout << (int)result.sequence[i] - startingChar << " ";
if ((int)result.sequence[i] - startingChar >= 10)
path.append(numberToStr((int)result.sequence[i] - startingChar, 0, true));
else
path += (char)(((int)result.sequence[i] - startingChar) + '0');
path += " ";
}
cout << "\nDistance: " << result.dist << endl;
string output = "DIMENSION : ";
output += numberToStr(numCities, 0, true);
output += "\n";
output += "TOUR_LENGTH : ";
output += numberToStr(0, result.dist, false);
output += "\n";
output += "TOUR_SECTION\n";
output.append(path);
output += "\n-1\nEOF\n";
return output;
}
int main(int arc, char const *argv[])
{
srand(time(NULL));

Graph g;
omp_set_num_threads(atoi(argv[2]));
g.makeGraph(argv[1]);
numCities = g.NumCities;
gGlobal = g;
baseStr = "";
for (int i = 1; i <= numCities; i++)
{
baseStr += (i + '0');
}
pair<Chromosome, double> Result = travellingSalesman(g);
dumpToFile(getResultAndShow(Result.first), Result.second);
return 0;
}