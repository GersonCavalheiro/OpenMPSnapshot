#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <queue>
#include <list>
#include <vector>
#include <fstream>
using namespace std;
int nReaders;
int nMappers;
int nReducers;
int nWriters;
char* fname[20]; 
char* lines[20]; 
char* mdata[20]; 
typedef queue<char*, list<char*> > Qreader;
Qreader rQ1;
int x[20]; 
int y[20]; 
int z[20]; 
int tc1;
int tc2;
int uw1;
int uw2;
int uw3;
int rsize[20]; 
int msrc[20]; 
int csrc[20]; 
int wsrc[20]; 
omp_lock_t l0;
omp_lock_t l1;
omp_lock_t l2;
omp_lock_t l3;
omp_lock_t l4;
omp_lock_t l5;
omp_lock_t l6;
typedef queue<int, list<int> > Qid;
Qid fQids; 
Qid rQids; 
Qid ENDr; 
Qid ENDm[20]; 
Qid ENDc; 
typedef struct record
{
char* words;
int wc;
} record;
typedef vector<record> Lrecords;
typedef struct wfreq
{
Lrecords wmap[128];
} wfreq;
wfreq W[20];
Lrecords Crecords[20];
void pushRQ(Qreader *rQ, char* s)
{
char* str = (char*) malloc((strlen(s)+1)*sizeof(char));
strcpy(str, s);
(*rQ).push(str);	
}
char* popRQ(Qreader *rQ)
{
char* str;
str = (char*) malloc( (strlen((*rQ).front())+1) * sizeof(char) );
strcpy(str, (*rQ).front());
free((*rQ).front());
(*rQ).pop();
return str;
}
int isRQfull(Qreader *rQ)
{
if((*rQ).size() >= (2*nReaders)) return 1;
else return 0;
}
int H(char* x)
{
int i;
unsigned int h;
unsigned long g;
g = 5381;
for(i = 0; x[i] != '\0'; i++)
{
g = ( (33*g) + tolower(x[i]) );
}
g = g%128;
h = (unsigned int)g;
return h;
}
char* readFile(string fname, int& cc)
{
int i, j;
string line;
char* fdata;
ifstream fin;
fin.open(fname.c_str());
if(!fin)
{
printf("Stale Input File Handle!\n");
return NULL;
}
i = 0;
while(!fin.eof())
{
getline(fin, line);
if(line.empty()) continue;
fdata = (char*) malloc((strlen(line.c_str())+1)*sizeof(char));
strcpy(fdata, line.c_str());
}
cc = strlen(fdata);
fin.close();
return fdata;
}
int writeFile(string fname, int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
int i;
char* test;
ofstream fout;
fout.open(fname.c_str(), ios::app);
if(!fout)
{
printf("Stale Output File Handle!\n");
return 0;
}
test = (char*) malloc(10*sizeof(char));
strcpy(test, "ThE");
i = 0;
it1 = Crecords[ct].begin();
it2 = Crecords[ct].end();
while(it1 != it2)
{
if(strcasecmp((*it1).words, test) == 0)
{
printf("CHECK %s : wc(THE) = %d\n", fname.c_str(), (*it1).wc);
}
fout << "< " << (*it1).words << ", " << (*it1).wc << " >" << endl;
++it1;
++i;
}
fout.close();
free(test);
omp_set_lock(&l6);
uw3 += i;
omp_unset_lock(&l6);
return i;
}
void DestroyWordFrequency(int mt)
{
int i;
Lrecords::iterator it1;
Lrecords::iterator it2;
for(i = 0; i < 128; i++)
{
it1 = W[mt].wmap[i].begin();
it2 = W[mt].wmap[i].end();
while(it1 != it2)
{
free((*it1).words);
(*it1).words = NULL;
++it1;
}
W[mt].wmap[i].clear();
}
}
void CreateWordFrequency(int mt)
{
char* word;
char* sentence;
char* split;
unsigned int key;
record t1;
Lrecords::iterator it1;
Lrecords::iterator it2;
int i;
int nUniqueWords;
nUniqueWords = 0;
sentence = (char*) malloc((strlen(mdata[mt]) + 1)*sizeof(char));
strcpy(sentence, mdata[mt]);
split = strtok_r(sentence, " ", &word);
while(split != NULL)
{
key = H(split);
it1 = W[mt].wmap[key].begin();
it2 = W[mt].wmap[key].end();
while(it1 != it2)
{
if(strcasecmp((*it1).words, split) == 0)
{
(*it1).wc += 1;
break;
}
++it1;
}
if(it1 == it2)
{
t1.words = (char*) malloc((strlen(split) + 1) * sizeof(char));
strcpy(t1.words, split);
t1.wc = 1;
W[mt].wmap[key].push_back(t1);
++nUniqueWords;
}
split = strtok_r(NULL, " ", &word);
}
omp_set_lock(&l3);
uw1 += nUniqueWords;
omp_unset_lock(&l3);
free(sentence);
printf("Map %02d : Unique Words (%d)\n", omp_get_thread_num(), nUniqueWords);
}
void MapRecordsAndReduce(int mt, int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
record t1;
int i;
Lrecords::iterator it3;
Lrecords::iterator it4;
for(i = ct; i < 128; i = i+nReducers)
{
it1 = W[mt].wmap[i].begin();
it2 = W[mt].wmap[i].end();
while(it1 != it2)
{
it3 = Crecords[ct].begin();
it4 = Crecords[ct].end();
while(it3 != it4)
{
if(strcasecmp((*it1).words, (*it3).words) == 0)
{
(*it3).wc += (*it1).wc;
break;
}
++it3;
}
if(it3 == it4)
{
t1.words = (char*) malloc((strlen((*it1).words)+1) * sizeof(char));
strcpy(t1.words, (*it1).words);
t1.wc = (*it1).wc;
Crecords[ct].push_back(t1);
}
++it1;
}
}
}
void DestroyReducerRecords(int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
it1 = Crecords[ct].begin();
it2 = Crecords[ct].end();
while(it1 != it2)
{
free((*it1).words);
(*it1).words = NULL;
++it1;
}
Crecords[ct].clear();
}
int FindRecord(char* test, int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
it1 = Crecords[ct].begin();
it2 = Crecords[ct].end();
while(it1 != it2)
{
if(strcasecmp((*it1).words, test) == 0)
{
return (*it1).wc;
}
++it1;
}
return 0;
}
int main(int argc, char* argv[])
{
int i, j, k, l, r1, m1, c1, w1;
int rdone;
int mdone[20];
int cdone;
int q[20];
char fid[2];
char* test;
int check;
char* ifiles[20];
char* ofiles[20];
double time1;
ofstream fout;
if(argc == 5)
{
nReaders = atoi(argv[1]);
nMappers = atoi(argv[2]);
nReducers = atoi(argv[3]);
nWriters = atoi(argv[4]);
}
else
{
nReaders = 1;
nMappers = 1;
nReducers = 1;
nWriters = 1;
}
for(i = 0; i < 20; i++)
{
ifiles[i] = (char*) malloc(20*sizeof(char));
strcpy(ifiles[i], "CleanText/");
sprintf(fid, "%d", (i+1));
strcat(ifiles[i], fid);
strcat(ifiles[i], ".txt");
fQids.push(i);
ofiles[i] = (char*) malloc(20*sizeof(char));
strcpy(ofiles[i], "Output/");
strcat(ofiles[i], fid);
strcat(ofiles[i], ".o");
}
for(i = 0; i < nWriters; i++)
{
fout.open(ofiles[i], ios::out);
fout.close();
}
omp_init_lock(&l0);
omp_init_lock(&l1);
omp_init_lock(&l2);
omp_init_lock(&l3);
omp_init_lock(&l4);
omp_init_lock(&l5);
omp_init_lock(&l6);
omp_set_num_threads(20);
#pragma omp parallel
{
#pragma omp master
{
r1 = 0;
m1 = 0;
c1 = 0;
w1 = 0;
uw1 = 0;
uw2 = 0;
uw3 = 0;
rdone = 0;
cdone = 0;
printf("Master %02d : ", omp_get_thread_num());
printf("nReaders (%02d); ", nReaders);
printf("nMappers (%02d); ", nMappers);
printf("nReducers (%02d); ", nReducers);
printf("nWriters (%02d)\n", nWriters);
time1 = omp_get_wtime();
for(i = 0; i < nReaders; i++)
{
#pragma omp task 
{
int rt, rid;
rid = omp_get_thread_num();
omp_set_lock(&l0);
rt = r1;
r1++;
omp_unset_lock(&l0);
x[rt] = 0;
while(!fQids.empty())
{
omp_set_lock(&l0);
if(!fQids.empty())
{
fname[rt] = ifiles[ fQids.front() ];
fQids.pop();
}
omp_unset_lock(&l0);
if(fname[rt])
{
lines[rt] = readFile(fname[rt], rsize[rt]);
omp_set_lock(&l1);
while(isRQfull(&rQ1))
{
usleep(500);
}
if(!isRQfull(&rQ1))
{
pushRQ(&rQ1, lines[rt]);
}
omp_unset_lock(&l1);
omp_set_lock(&l2);
rQids.push(rt);
omp_unset_lock(&l2);
x[rt] += rsize[rt];
free(lines[rt]);
lines[rt] = NULL;
}
}
omp_set_lock(&l3);
ENDr.push(rt);
tc1 += x[rt];
omp_unset_lock(&l3);
printf("Read %02d : Total Chars (%d)\n", rid, x[rt]);
}
}
for(j = 0; j < nMappers; j++)
{
#pragma omp task 
{
int mt, mid;
mid = omp_get_thread_num();
omp_set_lock(&l3);
mt = m1;
m1++;
omp_unset_lock(&l3);
y[mt] = 0;
while(rdone < nReaders)
{
omp_set_lock(&l3);
if(!ENDr.empty())
{
++rdone;
ENDr.pop();
}
omp_unset_lock(&l3);
while(!rQ1.empty())
{
omp_set_lock(&l4);
if(!rQ1.empty())
{
mdata[mt] = popRQ(&rQ1);
tc2 += strlen(mdata[mt]);
}
omp_unset_lock(&l4);
if(mdata[mt] != NULL)
{
omp_set_lock(&l2);
msrc[mt] = rQids.front();
rQids.pop();
omp_unset_lock(&l2);
y[mt] += strlen(mdata[mt]);
CreateWordFrequency(mt);
free(mdata[mt]);
mdata[mt] = NULL;
}
}
usleep(500);
}
omp_set_lock(&l4);
for(q[mt] = 0; q[mt] < nReducers; q[mt] += 1)
{
ENDm[q[mt]].push(mt);
}
omp_unset_lock(&l4);
printf("Map %02d :  Total Chars (%d)\n", mid, y[mt]);
}
}
for(k = 0; k < nReducers; k++)
{
#pragma omp task 
{
int ct, cid;
cid = omp_get_thread_num();
omp_set_lock(&l4);
ct = c1;
++c1;
omp_unset_lock(&l4);
mdone[ct] = 0;
while(mdone[ct] < nMappers)
{
omp_set_lock(&l4);
if(!ENDm[ct].empty())
{
++mdone[ct];
csrc[ct] = ENDm[ct].front();
ENDm[ct].pop();
}
else csrc[ct] = -1;
omp_unset_lock(&l4);
if(csrc[ct] > -1)
{
MapRecordsAndReduce(csrc[ct], ct);
}
else usleep(500);
}
omp_set_lock(&l5);
ENDc.push(ct);
omp_unset_lock(&l5);
printf("Reduce %02d : Reduced Words (%d)\n", cid, Crecords[ct].size());
}
}
for(l = 0; l < nWriters; l++)
{
#pragma omp task 
{
int wid, wt;
wid = omp_get_thread_num();
omp_set_lock(&l6);
wt = w1;
++w1;
omp_unset_lock(&l6);
z[wt] = 0;
while(cdone < nReducers)
{
omp_set_lock(&l5);
if(!ENDc.empty())
{
wsrc[wt] = ENDc.front();
ENDc.pop();
++cdone;
}
else wsrc[wt] = -1;
omp_unset_lock(&l5);
if(wsrc[wt] > -1)
{
z[wt] += writeFile(ofiles[wt], wsrc[wt]);
}
else usleep(500);
}
printf("Writer %02d : Words Written (%d)\n", wid, z[wt]);
}
}
}
}
time1 = omp_get_wtime() - time1;
for(i = 0; i < nMappers; i++)
{
DestroyWordFrequency(i);
}
test = (char*) malloc(10*sizeof(char));
strcpy(test, "ThE");
for(i = 0; i < nReducers; i++)
{
uw2 += Crecords[i].size();
check = FindRecord(test, i);
printf("CHECK %02d : wc(THE) = %d\n", i, check);
DestroyReducerRecords(i);
}
free(test);
printf("Total Chars : (Reader) %d = (Mapper) %d\n", tc1, tc2);
printf("Total Words : (Mapper)  %d -> (Reducer) %d\n", uw1, uw2);
printf("Total Words : (Reducer) %d  = (Writer)  %d\n", uw2, uw3);
printf("Total Elapsed Time is %fs\n", time1);
for(i = 0; i < 20; i++) free(ifiles[i]);
for(i = 0; i < 20; i++) free(ofiles[i]);
omp_destroy_lock(&l0);
omp_destroy_lock(&l1);
omp_destroy_lock(&l2);
omp_destroy_lock(&l3);
omp_destroy_lock(&l4);
omp_destroy_lock(&l5);
omp_destroy_lock(&l6);
return 0;
}
