#include <mpi.h>
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
int node, P;
int nReaders;
int nMappers;
int nReducers;
int nWriters;
char* mdata[20]; 
typedef queue<char*, list<char*> > Qreader;
Qreader rQ1;
int* x;
int* y;
int* z;
int tc1;
int tc2;
int uw1;
int uw2;
int uw3;
int* uw4;
int* msrc; 
int* csrc; 
int* wsrc; 
omp_lock_t l0;
omp_lock_t l1;
omp_lock_t l2;
omp_lock_t l3;
omp_lock_t l4;
omp_lock_t l5;
omp_lock_t l6;
omp_lock_t l7;
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
typedef struct wordarray
{
int wc;
char word[27];
} wordarray;
typedef vector<wordarray> Lwords;
Lwords Swords[8][160], Rwords[20];
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
string longline;
char* fdata;
ifstream fin;
fin.open(fname.c_str());
if(!fin)
{
printf("<n%02d> Stale Input File Handle!\n", node);
return NULL;
}
i = 0;
while(!fin.eof())
{
getline(fin, line);
if(line.empty()) continue;
longline += " ";
longline += line;
}
fdata = (char*) malloc((strlen(longline.c_str()) + 1)*sizeof(char));
strcpy(fdata, longline.c_str());
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
printf("<n%02d> Stale Output File Handle!\n", node);
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
printf("<n%02d> CHECK %s : wc(THE) = %d\n", node, fname.c_str(), (*it1).wc);
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
}
void MapRecordsAndReduce(int mt, int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
record t1;
int i;
Lrecords::iterator it3;
Lrecords::iterator it4;
for(i = (node*nReducers + ct); i < 128; i += (P*nReducers))
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
void ReduceRecords(int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
wordarray w;
record t;
int i;
i = 0;
while(!Rwords[ct].empty())
{
it1 = Crecords[ct].begin();
it2 = Crecords[ct].end();
w = Rwords[ct].back();
while(it1 != it2)
{
if(strcasecmp((*it1).words, w.word) == 0)
{
(*it1).wc += w.wc;
break;
}
++it1;
}
if(it1 == it2)
{
t.words = (char*) malloc((strlen(w.word)+1)*sizeof(char));
strcpy(t.words, w.word);
t.wc = w.wc;
Crecords[ct].push_back(t);
++i;
}
Rwords[ct].pop_back();
}
}
void ReduceRecords(int ct, int csrc)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
wordarray w;
record t;
int i;
i = 0;
while(!Swords[csrc][(ct+node*nReducers)].empty())
{
it1 = Crecords[ct].begin();
it2 = Crecords[ct].end();
w = Swords[csrc][(ct+node*nReducers)].back();
while(it1 != it2)
{
if(strcasecmp((*it1).words, w.word) == 0)
{
(*it1).wc += w.wc;
break;
}
++it1;
}
if(it1 == it2)
{
t.words = (char*) malloc((strlen(w.word)+1)*sizeof(char));
strcpy(t.words, w.word);
t.wc = w.wc;
Crecords[ct].push_back(t);
++i;
}
Swords[csrc][(ct+node*nReducers)].pop_back();
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
void MapRecordsToSend(int mt, int pid, int ct)
{
Lrecords::iterator it1;
Lrecords::iterator it2;
int i;
wordarray record1;
for(i = (pid*nReducers + ct); i < 128; i += (P*nReducers))
{
it1 = W[mt].wmap[i].begin();
it2 = W[mt].wmap[i].end();
while(it1 != it2)
{
record1.wc = (*it1).wc;
strcpy(record1.word, (*it1).words);
Swords[mt][(pid*nReducers + ct)].push_back(record1);
++it1;
}
}
return;
}
int main(int argc, char* argv[])
{
int i, j, k, l, r1, m1, c1, w1, master;
int rdone;
int cdone;
int* rsize;
char fid[2];
char* test;
int check;
double time1, time2, time3, time4, time5;
double fmapreadytime, lmapreadytime;
ofstream fout;
int* fileIDs;
int s1;
int* rcomplete;
int allreaddone;
int requestreads;
int readrequest;
int* ReadIDs;
int readalldone;
int sdone, nSi, nSj, ssrc;
int *nMi, *nMj;
int *nC, *receivecount;
int maxuw4;
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
MPI_Comm_rank(MPI_COMM_WORLD, &node);
MPI_Comm_size(MPI_COMM_WORLD, &P);
int blocks[2] = {1, 27};
MPI_Datatype type[2] = {MPI_INT, MPI_CHAR};
MPI_Aint disp[2];
MPI_Aint intex;
MPI_Type_extent(MPI_INT, &intex);
disp[0] = static_cast<MPI_Aint>(0);
disp[1] = intex;
MPI_Datatype MPI_RECORDS_TYPE;
MPI_Type_struct(2, blocks, disp, type, &MPI_RECORDS_TYPE);
MPI_Type_commit(&MPI_RECORDS_TYPE);
if(argc == 4)
{
nReaders = atoi(argv[1]);
nMappers = atoi(argv[2]);
nReducers = atoi(argv[3]);
nWriters = nReaders;       
}
else
{
nReaders = 1;
nMappers = 1;
nReducers = 1;
nWriters = 1;
}
typedef queue<int, list<int> > Qid;
Qid gQids; 
Qid fQids; 
Qid rQids; 
Qid ENDf[nReaders]; 
Qid ENDr; 
Qid ENDc; 
Qid ENDmap; 
MPI_Request sendreqs[(P*nReducers)];
MPI_Status requeststats;
MPI_Status receivestats[nReducers];
fileIDs = (int*) malloc(2*nReaders*sizeof(int));
ReadIDs = (int*) malloc(2*nReaders*sizeof(int));
rcomplete = (int*) malloc(nReaders*sizeof(int));
x = (int*) malloc(nReaders*sizeof(int));
y = (int*) malloc(nMappers*sizeof(int));
z = (int*) malloc(nWriters*sizeof(int));
uw4 = (int*) malloc(nReducers*sizeof(int));
rsize = (int*) malloc(nReaders*sizeof(int));
msrc = (int*) malloc(nMappers*sizeof(int));
csrc = (int*) malloc(nReducers*sizeof(int));
wsrc = (int*) malloc(nWriters*sizeof(int));
nMi = (int*) malloc(nMappers*sizeof(int));
nMj = (int*) malloc(nMappers*sizeof(int));
nC = (int*) malloc(nReducers*sizeof(int));
receivecount = (int*) malloc(nReducers*sizeof(int));
double *readtime = (double*) malloc(nReaders*sizeof(double));
double *maptime = (double*) malloc(nMappers*sizeof(double));
double *mapidletime = (double*) malloc(nMappers*sizeof(double));
char* ifiles[20];
char* ofiles[P*nWriters];
char* fname[nReaders]; 
char* lines[nReaders]; 
j = 0;
for(i = 0; i < 20; i++)
{
ifiles[i] = (char*) malloc(20*sizeof(char));
{
strcpy(ifiles[j], "CleanText/");
sprintf(fid, "%d", (i+1));
strcat(ifiles[j], fid);
strcat(ifiles[j], ".txt");
if(node == 0) gQids.push(j);
j++;
}
}
for(i = 0; i < 80; i++)
{
if(node == 0) gQids.push(i%20);
}
j = 0;
for(i = 0; i < (P*nWriters); i++)
{
ofiles[i] = (char*) malloc(20*sizeof(char));
if(i%P != node) continue;
{
strcpy(ofiles[j], "Output/");
sprintf(fid, "%d", (i+1));
strcat(ofiles[j], fid);
strcat(ofiles[j], ".o");
j++;
}
}
for(i = 0; i < nWriters; i++)
{
fout.open(ofiles[i], ios::out);
fout.close();
}
MPI_Barrier(MPI_COMM_WORLD);
if(node < P)
{
omp_init_lock(&l0);
omp_init_lock(&l1);
omp_init_lock(&l2);
omp_init_lock(&l3);
omp_init_lock(&l4);
omp_init_lock(&l5);
omp_init_lock(&l6);
omp_init_lock(&l7);
omp_set_num_threads(20);
#pragma omp parallel
{
#pragma omp master
{
master = omp_get_thread_num();
r1 = 0;
m1 = 0;
c1 = 0;
w1 = 0;
uw1 = 0;
uw2 = 0;
uw3 = 0;
rdone = 0;
cdone = 0;
allreaddone = 0;
fmapreadytime = 0.0;
lmapreadytime = 0.0;
printf("<n%02d> of %02d Master %02d : ", node, P, omp_get_thread_num());
printf("nReaders (%02d); ", nReaders);
printf("nMappers (%02d); ", nMappers);
printf("nReducers (%02d); ", nReducers);
printf("nWriters (%02d)\n", nWriters);
time1 = MPI_Wtime();
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
rcomplete[rt] = 0;
readtime[rt] = MPI_Wtime();
while(!rcomplete[rt])
{
omp_set_lock(&l0);
if(!fQids.empty())
{
fname[rt] = ifiles[ fQids.front() ];
fQids.pop();
}
else
{
fname[rt] = NULL;
if(!ENDf[rt].empty()) rcomplete[rt] = 1;
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
else
{
usleep(500);
}
}
readtime[rt] = MPI_Wtime() - readtime[rt];
omp_set_lock(&l3);
ENDr.push(rt);
tc1 += x[rt];
omp_unset_lock(&l3);
printf("<n%02d> Read %02d : Total Chars (%d)\n", node, rid, x[rt]);
z[rt] = 0;
while(cdone < nReducers)
{
omp_set_lock(&l5);
if(!ENDc.empty())
{
wsrc[rt] = ENDc.front();
ENDc.pop();
++cdone;
}
else wsrc[rt] = -1;
omp_unset_lock(&l5);
if(wsrc[rt] > -1)
{
z[rt] += writeFile(ofiles[rt], wsrc[rt]);
}
else usleep(500);
}
printf("<n%02d> Writer %02d : Words Written (%d)\n", node, rid, z[rt]);
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
maptime[mt] = 0.0;
mapidletime[mt] = -MPI_Wtime();
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
maptime[mt] -= MPI_Wtime();
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
maptime[mt] += MPI_Wtime();
}
}
usleep(500);
}
maptime[mt] -= MPI_Wtime();
for(nMi[mt] = 0; nMi[mt] < P; nMi[mt]++)
{
{
for(nMj[mt] = 0; nMj[mt] < nReducers; nMj[mt]++)
{
MapRecordsToSend(mt, nMi[mt], nMj[mt]);
}
}
}
maptime[mt] += MPI_Wtime();
mapidletime[mt] += (MPI_Wtime() - maptime[mt]);
omp_set_lock(&l7);
if(!fmapreadytime) fmapreadytime = MPI_Wtime();
ENDmap.push(mt);
lmapreadytime = MPI_Wtime();
omp_unset_lock(&l7);
printf("<n%02d> Map %02d :  Total Chars (%d)\n", node, mid, y[mt]);
}
}
if(node == 0)
{
#pragma omp task 
{
int sid;
sid = omp_get_thread_num();
while(!allreaddone)
{
omp_set_lock(&l0);
if(fQids.empty())
{
if(!allreaddone)
{
MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
MPI_Recv(ReadIDs, (2*nReaders), MPI_INT, 0, node, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
for(s1 = 0; s1 < (2*nReaders); s1++)
{
if(ReadIDs[s1] > -1) fQids.push( ReadIDs[s1] );
else
{
allreaddone = 1;
}
}
if(allreaddone)
{
for(s1 = 0; s1 < nReaders; s1++)
{
ENDf[s1].push(sid);
}
}
}
}
omp_unset_lock(&l0);
usleep(500);
}
printf("<n%02d> Sender %02d : Completed Reads\n", node, sid);
while(sdone < nMappers)
{
omp_set_lock(&l7);
if(!ENDmap.empty())
{
ssrc = ENDmap.front();
ENDmap.pop();
}
else ssrc = -1;
omp_unset_lock(&l7);
if(ssrc > -1)
{
nSj = 0;
for(nSi = 0; nSi < (P*nReducers); nSi++)
{
if((nSi/nReducers) != node)
{
MPI_Isend(Swords[ssrc][nSi].data(), Swords[ssrc][nSi].size(), MPI_RECORDS_TYPE, (nSi/nReducers), (100+nSi), MPI_COMM_WORLD, &sendreqs[nSj]);
nSj++;
}
else
{
MPI_Isend(&ssrc, 1, MPI_INT, (nSi/nReducers), (100+nSi), MPI_COMM_WORLD, &sendreqs[nSj]);
nSj++;
}
}
MPI_Waitall(P*nReducers, sendreqs, MPI_STATUSES_IGNORE);
for(nSi = 0; nSi < (P*nReducers); nSi++)
{
if((nSi/nReducers) != node) Swords[ssrc][nSi].clear();
}
++sdone;
}
else
{
usleep(500);
}
}
printf("<n%02d> Sender %02d : Completed Sends\n", node, sid);
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
uw4[ct] = 0;
for(nC[ct] = 0; nC[ct] < (nMappers*P); nC[ct]++)
{
MPI_Probe(MPI_ANY_SOURCE, (100+ct+(node*nReducers)), MPI_COMM_WORLD, &receivestats[ct]);
if(receivestats[ct].MPI_SOURCE == node)
{
MPI_Recv(&csrc[ct], 1, MPI_INT, node, receivestats[ct].MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
uw4[ct] += Swords[csrc[ct]][(ct+node*nReducers)].size();
ReduceRecords(ct, csrc[ct]);
Swords[csrc[ct]][(ct+node*nReducers)].clear();
}
else
{
MPI_Get_count(&receivestats[ct], MPI_RECORDS_TYPE, &receivecount[ct]);
Rwords[ct].resize(receivecount[ct]);
MPI_Recv(Rwords[ct].data(), receivecount[ct], MPI_RECORDS_TYPE, receivestats[ct].MPI_SOURCE, receivestats[ct].MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
uw4[ct] += receivecount[ct];
ReduceRecords(ct);
Rwords[ct].clear();
}
}
omp_set_lock(&l5);
ENDc.push(ct);
omp_unset_lock(&l5);
printf("<n%02d> Reduce %02d : Reduced Words (%d)\n", node, cid, Crecords[ct].size());
}
}
if(node == 0) 
{
printf("<n%02d> Number of Files to be processed (%02d)\n", node, gQids.size());
readalldone = 0;
while(readalldone < P)
{
MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requeststats);
for(j = 0; j < (2*nReaders); j++)
{
if(!gQids.empty())
{
fileIDs[j] = gQids.front();
gQids.pop();
}
else
{
fileIDs[j] = -1;
}
}
MPI_Send(fileIDs, (2*nReaders), MPI_INT, requeststats.MPI_SOURCE, (requeststats.MPI_SOURCE), MPI_COMM_WORLD);
if(fileIDs[((2*nReaders)-1)] == -1) ++readalldone;
usleep(500);
}
}
else 
{
while(!allreaddone)
{
omp_set_lock(&l0);
if(fQids.empty())
{
if(!allreaddone)
{
MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
MPI_Recv(ReadIDs, (2*nReaders), MPI_INT, 0, node, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
for(s1 = 0; s1 < (2*nReaders); s1++)
{
if(ReadIDs[s1] > -1) fQids.push( ReadIDs[s1] );
else
{
allreaddone = 1;
}
}
if(allreaddone)
{
for(s1 = 0; s1 < nReaders; s1++)
{
ENDf[s1].push(master);
}
}
}
}
omp_unset_lock(&l0);
usleep(500);
}
printf("<n%02d> Sender %02d : Completed Reads\n", node, master);
while(sdone < nMappers)
{
omp_set_lock(&l7);
if(!ENDmap.empty())
{
ssrc = ENDmap.front();
ENDmap.pop();
}
else ssrc = -1;
omp_unset_lock(&l7);
if(ssrc > -1)
{
nSj = 0;
for(nSi = 0; nSi < (P*nReducers); nSi++)
{
if((nSi/nReducers) != node)
{
MPI_Isend(Swords[ssrc][nSi].data(), Swords[ssrc][nSi].size(), MPI_RECORDS_TYPE, (nSi/nReducers), (100+nSi), MPI_COMM_WORLD, &sendreqs[nSj]);
nSj++;
}
else
{
MPI_Isend(&ssrc, 1, MPI_INT, (nSi/nReducers), (100+nSi), MPI_COMM_WORLD, &sendreqs[nSj]);
nSj++;
}
}
MPI_Waitall(P*nReducers, sendreqs, MPI_STATUSES_IGNORE);
for(nSi = 0; nSi < (P*nReducers); nSi++)
{
if((nSi/nReducers) != node) Swords[ssrc][nSi].clear();
}
++sdone;
}
else
{
usleep(500);
}
}
printf("<n%02d> Sender %02d : Completed Sends\n", node, master);
}
}
}
time1 = MPI_Wtime() - time1;
for(i = 0; i < nMappers; i++)
{
DestroyWordFrequency(i);
}
int max;
max = 0;
Lrecords::iterator it1;
Lrecords::iterator it2;
for(i = 0; i < nReducers; i++)
{
it1 = Crecords[i].begin();
it2 = Crecords[i].end();
while(it1 != it2)
{
if(max < strlen((*it1).words)) max = strlen((*it1).words);
++it1;
}
}
printf("<n%02d> MAXIMUM STRING SIZE %d\n", node, max);
test = (char*) malloc(10*sizeof(char));
strcpy(test, "ThE");
for(i = 0; i < nReducers; i++)
{
uw2 += Crecords[i].size();
check = FindRecord(test, i);
printf("<n%02d> CHECK %02d : wc(THE) = %d\n", node, i, check);
DestroyReducerRecords(i);
}
free(test);
printf("<n%02d> Total Chars : (Reader) %d = (Mapper) %d\n", node, tc1, tc2);
printf("<n%02d> Total Words : (Mapper)  %d -> (Reducer) %d\n", node, uw1, uw2);
printf("<n%02d> Total Words : (Reducer) %d  = (Writer)  %d\n", node, uw2, uw3);
printf("<n%02d> Total Elapsed Time is %fs\n", node, time1);
for(i = 0; i < 20; i++) free(ifiles[i]);
for(i = 0; i < (P*nWriters); i++) free(ofiles[i]);
omp_destroy_lock(&l0);
omp_destroy_lock(&l1);
omp_destroy_lock(&l2);
omp_destroy_lock(&l3);
omp_destroy_lock(&l4);
omp_destroy_lock(&l5);
omp_destroy_lock(&l6);
omp_destroy_lock(&l7);
}
else
{
printf("<n%02d> : No Work To DO!\n", node);
usleep(500);
}
free(fileIDs);
free(ReadIDs);
free(rcomplete);
free(x);
free(y);
free(z);
free(rsize);
free(msrc);
free(csrc);
free(wsrc);
free(nMi);
free(nMj);
free(nC);
free(receivecount);
MPI_Type_free(&MPI_RECORDS_TYPE);
time2 = readtime[0];
for(i = 1; i < nReaders; i++)
{
if(time2 < readtime[i]) time2 = readtime[i];
}
time3 = maptime[0];
time4 = mapidletime[0];
for(i = 1; i < nMappers; i++)
{
if(time3 < maptime[i]) time3 = maptime[i];
if(time4 < mapidletime[i]) time4 = mapidletime[i];
}
maxuw4 = 0;
for(i = 0; i < nReducers; i++)
{
if(maxuw4 < uw4[i]) maxuw4 = uw4[i];
}
MPI_Barrier(MPI_COMM_WORLD);
if(node > 0) MPI_Reduce(&time2, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
else MPI_Reduce(MPI_IN_PLACE, &time2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if(node > 0) MPI_Reduce(&time3, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
else MPI_Reduce(MPI_IN_PLACE, &time3, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if(node > 0) MPI_Reduce(&time4, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
else MPI_Reduce(MPI_IN_PLACE, &time4, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if(node > 0) MPI_Reduce(&fmapreadytime, NULL, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
else MPI_Reduce(MPI_IN_PLACE, &fmapreadytime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
if(node > 0) MPI_Reduce(&lmapreadytime, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
else MPI_Reduce(MPI_IN_PLACE, &lmapreadytime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if(node > 0) MPI_Reduce(&maxuw4, NULL, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
else MPI_Reduce(MPI_IN_PLACE, &maxuw4, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
if(node == 0)
{
time5 = lmapreadytime - fmapreadytime;
printf("<n%02d> Statistics : Max Reader Time %fs\n", node, time2);
printf("<n%02d> Statistics : Max Map Busy Time %fs\n", node, time3);
printf("<n%02d> Statistics : Max Map Idle Time %fs\n", node, time4);
printf("<n%02d> Statistics : First-to-Last MapReady Time %fs\n", node, time5);
printf("<n%02d> Statistics : Max Mapper->Reducer words %d\n", node, maxuw4);
}
free(readtime);
free(maptime);
free(mapidletime);
free(uw4);
MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
return 0;
}
