#include <stdio.h>
#include <string.h>
#include <string>
#include <fstream>
#include <unistd.h>
#include <cmath>
#include <omp.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <vector>
#include <immintrin.h> 
#include "HyperLogLog.h"
#define lim 4294967296 
#define lim_int 2147483647

using namespace std;
typedef unsigned long long int ullint;

unsigned char k=31; 
ullint bits_G;
ullint bits_T;
ullint bits_C;
ullint BITS;

inline float hsum_sse3(__m128 v) {
__m128 shuf = _mm_movehdup_ps(v);        
__m128 maxs = _mm_add_ps(v, shuf);
shuf        = _mm_movehl_ps(shuf, maxs); 
maxs        = _mm_add_ss(maxs, shuf);
return        _mm_cvtss_f32(maxs);
}

inline float hsum_avx(__m256 v) {
__m128 lo = _mm256_castps256_ps128(v);   
__m128 hi = _mm256_extractf128_ps(v, 1); 
lo = _mm_add_ps(lo, hi);          
return hsum_sse3(lo);                    
}

vector<vector<float>> estJaccard(vector<HyperLogLog*> hll, vector<float> &cards,int b,int N, int numThreads){
omp_set_num_threads(numThreads);
__m256i vec3; 
ullint bits_and[4]={15,15,15,15};

float a_m=(0.7213/(1+(1.079/N)))*N*N;
int ciclos_red=(b+2)/8+(((b+2)%8)>0);

int tam=hll.size();
vector<vector<float>> jaccards(tam);

#pragma omp parallel for 
for(int j1=0;j1<tam;j1++){
vector<float> temp_jaccards; 
temp_jaccards.resize(tam-j1-1);
for(int j2=j1+1;j2<tam;j2++){
vector<ullint> s1ref=(hll[j1]->getSketch());
vector<ullint> s2ref=(hll[j2]->getSketch());
vector<ullint>::iterator it1=s1ref.begin();
vector<ullint>::iterator it2=s2ref.begin();
vector<ullint>::iterator fin=s1ref.end();
vector<float> wU(32,0.0);
while(it1!=fin){ 
ullint i1=*it1,i2=*it2;



for(char i=0;i<16;++i){ 
ullint temp1=i1&0xF,temp2=i2&0xF;
(temp1>temp2) ? wU[temp1]++ : wU[temp2]++;
i1=i1>>4;
i2=i2>>4;
}
++it1;
++it2; 
}

float cardU=0.0;


float w2[32];
for(int i=0;i<32;++i) w2[i]=1.0;
int respow=1;
for(int i=0;i<b+2;++i){
w2[i]=(float)respow;
respow=respow<<1;
}
for(int i=0;i<32;++i){
printf("%d. %f %f\n",i,wU[i],w2[i]);
}

__m256 vec,vec2;
for(int i=0;i<ciclos_red;++i){
vec=_mm256_loadu_ps((const float *)&wU[i*8]);
vec2=_mm256_loadu_ps((const float *)&w2[i*8]);
vec=_mm256_div_ps(vec,vec2);
cardU+=hsum_avx(vec);
printf("sum: %f\n",hsum_avx(vec));
}

int ceros = wU[0];

cardU=(float)a_m/cardU;
if(ceros && cardU<=5*N/2) 
cardU=N*log(N/ceros);
else if(cardU>lim/30)
cardU=-lim*log(1-(cardU/lim));
printf("estimacion cardinalidad union: %f ceros: %d\n",cardU,ceros);

float jaccard=(cards[j1]+cards[j2]-cardU)/cardU;
if(jaccard<0) jaccard=0;
temp_jaccards[j2-j1-1]=jaccard;
}
jaccards[j1]=temp_jaccards;
}
return jaccards;
}

void printMatrix(vector<vector<float>> &jaccards, vector<string> names){
int tam=names.size();
char guion='-';
for(int j=0;j<tam;j++)
printf("%3s ",names[j].c_str());
printf("\n");
for(int j1=0;j1<tam;j1++){
printf("%3s ",names[j1].c_str());
for(int j2=0;j2<tam;j2++){
if(j2<j1+1){
printf("%3c ",guion);
continue;
}
printf("%3f ",jaccards[j1][j2-j1-1]);
}
printf("\n");
}
}

void saveOutput(char* filename,vector<string> names, vector<vector<float>> jaccards){ 
int tam=names.size();
FILE *fp=fopen(filename,"w");
fprintf(fp,"	");
for(int j=0;j<tam;j++)
fprintf(fp,"%s ",names[j].c_str());
fprintf(fp,"\n");
for(int j1=0;j1<tam;j1++){
fprintf(fp,"%s ",names[j1].c_str());
for(int j2=0;j2<tam;j2++){
if(j2<j1+1){
fprintf(fp,"- ");
continue;
}
fprintf(fp,"%f ",jaccards[j1][j2-j1-1]);
}
fprintf(fp,"\n");
}
fclose(fp);
}




void leer(char *genome,HyperLogLog *hll){
char c; 
ullint kmer=0,comp=0;
string linea;
ifstream indata(genome);
if(!indata){
printf("No se pudo abrir el archivo %s\n",genome);
exit(1);
}

hll->addSketch(genome); 

getline(indata,linea); 
getline(indata,linea); 
string::iterator it=linea.begin();

for(unsigned char j=0;j<k;++j){ 
kmer=kmer<<2;
comp=comp>>2;
c=*it;
++it;
if(c=='A') comp=comp|bits_T; 
else if(c=='C'){ 
kmer=kmer|0x1;
comp=comp|bits_G; 
}
else if(c=='G'){ 
kmer=kmer|0x2;
comp=comp|bits_C; 
}
else if(c=='T') kmer=kmer|0x3; 
}
(kmer>comp) ? hll->insert(comp) : hll->insert(kmer); 

while(!indata.eof()){
while(it!=linea.end()){
c=*it;
if(c=='A' || c=='C' || c=='G' || c=='T'){ 
kmer=(kmer<<2)&BITS; 
comp=comp>>2; 
if(c=='A') comp=comp|bits_T; 
else if(c=='C'){ 
kmer=kmer|0x1;
comp=comp|bits_G;
}
else if(c=='G'){ 
kmer=kmer|0x2;
comp=comp|bits_C;
}
else if(c=='T') kmer=kmer|0x3; 

(kmer>comp) ? hll->insert(comp) : hll->insert(kmer); 
}
else if(c=='>') break;
++it;
}
getline(indata,linea);
it=linea.begin();
}
indata.close();
}

vector<string> readFromFile(char* paths){
ifstream indata(paths);
if(!indata){
printf("No se pudo abrir el archivo %s\n",paths);
exit(1);
}
vector<string> genomes;
string filename;
while(!indata.eof()){
getline(indata,filename);
if(filename!="") genomes.push_back(filename);
}
indata.close();
return genomes;
}

vector<string> getPaths(char** argv, int argc){
vector<string> genomes;
for(int i=1;i<argc;++i){
if(!strcmp(argv[i],"-k") || !strcmp(argv[i],"-p") || !strcmp(argv[i],"-t") || !strcmp(argv[i],"-o") || !strcmp(argv[i],"-d") || !strcmp(argv[i],"-r")) ++i;
else if(strcmp(argv[i],"-s")) genomes.push_back(argv[i]);
}
return genomes;
}

vector<string> readCompressedFromFile(char* paths){
ifstream indata(paths);
if(!indata){
printf("No se pudo abrir el archivo %s\n",paths);
exit(1);
}
vector<string> genomes;
string filename;
while(!indata.eof()){
getline(indata,filename);
if(filename!="") genomes.push_back(filename);
}
indata.close();
return genomes;
}

vector<string> getCompressed(char** argv, int argc){
vector<string> genomes;
for(int i=1;i<argc;++i){
if(!strcmp(argv[i],"-k") || !strcmp(argv[i],"-p") || !strcmp(argv[i],"-t") || !strcmp(argv[i],"-o") || !strcmp(argv[i],"-f") || !strcmp(argv[i],"-r")) ++i;
else if(!strcmp(argv[i],"-d")) genomes.push_back(argv[i+1]);
}
return genomes;
}

int main(int argc, char *argv[]){
if(argc<3) {
printf("No hay suficientes argumentos\n");
exit(1);
}
unsigned char p=12;

char** option;
char** end=argv+argc;
option=std::find((char**)argv,end,(const std::string&)"-k");
if(option!=end){
char val=atoi(*(option+1));
if(val<32 && val>19) k=val;
}
option=std::find((char**)argv,end,(const std::string&)"-p");
if(option!=end){
char val=atoi(*(option+1));
if(val<16 && val>8) p=val;
}

vector<string> genomes,compressed;
option=std::find((char**)argv,end,(const std::string&)"-f");
if(option!=end) genomes=readFromFile((char*)(*(option+1)));
else genomes=getPaths(argv,argc);

option=std::find((char**)argv,end,(const std::string&)"-r");
if(option!=end) compressed=readCompressedFromFile((char*)(*(option+1)));
else compressed=getCompressed(argv,argc);

int tam=genomes.size(),tam2=compressed.size();
printf("tam: %d, tam2: %d\n",tam,tam2);
printf("k: %d p: %d\n",k,p);

int numThreads=min(tam+tam2,(int)std::thread::hardware_concurrency());

option=std::find((char**)argv,end,(const std::string&)"-t");
if(option!=end) numThreads=atoi((*(option+1)));

printf("threads: %d\n",numThreads);

vector<HyperLogLog*> v_hll;
for(int i=0;i<tam+tam2;++i){
HyperLogLog *hll;
hll = new HyperLogLog(p,32-p,k);
v_hll.push_back(hll);
}

const ullint desp=(2*(k-1));
bits_G=(ullint)2<<desp;
bits_T=(ullint)3<<desp;
bits_C=(ullint)1<<desp;
BITS=(bits_C-1)<<2;

omp_set_num_threads(numThreads);
#pragma omp parallel
{
#pragma omp single
{
for(int i=0;i<tam;i++){
#pragma omp task
leer((char*)genomes[i].c_str(),v_hll[i]);
}
for(int i=0;i<tam2;i++){
#pragma omp task
v_hll[i+tam]->loadSketch((char*)compressed[i].c_str());
}
}
}

option=std::find((char**)argv,end,(const std::string&)"-s");
if(option!=end){
for(int i=0;i<tam+tam2;++i)
v_hll[i]->saveSketch();
}

vector<string> names(tam+tam2);
vector<float> cards(tam+tam2);
#pragma omp parallel for
for(int i=0;i<tam+tam2;i++){
names[i]=v_hll[i]->getName();
cards[i]=v_hll[i]->estCard();
}
vector<vector<float>> jaccards=estJaccard(v_hll,cards,32-p,1<<p,numThreads);
printMatrix(jaccards,names);

for(int i=0;i<tam+tam2;++i)
delete v_hll[i];

return 0;
}
