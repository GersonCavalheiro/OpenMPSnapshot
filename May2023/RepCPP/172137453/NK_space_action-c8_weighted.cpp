#ifdef _OPENMP
# include <omp.h>
#endif

#include <iostream>

using std::cout;
using std::endl;
using std::to_string;
using std::string;

#include <fstream>
#include <utility>
#include <set>
#include <vector>  

using std::vector;
#include <iomanip>
#include <bitset> 
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>
#include "Agents.h"
int na = pow(2, 20);
const int nbgha = 100;
const int agentcounta = 100;
int matrixa[100][100]={};

void open_space_scores(int j, vector<double> &input_vec) {
std::ifstream in_scores;
double i_score;
in_scores.open("NK_space_scores_" + to_string(j) + ".txt",std::ios::in | std::ios::binary);
for (int i = 0; i < ::na; ++i) {
in_scores >> i_score;
input_vec[i] = i_score;
}
in_scores.close();
}

void open_space_string(vector <string> &string_vec) {
std::fstream strings;
strings.open("NK_space_strings.txt");
for (int i = 0; i < ::na; ++i) {
string i_str;
strings >> i_str;
string_vec[i] = i_str;
}
strings.close();
}

void A_list(vector<char> &types){
std::vector<char> sp(100);
std::fill(sp.begin(),sp.end(),'A');
for (unsigned int i = 0; i < sp.size(); ++i)
{
types[i]=sp[i];
}   
}
void AB_list(vector<char> &types){
std::vector<char> sp={'A','A','B','B','B','A','A','B','A','B','B','A','A','A','B','B','B','B','A','A','B','B','A','B','B','A','A','B','B','B','B','B','B','B','A','A','A','B','B','B','A','A','A','B','B','B','A','A','B','B','B','B','A','A','B','B','A','B','B','A','B','A','B','A','A','B','B','A','B','A','B','A','A','B','A','B','B','A','A','B','B','A','B','B','A','B','A','B','B','B','B','A','A','A','A','B','B','A','A','B'};

for (unsigned int i = 0; i < sp.size(); ++i)
{
types[i]=sp[i];
}   
}

void AB_random_list(vector<char> &types, int inksp){
int ac=0;
int bc=0;
int temp=0;
for (int i = 0; i < 100; ++i)
{
std::bernoulli_distribution d(0.5);
std::mt19937 gen;
gen.seed(inksp+i);

temp=d(gen);
if(temp==1) {types[i]='A'; ac++;}
if(temp==0) {types[i]='B'; bc++;}
d.reset();
}
}


void ABCDE_list(vector<char> &types){
vector<char> AB(::agentcounta);
for (int i = 0; i < ::agentcounta; i++) {
if (i%5==0) AB[i] = 'A';
if (i%5==1) AB[i] = 'B';
if (i%5==2) AB[i] = 'C';
if (i%5==3) AB[i] = 'D';
if (i%5==4) AB[i] = 'E';
}
int acount=0;
int bcount=0;
int ccount=0;
int dcount=0;
int ecount=0;
int temp = 0;
for (int i = 0; i < ::agentcounta; ++i) {
temp = rand() % (::agentcounta - i);
types[i] = AB[temp];
AB.erase(AB.begin() + temp);
if(types[i]=='A') acount++;
if(types[i]=='B') bcount++;
if(types[i]=='C') ccount++;
if(types[i]=='D') dcount++;
if(types[i]=='E') ecount++;

}
}
void AB_list_deseg(vector<char> &types){
vector<char> AB(::agentcounta);
for (int i = 0; i < ::agentcounta; ++i) {
if (i%2==0) types[i] = 'A';
if (i%2==1) types[i] = 'B';
}
}

void AB_list_seg(vector<char> &types){
vector<char> AB(::agentcounta);
for (int i = 0; i < ::agentcounta; ++i) {
if (i < 50) types[i] = 'A';
if (i >= 50) types[i] = 'B';
}
}

void output_connections(int NKspace,vector<Agent> &Agents,int rounds){ 
std::ofstream out;
out.open("agent connections "+to_string(NKspace)+" "+to_string(rounds)+".txt");
out<<"Agent id #"<<","<<"species"<<","<<"connection 0"<<","<<"connection 1"<<","<<"connection 2"<<","<<"connection 3"<<","<<"connection 4"<<","<<"connection 5"<<","<<"connection 6"<<","<<"connection 7"<<","<<"minority status"<<"\n";
for (vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); i++)
{
out<<i->id<<","<<i->species<<","<<i->connections[0]<<","<<i->connections[1]<<","<<i->connections[2]<<","<<i->connections[3]<<","<<i->connections[4]<<","<<i->connections[5]<<","<<i->connections[6]<<","<<i->connections[7];
if(i->minority==1) 
{out<<","<<"M"<<"\n";} 
else
{out<<"\n";}
}
}

void output_scores(int NKspace,vector<Agent> &Agents,int rounds){ 
std::ofstream out;
out.open("agent scores round "+to_string(NKspace)+" "+to_string(rounds)+".txt");
out<<"Agent id #"<<","<<"scores\n";
for (vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); i++)
{
out<<std::setprecision(15)<<i->id<<","<<i->score<<"\n";
}
}

void output_round(int NKspace,int round,vector<int> rounds,vector<double> scr,vector<double> ag,vector<int> mc,vector<int> us,vector<double> pu,int search,int typeout,vector<double> same,vector<double> diff,char method){
string outtype="";
string outmethod;
if(typeout==-9)  outtype ="baseline";
if(typeout==-1)  outtype="minority";
if(typeout==1)  outtype ="majority";
if(typeout==0)  outtype ="Swap";
if(method=='m') outmethod="matrix";
if(method=='c') outmethod="connections";
if(method=='s') outmethod="info_swap";
if(method=='b') outmethod="basline";
if(method=='z') {outtype="morph"; outmethod="morph";}
std::fstream out("8c-NK_space_weighted_"+to_string(NKspace)+"_"+outtype+"_SH_"+to_string(search)+"_"+outmethod+".txt",std::ios::out | std::ios::binary);
out<<"round,"<<"max score,"<<"avg score,"<<"Number of unique solutions,"<<"percent with max score,"<<"avg similiar species,"<<"avg different species,"<<"minority count"<<"\n";
for (int i=0;i<=round; i++){
out<<std::setprecision(15)<<rounds[i]<<","<<scr[i]<<","<<ag[i]<<","<<us[i]<<","<<pu[i]<<","<<same[i]<<","<<diff[i]<<","<<mc[i]<<"\n";
}
}
void swapper(vector<Agent> &v,vector<int> &a,vector<int> &b){ 
for(unsigned int i=0;i<a.size();i++){
v[b[i]].species=v[a[i]].species;
v[b[i]].tempstring=v[a[i]].binarystring;
v[b[i]].tempscore=v[a[i]].score;
}
}
void swap_agents(vector<Agent> &v,int mode){
vector<int> minor;
vector<int> permute;
for (int i = 0; i < 100; ++i)
{
if(mode==1)
{
if(v[i].minority==1)
{
minor.push_back(v[i].id);
v[i].minority=0; 
}
}
if(mode==-1)
{
if(v[i].minority!=1)
{
minor.push_back(v[i].id);
v[i].minority=0; 
}
}
}
permute=minor;
std::random_shuffle ( permute.begin(), permute.end() );
swapper(v,minor,permute);
for(unsigned int i=0;i<minor.size();++i)
{
v[permute[i]].Agent::agent_swap_interal_info(v[permute[i]]);
}
}


int main(int argc, char *argv[]) {
std::ios::sync_with_stdio(false); 
std::stringstream convert1(argv[1]);
std::stringstream convert2(argv[2]);
std::string convert3(argv[3]);
int start;
convert1 >> start;
int end;
convert2 >> end;
int searchm;
double prob=0.0;
int condition;
int mode;
char method='z';
if(convert3=="-S" && atoi(argv[4])<=3){
searchm=atoi(argv[4]);
}
else{
searchm=0;
}

if(argc>=6 && atof(argv[5])>0)
{
if(atof(argv[5])>=0.0) 
{
prob=atof(argv[5]);
condition=0;
cout<<"using "<<prob<<endl;
} 
else
{
prob=0.05; 
condition=0;
cout<<"using default probability of 0.05 for species switch\n";
}
}
else
{
cout<<"skipping species swap \n";
prob=-1;
if(argc==8)
{
mode=atoi(argv[7]);
condition=atoi(argv[6]);
}
}
int NKspace_num=start;
vector <Agent> agent_array(::agentcounta);
cout.setf(std::ios::fixed);
cout.setf(std::ios::showpoint);
cout.precision(15);
vector <string> NKspacevals(::na);
vector<double> NKspacescore(::na);
open_space_string(NKspacevals);

double max = 0.0;
double avgscore = 0.0;
int percuni=0;
vector<double> maxscore(100);
vector<int> maxround(100);
vector<int> minoritycount(100);
vector<double> avgscores(100);
vector<char> type(100);
vector<int> uniquesize(100);
vector<double> percentuni(100);
vector<double> elts(100);
vector<double> same(100);
vector<double> diff(100);

int rounds = 0;
int nums = 0;
int mcount=0;

#pragma omp parallel for default(none) shared(end,searchm,::na,NKspacevals,NKspace_num,cout,prob,argc,condition,mode,method) firstprivate(max,avgscore,percuni,maxscore,maxround,minoritycount,avgscores,uniquesize,percentuni,NKspacescore,rounds,mcount,agent_array,type,nums,elts,same,diff)   schedule(dynamic,25)
for(int inksp=NKspace_num;inksp<end;++inksp){
cout<<"NK_space #:"<<inksp<<endl;
#ifdef _OPENMP
cout<<"\t thread #: "<<omp_get_thread_num()<<endl;
#endif
if(inksp>0)
{
agent_array.clear();
NKspacescore.clear();
maxscore.clear();
maxround.clear();
minoritycount.clear();
avgscores.clear();
type.clear();
uniquesize.clear();
percentuni.clear();
agent_array.resize(100);
NKspacescore.resize(100);
elts.clear();
elts.resize(100);
maxscore.resize(100);
maxround.resize(100);
minoritycount.resize(100);
avgscores.resize(100);
type.resize(100);
uniquesize.resize(100);
percentuni.resize(100);
same.clear();
diff.clear();
same.resize(100);
diff.resize(100);
percuni=0;
rounds = 0;
nums = 0;
mcount=0;
max=0.0;
}
AB_list_seg(type);
open_space_scores(inksp, NKspacescore);
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
std::uniform_int_distribution<> ids(0,::na); 
std::random_device rdm;
std::mt19937 gen(rdm());
int rnums=ids(gen);
i->Agent::agent_connections(nums, *i);
i->species = type[nums];
i->Agent::agent_change(rnums, *i, NKspacevals, NKspacescore);

nums++;
}

for (; rounds < 100; ++rounds) {
elts.clear();
elts.resize(100);

if(rounds<=0){
for (int i=0; i<::agentcounta;++i)
{
elts[i]=agent_array[i].score;
}
std::sort(elts.begin(),elts.end());
auto ip=std::unique(elts.begin(),elts.end());
elts.erase(ip,elts.end());
uniquesize[rounds]=elts.size();

max=0;
for (int i = 0; i < ::agentcounta; ++i) 
{
if (max < agent_array[i].score) 
{
max = agent_array[i].score;
}

}
percuni=0;

for (int i = 0; i < 100; ++i)
{
if(max==agent_array[i].score)
{
percuni++;
}

}
percentuni[rounds]=(percuni);
}


mcount = 0;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
i->Agent::agent_minority_status(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]]);
if (i->minority == 1) 
{
mcount++;
}
i->minority=0;
}


minoritycount[rounds] = mcount;

for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
{
if(mode==1)
{
i->Agent::agent_exploit_weighted(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]]);
}
if(mode==-1)
{
i->Agent::agent_exploit_weighted_inverse(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]]);
}
}


for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) {
i->Agent::agent_swap_interal_info(*i);
}


for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
if (i->flag == -1) 
{
i->Agent::agent_explore(*i, NKspacescore,searchm);
}
}
if(condition==-9)
{
method='b';
}
else
{
if(prob==-1)
{ 
if(condition==0)
{
swap_agents(agent_array,mode);
method='s';
}
int pick=rand() %8;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
if (condition==1)
{

if (i->minority == 1) 
{

if(mode==1)
{

method='m';
i->Agent::agent_swap_con(agent_array,*i,
agent_array[i->connections[0]],agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]],pick);
}
if(mode==-1)
{

method='c';
i->Agent::agent_minority_swap(i->id,*i,
agent_array[i->connections[0]],agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]]);
}
}
}
if (condition==-1)
{

if (i->minority != 1) 
{

if(mode==1)
{

method='m';
i->Agent::agent_swap_con(agent_array,*i,
agent_array[i->connections[0]],agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]],pick);
}
if(mode==-1)
{

method='c';
i->Agent::agent_minority_swap(i->id,*i,
agent_array[i->connections[0]],agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]],
agent_array[i->connections[4]], agent_array[i->connections[5]],
agent_array[i->connections[6]], agent_array[i->connections[7]]);
}
}
}
}
}
}
if(argc>=6 && prob!=-1)
{
}



double samespecies=0;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
{
for (uint j = 0; j < i->connections.size(); ++j)
{
if(i->species==agent_array[i->connections[j]].species) samespecies++;
}   
}
same[rounds]=(samespecies/100);
diff[rounds]=8-(samespecies/100);
if(rounds>0)
{
for (int i=0; i<::agentcounta;++i)
{
elts[i]=agent_array[i].score;
}
std::sort(elts.begin(),elts.end());
auto ip=std::unique(elts.begin(),elts.end());
elts.erase(ip,elts.end());
uniquesize[rounds]=elts.size();

max=0;
for (int i = 0; i < ::agentcounta; ++i) 
{
if (max < agent_array[i].score) 
{
max = agent_array[i].score;
}

}
percuni=0;

for (int i = 0; i < 100; ++i)
{
if(max==agent_array[i].score)
{
percuni++;
}

}
percentuni[rounds]=(percuni);
}
maxscore[rounds] = max;
maxround[rounds] = rounds;
avgscore = 0.0;

for (int i = 0; i < ::agentcounta; ++i) 
{
avgscore += agent_array[i].score;
}

avgscores[rounds] = (avgscore / (::agentcounta));
int eqflag=0;

if((uniquesize[rounds]==1) || rounds>=99)
{
eqflag=1;
}

if(eqflag==1){break;}

}

output_round(inksp,rounds,maxround,maxscore,avgscores,minoritycount,uniquesize,percentuni,searchm,condition,same,diff,method);
}
return 0;
}
