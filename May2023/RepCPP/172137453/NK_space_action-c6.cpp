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
#include <algorithm>
#include <random> 
#include <chrono>
#include <unistd.h>
#include <filesystem>

#include "Agents6.h"
int na = (1<<15);
const int agentcounta = 100;
int matrixa[agentcounta][agentcounta]={};

void open_space_scores(int j, vector<double> &input_vec,int k) {
std::ifstream in_scores;
double i_score;
in_scores.open("4_NK_space_scores_"+to_string(k)+"_"+to_string(j)+"_cpp.txt",std::ios::in | std::ios::binary);
for (int i = 0; i < ::na; ++i) {
in_scores >> i_score;
input_vec[i] = i_score;
}
in_scores.close();
}

void open_space_scores_R(int j, vector<double> &input_vec,int k) {
std::ifstream in_scores;
double i_score;
in_scores.open("4_NK_space_scores_"+to_string(k)+"_"+to_string(j)+"_R.txt",std::ios::in | std::ios::binary);
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
std::vector<char> sp(::agentcounta);
std::fill(sp.begin(),sp.end(),'A');
for (unsigned int i = 0; i < types.size(); ++i)
{
types[i]=sp[i];
}   
}
void AB_list(vector<char> &types){
std::vector<char> sp={'A','A','B','B','B','A','A','B','A','B','B','A','A','A','B','B','B','B','A','A','B','B','A','B','B','A','A','B','B','B','B','B','B','B','A','A','A','B','B','B','A','A','A','B','B','B','A','A','B','B','B','B','A','A','B','B','A','B','B','A','B','A','B','A','A','B','B','A','B','A','B','A','A','B','A','B','B','A','A','B','B','A','B','B','A','B','A','B','B','B','B','A','A','A','A','B','B','A','A','B'};

for (unsigned int i = 0; i < types.size(); ++i)
{
types[i]=sp[i];
}   
}

void AB_part_25_list(vector<char> &types){
std::vector<char> sp={'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'};

for (unsigned int i = 0; i < types.size(); ++i)
{
types[i]=sp[i];
}   
}

void AB_part_5_list(vector<char> &types){
std::vector<char> sp={'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'};
for (unsigned int i = 0; i < types.size(); ++i)
{
types[i]=sp[i];
}   
}

void AB_random_list(vector<char> &types, int inksp){
int temp=0;
for (int i = 0; i < ::agentcounta; ++i)
{
std::bernoulli_distribution d(0.5);
std::mt19937 gen;
gen.seed(inksp+i);

temp=d(gen);
if(temp==1) {types[i]='A';}
if(temp==0) {types[i]='B';}
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
for (int i = 0; i < ::agentcounta; ++i) {
if (i < (::agentcounta)/2) types[i] = 'A';
if (i >= (::agentcounta)/2) types[i] = 'B';
}
}

void AB_list_permute(vector<char> &types,int inksp){
std::mt19937 gen;
gen.seed(inksp+1);
for (int i = 0; i < ::agentcounta; ++i) {
if (i < (::agentcounta/2)) types[i] = 'A';
if (i >= (::agentcounta/2)) types[i] = 'B';
}
std::shuffle ( types.begin(), types.end(),gen);
}

void agent_connections_replacement_number(vector<Agent> &Agents,int inksp){
std::uniform_int_distribution<> replace(1, 5);
std::mt19937 gen;
gen.seed(inksp+1);
for (std::vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); ++i)
{
i->connection_replace=replace(gen);
replace.reset();
}
}

void output_matrix_connections(int NKspace,vector<Agent> Agents,int rounds)
{
std::ofstream out;
std::ostringstream str;
str << std::setw(3) << std::setfill('0') << NKspace;
out.open("matrix_connections_"+str.str()+"_"+to_string(rounds)+".txt");
for (int i = 0; i < agentcounta; ++i)
{
if(i==0)out<<",";
out<<"cted "<<i;
if(i<=98) out<<",";
if(i==99) out<<"\n";
}
for (int i = 0; i < agentcounta; ++i)
{
out<<i<<",";
for (int j = 0; j < agentcounta; ++j)
{

for (int k = 0; k < 6; ++k)
{
if(Agents[i].connections[k]==j) out<<j;

}
out<<",";
if(j==agentcounta)out<<"\n";
}
}
}
void output_connections(int NKspace,vector<Agent> Agents,int rounds,int k,char method, int typeout , int Criterion,char vec_method,std::filesystem::path path,int global,char special){ 
string outtype="";
string outmethod="asymmetric";
string submethod="";
string globalstring="";
if(method=='c') outmethod="asymmetric";
if(method=='m') outmethod="symmetric";
if(method=='s') outmethod="info_swap";
if(method=='b') outmethod="basline";
if(special=='W') {outmethod="weighted";}
if(special=='B') {outmethod="bitflip";}
if(outmethod=="asymmetric")
{
if(Criterion==4)submethod="-4";
if(Criterion==5)submethod="-5";
if(Criterion==6)submethod="-6";
}
if(typeout==-9)  outtype ="baseline";
if(outtype=="baseline") outtype.append("-"+to_string(Criterion));
if(typeout==-1)  outtype="minority";
if(typeout==1)  outtype ="majority";
if(typeout==0)  outtype ="Swap";
if(typeout==100) outtype="morph_exp";
if(typeout==-100) outtype="morph_raw";
if(vec_method=='M') outtype+="merit";
if(global==1) globalstring="-global";
else globalstring="-local";
string edges="edge-list";
edges=edges.append(globalstring);

std::fstream out(path.string()+"/Network_"+outmethod+submethod+"_"+outtype+".txt",std::ios::binary|std::ios::out|std::ios::app);
out.seekg(0, std::ios::end);
if (out.tellg() == 0) out<<"Agent id #"<<","<<"species"<<","<<"connection 0"<<","<<"connection 1"<<","<<"connection 2"<<","<<"connection 3"<<","<<"connection 4"<<","<<"connection 5"<<","<<"NK#"<<","<<"K#"<<","<<"Round#"<<"\n";
for (vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); i++)
{
out<<i->id<<","<<i->species<<","<<i->connections[0]<<","<<i->connections[1]<<","<<i->connections[2]<<","<<i->connections[3]<<","<<i->connections[4]<<","<<i->connections[5]<<","<<NKspace<<","<<k<<","<<rounds<<endl;
out.flush();
}
}

void output_scores(int NKspace,vector<Agent> Agents,int rounds){ 
std::ofstream out;
out.open("agent scores round "+to_string(NKspace)+" "+to_string(rounds)+".txt");
out<<"Agent id #"<<","<<"scores\n";
for (vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); i++)
{
out<<std::setprecision(15)<<i->id<<","<<i->score<<"\n";
}
}

void output_round(int NKspace,int round,vector<int> rounds,vector<double> scr,vector<double> ag,vector<double> mc,vector<int> us,vector<double> pu,int search,int typeout,vector<double> same,vector<double> diff,char method,int k_val,int Criterion,char vec_method,std::filesystem::path path,int global, char special, vector<int> count_swapped){

string outtype="";
string outmethod="asymmetric";
string submethod="";
string globalstring="";
if(method=='c') outmethod="asymmetric";
if(method=='m') outmethod="symmetric";
if(method=='s') outmethod="info_swap";
if(method=='b') outmethod="basline";
if(special=='W') {outmethod="weighted";}
if(special=='B') {outmethod="bitflip";}
if(outmethod=="asymmetric")
{
if(Criterion==4)submethod="-4";
if(Criterion==5)submethod="-5";
if(Criterion==6)submethod="-6";
}

if(typeout==-9)  outtype ="baseline";
if(outtype=="baseline") outtype.append("-"+to_string(Criterion));
if(typeout==-1)  outtype="minority";
if(typeout==1)  outtype ="majority";
if(typeout==0)  outtype ="Swap";
if(typeout==100) outtype="morph_exp";
if(typeout==-100) outtype="morph_raw";
if(vec_method=='M') outtype+="merit";
if(global==1) globalstring="-global";
else globalstring="-local";
std::fstream out(path.string()+"/6c-NK_space_"+to_string(k_val)+'_'+to_string(NKspace)+"_"+outtype+"_SH_"+to_string(search)+"_"+outmethod+submethod+globalstring+".txt",std::ios::out | std::ios::binary);
out<<"round,"<<"max score,"<<"avg score,"<<"Number of unique solutions,"<<"percent with max score,"<<"avg similiar species,"<<"avg different species,"<<"minority count,"<<"tie swapped"<<"\n";
for (int i=0;i<=round; i++){
out<<std::setprecision(15)<<rounds[i]<<","<<scr[i]<<","<<ag[i]<<","<<us[i]<<","<<pu[i]<<","<<same[i]<<","<<diff[i]<<","<<mc[i]<<","<<count_swapped[i]<<"\n";
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
for (int i = 0; i < ::agentcounta ; ++i)
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
int opt;
int start=-1;
int end=-1;
int searchm=-1;
double prob=-1;
int condition=-10;
int mode=-10;
int Criterion=-1;
char method='q';
int score_vec=0;
int globaltest=-1;
char special='N';
while((opt = getopt(argc, argv, "s:e:S:c:m:C:M:P:G:U:")) != -1)  
{  
switch(opt)  
{  
case 's': start=atoi(optarg); break; 
case 'e':  end=atoi(optarg); break; 
case 'S':  
searchm=atoi(optarg); 
break;     
case 'c':  
condition=atoi(optarg); 
break;  
case 'm':
mode=atoi(optarg); 
break;
case 'C':  
Criterion=atoi(optarg); 
break;  
case 'M': method=*optarg; break; 
case 'P': prob=atof(optarg); break;
case 'G': if(atoi(optarg)==1) globaltest=1;; break; 
case 'U': special=*optarg; break;
}  
}
if(score_vec==1 && mode==1){
cout<<"Conflict between Vector list and mode\n";
return 0;
}
if(start<=-1){
cout<<" start value must be 0 or greater\n";
return 0;
}
if(end==-1){
cout<<" end value must be given\n";
return 0;
}    
if(end<start)
{
cout<<" end value must be greater than start\n";
return 0;
}  
if(searchm==-1){
cout<<"Search heuristic required i.e 0,1,2 or 3\n";
return 0; 
}
if(Criterion==-1||Criterion<4||Criterion>6){
if(condition!=9){
cout<<"Valid Criterion values are 4,5,6 using default value of 4\n";
Criterion=4;
}
else Criterion=4;
}
if(method=='z'){
cout<<"Morhping will be used\nIf not desired, pass -M q\n";
if(prob==-1.0) {
cout<<"using exponetial probability function in Morhping\n Pass # 0<=P<=1 for raw probability\n";
condition=100;
}
else condition=-100;
}
else{
cout<<"Morhping will not be used\n";
cout<<condition;
if(abs(condition)!=1){
if(condition==-9){
cout<<"using baseline\n";
}
else if(condition==9){
score_vec=1;
cout<<"using merit based connection swapping\n";
}
else if(condition==0) cout<<"using swapping function\n";
else{
cout<<"Condition for seeking must be either -1 or 1 or 9 or -9 ;\nMajority seeking is 1 and Minority seeking is -1,Merit is 9 baseline is -9\n";
return 0; 
}
}
if(abs(mode)!=1 && condition!=-9 && mode!=0){
cout<<"mode must be either -1 or 1;\nMatrix is 1 and connections is -1\n";
return 0; 
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
vector<double> minoritycount(100);
vector<double> avgscores(100);
vector<char> type(::agentcounta);
vector<int> uniquesize(100);
vector<double> percentuni(100);
vector<double> elts(100);
vector<double> same(100);
vector<double> diff(100);
double samespecies=0;
vector<int> tieswapped(100);
int counter_tieswapping = 0;

int rounds = 0;
int nums = 0;
double mcount=0;
vector<int> k_opts={1,3,5,10};
#pragma omp parallel for default(none) shared(end,searchm,::na,NKspacevals,NKspace_num,cout,prob,argc,condition,mode,method,::matrixa,k_opts,std::cin,::agentcounta,Criterion,score_vec,globaltest,special) firstprivate(max,avgscore,percuni,maxscore,maxround,minoritycount,avgscores,uniquesize,percentuni,NKspacescore,rounds,mcount,agent_array,type,nums,elts,same,diff,samespecies,tieswapped,counter_tieswapping) schedule(static,end/24)
for(int inksp=NKspace_num;inksp<end;++inksp){

for (int opts = 0; opts < k_opts.size(); ++opts)
{
char merit_method='0';
if(score_vec==1) merit_method='M';    
string outtype="";
string outmethod="";
string submethod="";
string globalstring="";
string edges="edge-list";
if(mode==-1) outmethod="asymmetric";
if(mode==1) outmethod="symmetric";
if(outmethod=="asymmetric")
{
if(Criterion==4)submethod="-4";
if(Criterion==5)submethod="-5";
if(Criterion==6)submethod="-6";
}
if(outmethod=="symmetric")
{
if(Criterion==4)submethod="-4";
if(Criterion==5)submethod="-5";
if(Criterion==6)submethod="-6";
}
if(merit_method=='M') outmethod+="_merit";
if(globaltest==1) globalstring="-global";
else globalstring="-local";
edges.append(globalstring);
if(condition==-9) outmethod="baseline";
if(special=='W')outmethod+=("_weighted_"+to_string(prob));
if(special=='B')outmethod+=("_bitflip"+to_string(prob));   
std::filesystem::path path=std::filesystem::current_path();
std::filesystem::path edgesfolder=std::filesystem::current_path();
std::filesystem::path combine;
edgesfolder/=edges;
path/=outmethod+submethod+globalstring;
combine=path.string()+"-"+edges;

if(!std::filesystem::exists(path)) std::filesystem::create_directory(path);
if(!std::filesystem::exists(combine)) std::filesystem::create_directory(combine);
cout<<"NK_space #:"<<inksp<<endl;
#ifdef _OPENMP
#endif

std::mt19937 genAs(inksp+1);
std::uniform_int_distribution<> Asearch(0,3);
std::mt19937 genBs(2*inksp+1);
std::uniform_int_distribution<> Bsearch(0,3);
int A_sel= Asearch(genAs);
int B_sel= Bsearch(genBs);
while(B_sel==A_sel){
B_sel= Bsearch(genBs);
Bsearch.reset();
}
Asearch.reset();
Bsearch.reset();
std::mt19937 A(inksp+1);
std::mt19937 B(2*inksp+1);

std::uniform_int_distribution<> Astart(2,7);
int startindex=Astart(A);
Astart.reset();
std::uniform_int_distribution<> Aend(startindex+1,13);
int endindex=Aend(A);
Aend.reset();
std::uniform_int_distribution<> Aselection(startindex,endindex);
int choice=Aselection(A);
Aselection.reset();
std::vector<int> Aseq;
std::vector<int> Bseq;
std::vector<int> options = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};

std::sample(options.begin()+2,options.begin()+13,std::back_inserter(Aseq),choice,A);

std::set_difference(options.begin(),options.end(),Aseq.begin(),Aseq.end(),std::inserter(Bseq,Bseq.end()));

if(inksp>0||opts>0)
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
agent_array.resize(::agentcounta);
NKspacescore.resize(::na);
elts.clear();
elts.resize(::agentcounta);
maxscore.resize(100);
maxround.resize(100);
minoritycount.resize(100);
avgscores.resize(100);
type.resize(::agentcounta);
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
samespecies=0;
tieswapped.clear();
tieswapped.resize(100);

}
AB_list_permute(type,inksp);
open_space_scores(inksp, NKspacescore,k_opts[opts]);
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
unsigned seed = static_cast<int> (std::chrono::system_clock::now().time_since_epoch().count());
std::random_device rdm;
std::mt19937 gen(seed);
std::uniform_int_distribution<> ids(0,(::na));

int rnums=ids(gen);
i->Agent::agent_connections(nums, *i);
i->species = type[nums];
i->Agent::agent_change(rnums, *i, NKspacevals, NKspacescore);
i->connection_replace=Criterion;
i->move_avg=i->score;
nums++;
ids.reset();
if(nums==(::agentcounta)) break;
}


for (; rounds < 100; ++rounds) {
elts.clear();
elts.resize(::agentcounta);

if(rounds==0){
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

for (int i = 0; i < ::agentcounta; ++i)
{
if(max==agent_array[i].score)
{
percuni++;
}

}
percentuni[rounds]=(percuni);

samespecies=0;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
{
for (uint j = 0; j < i->connections.size(); ++j)
{
if(i->species==agent_array[i->connections[j]].species) samespecies++;
}   
}
same[rounds]=(samespecies/::agentcounta);
diff[rounds]=6-(samespecies/::agentcounta);


mcount = 0;

for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
if(mode==1 && condition!=9){

i->Agent::agent_minority_status_symmetric(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],condition);
if (i->minority == 1) 
{
mcount++;
}
}
if(mode==-1 && condition!=9){
i->Agent::agent_minority_status_asymmetric(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion,condition);
if (i->minority == 1) 
{
mcount++;
}

}
if(condition==9){
i->Agent::agent_minority_status_merit(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion);
if (i->minority == 1) 
{
mcount++;
}
}

}


for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
i->minority=0;
}

max=0;
for (int i = 0; i < ::agentcounta; ++i) 
{
if (max < agent_array[i].score) 
{
max = agent_array[i].score;
}

}
percuni=0;

for (int i = 0; i < ::agentcounta; ++i)
{
if(max==agent_array[i].score)
{
percuni++;
}

}
percentuni[rounds]=(percuni);

maxscore[rounds] = max;
maxround[rounds] = rounds;
avgscore = 0.0;

for (int i = 0; i < ::agentcounta; ++i) 
{
avgscore += agent_array[i].score;
}

avgscores[rounds] = (avgscore / (::agentcounta));

minoritycount[rounds] = mcount;
}

if(rounds==0) continue;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
{
if(special=='N')
{
if(condition==9){
i->Agent::agent_exploit(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],prob,method,condition,Criterion,condition,NKspacescore);
}
if(condition!=9){
i->Agent::agent_exploit(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],prob,method,mode,Criterion,condition,NKspacescore);
}
}
if (special=='W')
{
i->Agent::agent_exploit_weighted_inverse(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],mode,Criterion,prob,condition);
}    


}


for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) {
if (i->flag == 1)
i->Agent::agent_swap_interal_info(*i);
}
std::vector<int> seq;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
if(i->species=='A') {seq=Aseq;} 
if(i->species=='B') {seq=Bseq;}
if (i->flag == -1) 
{
i->Agent::agent_explore_new(*i, NKspacescore,A_sel,B_sel,seq);
}
}
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i){
i->move_avg=(i->move_avg+(i->score-i->move_avg)/(rounds+1));
}


if(condition==-9 && method!='z')
{
method='b';

}
else
{

if(condition==0)
{
swap_agents(agent_array,mode);
method='s';
}
else
{
counter_tieswapping = 0;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{

if (condition==1)

{    vector<int> list;
list=(i->Agent::agent_connections_species_match(agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],condition));

if (i->minority == 1) 
{
++counter_tieswapping;
if(mode==1)
{
method='m';
i->Agent::agent_swap_hack_symmetric(agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],list);
}
if(mode==-1)
{

if(globaltest==1)
{

i->Agent::agent_swap_hack_asymmetric_global(i->id,agent_array,*i,Criterion);
}  
if(globaltest!=1)
{ 

method='c';
i->Agent::agent_swap_hack_asymmetric(i->id,agent_array,*i,
agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion,list);
}
}
}
}
if (condition==-1)
{
vector<int> list;
list=(i->Agent::agent_connections_species_match(agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],condition));

if (i->minority == 1)
{
++counter_tieswapping;
if(mode==1)
{

method='m';
i->Agent::agent_swap_hack_symmetric(agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],list);
}
if(mode==-1)
{
if(globaltest==1)
{
i->Agent::agent_swap_hack_asymmetric_global(i->id,agent_array,*i,Criterion);
}
if(globaltest!=1)
{
method='c';
i->Agent::agent_swap_hack_asymmetric(i->id,agent_array,*i,
agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion,list);
}
}
}
}
if(condition==9)
{
vector<int> list;
list=(i->Agent::agent_connections_score_match(agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]]));
if(i->minority==1)
{
++counter_tieswapping;

if(mode==1)
{
method='m';
i->Agent::agent_swap_hack_symmetric(agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],list);
}
if(mode==-1)
{
if(globaltest==1)
{
i->Agent::agent_swap_hack_asymmetric_global(i->id,agent_array,*i,Criterion);
}
if(globaltest!=1)
{
method='c';
i->Agent::agent_swap_hack_asymmetric(i->id,agent_array,*i,agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion,list);
}
}
}   
}

tieswapped[rounds]=counter_tieswapping;    
}
}

}





if(rounds>0)
{
mcount = 0;

for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
if(mode==1 && condition!=9){

i->Agent::agent_minority_status_symmetric(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],condition);
if (i->minority == 1) 
{
mcount++;
}
}
if(mode==-1 && condition!=9){
i->Agent::agent_minority_status_asymmetric(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion,condition);
if (i->minority == 1) 
{
mcount++;
}

}
if(condition==9){
i->Agent::agent_minority_status_merit(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
agent_array[i->connections[2]], agent_array[i->connections[3]], agent_array[i->connections[4]],agent_array[i->connections[5]],Criterion);
if (i->minority == 1) 
{
mcount++;
}
}

}
minoritycount[rounds] = mcount;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
{
i->minority=0;
}
samespecies=0;
for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
{
for (uint j = 0; j < i->connections.size(); ++j)
{
if(i->species==agent_array[i->connections[j]].species) samespecies++;
}   
}
same[rounds]=(samespecies/::agentcounta);
diff[rounds]=6-(samespecies/::agentcounta);
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

for (int i = 0; i < ::agentcounta; ++i)
{
if(max==agent_array[i].score)
{
percuni++;
}

}
percentuni[rounds]=(percuni);

maxscore[rounds] = max;
maxround[rounds] = rounds;
avgscore = 0.0;

for (int i = 0; i < ::agentcounta; ++i) 
{
avgscore=avgscore + (agent_array[i].score-avgscore)/(i+1);
}
avgscores[rounds] = (avgscore);
}
int eqflag=0;


output_connections(inksp,agent_array,rounds,k_opts[opts],method,condition,Criterion,merit_method,combine,globaltest,special);

if((uniquesize[rounds]==1) || rounds>=70)
{
eqflag=1;
}

if(eqflag==1){break;}

}


output_round(inksp,rounds,maxround,maxscore,avgscores,minoritycount,uniquesize,percentuni,searchm,condition,same,diff,method,k_opts[opts],Criterion,merit_method,path,globaltest,special,tieswapped);
}
}
return 0;
}
