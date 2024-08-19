#include <iostream>
using std::cout;
using std::endl; 
using std::to_string;
using std::string;
#include <vector>  
using std::vector;
#include <bitset> 
#include <fstream> 
using std::ofstream;
#include <cmath> 
#include <random> 
#include <algorithm>
#include <ios>
#include <iomanip>
#include <cstdlib>

double num=240;
double sigma=sqrt(1/num);
std::normal_distribution<double> uni(0.4,.09);
std::random_device rd;
std::mt19937 norm(rd());
void NKspacevals_gen(vector<double>& input_vec,int n){

for (int i=0;i<n;++i)
{
input_vec[i]=uni(norm);
uni.reset();
}
};
void NKspacevals_lowk(vector<double>& input_vec,int k_val){
int positions=(20-k_val)/2;
int max_pos=positions*2;
int max_num=(1<<(k_val+1));
for (uint i = 0; i < max_num; ++i)
{
std::uniform_int_distribution<int> selection(max_pos,input_vec.size()-max_pos);
std::random_device selrd;
std::mt19937 mtsel(selrd());
std::uniform_real_distribution<double> noise(.05,.015);
std::random_device nrd;
std::mt19937 mtnoise(nrd());
int hole_sel=selection(mtsel);
while(input_vec[hole_sel]==static_cast<double>(1.000))
{
hole_sel=selection(mtsel);
}
input_vec[hole_sel]+=noise(mtnoise);
selection.reset();
noise.reset();
}
};
void NKspacevals_highk(vector<double>& input_vec,int k_val){
int positions=(50-k_val)/2;
int max_pos=positions*2;
int max_num=(1<<(k_val+1));
for (uint i = 0; i < max_num; ++i)
{
std::uniform_real_distribution<double> replace(.65,.85);
std::random_device rprd;
std::mt19937 mt(rprd());
std::uniform_int_distribution<int> selection(max_pos,input_vec.size()-max_pos);
std::random_device selrd;
std::mt19937 mtsel(selrd());
std::uniform_real_distribution<double> noise(.001,.01);
std::random_device nrd;
std::mt19937 mtnoise(nrd());
double hole_score=replace(mt);
int hole_sel=selection(mtsel);
while(input_vec[hole_sel]==static_cast<double>(1.000))
{
hole_sel=selection(mtsel);
}
input_vec[hole_sel]=hole_score;
int j=0;
double exp_pos=0.0;
for (j=1, exp_pos =5; j<positions; exp_pos-=(.2500),++j)
{
input_vec[hole_sel+j]=(hole_score*(1-std::exp(-exp_pos))-noise(mtnoise));
input_vec[hole_sel-j]=(hole_score*(1-std::exp(-exp_pos))-noise(mtnoise));
}
replace.reset();
selection.reset();
noise.reset();
}
};

double stdev_calc(vector<int> vec){

double sum = std::accumulate(vec.begin(), vec.end(), 0.0); 
double mean = sum / vec.size();							   

std::vector<double> diff(vec.size());
std::transform(vec.begin(), vec.end(), diff.begin(), [mean](double x) { return x - mean; });
double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0); 
double stdev = std::sqrt(sq_sum / (vec.size()-1)); 
return stdev;
};

void NKspacevals_sine_normal(vector<double>& input_vec,int k_val){
vector<int> sol(input_vec.size());
for (uint i = 0; i < (input_vec.size()); ++i)
{
sol[i]=i+1;	
}
double per1=(input_vec.size())/((1<<k_val));
double per2=(input_vec.size())/(1<<(k_val+3));
std::uniform_real_distribution<double> uniform1(0,1);
std::random_device rnd1;
std::mt19937 mt1(rnd1());
std::uniform_real_distribution<double> uniform2(0,1);
std::random_device rnd2;
std::mt19937 mt2(rnd2());
vector<double> score(input_vec.size());
vector<double> std_norm(input_vec.size());
std::uniform_int_distribution<int> norm_mean(1,(input_vec.size()));
std::random_device rndn_mean;
std::mt19937 mt_normmean(rndn_mean());
std::uniform_real_distribution<double> perturbation(.00005,.3);
std::random_device rndn_per;
std::mt19937 mt_per(rndn_per());
std::random_device rndn_norm;
std::mt19937 mt_norm(rndn_norm());
double uni1=uniform1(mt1);
double uni2=uniform2(mt2);
for (uint i = 0; i < input_vec.size(); ++i)
{
score[i]=(std::sin((1/per1)*(i+1))*uni1+std::sin((1/per2)*(i+1))*uni2);

}
vector<double>::iterator maxresult;
vector<double>::iterator minresult;
maxresult = max_element(score.begin(), score.end());
minresult = min_element(score.begin(), score.end());
double maxscore=score[int(distance(score.begin(), maxresult))];
double minscore=score[int(distance(score.begin(), minresult))];
minscore=std::fabs(minscore);
double inverse_k= (1/k_val);
for (uint i = 0; i < input_vec.size(); ++i)
{
score[i]=((score[i]+minscore)/maxscore)*inverse_k;
}

double stdev = stdev_calc(sol);
stdev/=(1<<k_val);
double N_mean=static_cast<double>(norm_mean(mt_normmean));
double pi=atan(1) * 4;
double frac=(stdev*sqrtl(2.0*pi));
for (uint i = 0; i < input_vec.size(); ++i)
{
std_norm[i]=expl(pow((sol[i]-N_mean)/stdev,2)*(-0.5))/frac;

}
vector<double>::iterator maxresult_norm;
maxresult_norm = max_element(std_norm.begin(), std_norm.end());
double maxscore_norm=std_norm[int(distance(std_norm.begin(), maxresult_norm))];
for (uint i = 0; i < input_vec.size(); ++i)
{
std_norm[i]=std_norm[i]/maxscore_norm;

}

std::vector<double> normd_scores(input_vec.size());
for (uint i = 0; i < input_vec.size(); ++i)
{
normd_scores[i]=std_norm[i]+score[i];
}
vector<double>::iterator maxresult_normd_score;
maxresult_normd_score = max_element(normd_scores.begin(), normd_scores.end());
double maxscore_normd_score=normd_scores[int(distance(normd_scores.begin(), maxresult_normd_score))];
for (uint i = 0; i < input_vec.size(); ++i)
{
normd_scores[i]=(normd_scores[i]/maxscore_normd_score)+perturbation(mt_per);
perturbation.reset();
}




vector<double>::iterator pert_normd;
pert_normd = max_element(normd_scores.begin(), normd_scores.end());
double pert_normd_score=normd_scores[int(distance(normd_scores.begin(), pert_normd))];

for (uint i = 0; i < input_vec.size(); ++i)
{
normd_scores[i]=normd_scores[i]/pert_normd_score;
}
input_vec=normd_scores;

};


void NKspacevals_unit(vector<double>& input_vec,int n){
vector<double>::iterator maxresult;
vector<double>::iterator minresult;
maxresult = max_element(input_vec.begin(), input_vec.end());
minresult = min_element(input_vec.begin(), input_vec.end());
input_vec[int(distance(input_vec.begin(), maxresult))]=double(1.000);
input_vec[int(distance(input_vec.begin(), minresult))]=double(0.000);

};

int main(int argc, char *argv[]){
cout.setf(std::ios::fixed);
cout.setf(std::ios::showpoint);
cout.precision(15);
int nk=(1<<15);
vector<double> v(nk);
std::fstream file;
file.open("NK_space_strings.txt", std::ios_base::out | std::ios_base::in);
if (!file.is_open())
{
cout<<"does not exists"<<endl;
ofstream strings;
strings.open("NK_space_strings.txt");
for (int s=0; s<nk;++s)
{
strings<<std::bitset< 15 >(s).to_string()<<endl;

}
strings.close();
}

int loop=10000;
std::vector<int> k_opts={1,3,5,10};
int start = 0;
#pragma omp parallel for default(none) shared(k_opts,cout,nk,start,loop) firstprivate(v)
for (int j=start; j < loop; ++j)
{
for (int k = 0; k < k_opts.size(); ++k)
{
NKspacevals_sine_normal(v,k_opts[k]);




std::fstream scores;
scores.open("4_NK_space_scores_"+to_string(k_opts[k])+"_"+to_string(j)+"_cpp.txt",std::ios::out);
for (vector<double>::iterator i = v.begin(); i != v.end(); i++) 
{

scores<< std::fixed << std::setprecision(15)<<*i<<"\n";

}
scores.close();
}
}


return 0;
}