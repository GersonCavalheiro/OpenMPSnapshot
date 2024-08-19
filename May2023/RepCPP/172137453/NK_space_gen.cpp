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

double num=240;
double sigma=sqrt(1/num);
std::normal_distribution<double> uni(0.4,sigma);
std::random_device rd;
std::mt19937 norm(rd());
void NKspacevals_gen(vector<double>& input_vec,int n){
for (int i=0;i<n;++i)
{
input_vec[i]=uni(norm);
uni.reset();
}
};
void NKspacevals_k(vector<double>& input_vec,int k_val){
int max_values=0;
if(k_val%2==0) max_values=1<<((k_val+3));
else max_values=1<<(k_val+3);
cout<<max_values<<endl;
std::normal_distribution<double> hole1(.55,.009);
std::random_device rdn1;
std::mt19937 nor1(rdn1());
std::normal_distribution<double> hole2(.85,.009);
std::random_device rdn2;
std::mt19937 nor2(rdn2());
std::uniform_int_distribution<int> choice(0,(1<<20)-1);
std::random_device rdu;
std::mt19937 uniform(rdu());
int selection=0;
double replace=0.0;
for (int i = 0; i < max_values; ++i)
{
selection=choice(uniform);
while(input_vec[selection]==1) selection=choice(uniform);
replace=hole1(nor1);
while(replace>=1) replace=hole1(nor1);
input_vec[selection]=replace;
}
for (int i = 0; i < max_values; ++i)
{
selection=choice(uniform);
while(input_vec[selection]==1) selection=choice(uniform);
replace=hole2(nor2);
while(replace>=1) replace=hole2(nor2);
input_vec[selection]=replace;
}
};
void NKspacevals_unit(vector<double>& input_vec,int n){
vector<double>::iterator maxresult;
vector<double>::iterator minresult;
maxresult = max_element(input_vec.begin(), input_vec.end());
minresult = min_element(input_vec.begin(), input_vec.end());
input_vec[int(distance(input_vec.begin(), minresult))]=double(0.000);
for (int i = 0; i < n; ++i)
{	
input_vec[i]=input_vec[i]/input_vec[int(distance(input_vec.begin(), maxresult))];
}
};

int main(int argc, char *argv[]){
cout.setf(std::ios::fixed);
cout.setf(std::ios::showpoint);
cout.precision(15);
int nk=(1<<20);
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
strings<<std::bitset< 20 >(s).to_string()<<endl;

}
strings.close();
}

int loop=1000;
std::vector<int> k_opts={1,5,10,15};
int start = 0;
#pragma omp parallel for default(none) shared(k_opts,cout,nk,start,loop) firstprivate(v)
for (int j=start; j < loop; ++j)
{
for (int k = 0; k < k_opts.size(); ++k)
{
NKspacevals_gen(v,nk);
NKspacevals_k(v,k_opts[k]);


std::fstream scores;
scores.open("NK_space_scores_"+to_string(k_opts[k])+"_"+to_string(j)+".txt",std::ios::out);
for (vector<double>::iterator i = v.begin(); i != v.end(); i++) 
{

scores<< std::fixed << std::setprecision(15)<<*i<<"\n";

}
scores.close();
}
}


return 0;
}