#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <omp.h>
using namespace std;

#define N 500
#define max_iter 100000


double func(vector<double> &x,int fnum){
double res=0;
switch(fnum){
case 1:    
for(int n = 0; n < N; n++)
{
res+=(pow(10,6.0*(n/float(N)))*x[n]*x[n]);
}
break;
case 2:
res+=x[0]*x[0];
for(int n = 0; n < N-1; n++)
{
res+=1000000*(x[n+1]*x[n+1]);
}
break;

case 3:
res+=x[0]*x[0];
for(int n = 0; n < N-2; n++)
{
res+=10000*(x[n+1]*x[n+1]);
}
res+=1000000*x[N-1]*x[N-1];
break;
case 4:

res+=1000000*x[0]*x[0];
for(int n = 0; n < N-1; n++)
{
res+=(x[n+1]*x[n+1]);
}
break;

case 5:
for(int n = 0; n < N; n++)
{
double temp=0;
for(int j=0;j<n;j++)
temp+=x[n]*x[n];
res+=temp*temp;
}
break;
case 6:

for(int n = 0; n < N-1; n++)
{
res+=(100*(x[n+1]-x[n]*x[n])*(x[n+1]-x[n]*x[n])+(x[n]-1)*(x[n]-1));
}
break;
}
return res;
}

int partition(vector<vector<double>> &a,int l,int u,int num)
{
int i = (l - 1); 
double fpivot = func(a[u],num);  

for (int j = l; j <= u-1; j++) 
{ 
if (func(a[j],num) <= fpivot) 
{ 
i++; 
a[i].swap(a[j]); 
} 
} 
a[i + 1].swap(a[u]); 
return (i + 1); 
}

void quick_sort(vector<vector<double>> &a,int l,int u,int num)
{
int j;
if(l<u)
{
j=partition(a,l,u,num);
#pragma omp parallel sections
{
#pragma omp section
{
quick_sort(a,l,j-1,num);
}
#pragma omp section
{
quick_sort(a,j+1,u,num);
}
}
}
}

void find_rank(vector<vector<double>> &F,vector<vector<double>> &R,int &t,int &mu){
set<double> Funion;
for(int i=0;i<mu;i++){
Funion.insert(F[t&1][i]);
Funion.insert(F[(t+1)&1][i]);
}

#pragma omp parallel for
for(int i = 0;i<mu;i++){
for(auto it = Funion.begin();it!=Funion.end();it++){
if(F[t&1][i] == *it){
R[t&1][i] =distance(Funion.begin(),it);
break;
}
}
for(auto it = Funion.begin();it!=Funion.end();it++){
if(F[(t+1)&1][i] == *it){
R[(t+1)&1][i] =distance(Funion.begin(),it);
break;
}
}
}

}

vector<double> R1ES(int num){

double sigma;
vector<vector<double>> m(2,vector<double>(N,0));
vector<vector<double>> p(2,vector<double>(N,0));
vector<double> xbest(N,0);
double s;

sigma=20/3.0;
srand (time(NULL));
#pragma omp parallel for  
for(int n=0;n<N;n++)
{
double fn = ((float)rand())/RAND_MAX;
float fr = 2*(fn-0.5);
m[0][n]= 10*fr;
xbest[n]=m[0][n];
}

int lambda = 4 + floor(3*log(N));
int mu = floor(lambda/2.0);
vector<double> w(mu,0);
double mueffdem = 0;

#pragma omp parallel for shared(mu)
for (int i = 1; i <= mu; i++){
double wnum = log(mu+1) - log(i);
double wdem = mu*log(mu+1);
#pragma omp parallel for shared(mu) reduction (-:wdem) 
for(int j = 1; j <= mu; j++){
wdem -= log(j);
}
w[i-1] = wnum/wdem;
mueffdem = mueffdem + w[i-1]*w[i-1];
} 
double mueff = 1.0/mueffdem;
double ccov =  1.0/(3*sqrt(N) + 5);
double cc = 2.0/(N+7);
double qstar = 0.3;
double q = 0;
double dsigma = 1000;
double cs = 0.3;

vector<vector<double>> x (lambda,vector<double>(N,0));
vector<vector<double>> R (2,vector<double>(mu,0));
vector<vector<double>> F (2,vector<double>(mu,0));
double fm = func(m[0],num);
#pragma omp parallel for shared(fm,mu)
for(int i = 0;i<mu;i++){
F[0][i] = fm;
}

int t=0;
double fbest = func(xbest,num);
srand(time(NULL));
while(t<max_iter&&fbest>0.001){
#pragma omp parallel for shared(lambda,fbest,sigma,ccov) 
for(int i = 0; i<lambda; i++){

vector<double> z(N);
std::random_device rd{};
std::mt19937 gen{rd()};
default_random_engine generator;
normal_distribution<double> distribution(0,1.0);
double r  = distribution(gen);
#pragma omp prallel for shared(x,m,sigma,ccov,p,r,z)
for(int n=0; n <N;n++){
z[n]=distribution(gen);
x[i][n] = m[t&1][n] + sigma*(sqrt(1-ccov)*z[n] + sqrt(ccov)*r*p[t&1][n]);
}
#pragma omp critical
if(func(x[i],num) < fbest){
#pragma omp parallel for
for(int n=0; n<N;n++){
xbest[n] = x[i][n];
}
fbest = func(xbest,num);
}
}



quick_sort(x,0,lambda-1,num);
#pragma omp parallel for shared(mu,t)
for(int i = 0;i<mu;i++){
F[(t+1)&1][i] = func(x[i],num);
}

#pragma omp parallel for shared(mu,sigma,cc,m,p,mueff)
for(int n=0;n<N;n++){
double mnew=0;
#pragma omp parallel for shared(mu,w,n) reduction(+:mnew)
for(int i=0;i<mu;i++){
mnew += w[i]*x[i][n];
}
m[(t+1)&1][n] = mnew; 
p[(t+1)&1][n] = (1-cc)*p[t&1][n] + sqrt(cc*(2-cc)*mueff)*(m[(t+1)&1][n] - m[t&1][n])/sigma;
}

find_rank(p,R,t,mu);
q=0;
#pragma omp parallel for shared(mu,R,w) reduction(+:q)
for(int i=0;i<mu;i++){
q += w[i]*(R[t&1][i]-R[(t+1)&1][i]);
}
q /= mu;

s = (1-cs)*s +cs*(q-qstar);
sigma = sigma*exp(s/dsigma);
t++;
}
return xbest;
}

int main()
{
omp_set_num_threads(8);
int i;
double T[2];
for(i=1;i<=6;i++){
for(int j = 0;j<2;j++){
double t1=omp_get_wtime(),t2;
vector<double> v = R1ES(i);
t2=omp_get_wtime();
T[j]=(double)(t2-t1);
cout<<i<<","<<j<<","<<T[j]<<endl;
}
cout<<"Avg="<<(T[0]+T[1])/2.0<<endl;
}
return 0;
}

