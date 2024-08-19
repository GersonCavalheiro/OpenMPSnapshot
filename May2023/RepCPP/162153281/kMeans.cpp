#include<iostream>
#include<math.h>
#include<time.h>
#include<omp.h>
using namespace std;
#define MAX 100000 

void find_clusters(int n,int k,int *x,int *y,int *kc){
float mn=999999,max_it=50,it=0;
int xcnt=0,cnt=0,i;
int cenx[n],ceny[n],kc1[n];
double d[n],euc_distance[k][n];

srand(time(0));
#pragma omp parallel for shared(i,kc1,kc,cenx,ceny)
for(i=0;i<n;i++){
kc1[i]=rand()%k;
kc[i]=rand()%k;
cenx[i]=x[i];
ceny[i]=y[i];
}

do
{
int m,j;
#pragma parallel omp for collapse(2) shared(m,j,d,euc_distance)
for(m=0;m<k;m++) 
{
for(j=0;j<n;j++) 
{
d[j]=(((cenx[m]-x[j])*(cenx[m]-x[j]))+((ceny[m]-y[j])*(ceny[m]-y[j])));

euc_distance[m][j] = sqrt(d[j]);
}
}
int l=0;
#pragma parallel omp for shared(j,l,kc,kc1)
for(j=0;j<n;j++)
{
if(kc1[j]==kc[j]) 
{
l++;
}
}
if(l==n)
{
return;

}
else 
{
#pragma omp parallel for shared(j,kc,kc1)
for(j=0;j<n;j++)
kc1[j]=kc[j];
}

for(j=0;j<n;j++) 
{
for(m=0;m<k;m++) 
{
if(euc_distance[m][j]<mn)
{
mn=euc_distance[m][j];
kc[j]=m;
}
}
mn=999999;
cenx[j]=ceny[j]=0; 

}

for(m=0;m<k;m++) 
{
xcnt=0;
#pragma omp parallel for shared(kc,cenx,ceny,xcnt,m,j)
for(j=0;j<n;j++) 
{
if(kc[j]==m)
{
cenx[m]=x[j]+cenx[m];
ceny[m]=y[j]+ceny[m];
xcnt++;
}
}
cenx[m]=cenx[m]/xcnt;
ceny[m]=ceny[m]/xcnt;
}   
it++;
}while(it<max_it);
}
