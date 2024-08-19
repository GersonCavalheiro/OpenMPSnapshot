#include<iostream>
#include<math.h>
#include<time.h>
#include<mpi.h>

#include <cstdlib>
using namespace std;
#define MAX 100000 

void find_clusters(int n,int k,int *x,int *y,int *kcen){
struct { 
double value; 
int   rank; 
} kc1[n], o[n],kc[n],d[n];
int min=999999,max_it=5,it=0,i,j;
int xcnt=0,cnt=0;
int cenx[k],ceny[k];
double euc_distance[k][n];
int rank,size;

int root=0,flag=0;

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

srand(time(0));
for(int i=0;i<n;i++){
kc1[i].rank=rand()%k;
kc[i].rank=rand()%k;
if(i<k)
{
cenx[i]=rand()%100;
ceny[i]=rand()%100;
}
}
MPI_Bcast(cenx, k, MPI_INT, root, MPI_COMM_WORLD);
MPI_Bcast(cenx, k, MPI_INT, root, MPI_COMM_WORLD);

do
{
#pragma omp parallel for shared(d)
for(j=0;j<n;j++) 
{
d[j].value=(((cenx[rank]-x[j])*(cenx[rank]-x[j]))+((ceny[rank]-y[j])*(ceny[rank]-y[j])));

d[j].value = sqrt(d[j].value);
d[j].rank=rank;
}

MPI_Barrier(MPI_COMM_WORLD);



MPI_Reduce( d, kc, n, MPI_DOUBLE_INT, MPI_MINLOC, root,  MPI_COMM_WORLD);
if(rank==root)
{




int l=0;
#pragma omp parallel for shared(l)
for(j=0;j<n;j++)
{
if(kc1[j].rank==kc[j].rank) 
{
#pragma omp atomic
l++;
}
}
if(l==n)
{

flag=1;
#pragma omp parallel for
for(i=0;i<n;i++)
kcen[i]=kc[i].rank;


}
else 
{
#pragma omp parallel for    
for(j=0;j<n;j++)
kc1[j].rank=kc[j].rank;
}




for(int m=0;m<k;m++) 
{
xcnt=0;
#pragma omp parallel for shared(kc,cenx,ceny,xcnt,m)
for(int j=0;j<n;j++) 
{
if(kc[j].rank==m)
{
cenx[m]=x[j]+cenx[m];
ceny[m]=y[j]+ceny[m];
xcnt++;
}
}
if(xcnt!=0)
{
cenx[m]=(int)(cenx[m]/xcnt);
ceny[m]=(int)(ceny[m]/xcnt);
}
}
}
MPI_Bcast(cenx, k, MPI_INT, root, MPI_COMM_WORLD);
MPI_Bcast(ceny, k, MPI_INT, root, MPI_COMM_WORLD);
MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD); 
it++;

}while(it<max_it && flag==0);

if(rank==0)
{
#pragma omp parallel for
for(i=0;i<n;i++)
kcen[i]=kc[i].rank;
}

}
