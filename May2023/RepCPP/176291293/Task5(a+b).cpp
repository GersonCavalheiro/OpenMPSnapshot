
#include <iostream>
#include<cstdlib>
#include<ctime>
#include<omp.h>

using namespace std;

double seqtime()                                            
{   double start=omp_get_wtime();                           
const int n=1000;
static unsigned int a[n][n],b[n][n],c[n][n];
static unsigned int multi[n][n],sum[n][n];
int i,j,k;

srand(1234);                                            
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
a[i][j]=rand()%10;                              
b[i][j]=rand()%10;                              
c[i][j]=rand()%10;                              
}
}

for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
multi[i][j]=0;                                  
for(k=0;k<n;k++)
{
multi[i][j]=multi[i][j]+a[i][k]*b[k][j];    
}
}
}

for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
sum[i][j]=multi[i][j]+c[i][j];                  
}
}

double end = omp_get_wtime();                           
double elapsed;
elapsed = (end - start);
return elapsed;                                         
}

double paratime()                                           
{   double start=omp_get_wtime();                           
const int n=1000;
static unsigned int a[n][n],b[n][n],c[n][n];
static unsigned int multi[n][n],sum[n][n];
int i,j,k,threads;

srand(1234);                                            
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
a[i][j]=rand()%10;                              
b[i][j]=rand()%10;                              
c[i][j]=rand()%10;                              
}
}

cout<<endl<<"\nMax number of threads used: "<<omp_get_max_threads();

#pragma omp parallel                                                
threads=omp_get_num_threads();

cout<<endl<<"\nNumber of threads: "<<threads<<endl;

# pragma omp parallel shared (a,b,c) private (i,j,k)    
{                                                       
# pragma omp for schedule(dynamic)                  
for(i=0;i<n;i++)                                    
{
for(j=0;j<n;j++)
{
multi[i][j]=0;                              
for(k=0;k<n;k++)
{
multi[i][j]=multi[i][j]+a[i][k]*b[k][j];
}
}
}

#pragma omp for
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
sum[i][j]=multi[i][j]+c[i][j];              
}
}

}

double end = omp_get_wtime();                           
double elapsed;
elapsed = (end - start);
return elapsed;                                         
}

int main ()
{   double seq_time,para_time;
string ch;
cout<<endl<<"\t\t\t Large Martrix (1000x1000) Operation\n"<<endl;
do{
cout<<endl<<"\nCalculating Sequentially...";
seq_time=seqtime();                                 
cout<<endl<<"\nThe time taken by Operation Sequentially : "<<seq_time<<"s";
cout<<endl<<"\n\nCalculating Parallely...";
para_time=paratime();                               
cout<<endl<<"\nThe time taken by Operation Parallely : "<<para_time<<"s";
float factor=(seq_time/para_time);                  
cout<<endl<<"\n\nThe Parallel process is "<<factor<<" times faster than Sequential"<<endl;
cout<<endl<<"\nWould you like to re-run the test? (y/n) : ";
cin>>ch;                                            
}while(ch=="y"||ch=="Y");

return 0;

}

