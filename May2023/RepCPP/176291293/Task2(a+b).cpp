

#include <iostream>
#include <omp.h>

using namespace std;

void seqmultisum(int n, int m, int p)							        
{
int a[n][m],b[m][p],mult[n][p],c[n][p],sum[n][p],i,j,k;

cout<<endl<<"Enter elements of matrix A:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<m;++j)
{
cout<<"Enter element A"<<i+1<<j+1<<" : ";
cin>>a[i][j];                                           
}
}

cout<<endl<<"Enter elements of matrix B:"<<endl;

for(i=0;i<m;++i)
{   for(j=0;j<p;++j)
{
cout<<"Enter element B"<<i+1<<j+1<<" : ";
cin>>b[i][j];                                           
}
}

cout<<endl<<"Enter elements of matrix C:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<"Enter element C"<<i+1<<j+1<<" : ";
cin>>c[i][j];                                           
}
}

for(i = 0; i < n; ++i)
{   for(j = 0; j < p; ++j)							        
{   mult[i][j]=0;
for(k=0;k<m;++k)
{
mult[i][j]+=a[i][k]*b[k][j];			        

}
sum[i][j]=mult[i][j]+c[i][j];				        
}

}

cout<<endl<<"A:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<m;++j)
{
cout<<" "<<a[i][j];							            
if(j==m-1)
cout<<endl;
}
}

cout<<endl<<"B: "<<endl;

for(i=0;i<m;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<b[i][j];							            
if(j==p-1)
cout<<endl;
}
}


cout<<endl<<"C: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<c[i][j];							            
if(j==p-1)
cout<<endl;
}
}

cout<<endl<<"Output Matrix  AB: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<mult[i][j];					                
if(j==p-1)
cout<<endl;
}
}

cout<<endl<<"Output Matrix  AB+C: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<sum[i][j];				                    
if(j==p-1)
cout<<endl;
}
}



}

void paramultisum(int n, int m, int p)					                
{
int a[n][m],b[m][p],multi[n][p],c[n][p],sum[n][p],i,j,k,threads;

cout<<endl<<"Enter elements of matrix A:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<m;++j)
{
cout<<"Enter element A"<<i+1<<j+1<<" : ";
cin>>a[i][j];                                           
}
}

cout<<endl<<"Enter elements of matrix B:"<<endl;

for(i=0;i<m;++i)
{   for(j=0;j<p;++j)
{
cout<<"Enter element B"<<i+1<<j+1<<" : ";
cin>>b[i][j];                                           
}
}

cout<<endl<<"Enter elements of matrix C:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<"Enter element C"<<i+1<<j+1<<" : ";
cin>>c[i][j];                                           
}
}

cout<<endl<<"\nMax number of threads used: "<<omp_get_max_threads();

#pragma omp parallel                                                
threads=omp_get_num_threads();

cout<<endl<<"\nNumber of threads: "<<threads<<endl;

# pragma omp parallel private (j,k) shared(a,b,c,multi)		        
{															        
# pragma omp for schedule(dynamic)						        
for(i=0;i<n;i++)									        
{													        
for(j=0;j<p;j++)
{
multi[i][j]=0;						                    
for(k=0;k<m;k++)
{
multi[i][j]=multi[i][j]+a[i][k]*b[k][j];
}
}
}

#pragma omp barrier                                             
#pragma omp for	                                                
for(i=0;i<n;i++)
{
for(j=0;j<p;j++)
{
sum[i][j]=multi[i][j]+c[i][j];			                
}
}

}

cout<<endl<<"A:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<m;++j)
{
cout<<" "<<a[i][j];						                
if(j==m-1)
cout<<endl;
}
}

cout<<endl<<"B: "<<endl;

for(i=0;i<m;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<b[i][j];						                
if(j==p-1)
cout<<endl;
}
}


cout<<endl<<"C: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<c[i][j];                                     
if(j==p-1)
cout<<endl;
}
}

cout<<endl<<"Output Matrix  AB: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<multi[i][j];                                 
if(j==p-1)
cout<<endl;
}
}

cout<<endl<<"Output Matrix  AB+C: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<sum[i][j];                                   
if(j==p-1)
cout<<endl;
}
}
}

int main ()
{
int r1, c1, r2, c2, row3, col3;
int select,count=1;
string ch;

cout<<endl<<"\t\t\t Multiplication and Summation ";
cout<<endl<<"\nEnter rows and columns for first matrix: ";			
cin>>r1>>c1;
cout<<endl<<"Enter rows and columns for second matrix: ";			
cin>>r2>>c2;
cout<<endl<<"Enter rows and columns for third matrix: ";			
cin>>row3>>col3;

while (c1!=r2)											            
{                                                                   
cout<<endl<<"Error! column of first matrix not equal to row of second.";
cout<<endl<<"Enter rows and columns for first matrix: ";
cin>>r1>>c1;
cout<<"Enter rows and columns for second matrix: ";
cin>>r2>>c2;
}

while ((c2!=col3)||(r1!=row3))						                
{
cout<<endl<<"Error! Size of Summation matrix is not equal to the Multiplied matrix";
cout<<endl<<"Enter rows and columns for third matrix: ";
cin>>row3>>col3;
}


do
{
cout<<endl<<endl<<"Select your option:";                    
cout<<endl<<"1. Sequential Multiplication and Summation"<<"\t 2.Parallely Multiplication and Summation (OpenMP)"<<endl<<"\nElse Press 0 to exit"<<endl<<"Choose : ";
cin>>select;

switch (select)                                             

{
case 1 :    cout<<endl<<"Using Sequential Matrix Multiplication"<<endl;

seqmultisum(r1,c1,c2);				        

break;

case 2 :    cout<<endl<<"Use Parallel Matrix Multiplication"<<endl;

paramultisum(r1,c1,c2);				        

break;

case 0 :    break;

default : cout<<"Invalid selection" << endl;

}
if(count>=2)                                               
{
cout<<endl<<"\nExit?? (y/n): ";
cin>>ch;
if(ch=="y"||ch=="Y")
break;
else
continue;
}
count++;

}while (select != 0);

return 0;

}
