
#include <iostream>
#include<cstdlib>
#include<omp.h>

using namespace std;

double randomgenerator()                                        
{
double x;
x=rand()/double(RAND_MAX);	                                
return x;                                                   
}

double randomgenerator(double a, double b)                      
{
double t;
t=(b-a)*randomgenerator() + a;                             
return t;
}

void seqrand(int n, int m)                                      
{
srand(101);									                
double mat[n][m];
int i,j;
for (i=0;i<n;++i)
{
for(j=0;j<m;j++)
{
mat[i][j]=randomgenerator(1.0,0.0);                 
}
}

cout<<endl<<"Randomly generated Matrix:"<<endl;

for(i=0;i<n;++i)
{
for(j=0;j<m;++j)
{
cout<<" "<<mat[i][j];                               
if(j==m-1)
cout<<endl;
}
}
}

void pararand(int n,int m)				                        
{
double mat[n][m];
int i,j,threads;
srand(101);                                                 

cout<<endl<<"\nMax number of threads used: "<<omp_get_max_threads();

#pragma omp parallel                                                
threads=omp_get_num_threads();

cout<<endl<<"\nNumber of threads: "<<threads<<endl;

#pragma omp parallel for schedule(dynamic)	                        
for (i=0;i<n;++i)
{
for(j=0;j<m;j++)
{
mat[i][j]=randomgenerator(1.0,0.0);         
}
}

cout<<endl<<"Randomly generated Matrix:"<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<m;++j)
{
cout<<" "<<mat[i][j];                           
if(j==m-1)
cout<<endl;
}
}
}

int main ()
{
int r1, c1;
int select,count=1;
string ch;

cout<<endl<<"\t\t\t\t Random Matrix Generation";
cout<<endl<< "\nEnter rows and columns for required random matrix: "; 
cin>>r1>>c1;


do
{
cout<<endl<<endl<<"Select your option:"<<endl;		        
cout<<endl<<"1.Sequential Allocation"<<"\t 2.Parallel Allocation(OpenMP) "<<endl<<"\nElse Press 0 to exit"<<endl<<"Choose : ";
cin>>select;

switch (select)                                             
{
case 1 :    cout<<endl<<"Allocating Sequentially...:"<<endl;

seqrand(r1,c1);						        

break;

case 2 :    cout<<endl<<"Allocating Parallely..."<<endl;

pararand(r1,c1);					        

break;

case 0 :    break;

default :   cout<<"Invalid selection... Press 0 to exit" << endl;

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
