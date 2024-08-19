

#include <iostream>
#include <omp.h>

using namespace std;

void seqmulti(int n, int m, int p)                                      
{
int a[n][m],b[m][p],multi[n][p],i,j,k;

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

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{   multi[i][j]=0;                                          
for(k=0;k<m;++k)
{
multi[i][j]+=a[i][k]*b[k][j];                       
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

cout<<endl<<"Output Matrix  AB: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<multi[i][j];                                  
if(j==p-1)
cout<<endl;
}
}
}

void paramulti(int n, int m, int p)                                     
{
int a[n][m],b[m][p],multi[n][p],i,j,k,threads;

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

cout<<endl<<"\nMax number of threads used: "<<omp_get_max_threads();

#pragma omp parallel                                                
threads=omp_get_num_threads();

cout<<endl<<"\nNumber of threads: "<<threads<<endl;

#pragma omp parallel for private(j,k) schedule(dynamic)				
for(i=0;i<n;++i)                                                
{   for(j=0;j<p;++j)                                        
{   multi[i][j]=0;                                      
for(k=0;k<m;++k)
{
multi[i][j]+=a[i][k]*b[k][j];                   
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

cout<<endl<<"Output Matrix  AB: "<<endl;

for(i=0;i<n;++i)
{   for(j=0;j<p;++j)
{
cout<<" "<<multi[i][j];                                  
if(j==p-1)
cout<<endl;
}
}

}

int main ()
{
int r1, c1, r2, c2;
int select,count=1;
string ch;

cout<<endl<<"\t\t\t Matrix Multiplication";
cout<<endl<< "\nEnter rows and columns for first matrix: ";
cin>>r1>>c1;                                                        
cout<<endl<< "Enter rows and columns for second matrix: ";
cin>>r2>>c2;                                                        

while (c1!=r2)                                                      
{                                                                   
cout << "Error! column of first matrix not equal to row of second.";
cout << "Enter rows and columns for first matrix: ";
cin>> r1 >> c1;
cout << "Enter rows and columns for second matrix: ";
cin >> r2 >> c2;
}


do
{
cout<<endl<<endl<<"Select your option:";                    
cout<<endl<<"1. Sequential Matrix Multiplication"<<"\t 2.Parallely Multiply Matrix (OpenMP)"<<endl<<"Else Press 0 to exit"<<endl<<" Choose : ";
cin>>select;

switch (select)                                             

{
case 1 :    cout<<endl<<"Using Sequential Matrix Multiplication"<<endl;

seqmulti(r1,c1,c2);                         

break;

case 2 :    cout<<endl<<"Use Parallel Matrix Multiplication"<<endl;

paramulti(r1,c1,c2);                        

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
}while (select!=0);

return 0;

}


