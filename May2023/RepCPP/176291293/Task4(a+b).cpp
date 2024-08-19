
#include <iostream>
#include<cstdlib>
#include<unistd.h>
#include<omp.h>
#include<fstream>
#include<windows.h>
#include<ctime>

using namespace std;

void seqrand()                                                  
{   const int s=1000;
srand(time(0));                                             
static int mat[s][s];
int i,j;
string ch,ch2;
for (i=0;i<s;++i)
{
for(j=0;j<s;j++)
{
mat[i][j]=rand();                                   
}
}

ofstream fout("seq.csv");                                   

if(fout.is_open())                                          
{
cout<<endl<<"File Opened successfully!!!. Writing data from array to file"<<endl;

for (i=0;i<s;++i)
{
for(j=0;j<s;j++)
{
fout<<mat[i][j]<<",";                           
if(j==s-1)
fout<<endl;
}
}

cout<<endl<<"Array data successfully saved into the file seq.csv"<<endl;
}

else                                                        

{

cout <<endl<< "File could not be opened." << endl;

}

cout<<endl<<"Create heatmap?  (y/n): ";

cin>>ch;

if(ch=="y"||ch=="Y")                                        
{
ShellExecuteA(0,"open",".\\seq_heatmap.R",NULL,NULL,SW_SHOW); 

sleep(6);

cout<<endl<<"Heat map Created...";

cout<<endl<<"View the heatmap in a separate window as .TIFF file?  (y/n): ";

cin>>ch2;

if(ch2=="y"||ch2=="Y")                              
{
ShellExecuteA(0,"open",".\\seq_heatmap.tiff",NULL,NULL,SW_SHOW); 

sleep(2);
}

else
{
return;
}
}
else
{
return;
}

}

void pararand()                                                 
{   const int s=1000;
srand(time(0));                                             
static int mat[s][s];
int i,j,threads;
string ch,ch2;

cout<<endl<<"\nMax number of threads used: "<<omp_get_max_threads();

#pragma omp parallel                                                
threads=omp_get_num_threads();

cout<<endl<<"\nNumber of threads: "<<threads<<endl;

#pragma omp 
{
#pragma omp prallel for                                 
for (i=0;i<s;++i)
{
for(j=0;j<s;j++)
{
mat[i][j]=rand();                               
}

}
#pragma omp barrier                                     
ofstream fout("para.csv");                              

if(fout.is_open())                                      

{
cout<<endl<<"File Opened successfully!!!. Writing data from array to file"<<endl;

for (i=0;i<s;++i)
{
for(j=0;j<s;j++)
{
fout<<mat[i][j]<<",";                       
if(j==s-1)
fout<<endl;
}
}

cout<<endl<<"Array data successfully saved into the file para.csv"<<endl;
}

else                                                    

{

cout<<endl<<"File could not be opened."<<endl;

}

cout<<endl<<"Create heatmap?  (y/n): ";

cin>>ch;

if(ch=="y"||ch=="Y")                                    
{
ShellExecuteA(0,"open",".\\para_heatmap.R",NULL,NULL,SW_SHOW); 

sleep(6);

cout<<endl<<"Heat map Created...";

cout<<endl<<"View the heatmap in a separate window as .TIFF file?  (y/n): ";

cin>>ch2;

if(ch2=="y"||ch2=="Y")                          
{
ShellExecuteA(0,"open",".\\para_heatmap.tiff",NULL,NULL,SW_SHOW); 

sleep(2);
}
else
{
return;
}
}

else
{
return;
}
}
}
int main ()
{
int select,count=1;;
string ch;
cout<<endl<<"\t\t\tRandom Matrix Generation";

do
{
cout<<endl<<endl<<"Select your option: ";           
cout<<endl<<"1. Sequential Allocation"<<"\t 2.Parallel Allocation (OpenMP)"<<endl<<"Else Press 0 to exit"<<endl<<" Choose : ";
cin>>select;
count++;
switch (select)                                     

{
case 1 :    cout<<endl<<"Allocating Sequentially..."<<endl;

seqrand();                          

break;

case 2 :    cout<<endl<<"Allocating Parallely..."<<endl;

pararand();                         

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
