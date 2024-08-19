#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
bool consensus = true;	

while(true)
{
consensus = true;

#pragma omp parallel num_threads(4)		
{
int choice;
#pragma omp critical
{
cout<<"Enter choice (0=yes / 1 = no) for thread"<< omp_get_thread_num()<<" = ";
cin>>choice;
if (choice == 1)
{
consensus = false;
}
}
}
if (consensus)
{
cout<<"Consensus has been reached :)"<<endl<<endl;
}
else
{
cout<<"Consensus couldn't be reached"<<endl<<endl;
}
}
}