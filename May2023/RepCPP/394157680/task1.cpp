#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
int count = 0;	

#pragma omp parallel num_threads(15)		
{
#pragma omp critical
{
int temp = count;	
temp++;
count = temp;		
}
}
cout<<"Thread Count = "<<count<<endl;
cout<<"Name = Hassan Shahzad"<<endl;
cout<<"Roll No = 18i-0441"<<endl;
}