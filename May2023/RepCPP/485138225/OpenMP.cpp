#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char** argv){

#pragma omp parallel num_threads(4)
{
#pragma omp critical
cout<<"Threads ID is OpemMP stage 1 = "<<omp_get_thread_num()<<endl; 
} 

cout<<"I am Muhammad Allah Rakha"<<endl;

#pragma omp parralel num_threads(2)
{
cout<<"Thread ID in OpenMP stage 2 = "<<omp_get_thread_num()<<endl;
}

}
