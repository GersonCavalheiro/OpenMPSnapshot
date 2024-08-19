#include<iostream>
#include<omp.h>
using namespace std;

int main(int argc, char* argv[]) {
cout << "Name: Sameet Asadullah\nRoll Number: 18i-0479\n";	
int threadCount = 0;

#pragma omp parallel num_threads(100)
{
threadCount++;
}
cout << "Total Number of Threads: " << threadCount  << endl;
}
