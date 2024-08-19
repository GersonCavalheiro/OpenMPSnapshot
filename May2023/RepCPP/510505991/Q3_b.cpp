#include <omp.h>
#include<iostream>
#include<string.h>
#include <unistd.h>
using namespace std;

int parallel_count_occurrence(string text_string, string pattern)
{
int count = 0;
#pragma omp parallel for num_threads(6)
for(int i=0; i<text_string.length()-pattern.length()+1; i++)
{
string temp = "";
#pragma omp parallel for num_threads(6)
for(int j=0; j<pattern.length(); j++)
{
temp += text_string[i+j];
}
usleep(5000 * omp_get_thread_num());
cout<<"Thread "<<omp_get_thread_num()<<": "<<temp<<endl;
if(temp==pattern){
count++;
}
}
cout<<endl<<"Text: "<<text_string<<endl;
cout<<"Pattern: "<<pattern<<endl;
cout<<"Count: "<<count<<endl<<endl;
return count;
}

int main()
{
parallel_count_occurrence("ATTTGCGCAGACCTAAGCA", "GCA");

return 0;
}