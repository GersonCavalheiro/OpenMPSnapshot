#include <iostream>
#include <unistd.h>
#include <algorithm>
#include <string>

using namespace std;

int print_syntax();

int soft2hard();
string a2n();

int soft2hard()
{

string buffer;


while (getline(cin, buffer))
{
if (buffer.size() == 0)
continue;
if (buffer[0] == '>')
{
cout << buffer << endl;
}

else
{

#pragma omp parallel
{
std::replace(buffer.begin(), buffer.end(), 'a', 'N');
}
#pragma omp parallel
{
std::replace(buffer.begin(), buffer.end(), 't', 'N');
}
#pragma omp parallel
{
std::replace(buffer.begin(), buffer.end(), 'c', 'N');
}
#pragma omp parallel
{
std::replace(buffer.begin(), buffer.end(), 'g', 'N');
}



#pragma omp barrier
cout << buffer << endl;

}

}

return 0;
}

int print_syntax()
{

cerr << "Usage: cat/zcat/bzcat *.fa | soft2hard | or > into another program or file" << endl;
return 0;
}

int main()
{

(isatty(fileno(stdin)) ? print_syntax : soft2hard)();
return 0;
}
