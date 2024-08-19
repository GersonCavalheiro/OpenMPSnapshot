#include<iostream>
#include<omp.h>
#include<stdlib.h>
using namespace std;

int main(int argc, char* argv[]) {
cout << "Name: Sameet Asadullah\nRoll Number: 18i-0479\n\n";	

srand(time(NULL));	
int day, month, year = 2021, hour, agree = 1;
bool confirm = false;	

while(!confirm) {
day = (rand() % 30) + 1;
month = (rand() % 9) + 4;
hour = rand() % 24;
cout << "Proposed date and time is: " << day << "/" << month << "/" << year << ", " << hour << ":00:00." << endl;

#pragma omp parallel reduction(*:agree) num_threads(2)
{
agree = rand() % 2;
if (agree == 1) {
cout << "Friend " << omp_get_thread_num() << " agreed." << endl;
} else {
cout << "Friend " << omp_get_thread_num() << " didn't agree." << endl;
}	
}

if (agree == 1) {
confirm = true;
cout << "\nAll friends agreed on " << day << "/" << month << "/" << year << ", " << hour << ":00:00." << endl;
break;

} else {
agree = 1;
cout << endl;
}
}
}
