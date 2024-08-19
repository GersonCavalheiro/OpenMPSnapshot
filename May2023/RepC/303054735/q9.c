#include <stdio.h>
#include <omp.h>
#include <unistd.h>
char buffer[5];
int nextin = 0;
int nextout = 0;
int count = 0;
int empty = 1;
int full = 0;
void add(char item)
{
buffer[nextin] = item;
nextin = (nextin + 1) % 5;
count++;
if (count == 5)
full = 1;
if (count == 1)
empty = 0;
}
void producer(int tid)
{
char item;
int i=0;
while( i < 50)
{
#pragma omp critical
{
if(full)
printf("Buffer already full\n");
else
{
item = 'A' + (i % 26);
add(item);
i++;
printf("Thread %d Prodused %c\n",tid, item);
}
}
sleep(1);
}
}
char removee()
{
char item;
item = buffer[nextout];
nextout = (nextout + 1) % 5;
count--;
if (count == 0)
empty = 1;
if (count == 4)
full = 0;
return item;
}
void consumer(int tid)
{
char item;
int j=0;
while(j < 50)
{
#pragma omp critical
{
if(empty)
printf("Buffer empty\n");
else
{
j++;
item = removee();
printf("Thread %d Consumed %c\n",tid, item);
}
}
sleep(1);
}
}
int main()
{
#pragma omp parallel num_threads(2)
{
int tid=omp_get_thread_num();
if(tid==1)
producer(tid);
else if(tid==0)
consumer(tid);
}
}
