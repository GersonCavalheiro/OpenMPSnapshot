#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>
#include "sys/time.h"


using namespace std;
int min(float a, float b, float c);
#define num_loops 10000
#define num_data 256
#define data_max_value 10000



int main()
{

struct timeval start,end;
double elapsedTime;
int data[num_data][3];
int centroids[3][2];
srand(time(NULL));


for (int i = 0; i < num_data; i++)
{
for (int j = 0; j < 2; j++)
{

data[i][j] = (rand() % data_max_value) + 1;
}
}
int index;

for (int i = 0; i < 3; i++)
{

index = rand() % num_data;
centroids[i][0] = data[index][0];
centroids[i][1] = data[index][1];

}
gettimeofday(&start, NULL);

float d_c0, d_c1, d_c2;
int loop;
for (loop=0; loop < num_loops;loop++)
{
d_c0=0;
d_c1=0;
d_c2=0;

#pragma acc kernels
for (int i = 0; i < num_data; i++)
{
d_c0 = sqrt(pow((data[i][0] - centroids[0][0]),2.0) + pow((data[i][1] - centroids[0][1]),2.0));
d_c1 = sqrt(pow((data[i][0] - centroids[1][0]),2.0) + pow((data[i][1] - centroids[1][1]),2.0));
d_c2 = sqrt(pow((data[i][0] - centroids[2][0]),2.0) + pow((data[i][1] - centroids[2][1]),2.0));
data[i][2] = min(d_c0, d_c1, d_c2);
}

float c0x_avg = 0, c0y_avg = 0, c1x_avg = 0, c1y_avg = 0, c2x_avg = 0, c2y_avg = 0;
int c0_count = 0, c1_count = 0, c2_count = 0;





for (int i = 0; i < num_data; i++)
{
if (data[i][2] == 0)
{
c0x_avg = c0x_avg + data[i][0];
c0y_avg = c0y_avg + data[i][1];
c0_count++;
}
else if (data[i][2] == 1)
{
c1x_avg = c1x_avg + data[i][0];
c1y_avg = c1y_avg + data[i][1];
c1_count++;
}
else if (data[i][2] == 2)
{
c2x_avg = c2x_avg + data[i][0];
c2y_avg = c2y_avg + data[i][1];
c2_count++;
}
else{
}
}



centroids[0][0] = c0x_avg / c0_count;
centroids[0][1] = c0y_avg / c0_count;
centroids[1][0] = c1x_avg / c1_count;
centroids[1][1] = c1y_avg / c1_count;
centroids[2][0] = c2x_avg / c2_count;
centroids[2][1] = c2y_avg / c2_count;

}


gettimeofday(&end, NULL);
elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
elapsedTime /= 1000.0;


printf("OpenACC         : %.3f seconds\n", elapsedTime);


return 0;

}

#pragma acc routine seq
int min(float a, float b, float c)
{
if (a < b && a < c)
return 0;
if (b < a && b < c)
return 1;
if (c < a && c < b)
return 2;
else
return -1;
}
