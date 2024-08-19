USE DOUBLE VALUES MODDED WITH SOMETHING THAT PREVENTS OVERFLOW


cout << endl << "Dynamically allocating square matricies of size " << mat_size << "..." << endl;
int** A = new int*[mat_size];
for(int i = 0; i < mat_size; i++)
A[i] = new int[mat_size];

int** B = new int*[mat_size];
for(int i = 0; i < mat_size; i++)
B[i] = new int[mat_size];

int** C = new int*[mat_size];
for(int i = 0; i < mat_size; i++)
C[i] = new int[mat_size];



cout << endl << "Deleting dynamically allocated matricies from memory..." << endl;
for(int i = 0; i < mat_size; i++)
delete[] A[i];
delete[] A;

for(int i = 0; i < mat_size; i++)
delete[] B[i];
delete[] B;

for(int i = 0; i < mat_size; i++)
delete[] C[i];
delete[] C;


srand(time(NULL));


rand()


int A[4][4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};


for (int i=0; i<4; i++)
{
for(int j=0; j<4; j++)
{
cout << A[i][j] << "\t";
}
cout << endl;
}


for (int i=0; i<4; i++)
{
for(int j=0; j<4; j++)
{
curr_cell = 0; 
for(int k=0; k<4; k++)
{
curr_cell += A[i][k]*B[k][j];
}
C[i][j]=curr_cell; 
cout << curr_cell << "\t";
}
cout << endl;
}


#include <ctime>
time_t begin,end; 

time (&begin); 
CODE GOES HERE
time (&end); 

double difference = difftime (end,begin);
printf ("time taken for function() %.2lf seconds.\n", difference);




int mat_size = 10; 
int chunk_size = 3; 
int thread_count = 3; 
int curr_cell = 0; 
int start; 
int end; 
#pragma omp parallel
{
start = (chunk_size * thread_ID);
end = (start + chunk_size - 1);
if(thread_ID == thread_count - 1)
{
end = mat_size - 1;
}
for(i = start; i <= end; i++)
{
for(j = start; j <= end; j++)
{
curr_cell = 0;
for(k = 0; k <= mat_size; k++)
{
curr_cell += A[i][k]*B[k][j];
}
C[i][j] = curr_cell;
}
}
}





cout << endl << "Initializing static values..." << endl;
A[0][0] = 1;
A[0][1] = 2;
A[0][2] = 3;
A[0][3] = 4;
A[1][0] = 5;
A[1][1] = 6;
A[1][2] = 7;
A[1][3] = 8;
A[2][0] = 9;
A[2][1] = 10;
A[2][2] = 11;
A[2][3] = 12;
A[3][0] = 13;
A[3][1] = 14;
A[3][2] = 15;
A[3][3] = 16;

B[0][0] = 16;
B[0][1] = 15;
B[0][2] = 14;
B[0][3] = 13;
B[1][0] = 12;
B[1][1] = 11;
B[1][2] = 10;
B[1][3] = 9;
B[2][0] = 8;
B[2][1] = 7;
B[2][2] = 6;
B[2][3] = 5;
B[3][0] = 4;
B[3][1] = 3;
B[3][2] = 2;
B[3][3] = 1;



cout << endl << "Printing Matrix A:" << endl;
for(int i = 0; i < mat_size; i++)
{
for(int j = 0; j < mat_size; j++)
{
cout << A[i][j] << "\t";
}
cout << endl;
}


cout << endl << "Printing Matrix B:" << endl;
for(int i = 0; i < mat_size; i++)
{
for(int j = 0; j < mat_size; j++)
{
cout << B[i][j] << "\t";
}
cout << endl;
}




















