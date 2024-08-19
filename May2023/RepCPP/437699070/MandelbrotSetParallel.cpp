#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <stdlib.h>
using namespace std;

const int dispWidth = 768;
const int dispHeight = 768;
const int neg_X = -2;
const int x = 2;
const int neg_Y = -2;
const int y = 2;

#define DATA_TAG 1
#define TERMINATE_TAG 0
#define RESULT_TAG 2

struct complex 
{
float real;
float imag;
};

int coords[dispWidth][dispHeight];

int cal_pixel(complex c) 
{
int count = 0;
int max_iter = 10000;
complex z;
float temp;
float lengthsq;
z.real = 0.0;
z.imag = 0.0;

do 
{
temp = z.real * z.real - z.imag * z.imag + c.real;
z.imag = 2 * z.real * z.imag + c.imag;
z.real = temp;
lengthsq = z.real * z.real + z.imag * z.imag;
count++;
} while ((lengthsq < 4.0) && (count < max_iter));

return count; 
}

int main(int argc, char* argv[])
{
int myrank;
int npes;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Comm_size(MPI_COMM_WORLD, &npes);

if (myrank == 0)
{

MPI_Status status;
int row = 0; 
int row_colors[dispWidth + 1]; 

double startTime;
double endTime; 

ofstream myFile;
myFile.open("MandelbrotSetParallelImg.ppm");
myFile << "P3\n";
myFile << dispWidth << " " << dispHeight << "\n";
myFile << "255\n";

startTime = MPI_Wtime(); 

for (int i = 1; i < npes; i++)
{
MPI_Send(&row, 1, MPI_INT, i, DATA_TAG, MPI_COMM_WORLD);
row++;
}

int doneRows = 0;

while (doneRows < dispHeight)
{
MPI_Recv(&row_colors, dispWidth + 1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);

int doneSlave = status.MPI_SOURCE;
int receivedRow = row_colors[0]; 

for (int col = 0; col < dispWidth; col++) 
{
coords[receivedRow][col] = row_colors[col + 1];
}

doneRows++;

if (row < dispHeight)
{
MPI_Send(&row, 1, MPI_INT, doneSlave, DATA_TAG, MPI_COMM_WORLD);
row++;
}

}

for (int i = 1; i < npes; i++) 
{
MPI_Send(0, 0, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
}

for (int x = 0; x < dispWidth; x++)
{
for (int y = 0; y < dispHeight; y++)
{
myFile << (coords[x][y] * coords[x][y]) % 126 << " " << (coords[x][y] * coords[x][y]) % 186 << " " << (coords[x][y]) % 16 << "   ";
}
myFile << endl;
}

endTime = MPI_Wtime();
cout << dispHeight << "x" << dispWidth << " con " << npes << " procesos tomo... " << (endTime - startTime) << " segundos" << endl;

myFile.close();
}
else 
{
float scale_real = (float)(x - neg_X) / dispWidth;
float scale_imag = (float)(y - neg_Y) / dispHeight;

complex z;
int slave_colors[dispWidth + 1];
int row = 0;
MPI_Status status;

MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

while (status.MPI_TAG == DATA_TAG)
{
if (status.MPI_TAG == TERMINATE_TAG) 
{
exit(0);
}

slave_colors[0] = row; 

z.imag = neg_Y + ((float)row * scale_imag);

#pragma omp parallel 
{
#pragma omp parallel for
for (int x = 0; x < dispWidth; x++) 
{
#pragma omp critical 
{
z.real = neg_X + ((float)x * scale_real);
slave_colors[x + 1] = cal_pixel(z);
}
}
}

MPI_Send(slave_colors, dispWidth + 1, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

}
}

MPI_Finalize();
return 0;
}