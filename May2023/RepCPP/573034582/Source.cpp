#include <iostream>
#include <sstream>
#include <omp.h>
#include "BMPFileRW.h"

using namespace std;

template <typename t>
void shellsort(t*& arr_def, int size);

template <typename t>
void qsortSection(t*& arr, int start, int end);

double median_filtering(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int imWidth, int  imHeight, int RH, int RW, string sorttype)
{
double time_start = omp_get_wtime();
#pragma omp parallel for
for (int y = 0; y < imHeight; y++)
{
int size = (RH * 2 + 1) * (RW * 2 + 1);
int* MEDMAS_R = new int[size], * MEDMAS_G = new int[size], * MEDMAS_B = new int[size];

for (int x = 0; x < imWidth; x++)
{
int Masind = 0;
for (int DY = -RH; DY <= RH; DY++)
{
int KY = y + DY;
if (KY < 0)
KY = 0;
if (KY > imHeight - 1)
KY = imHeight - 1;
for (int DX = -RW; DX <= RW; DX++)
{
int KX = x + DX;
if (KX < 0)
KX = 0;
if (KX > imWidth - 1)
KX = imWidth - 1;
MEDMAS_R[Masind] = rgb_in[KY][KX].rgbtRed;
MEDMAS_G[Masind] = rgb_in[KY][KX].rgbtGreen;
MEDMAS_B[Masind] = rgb_in[KY][KX].rgbtBlue;
Masind++;
}
}

if (sorttype == "shellsort")
{
shellsort(MEDMAS_R, size);
shellsort(MEDMAS_G, size);
shellsort(MEDMAS_B, size);
}
else if (sorttype == "qsortSection")
{
qsortSection(MEDMAS_R, 0, size);
qsortSection(MEDMAS_G, 0, size);
qsortSection(MEDMAS_B, 0, size);
}
rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];
}
delete[] MEDMAS_R; delete[] MEDMAS_G; delete[] MEDMAS_B;
}
return omp_get_wtime() - time_start;
}

double median_Section(int start, RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int imWidth, int  imHeight, int RH, int RW, string sorttype)
{
double time_start = omp_get_wtime();

for (int y = start; y < imHeight; y++) {
int size = (RH * 2 + 1) * (RW * 2 + 1);
int* MEDMAS_R = new int[size], * MEDMAS_G = new int[size], * MEDMAS_B = new int[size];

for (int x = 0; x < imWidth; x++)
{
int Masind = 0;
for (int DY = -RH; DY <= RH; DY++)
{
int KY = y + DY;
if (KY < 0)
KY = 0;
if (KY > imHeight - 1)
KY = imHeight - 1;
for (int DX = -RW; DX <= RW; DX++)
{
int KX = x + DX;
if (KX < 0)
KX = 0;
if (KX > imWidth - 1)
KX = imWidth - 1;
MEDMAS_R[Masind] = rgb_in[KY][KX].rgbtRed;
MEDMAS_G[Masind] = rgb_in[KY][KX].rgbtGreen;
MEDMAS_B[Masind] = rgb_in[KY][KX].rgbtBlue;
Masind++;
}
}

if (sorttype == "shellsort")
{
shellsort(MEDMAS_R, size);
shellsort(MEDMAS_G, size);
shellsort(MEDMAS_B, size);
}
else if (sorttype == "qsortSection")
{
qsortSection(MEDMAS_R, 0, size);
qsortSection(MEDMAS_G, 0, size);
qsortSection(MEDMAS_B, 0, size);
}
rgb_out[y][x].rgbtRed = MEDMAS_R[size / 2];
rgb_out[y][x].rgbtGreen = MEDMAS_G[size / 2];
rgb_out[y][x].rgbtBlue = MEDMAS_B[size / 2];
}
delete[] MEDMAS_R; delete[] MEDMAS_G; delete[] MEDMAS_B;
}
return omp_get_wtime() - time_start;
}

double median_filteringSection(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int imWidth, int  imHeight, int RH, int RW, string sorttype)
{
double t_start = omp_get_wtime();
int n_t = omp_get_max_threads();
int st = 0;
int s1 = imHeight / n_t;

int s2 = imHeight * 2 / n_t;
int s3 = imHeight * 3 / n_t;
int se = imHeight;
#pragma omp parallel sections
{
#pragma omp section
{
median_Section(st, rgb_in, rgb_out, imWidth, s1, RH, RW, sorttype);
}
#pragma omp section
{
if (n_t > 1)
median_Section(s1, rgb_in, rgb_out, imWidth, s2, RH, RW, sorttype);

}
#pragma omp section
{
if (n_t > 2)
median_Section(s2, rgb_in, rgb_out, imWidth, s3, RH, RW, sorttype);

}
#pragma omp section
{
if (n_t > 3)
median_Section(s3, rgb_in, rgb_out, imWidth, se, RH, RW, sorttype);

}
}
return omp_get_wtime() - t_start;
}

template <typename t>
void insertionsort(t*& a, int n, int stride)
{
for (int j = stride; j < n; j += stride)
{
int key = a[j];
int i = j - stride;
while (i >= 0 && a[i] > key)
{
a[i + stride] = a[i];
i -= stride;
}
a[i + stride] = key;
}
}

template <typename t>
void shellsort(t*& a, int n)
{
int i, m;

for (m = n / 2; m > 0; m /= 2)
{
#pragma omp parallel for shared(a,m,n) private (i) default(none)
for (i = 0; i < m; i++)
insertionsort(a, n - i, m);
}
}

template <typename t>
int partition(t*& arr, int start, int end)
{
int pivot = arr[end];
int i = (start - 1);

for (int j = start; j <= end - 1; j++) {
if (arr[j] < pivot) {
i++;
swap(arr[i], arr[j]);
}
}
swap(arr[i + 1], arr[end]);

return (i + 1);
}

template <typename t>
void qsortSection(t*& arr, int start, int end)
{
int index;

if (start < end) {

index = partition(arr, start, end);

#pragma omp parallel sections
{
#pragma omp section
{
qsortSection(arr, start, index - 1);
}
#pragma omp section
{
qsortSection(arr, index + 1, end);
}
}
}
}

int main()
{

double* times = new double[2];

for (int dataset = 1; dataset <= 4; dataset++)
{
stringstream ss1;
ss1 << dataset;

RGBTRIPLE** rgb_in, ** rgb_out;
BITMAPFILEHEADER header;
BITMAPINFOHEADER bmiHeader;
int imWidth = 0, imHeight = 0;

BMPRead(rgb_in, header, bmiHeader, "c:\\temp\\sample" + ss1.str() + ".bmp");

imWidth = bmiHeader.biWidth;
imHeight = bmiHeader.biHeight;

cout << "\nImage params:" << imWidth << "x" << imHeight << endl;

rgb_out = new RGBTRIPLE * [imHeight];
rgb_out[0] = new RGBTRIPLE[imWidth * imHeight];

for (int i = 1; i < imHeight; i++)
{
rgb_out[i] = &rgb_out[0][imWidth * i];
}


for (int threads = 1; threads <= 4; threads++)
{
omp_set_num_threads(threads);
cout << "\nТекущее число потоков " << omp_get_max_threads() << endl;
cout << "Набор данных " << dataset << endl;

stringstream ss2;
ss2 << omp_get_max_threads();

for (int RH = 3, RW = 3; RH <= 7; RH += 2, RW += 2)
{
stringstream ss3;
ss3 << RW;
times[0] = median_filteringSection(rgb_in, rgb_out, imWidth, imHeight, RH, RW, "shellsort");
BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_shellsort.bmp");

times[1] = median_filteringSection(rgb_in, rgb_out, imWidth, imHeight, RH, RW, "qsortSection");
BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_qsortSection.bmp");
cout << "\nДля RH = RW = " << RH << " время shellsort, qsortSection составило" << endl;
for (int i = 0; i < 2; i++)
cout << times[i] * 1000 << " ";
}
}
}

return 0;
}

