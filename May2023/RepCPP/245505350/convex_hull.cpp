#include <iostream>
#include <bitset>
#include <string>
#include <vector>
#include <thread>
#include <time.h>
#include <cmath>
#include <cfenv>
using namespace std;

#define POWER_OF_TWO 3 
#define HIGHEST_VALUE_IN_BYTE 256 
#define BIGGEST_SINGLE_BIT 128 
#define INFINITE 10000000

#pragma STDC FENV_ACCESS ON
double roundFloat(double x) {
std::fenv_t save_env;
std::feholdexcept(&save_env);
double result = std::rint(x);
if (std::fetestexcept(FE_INEXACT)) {
auto const save_round = std::fegetround();
std::fesetround(FE_TOWARDZERO);
result = std::rint(std::copysign(0.5 + std::fabs(x), x));
std::fesetround(save_round);
}
std::feupdateenv(&save_env);
return result;
}

typedef unsigned int IMAGE; 

class BitImage {
private:
struct Mask {
int A, B, C;
};

IMAGE *binimg;
int height, width, size, pixels;
int amount;

int bit = 1, bitMask = 1;
int byte = 0, bitInByte = 0;
int components = 0;
int** convex_hull = NULL;

string matrix = "";
std::vector<int*> result;

public:	
BitImage(int x, int y) {
if (x <= 0 || y <= 0) {
std::cout << "Either size cannot be less or equal 0" << std::endl;
exit(0);
}	
height = x;
width = y;
pixels = x * y;

size = (x * y) >> POWER_OF_TWO;

if (((x * y) % 8) != 0) {
size += 1;
}

binimg = new IMAGE[size];

for (int i = 0; i < size; i++) {
binimg[i] = 0;
std::bitset<8> bits(0);
matrix += bits.to_string();
}

int redundant = (size << POWER_OF_TWO) - pixels;

matrix.erase(matrix.size() - redundant);
};

void setUpAccess(int x, int y) {
bit = ((x - 1) * width) + y;
byte = 0;
bitMask = 1;
bitInByte = bit % 8;

byte = (bit >> POWER_OF_TWO);

if (bitInByte != 0) {
bitMask = BIGGEST_SINGLE_BIT >> (bitInByte - 1);
byte += 1;
}
}

void setRandom() {
srand(time(NULL)); 

int digit;
matrix = "";
for (int i = 0; i < size; i++) {
digit = rand() % HIGHEST_VALUE_IN_BYTE;
binimg[i] = digit;
std::bitset<8> bits(digit);
matrix += bits.to_string();
}
}

void printAll() {
printf("\nOur image");
for (int i = 0; i < pixels; i++) {
if ( (i % width) == 0 ) { putchar('\n'); }
printf("%c", matrix[i]);
}
putchar('\n');
}

void setPixel(int x, int y) {
if (x <= 0 || y <= 0 || pixels < (x * y)) {
cout << "You are accessing " << (x * y) << "th bit from only " << pixels << " available, ";
cout << "therefore it's a nonexistent bit, abort." << endl;
return;
}

setUpAccess(x, y);

cout << "Accessing " << byte << "th byte: " << std::bitset<8>(binimg[byte - 1]) << ", ";
cout << "setting up " << bit << "th bit:\t" << std::bitset<8>(bitMask) << endl;

binimg[byte - 1] |= bitMask;
matrix[bit - 1] = '1';
}

void fillImage() {
string s;
matrix = "";
for (int i = 0; i < size; i++) {
binimg[i] = HIGHEST_VALUE_IN_BYTE - 1; 
auto s = to_string(binimg[i]);
matrix += s;
}
}

void fillHalf() {
string s;
matrix = "";
for (int i = 0; i < size; i++) {
if (!(i % 4)) continue;
binimg[i] = HIGHEST_VALUE_IN_BYTE - 1; 
auto s = to_string(binimg[i]);
matrix += s;
}
}

bool getPixel(int x, int y) {
setUpAccess(x, y);

int nthByte = 7 - ((bit - 1) % 8);
std::bitset<8> ourByte(binimg[byte - 1]);

if (ourByte[nthByte]) return true;

return false;
}

double turnPoint(int* p1, int* p2, int* p3) {
double ax = p2[0] - p1[0];
double ay = p2[1] - p1[1];
double bx = p3[0] - p1[0];
double by = p3[1] - p1[1];

return (ax * bx + ay * by) / (sqrt(ax * ax + ay * ay) * sqrt(bx * bx + by * by));
}

double betweenPoints(int* p1, int* p2) {
return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]));
}
void findConnectedComponents() {
int kWidth = 0, kHeight = 0;
string s; 

amount = 1;
Mask localMask;

for (int i = 1; i <= width; i++) {
for (int j = 1; j <= height; j++) {
kWidth = j - 1;
if (kWidth <= 0) {
kWidth = 1;
localMask.B = 0;
} else {
localMask.B = getPixel(i, kWidth);
}

kHeight = i - 1;
if (kHeight <= 0) {
kHeight = 1;
localMask.C = 0;
} else {
localMask.C = getPixel(kHeight, j);
}

localMask.A = getPixel(i, j);

if (localMask.A == 0) {
} else if (localMask.B == 0 && localMask.C == 0) {
components += 1;
amount += 1;
s = to_string(amount);
matrix[((i - 1) * width) + (j - 1)] = s[0];	
} else if (localMask.B != 0 && localMask.C == 0) {
s = to_string(localMask.B);
matrix[((i - 1) * width) + (j - 1)] = s[0];
} else if (localMask.B == 0 && localMask.C != 0) {
s = to_string(localMask.C);
matrix[((i - 1) * width) + (j - 1)] = s[0];
} else if (localMask.B != 0 && localMask.C != 0) {
s = to_string(localMask.B);
matrix[((i - 1) * width) + (j - 1)] = s[0];

if (localMask.B != localMask.C) {
for (int a = 0; a < i; a++) {
for (int b = 0; b < height; b++) {
if (matrix[(a * width) + b] == localMask.C) {
s = to_string(localMask.B);
matrix[(a * width) + b] = s[0];	
}
}
}
}
}
}
}
}

void makeConvex() {
int tempo = 0;
convex_hull = new int*[components];

for (int i = 0; i < components; i++) {
convex_hull[i] = new int[3];
}

for (int i = 0; i < width; i++) {
for (int j = 0; j < height; j++) {

char access = matrix[(i * width) + j];
if (access != '0') {
convex_hull[tempo][0] = j;
convex_hull[tempo][1] = i;

auto s = (int)access;
convex_hull[tempo][2] = s;

if (tempo >= components - 1) break;
tempo += 1;
}
}
}
}

void Jarvis__STD(const int num_threads) {
int m = 0, minind = 0;
double mincos, cosine;
double len = 0, maxlen = 0;

if (components == 1) {
result.push_back(convex_hull[0]);
return;
}
if (components == 2) {
result.push_back(convex_hull[0]);
result.push_back(convex_hull[1]);
return;
}

double* first_elements = new double[2];
first_elements[0] = convex_hull[0][0];
first_elements[1] = convex_hull[0][1];

for (int i = 1; i < components; i++) {
if (convex_hull[i][1] < first_elements[1]) {
m = i;
} else {
if ((convex_hull[i][1] == first_elements[1]) && (convex_hull[i][0] < first_elements[0])) {
m = i;
}
}
}

result.push_back(convex_hull[m]);

int* last = new int[2];
int* beforelast = new int[2];

last = convex_hull[m];
beforelast[0] = convex_hull[m][0] - 2;
beforelast[1] = convex_hull[m][1];

for(;;) {
mincos = 2;
int step_i = 0;
auto fq = [&]() {
int core = step_i++;
const int start = (core * components) / num_threads;
const int finish = (core + 1) * components / num_threads;

for (int i = start; i < finish; i++) {
cosine = roundFloat(turnPoint(last, beforelast, convex_hull[i]) * INFINITE) / INFINITE;
if (cosine < mincos) {
minind = i;
mincos = cosine;
maxlen = betweenPoints(last, convex_hull[i]);
} else if (cosine == mincos) {
len = betweenPoints(last, convex_hull[i]);
if (len > maxlen) {
minind = i;
maxlen = len;
}
}
}

beforelast = last;
last = convex_hull[minind];
};

thread* ourThreads = new thread[num_threads];
for (int i = 0; i < num_threads; i++) {
ourThreads[i] = thread(fq);
}
for (int i = 0; i < num_threads; i++) {
ourThreads[i].join();
}

delete[]ourThreads;

if (last == convex_hull[m])
break;
result.push_back(convex_hull[minind]);
}
}

void printResult() {
int* temp = NULL;

for (int i = 0; i < result.size(); i++) {
temp = result[i];
}
putchar('\n');
for (int i = 0; i < sizeof(temp)/sizeof(temp[0]); i++)
cout << "(" << temp[i] << ")";

putchar('\n');
}

void findConnectedComponents__Threads(const int num_threads) { 
int kWidth = 0, kHeight = 0;
string s; 

amount = 1;
Mask localMask;

for (int i = 1; i <= width; i++) {
for (int j = 1; j <= height; j++) {
kWidth = j - 1;
if (kWidth <= 0) {
kWidth = 1;
localMask.B = 0;
} else {
localMask.B = getPixel(i, kWidth);
}

kHeight = i - 1;
if (kHeight <= 0) {
kHeight = 1;
localMask.C = 0;
} else {
localMask.C = getPixel(kHeight, j);
}

localMask.A = getPixel(i, j);

if (localMask.A == 0) {
} else if (localMask.B == 0 && localMask.C == 0) {
components += 1;
amount += 1;
s = to_string(amount);
matrix[((i - 1) * width) + (j - 1)] = s[0];	
} else if (localMask.B != 0 && localMask.C == 0) {
matrix[((i - 1) * width) + (j - 1)] = (char)localMask.B;
} else if (localMask.B == 0 && localMask.C != 0) {
matrix[((i - 1) * width) + (j - 1)] = (char)localMask.C;
} else if (localMask.B != 0 && localMask.C != 0) {
matrix[((i - 1) * width) + (j - 1)] = (char)localMask.B;

if (localMask.B != localMask.C) {
int step_i = 0;
auto f = [&]() {
int core = step_i++;
const int start = (core * i) / num_threads;
const int finish = (core + 1) * i / num_threads;
for (int a = start; a < finish; a++) {
for (int b = 0; b < height; b++) {
if (matrix[(a * width) + b] == localMask.C) {
matrix[(a * width) + b] = (char)localMask.B;	
}
}
}
};
thread* ourThreads = new thread[num_threads];
for (int i = 0; i < num_threads; i++) {
ourThreads[i] = thread(f);
}
for (int i = 0; i < num_threads; i++) {
ourThreads[i].join();
}
delete[]ourThreads;
}

}
}
}
}
void Jarvis() {
int m = 0, minind = 0;
double mincos, cosine;
double len = 0, maxlen = 0;

if (components == 1) {
result.push_back(convex_hull[0]);
return;
}
if (components == 2) {
result.push_back(convex_hull[0]);
result.push_back(convex_hull[1]);
return;
}

double* first_elements = new double[2];
first_elements[0] = convex_hull[0][0];
first_elements[1] = convex_hull[0][1];

for (int i = 1; i < components; i++) {
if (convex_hull[i][1] < first_elements[1]) {
m = i;
} else {
if ((convex_hull[i][1] == first_elements[1]) && (convex_hull[i][0] < first_elements[0])) {
m = i;
}
}
}

result.push_back(convex_hull[m]);

int* last = new int[2];
int* beforelast = new int[2];

last = convex_hull[m];
beforelast[0] = convex_hull[m][0] - 2;
beforelast[1] = convex_hull[m][1];

for(;;) {
mincos = 2;
for (int i = 0; i < components; i++) {
cosine = roundFloat(turnPoint(last, beforelast, convex_hull[i]) * INFINITE) / INFINITE;
if (cosine < mincos) {
minind = i;
mincos = cosine;
maxlen = betweenPoints(last, convex_hull[i]);
} else if (cosine == mincos) {
len = betweenPoints(last, convex_hull[i]);
if (len > maxlen) {
minind = i;
maxlen = len;
}
}
}

beforelast = last;
last = convex_hull[minind];
if (last == convex_hull[m])
break;
result.push_back(convex_hull[minind]);
}
}
~BitImage() {
matrix.clear();
}
};

int main() {
int height, width;

cout << "We represent our binary image as a packed one-dimensional array in which each pixel is a bit." << endl;




width = 4500;
height = 4500;


BitImage be(height, width);
BitImage ae(height, width);

be.setRandom(); 
ae = be;

int num_threads = 1;

clock_t begin = clock();
be.findConnectedComponents__Threads(num_threads);
be.makeConvex();
be.Jarvis();
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

be.printResult();

cout << "(Threads) Time spent on finding connected components and making the convex hull is: " << time_spent << " seconds" << endl;

begin = clock();
ae.findConnectedComponents();
ae.makeConvex();
ae.Jarvis();
end = clock();
time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

ae.printResult();
cout << "(Seq) Time spent on finding connected components and making the convex hull is: " << time_spent << " seconds" << endl;

return 0;
}
