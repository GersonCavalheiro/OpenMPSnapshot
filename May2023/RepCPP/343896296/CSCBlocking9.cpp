


#include "CSCBlocking9.hpp"

int CSCBlocking9::GetBlockValue(CSCMatrix *M, int linBegin, int colBegin)
{



int value = 0;
int _a, _b, _c;

for (int i = 0; i < BLOCK_LINE_SIZE; i++)
{

int pStart = M->cscp[colBegin + 2-i];       
int pEnd = M->cscp[colBegin + 2-i + 1] - 1; 

if (pStart > pEnd)
{
continue;
}

_a = CSCBlocking9::BinarySearch(M->csci, pStart, pEnd, linBegin);

if (_a == -1)
{
_b = CSCBlocking9::BinarySearch(M->csci, pStart, pEnd, linBegin + 1);

if (_b == -1)
{

if (CSCBlocking9::BinarySearch(M->csci, pStart, pEnd, linBegin + 2) == -1)
{
continue;
}
else
{
value |= B001 << i*3;
continue;


}
}
else
{



if (_b + 1 <= pEnd && M->csci[_b + 1] == linBegin + 2)
{
value |= B011 << i*3;
continue;

}
}
}
else
{

if (_a + 1 <= pEnd && M->csci[_a + 1] == linBegin + 1)
{

if (_a + 2 <= pEnd && M->csci[_a + 2] == linBegin + 2)
{

value |= B111 << i*3;

continue;

}
else
{
value |= B110 << i*3;

continue;

}
}
else
{

if (_a + 1 <= pEnd && M->csci[_a + 1] == linBegin + 2)
{
value |= B101 << i*3;

continue;


}
else
{
value |= B100 << i*3;

continue;
}
}
}
}

return value;
}


int CSCBlocking9::GetFilterBlockValue(CSCMatrix* M, int linBegin, int colBegin){

int value = CSCBlocking9::GetBlockValue(M, linBegin, colBegin);

return value ^ 0x1FF;

}


int CSCBlocking9::BinarySearch(int *arr, int l, int r, int x)
{

if (r >= l)
{
int mid = l + (r - l) / 2;

if (arr[mid] == x)
return mid;

if (arr[mid] > x)
return CSCBlocking9::BinarySearch(arr, l, mid - 1, x);

return CSCBlocking9::BinarySearch(arr, mid + 1, r, x);
}

return -1;
}

void CSCBlocking9::AddCOOfromBlockValue(COOMatrix *M, int blockValue, int linBegin, int colBegin)
{

#if BLOCK_LINE_SIZE != 3
printf("[Error] Block size is not 3 (CSCBlocking)\m");
exit(EXIT_FAILURE);
#endif


bool aL = (blockValue >> 8) & 1;
bool bL = (blockValue >> 7) & 1;
bool cL = (blockValue >> 6) & 1;
bool dL = (blockValue >> 5) & 1;
bool eL = (blockValue >> 4) & 1;
bool fL = (blockValue >> 3) & 1;
bool gL = (blockValue >> 2) & 1;
bool hL = (blockValue >> 1) & 1;
bool iL = (blockValue)&1;

if (aL)
M->addPoint(linBegin, colBegin);

if (bL)
M->addPoint(linBegin + 1, colBegin);

if (cL)
M->addPoint(linBegin + 2, colBegin);

if (dL)
M->addPoint(linBegin, colBegin + 1);

if (eL)
M->addPoint(linBegin + 1, colBegin + 1);

if (fL)
M->addPoint(linBegin + 2, colBegin + 1);

if (gL)
M->addPoint(linBegin, colBegin + 2);

if (hL)
M->addPoint(linBegin + 1, colBegin + 2);

if (iL)
M->addPoint(linBegin + 2, colBegin + 2);
}
