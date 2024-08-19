#pragma once
class MergeSortTask
{
public:
MergeSortTask(int* buf, const int first, const int last);
void operator()();
private:
int* Buf;
int  First;
int  Last;
};
