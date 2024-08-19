#pragma once
void Merge(int* buf, const int lb, const int ll, const int rb, const int rl);
class MergeTask
{
public:
MergeTask(int* buf, const int lb, const int ll, const int rb, const int rl);
void operator()();
private:
int* Buf;
int  Lb;
int  Ll;
int  Rb;
int  Rl;
};
