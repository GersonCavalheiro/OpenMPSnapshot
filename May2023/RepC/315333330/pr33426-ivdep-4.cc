#include <vector>
template<class T, class T2>
void Loop(T *b, T2 c) {
#pragma GCC ivdep
for (auto &i : *b) {
i *= *c;
}
}
void foo(std::vector<int> *ar, int *b) {
Loop<std::vector<int>, int*>(ar, b);
}
