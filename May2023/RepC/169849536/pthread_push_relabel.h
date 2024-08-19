#pragma once
int push_relabel(int num_threads, int N, int src, int sink, int *cap, int *flow);
namespace utils {
int idx(int x, int y, int n);
}