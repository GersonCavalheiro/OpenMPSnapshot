#ifndef NODE
#define NODE

#include "dataset.hpp"
#include <array>
#include <mpi.h>

#define COUNT 10

using std::array;

template <typename T> struct node {
unsigned short axis{2};
array<T, 2> pnt;
unique_ptr<node> left_ptr;
unique_ptr<node> right_ptr;
#if defined(_OPENMP)
int left_prc{-1};
int right_prc{-1};
#endif

node() = default;
~node() = default;

void build(dataset<T> &data, size_t first, size_t last, unsigned short dir) {
size_t size = last - first + 1, mdn = 0;

if (size < 2) {
pnt[0] = data[first][0];
pnt[1] = data[first][1];
return;
}

axis = data.most_spreaded(first, last);

if (axis != dir) { 
data.sort(axis, first, last);
}

mdn = first + size / 2 - (size < 3);

pnt[0] = data[mdn][0];
pnt[1] = data[mdn][1];

#if defined(_OPENMP)
#pragma omp task shared(data)                                                  \
firstprivate(size, axis, first, mdn) 
#endif
if (size > 2) {
#if defined(_OPENMP)
MPI_Comm_rank(MPI_COMM_WORLD, &left_prc);
#endif
auto tmp = new node{};
tmp->build(data, first, mdn - (mdn > 0), axis);
left_ptr.reset(tmp);
}

#if defined(_OPENMP)
#pragma omp task shared(data) firstprivate(axis, last, mdn) 
#endif
{
#if defined(_OPENMP)
MPI_Comm_rank(MPI_COMM_WORLD, &right_prc);
#endif
auto tmp = new node{};
tmp->build(data, mdn + (mdn < last), last, axis);
right_ptr.reset(tmp);
}
}

friend ostream &operator<<(ostream &os, const unique_ptr<node> &x) {
if (x->axis != 2)
os << x->axis << " (" << x->pnt[0] << ", " << x->pnt[1] << ')';
else
os << '(' << x->pnt[0] << ", " << x->pnt[1] << ')';
return os;
}
};

template <typename T> node<T> *build(dataset<T> &data, unsigned short dir) {
auto tmp = new node<T>{};

#if defined(_OPENMP)
#pragma omp parallel shared(data) proc_bind(close)
#pragma omp single
#endif
tmp->build(data, 0, data.cardinality - 1, dir);

return tmp;
}

template <typename T> void print(const unique_ptr<T> &root) { print(root, 0); }

template <typename T>
void print(const unique_ptr<T> &root, unsigned short space) {
if (root == nullptr)
return;

space += COUNT;
print(root->right_ptr, space);
cout << endl;

for (unsigned short i = COUNT; i < space; i++)
cout << ' ';

cout << root  << endl;

print(root->left_ptr, space);
}

#if defined(_OPENMP)
template <typename T> int left(T *&head, int master) {
int recv_rank, rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == master)
recv_rank = head->left_prc;

MPI_Bcast(&recv_rank, 1, MPI_INT, master, MPI_COMM_WORLD);

if (rank == master && rank == recv_rank) {
head = head->left_ptr.get();
} else if (rank == recv_rank) {
auto shared_node = find_node(head, master, rank);
head = shared_node->left_ptr.get();
}

return recv_rank;
}

template <typename T> int right(T *&head, int master) {
int recv_rank = (head != nullptr) ? head->right_prc : -1, rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Bcast(&recv_rank, 1, MPI_INT, master, MPI_COMM_WORLD);

if (rank == master && rank == recv_rank) {
head = head->right_ptr.get();
return recv_rank;
} else if (rank == recv_rank) {
auto shared_node = find_node(head, master, rank);
head = shared_node->right_ptr.get();
return recv_rank;
}

return recv_rank;
}

template <typename T> T find_node(T head, int master, int slave) {
if (head->left_prc == master && head->right_prc == slave)
return head;

if (head->right_prc == master && head->left_prc == slave)
return head;

T common_node = nullptr;

if (head->left_ptr != nullptr)
common_node = find_node(head->left_ptr.get(), master, slave);

if (head->right_ptr != nullptr)
common_node = find_node(head->right_ptr.get(), master, slave);

return common_node;
}
#endif
#endif
