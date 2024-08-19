
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <set>

#if !defined(NTHREADS)
#define NTHREADS 2
#endif

typedef struct elem_t {
int * data;
elem_t * next;
} elem_t;

namespace pchase {

void generateAndStoreRandomNumber(elem_t * elem) {
*(elem->data) = std::rand();
}

void process_list(elem_t *elem) {
#pragma omp parallel num_threads(NTHREADS)
{
#pragma omp single
{
while ( elem != NULL ) {
#pragma omp task firstprivate(elem)
{
generateAndStoreRandomNumber( elem );
}
elem = elem->next;
}
}
}
}

elem_t * createNode(elem_t *next) {
elem_t *elem = new elem_t;
elem->data = new int;
elem->next = next;
return elem;
}

elem_t * initialize(int n) {
elem_t *head = NULL;

int *duplicate = new int;
int pos1 = 0, pos2 = 0;
while (pos1 == pos2) {
pos1 = std::rand() % n;
pos2 = std::rand() % n;
}

for (int i = 0 ; i < n; i++) {
head = createNode(head);
if (i == pos1 || i == pos2) {
delete head->data;
head->data = duplicate;
}
}
return head;
}

void clean(elem_t *elem) {
std::set<int *> pointers;

while (elem) {
elem_t * tmp = elem->next;
if (pointers.find(elem->data) == pointers.end()) { 
pointers.insert(elem->data);
delete elem->data;
}
delete elem;
elem = tmp;
}
}

}; 

int main(int argc, char **argv) {

int size;
if (argc < 2) {
size = 14;
}
else {
size = atoi( argv[1] );
}
std::srand(std::time(nullptr));
printf("List size %d\n", size);
elem_t * elems_head = pchase::initialize(size);
pchase::process_list(elems_head);
pchase::clean(elems_head);
return 0;
}
