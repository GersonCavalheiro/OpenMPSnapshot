#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
const int INFINITY = 1000000;
const int NO_CITY = -1;
const int FALSE = 0;
const int TRUE = 1;
const int MAX_STRING = 1000;
typedef int city_t;
typedef int cost_t;
typedef struct {
city_t* cities; 
int count;      
cost_t cost;    
} tour_struct;
typedef tour_struct* tour_t;
#define City_count(tour) (tour->count)
#define Tour_cost(tour) (tour->cost)
#define Last_city(tour) (tour->cities[(tour->count)-1])
#define Tour_city(tour,i) (tour->cities[(i)])
typedef struct {
tour_t* list;
int list_sz;
int list_alloc;
}  stack_struct;
typedef stack_struct* my_stack_t;
typedef struct {
tour_t* list;
int list_alloc;
int head;
int tail;
int full;
}  queue_struct;
typedef queue_struct* my_queue_t;
#define Queue_elt(queue,i) \
(queue->list[(queue->head + (i)) % queue->list_alloc])
int n;  
int thread_count;
cost_t* digraph;
#define Cost(city1, city2) (digraph[city1*n + city2])
city_t home_town = 0;
tour_t best_tour;
my_queue_t queue;
int queue_size;
int init_tour_count;
void Usage(char* prog_name);
void Read_digraph(FILE* digraph_file);
void Print_digraph(void);
void Par_tree_search(void);
void Partition_tree(int my_rank, my_stack_t stack);
void Set_init_tours(int my_rank, int* my_first_tour_p,
int* my_last_tour_p);
void Build_initial_queue(void);
void Print_tour(int my_rank, tour_t tour, char* title);
int  Best_tour(tour_t tour); 
void Update_best_tour(tour_t tour);
void Copy_tour(tour_t tour1, tour_t tour2);
void Add_city(tour_t tour, city_t);
void Remove_last_city(tour_t tour);
int  Feasible(tour_t tour, city_t city);
int  Visited(tour_t tour, city_t city);
void Init_tour(tour_t tour, cost_t cost);
tour_t Alloc_tour(my_stack_t avail);
void Free_tour(tour_t tour, my_stack_t avail);
my_stack_t Init_stack(void);
void Push(my_stack_t stack, tour_t tour);  
void Push_copy(my_stack_t stack, tour_t tour, my_stack_t avail); 
tour_t Pop(my_stack_t stack);
int  Empty_stack(my_stack_t stack);
void Free_stack(my_stack_t stack);
void Print_stack(my_stack_t stack, int my_rank, char title[]);
my_queue_t Init_queue(int size);
tour_t Dequeue(my_queue_t queue);
void Enqueue(my_queue_t queue, tour_t tour);
int Empty_queue(my_queue_t queue);
void Free_queue(my_queue_t queue);
void Print_queue(my_queue_t queue, int my_rank, char title[]);
int Get_upper_bd_queue_sz(void);
long long Fact(int k);
int main(int argc, char* argv[]) {
FILE* digraph_file;
double start, finish;
if (argc != 3) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
if (thread_count <= 0) {
fprintf(stderr, "Thread count must be positive\n");
Usage(argv[0]);
}
digraph_file = fopen(argv[2], "r");
if (digraph_file == NULL) {
fprintf(stderr, "Can't open %s\n", argv[2]);
Usage(argv[0]);
}
Read_digraph(digraph_file);
fclose(digraph_file);
#  ifdef DEBUG
Print_digraph();
#  endif   
best_tour = Alloc_tour(NULL);
Init_tour(best_tour, INFINITY);
#  ifdef DEBUG
Print_tour(-1, best_tour, "Best tour");
printf("City count = %d\n",  City_count(best_tour));
printf("Cost = %d\n\n", Tour_cost(best_tour));
#  endif
start = omp_get_wtime();
#pragma omp parallel num_threads(thread_count) default(none)
Par_tree_search();
finish = omp_get_wtime();
Print_tour(-1, best_tour, "Best tour");
printf("Cost = %d\n", best_tour->cost);
printf("Elapsed time = %e seconds\n", finish-start);
free(best_tour->cities);
free(best_tour);
free(digraph);
return 0;
}  
void Init_tour(tour_t tour, cost_t cost) {
int i;
tour->cities[0] = 0;
for (i = 1; i <= n; i++) {
tour->cities[i] = NO_CITY;
}
tour->cost = cost;
tour->count = 1;
}  
void Usage(char* prog_name) {
fprintf(stderr, "usage: %s <thread_count> <digraph file>\n", prog_name);
exit(0);
}  
void Read_digraph(FILE* digraph_file) {
int i, j;
fscanf(digraph_file, "%d", &n);
if (n <= 0) {
fprintf(stderr, "Number of vertices in digraph must be positive\n");
exit(-1);
}
digraph = malloc(n*n*sizeof(cost_t));
for (i = 0; i < n; i++)
for (j = 0; j < n; j++) {
fscanf(digraph_file, "%d", &digraph[i*n + j]);
if (i == j && digraph[i*n + j] != 0) {
fprintf(stderr, "Diagonal entries must be zero\n");
exit(-1);
} else if (i != j && digraph[i*n + j] <= 0) {
fprintf(stderr, "Off-diagonal entries must be positive\n");
fprintf(stderr, "diagraph[%d,%d] = %d\n", i, j, digraph[i*n+j]);
exit(-1);
}
}
}  
void Print_digraph(void) {
int i, j;
printf("Order = %d\n", n);
printf("Matrix = \n");
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++)
printf("%2d ", digraph[i*n+j]);
printf("\n");
}
printf("\n");
}  
void Par_tree_search(void) {
int my_rank = omp_get_thread_num();
city_t nbr;
my_stack_t stack;  
my_stack_t avail;  
tour_t curr_tour;
avail = Init_stack();
stack = Init_stack();
Partition_tree(my_rank, stack);
while (!Empty_stack(stack)) {
curr_tour = Pop(stack);
#     ifdef DEBUG
Print_tour(my_rank, curr_tour, "Popped");
#     endif
if (City_count(curr_tour) == n) {
if (Best_tour(curr_tour)) {
#           ifdef DEBUG
Print_tour(my_rank, curr_tour, "Best tour");
#           endif
#pragma omp critical
Update_best_tour(curr_tour);
}
} else {
for (nbr = n-1; nbr >= 1; nbr--) 
if (Feasible(curr_tour, nbr)) {
Add_city(curr_tour, nbr);
Push_copy(stack, curr_tour, avail);
Remove_last_city(curr_tour);
}
}
Free_tour(curr_tour, avail);
}
Free_stack(stack);
Free_stack(avail);
#pragma omp barrier
#pragma omp master
Free_queue(queue);
}  
void Partition_tree(int my_rank, my_stack_t stack) {
int my_first_tour, my_last_tour, i;
#pragma omp single
queue_size = Get_upper_bd_queue_sz();
#  ifdef DEBUG
printf("Th %d > queue_size = %d\n", my_rank, queue_size);
#  endif
if (queue_size == 0) exit(0);
#pragma omp master
Build_initial_queue();
#pragma omp barrier
Set_init_tours(my_rank, &my_first_tour, &my_last_tour);
#  ifdef DEBUG
printf("Th %d > init_tour_count = %d, first = %d, last = %d\n", 
my_rank, init_tour_count, my_first_tour, my_last_tour);
#  endif
for (i = my_last_tour; i >= my_first_tour; i--) {
#     ifdef DEBUG
Print_tour(my_rank, Queue_elt(queue,i), "About to push");
#     endif
Push(stack, Queue_elt(queue,i));
}
#  ifdef DEBUG
Print_stack(stack, my_rank, "After set up");
#  endif
}  
void Set_init_tours(int my_rank, int* my_first_tour_p,
int* my_last_tour_p) {
int quotient, remainder, my_count;
quotient = init_tour_count/thread_count;
remainder = init_tour_count % thread_count;
if (my_rank < remainder) {
my_count = quotient+1;
*my_first_tour_p = my_rank*my_count;
} else {
my_count = quotient;
*my_first_tour_p = my_rank*my_count + remainder;
}
*my_last_tour_p = *my_first_tour_p + my_count - 1;
}   
void Build_initial_queue(void) {
int curr_sz = 0;
city_t nbr;
tour_t tour = Alloc_tour(NULL);
Init_tour(tour, 0);
queue = Init_queue(2*queue_size);
Enqueue(queue, tour);  
Free_tour(tour, NULL);
curr_sz++;
while (curr_sz < thread_count) {
tour = Dequeue(queue);
curr_sz--;
for (nbr = 1; nbr < n; nbr++)
if (!Visited(tour, nbr)) {
Add_city(tour, nbr);
Enqueue(queue, tour);
curr_sz++;
Remove_last_city(tour);
}
Free_tour(tour, NULL);
}  
init_tour_count = curr_sz; 
#  ifdef DEBUG
Print_queue(queue, 0, "Initial queue");
#  endif
}  
int Best_tour(tour_t tour) {
cost_t cost_so_far = Tour_cost(tour);
city_t last_city = Last_city(tour);
if (cost_so_far + Cost(last_city, home_town) < Tour_cost(best_tour))
return TRUE;
else
return FALSE;
}  
void Update_best_tour(tour_t tour) {
if (Best_tour(tour)) {
Copy_tour(tour, best_tour);
Add_city(best_tour, home_town);
}
}  
void Copy_tour(tour_t tour1, tour_t tour2) {
memcpy(tour2->cities, tour1->cities, (n+1)*sizeof(city_t));
tour2->count = tour1->count;
tour2->cost = tour1->cost;
}  
void Add_city(tour_t tour, city_t new_city) {
city_t old_last_city = Last_city(tour);
tour->cities[tour->count] = new_city;
(tour->count)++;
tour->cost += Cost(old_last_city,new_city);
}  
void Remove_last_city(tour_t tour) {
city_t old_last_city = Last_city(tour);
city_t new_last_city;
tour->cities[tour->count-1] = NO_CITY;
(tour->count)--;
new_last_city = Last_city(tour);
tour->cost -= Cost(new_last_city,old_last_city);
}  
int Feasible(tour_t tour, city_t city) {
city_t last_city = Last_city(tour);
if (!Visited(tour, city) && 
Tour_cost(tour) + Cost(last_city,city) < Tour_cost(best_tour))
return TRUE;
else
return FALSE;
}  
int Visited(tour_t tour, city_t city) {
int i;
for (i = 0; i < City_count(tour); i++)
if ( Tour_city(tour,i) == city ) return TRUE;
return FALSE;
}  
void Print_tour(int my_rank, tour_t tour, char* title) {
int i;
char string[MAX_STRING];
if (my_rank >= 0)
sprintf(string, "Th %d > %s %p: ", my_rank, title, tour);
else
sprintf(string, "%s: ", title);
for (i = 0; i < City_count(tour); i++)
sprintf(string + strlen(string), "%d ", Tour_city(tour,i));
printf("%s\n\n", string);
}  
tour_t Alloc_tour(my_stack_t avail) {
tour_t tmp;
if (avail == NULL || Empty_stack(avail)) {
tmp = malloc(sizeof(tour_struct));
tmp->cities = malloc((n+1)*sizeof(city_t));
return tmp;
} else {
return Pop(avail);
}
}  
void Free_tour(tour_t tour, my_stack_t avail) {
if (avail == NULL) {
free(tour->cities);
free(tour);
} else {
Push(avail, tour);
}
}  
my_stack_t Init_stack(void) {
int i;
my_stack_t stack = malloc(sizeof(stack_struct));
stack->list = malloc(n*n*sizeof(tour_t));
for (i = 0; i < n*n; i++)
stack->list[i] = NULL;
stack->list_sz = 0;
stack->list_alloc = n*n;
return stack;
}  
void Push(my_stack_t stack, tour_t tour) {
if (stack->list_sz == stack->list_alloc) {
fprintf(stderr, "Stack overflow in Push!\n");
free(tour->cities);
free(tour);
} else {
#     ifdef DEBUG
printf("In Push, list_sz = %d, pushing %p and %p\n",
stack->list_sz, tour, tour->cities);
Print_tour(-1, tour, "About to be pushed onto stack");
printf("\n");
#     endif
stack->list[stack->list_sz] = tour;
(stack->list_sz)++;
}
}  
void Push_copy(my_stack_t stack, tour_t tour, my_stack_t avail) {
tour_t tmp;
if (stack->list_sz == stack->list_alloc) {
fprintf(stderr, "Stack overflow!\n");
exit(-1);
}
tmp = Alloc_tour(avail);
Copy_tour(tour, tmp);
stack->list[stack->list_sz] = tmp;
(stack->list_sz)++;
}  
tour_t Pop(my_stack_t stack) {
tour_t tmp;
if (stack->list_sz == 0) {
fprintf(stderr, "Trying to pop empty stack!\n");
exit(-1);
}
tmp = stack->list[stack->list_sz-1];
stack->list[stack->list_sz-1] = NULL;
(stack->list_sz)--;
return tmp;
}  
int  Empty_stack(my_stack_t stack) {
if (stack->list_sz == 0)
return TRUE;
else
return FALSE;
}  
void Free_stack(my_stack_t stack) {
int i;
for (i = 0; i < stack->list_sz; i++) {
free(stack->list[i]->cities);
free(stack->list[i]);
}
free(stack->list);
free(stack);
}  
void Print_stack(my_stack_t stack, int my_rank, char title[]) {
char string[MAX_STRING];
int i, j;
printf("Th %d > %s\n", my_rank, title);
for (i = 0; i < stack->list_sz; i++) {
sprintf(string, "Th %d > ", my_rank);
for (j = 0; j < stack->list[i]->count; j++)
sprintf(string + strlen(string), "%d ", stack->list[i]->cities[j]);
printf("%s\n", string);
}
}  
my_queue_t Init_queue(int size) {
my_queue_t new_queue = malloc(sizeof(queue_struct));
new_queue->list = malloc(size*sizeof(tour_t));
new_queue->list_alloc = size;
new_queue->head = new_queue->tail = new_queue->full = 0;
return new_queue;
}  
tour_t Dequeue(my_queue_t queue) {
tour_t tmp;
if (Empty_queue(queue)) {
fprintf(stderr, "Attempting to dequeue from empty queue\n");
exit(-1);
}
tmp = queue->list[queue->head];
queue->head = (queue->head + 1) % queue->list_alloc;
return tmp;
}  
void Enqueue(my_queue_t queue, tour_t tour) {
tour_t tmp;
if (queue->full == TRUE) {
fprintf(stderr, "Attempting to enqueue a full queue\n");
fprintf(stderr, "list_alloc = %d, head = %d, tail = %d\n",
queue->list_alloc, queue->head, queue->tail);
exit(-1);
}
tmp = Alloc_tour(NULL);
Copy_tour(tour, tmp);
queue->list[queue->tail] = tmp;
queue->tail = (queue->tail + 1) % queue->list_alloc; 
if (queue->tail == queue->head)
queue->full = TRUE;
}  
int Empty_queue(my_queue_t queue) {
if (queue->full == TRUE)
return FALSE;
else if (queue->head != queue->tail)
return FALSE;
else
return TRUE;
}  
void Free_queue(my_queue_t queue) {
free(queue->list);
free(queue);
}  
void Print_queue(my_queue_t queue, int my_rank, char title[]) {
char string[MAX_STRING];
int i, j;
printf("Th %d > %s\n", my_rank, title);
for (i = queue->head; i != queue->tail; i = (i+1) % queue->list_alloc) {
sprintf(string, "Th %d > %p = ", my_rank, queue->list[i]);
for (j = 0; j < queue->list[i]->count; j++)
sprintf(string + strlen(string), "%d ", queue->list[i]->cities[j]);
printf("%s\n", string);
}
}  
int Get_upper_bd_queue_sz(void) {
int fact = n-1;
int size = n-1;
while (size < thread_count) {
fact++;
size *= fact;
}
if (size > Fact(n-1)) {
fprintf(stderr, "You really shouldn't use so many threads for");
fprintf(stderr, "such a small problem\n");
size = 0;
}
return size;
}  
long long Fact(int k) {
long long tmp = 1;
int i;
for (i = 2; i <= k; i++)
tmp *= i;
return tmp;
}  
