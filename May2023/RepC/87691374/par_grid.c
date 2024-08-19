#include "par_grid.h"
int main(int argc, char* argv[]){
char* file;             
int generations = 0;    
int cube_size = 0;      
GraphNode*** graph;     
omp_lock_t** graph_lock;
int g, i, j;
GraphNode* it;
int live_neighbours;
parseArgs(argc, argv, &file, &generations);
debug_print("ARGS: file: %s generations: %d.", file, generations);
graph = parseFile(file, &cube_size);
graph_lock = (omp_lock_t**)malloc(cube_size * sizeof(omp_lock_t*));
for(i = 0; i < cube_size; i++){
graph_lock[i] = (omp_lock_t*) malloc(cube_size * sizeof(omp_lock_t));
for(j = 0; j < cube_size; j++){
omp_init_lock(&(graph_lock[i][j]));
}
}
double start = omp_get_wtime();  
for(g = 1; g <= generations; g++){
#pragma omp parallel
{
#pragma omp for private(i, j, it)   
for(i = 0; i < cube_size; i++){
for(j = 0; j < cube_size; j++){
for(it = graph[i][j]; it != NULL; it = it->next){
if(it->state == ALIVE)
visitNeighbours(graph, graph_lock, cube_size, i, j, it->z);
}
}
}
#pragma omp for private(i, j, it, live_neighbours)
for(i = 0; i < cube_size; i++){
for(j = 0; j < cube_size; j++){
for (it = graph[i][j]; it != NULL; it = it->next){
live_neighbours = it->neighbours;
it->neighbours = 0;
if(it->state == ALIVE){
if(live_neighbours < 2 || live_neighbours > 4){
it->state = DEAD;
}  
}else{
if(live_neighbours == 2 || live_neighbours == 3){
it->state = ALIVE; 
}
}
}
}
}
if(g % REMOVAL_PERIOD == 0){
#pragma omp for private(i, j)
for(i = 0; i < cube_size; i++){
for(j = 0; j < cube_size; j++){
GraphNode ** list = &graph[i][j];
graphListCleanup(list);
}
}
}
}
} 
double end = omp_get_wtime();   
printAndSortActive(graph, cube_size);
time_print("%f\n", end - start);
for(i = 0; i < cube_size; i++){
for(j = 0; j<cube_size; j++){
omp_destroy_lock(&(graph_lock[i][j]));
}
free(graph_lock[i]);
}
free(graph_lock);
freeGraph(graph, cube_size);
free(file);
}
void visitNeighbours(GraphNode*** graph, omp_lock_t** graph_lock, int cube_size, coordinate x, coordinate y, coordinate z){
GraphNode* ptr;
coordinate x1, x2, y1, y2, z1, z2;
x1 = (x+1)%cube_size; x2 = (x-1) < 0 ? (cube_size-1) : (x-1);
y1 = (y+1)%cube_size; y2 = (y-1) < 0 ? (cube_size-1) : (y-1);
z1 = (z+1)%cube_size; z2 = (z-1) < 0 ? (cube_size-1) : (z-1);
graphNodeAddNeighbour(&(graph[x1][y]), z, &(graph_lock[x1][y]));
graphNodeAddNeighbour(&(graph[x2][y]), z, &(graph_lock[x2][y]));
graphNodeAddNeighbour(&(graph[x][y1]), z, &(graph_lock[x][y1]));
graphNodeAddNeighbour(&(graph[x][y2]), z, &(graph_lock[x][y2]));
graphNodeAddNeighbour(&(graph[x][y]), z1, &(graph_lock[x][y]));
graphNodeAddNeighbour(&(graph[x][y]), z2, &(graph_lock[x][y]));
}
GraphNode*** initGraph(int size){
int i,j;
GraphNode*** graph = (GraphNode***) malloc(sizeof(GraphNode**) * size);
for (i = 0; i < size; i++){
graph[i] = (GraphNode**) malloc(sizeof(GraphNode*) * size);
for (j = 0; j < size; j++){
graph[i][j] = NULL;
}
}
return graph;
}
void freeGraph(GraphNode*** graph, int size){
int i, j;
if (graph != NULL){
for (i = 0; i < size; i++){
for (j = 0; j < size; j++){
graphNodeDelete(graph[i][j]);
}
free(graph[i]);
}
free(graph);
}
}
void printAndSortActive(GraphNode*** graph, int cube_size){
int x,y;
GraphNode* it;
for (x = 0; x < cube_size; ++x){
for (y = 0; y < cube_size; ++y){
graphNodeSort(&(graph[x][y]));
for (it = graph[x][y]; it != NULL; it = it->next){    
if (it->state == ALIVE)
out_print("%d %d %d\n", x, y, it->z);
}
}
}
}
void parseArgs(int argc, char* argv[], char** file, int* generations){
if (argc == 3){
char* file_name = malloc(sizeof(char) * (strlen(argv[1]) + 1));
strcpy(file_name, argv[1]);
*file = file_name;
*generations = atoi(argv[2]);
if (*generations > 0 && file_name != NULL)
return;
}    
printf("Usage: %s [data_file.in] [number_generations]", argv[0]);
exit(EXIT_FAILURE);
}
GraphNode*** parseFile(char* file, int* cube_size){
int first = 0;
char line[BUFFER_SIZE];
int x, y, z;
FILE* fp = fopen(file, "r");
if(fp == NULL){
err_print("Please input a valid file name");
exit(EXIT_FAILURE);
}
GraphNode*** graph;
while(fgets(line, sizeof(line), fp)){
if(!first){
if(sscanf(line, "%d\n", cube_size) == 1){
first = 1;
graph = initGraph(*cube_size);
}    
}else{
if(sscanf(line, "%d %d %d\n", &x, &y, &z) == 3){
graph[x][y] = graphNodeInsert(graph[x][y], z, ALIVE);
}
}
}
fclose(fp);
return graph;
}
