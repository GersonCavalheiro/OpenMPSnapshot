#include "par_grid_hash.h"
int main(int argc, char* argv[]){
char* input_name;           
int generations = 0;        
int cube_size = 0;          
GraphNode*** graph;         
Hashtable* hashtable;       
int g, i, j;
GraphNode* g_it = NULL;
Node* it = NULL;
omp_lock_t** graph_lock;
parseArgs(argc, argv, &input_name, &generations);
int initial_alive = getAlive(input_name);
hashtable = createHashtable(HASH_RATIO * initial_alive); 
graph = parseFile(input_name, hashtable, &cube_size);
debug_print("Hashtable: Occupation %.1f, Average %.2f elements per bucket", (hashtable->occupied*1.0) / hashtable->size, (hashtable->elements*1.0) /  hashtable->occupied);
graph_lock = (omp_lock_t**)malloc(cube_size * sizeof(omp_lock_t*));
for(i = 0; i < cube_size; i++){
graph_lock[i] = (omp_lock_t*) malloc(cube_size * sizeof(omp_lock_t));
for(j = 0; j < cube_size; j++){
omp_init_lock(&(graph_lock[i][j]));
}
}
double start = omp_get_wtime();  
for(g = 1; g <= generations; g++){
int num_alive = hashtable->elements;                        
Node** vector = (Node**) malloc(sizeof(Node*) * num_alive);
i = 0;
for(j = 0; j < hashtable->size; j++){
for (it = hashtable->table[j]; it != NULL; it = it->next){
vector[i++] = it;
}            
}
Node*** neighbour_vector = (Node***)malloc(sizeof(Node**) * num_alive);
#pragma omp parallel for private(i, j)
for(i = 0; i < num_alive; i++){
neighbour_vector[i] = (Node**)malloc(sizeof(Node*) * 6);
for(j = 0; j < 6; j++){
neighbour_vector[i][j] = NULL;
}
}
#pragma omp parallel for private(i, j)
for (i = 0; i < num_alive; i++){
coordinate x = vector[i]->x; 
coordinate y = vector[i]->y; 
coordinate z = vector[i]->z;
coordinate x1, x2, y1, y2, z1, z2;
x1 = (x+1) % cube_size; x2 = (x-1) < 0 ? (cube_size-1) : (x-1);
y1 = (y+1) % cube_size; y2 = (y-1) < 0 ? (cube_size-1) : (y-1);
z1 = (z+1) % cube_size; z2 = (z-1) < 0 ? (cube_size-1) : (z-1);
coordinate c[6][3] = {{x1,y,z}, {x2,y,z}, {x,y1,z}, {x,y2,z}, {x,y,z1}, {x,y,z2}};
GraphNode* ptr; 
for(j = 0; j < 6; j++){
if(graphNodeAddNeighbour( &(graph [c[j][X]] [c[j][Y]]), c[j][Z], &ptr, &(graph_lock [c[j][X]] [c[j][Y]]))){
neighbour_vector[i][j] = nodeInsert(NULL, c[j][X], c[j][Y], c[j][Z], ptr);
}else{
neighbour_vector[i][j] = NULL;
}
}
}
#pragma omp parallel for private(it, i, j)
for(i = 0; i < num_alive; i++){
it = vector[i];
unsigned char live_neighbours = it->ptr->neighbours;
it->ptr->neighbours = 0;
if(it->ptr->state == ALIVE){
if(live_neighbours < 2 || live_neighbours > 4){
it->ptr->state = DEAD;
graphNodeRemove(&(graph[it->x][it->y]), it->z, &(graph_lock[it->x][it->y]));
hashtableRemove(hashtable, it->x, it->y, it->z);
}                        
}
for(j = 0; j < 6; j++){
it = neighbour_vector[i][j];
if(it != NULL){
unsigned char live_neighbours = it->ptr->neighbours;
it->ptr->neighbours = 0;
if(it->ptr->state == DEAD){
if(live_neighbours == 2 || live_neighbours == 3){
it->ptr->state = ALIVE;
hashtableWrite(hashtable, it->x, it->y, it->z, it->ptr);
}
else{
graphNodeRemove(&(graph[it->x][it->y]), it->z, &(graph_lock[it->x][it->y]));
}
}
}
}
}
for(i = 0; i < num_alive; i++){
for(j = 0; j < 6; j++){
free(neighbour_vector[i][j]);
}
free(neighbour_vector[i]);
}
free(neighbour_vector);
free(vector);
}
double end = omp_get_wtime();   
printAndSortActive(graph, cube_size);
time_print(" %f\n", end - start);
freeGraph(graph, cube_size);
hashtableFree(hashtable);    
for(i = 0; i < cube_size; i++){
for(j=0; j<cube_size; j++){
omp_destroy_lock(&(graph_lock[i][j]));
}
}
free(input_name);
return 0;
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
void printSortedGraphToFile(GraphNode*** graph, int cube_size, char* input_name, int generations){
int x,y;
GraphNode* it;
char* output_name = generateOuputFilename(input_name, generations);
FILE* output = fopen(output_name, "w");
if (output == NULL){
err_print("Could not open output file");
exit(EXIT_FAILURE);
}
for (x = 0; x < cube_size; ++x){
for (y = 0; y < cube_size; ++y){
graphNodeSort(&(graph[x][y]));
for (it = graph[x][y]; it != NULL; it = it->next){    
fprintf(output, "%d %d %d\n", x, y, it->z);
}
}
}
free(output_name);
fclose(output);
}
char* generateOuputFilename(char* input_name, int generations){
char generation[GEN_BUFFER_SIZE];
sprintf(generation, "%d", generations);
char* in_ext = findLastDot(input_name);
int aux_length = strlen(input_name) - strlen(in_ext);
char* aux = malloc(sizeof(char) * (aux_length + 1));
strncpy(aux, input_name, aux_length);
aux[aux_length] = '\0';
char* output_name = malloc(sizeof(char) * (aux_length + strlen(generation) + strlen(OUT_EXT)) + 3);
sprintf(output_name, "%s.%s.%s", aux, generation, OUT_EXT);
free(aux);
return output_name;
}
char* findLastDot(char* str){
char *ptr = str;
char *dot = NULL;
while(*ptr++){
if (*ptr == '.')
dot = ptr;
}
return dot;
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
int getAlive(char* file){
FILE* fp = fopen(file, "r");
int alive_num = 0;
if (fp == NULL){
return EXIT_FAILURE;
}
while(!feof(fp)){
if(fgetc(fp) == '\n'){
alive_num++;
}
}
fclose(fp);
return alive_num - 1;
}
GraphNode*** parseFile(char* input_name, Hashtable* hashtable, int* cube_size){
int first = 0;
char line[BUFFER_SIZE];
int x, y, z;
FILE* fp = fopen(input_name, "r");
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
hashtableWrite(hashtable, x, y, z, (GraphNode*)(graph[x][y]));                
}
}
}
fclose(fp);
return graph;
}
