#include "openmpScc.h"
int sccCounter;
bool changedColor;
CooArray* readMtxFile(char* filename){
int ret_code;
MM_typecode matcode;
FILE *f;
fpos_t pos;
int M, N, nz;   
int i, *I, *J;
double* val;
char filepath[50];
strcpy(filepath, "../graphs/");
strcat(filepath, filename);
strcat(filepath, ".mtx");
int numOfCols = 1;
if ((f = fopen(filepath, "r")) == NULL){
printf("File with name <%s> not found in graphs/", filename);
exit(1);
}
if (mm_read_banner(f, &matcode) != 0){
printf("Could not process Matrix Market banner.\n");
exit(1);
}
if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ){
printf("Sorry, this application does not support ");
printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
exit(1);
}
if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0){
exit(1);
}
fgetpos(f, &pos);
char str[150];
if(fgets(str,150, f) == NULL){
printf("Error: Cannot read graph!\n");
}
i=0;
while(i<=str[i]){
if(str[i]==' '){
numOfCols++;
}
i++;
}
fsetpos(f, &pos);
I = (int*) malloc(nz * sizeof(int));
J = (int*) malloc(nz * sizeof(int));
val = (double *) malloc(nz * sizeof(double));
if(numOfCols == 3){
for(i=0; i<nz; i++){
if(fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) == 0){
printf("Error: Cannot read graph!\n");
}
I[i]--;  
J[i]--;
}
if (f !=stdin) fclose(f);
mm_write_banner(stdout, matcode);
mm_write_mtx_crd_size(stdout, M, N, nz);
}
else if(numOfCols == 2){
for(i=0; i<nz; i++){
if(fscanf(f, "%d %d\n", &I[i], &J[i]) == 0){
printf("Error: Cannot read graph!\n");
}
I[i]--;  
J[i]--;
}
if (f !=stdin) fclose(f);
mm_write_banner(stdout, matcode);
mm_write_mtx_crd_size(stdout, M, N, nz);
}
else{
printf("Error: Number of columns not 2 or 3!\n");
exit(1);
}
CooArray* cooArray = (CooArray*) malloc(sizeof(CooArray));
cooArray->i = I;
cooArray->j = J;
cooArray->iLength = nz;
cooArray->jLength = nz;
cooArray->numOfVertices = M;
printf("\n");
free(val);
return cooArray;
}
void calculateVertexDegrees(Graph* g){
#pragma omp parallel for
for(int i=0;i<g->endLength;i++){
int startId = g->startAll[i];
int endId = g->end[i];
if(g->vertices[startId] == -1 || g->vertices[endId] == -1) continue;
if(g->vertices[startId] == g->vertices[endId]) continue;
g->outDegree[startId]++;
g->inDegree[endId]++;
}
}
void trimGraph(Graph* g, int startingVertex, int endingVertex){
calculateVertexDegrees(g);
int sccTrimCounter = 0;
#pragma omp parallel for reduction(+:sccTrimCounter)
for(int i=startingVertex;i<endingVertex;i++){
if(g->vertices[i] == -1) continue;
if(g->inDegree[i] == 0 || g->outDegree[i] == 0){
deleteIndexfromArray(g->vertices, i);
g->sccIdOfVertex[i] = sccCounter + sccTrimCounter++;
}
g->inDegree[i] = 0;
g->outDegree[i] = 0;
}
sccCounter += sccTrimCounter;
g->numOfVertices -= sccTrimCounter;
}
Graph* initGraphFromCoo(CooArray* ca){
Graph* g = (Graph*) malloc(sizeof(Graph));
g->end = ca->i;
g->endLength = ca->iLength;
g->startAll = ca->j;
g->sccIdOfVertex = (int*) malloc(ca->numOfVertices * sizeof(int));
g->start = (int*) malloc(ca->jLength * sizeof(int));
g->startLength = 0;
g->startPointer = (int*) malloc(ca->jLength * sizeof(int));
g->startPointerLength = 0;
g->vertices = (int*) malloc(ca->numOfVertices * sizeof(int));
g->vertexPosInStart = (int*) malloc(ca->numOfVertices * sizeof(int));
g->verticesLength = ca->numOfVertices;
for(int i=0;i<g->verticesLength;i++){
g->vertices[i] = i;
g->vertexPosInStart[i] = -1;
}
g->inDegree = (int*) calloc(ca->numOfVertices, sizeof(int));
g->outDegree = (int*) calloc(ca->numOfVertices, sizeof(int));
int vid = -1;
for(int index=0;index<ca->jLength;index++){
if(vid != ca->j[index]){
vid = ca->j[index];
g->start[g->startLength] = vid;
g->vertexPosInStart[vid] = g->startLength;
g->startLength++;
g->startPointer[g->startPointerLength] = index;
g->startPointerLength++;
}
}
g->numOfVertices = g->verticesLength;
free(ca);
return g;
}
void initColor(Graph* g, int* vertexColor, int startingVertex, int endingVertex){
#pragma omp parallel for 
for(int i=startingVertex;i<endingVertex;i++){
vertexColor[i] = -1;
int vid = g->vertices[i];
if(vid != -1)
vertexColor[i] = vid;
}
}
void spreadColor(Graph* g, int* vertexColor, int startingVertex, int endingVertex){
#pragma omp parallel for 
for(int i=startingVertex;i<endingVertex;i++){
int vid = g->vertices[i];
if(vid == -1){
continue;
} 
int color = vertexColor[vid];
if(color == 0)
continue;
int startIndex = g->vertexPosInStart[vid];
int ifinish = startIndex + 1 < g->startPointerLength ? g->startPointer[startIndex+1] : g->endLength;
for(int endIndex=g->startPointer[startIndex];endIndex<ifinish;endIndex++){
int endvid = g->vertices[g->end[endIndex]];
if(endvid == -1){
continue;
}
int nextColor = vertexColor[endvid];
if(nextColor < color){
vertexColor[vid] = vertexColor[endvid];
changedColor = true;
}
}
}
}
void merge(int arr[], int l, int m, int r){
int i, j, k;
int n1 = m - l + 1;
int n2 = r - m;
int* L = (int*) malloc(n1 * sizeof(int));
int* R = (int*) malloc(n2 * sizeof(int));
for (i = 0; i < n1; i++)
L[i] = arr[l + i];
for (j = 0; j < n2; j++)
R[j] = arr[m + 1 + j];
i = 0; 
j = 0; 
k = l; 
while (i < n1 && j < n2) {
if (L[i] <= R[j]) {
arr[k] = L[i];
i++;
}
else {
arr[k] = R[j];
j++;
}
k++;
}
while (i < n1) {
arr[k] = L[i];
i++;
k++;
}
while (j < n2) {
arr[k] = R[j];
j++;
k++;
}
free(L);
free(R);
}
void mergeSort(int arr[], int l, int r){
if (l < r) {
int m = l + (r - l) / 2;
mergeSort(arr, l, m);
mergeSort(arr, m + 1, r);
merge(arr, l, m, r);
}
}
int* copyArray(int const* src, int len){
int* p = malloc(len * sizeof(int));
if(p == NULL)
printf("Error: malloc failed in copy array\n");
memcpy(p, src, len * sizeof(int));
return p;
}
Array* findUniqueColors(int* vertexColor, int size){
Array* uniqueColors = (Array*) malloc(sizeof(Array));
uniqueColors->arr = (int*) malloc(size * sizeof(int));
uniqueColors->length = 0;
int* temp = copyArray(vertexColor, size);
mergeSort(temp, 0, size - 1);
for(int i=0;i<size;i++){
int color = temp[i];
if(color == -1){
continue;
}
while(i < size - 1 && temp[i] == temp[i+1])
i++;
uniqueColors->arr[uniqueColors->length++] = color;
}
free(temp);
return uniqueColors;
}
void accessUniqueColors(Graph* g, Array* uc, int* vertexColor, int startingColor, int endingColor){
int sccUcCounter = 0;
int sccNumOfVertices = 0;
int n = g->verticesLength;
Queue* queueArr[endingColor - startingColor];
Array* sccArr[endingColor - startingColor];
#pragma omp parallel for reduction(+:sccUcCounter, sccNumOfVertices)
for(int i=startingColor;i<endingColor;i++){
int color = uc->arr[i];
Queue* queue = queueArr[i];
queue = (Queue*) malloc(sizeof(Queue));
queueInit(queue, n);
Array* scc = sccArr[i];
scc = (Array*) malloc(sizeof(Array));
scc->arr = (int*) malloc(n * sizeof(int));
scc->length = 0;
bfs(g, color, vertexColor, queue, scc);
if(scc->length > 0){
sccUcCounter++;
sccNumOfVertices += scc->length;
for(int j=0;j<scc->length;j++){
int vid = scc->arr[j];
g->sccIdOfVertex[vid] = sccCounter + sccUcCounter - 1;
deleteVertexFromGraph(g, vid);
}
}
else{
printf("Error: Did not find any SCCs for color=%d!\n", color);
exit(1);
}
free(queue->arr);
free(queue);
free(scc->arr);
free(scc);
}
sccCounter += sccUcCounter;
g->numOfVertices -= sccNumOfVertices;
}
int openmpColorScc(Graph* g, bool trimming){
sccCounter = 0;
int n = g->verticesLength;
int* vertexColor = (int*) malloc(n * sizeof(int));
while(g->numOfVertices > 0){
if(trimming){
trimGraph(g, 0, g->verticesLength);
}
initColor(g, vertexColor, 0, n);
changedColor = true;
while(changedColor){     
changedColor = false;
spreadColor(g, vertexColor, 0, n);
}
Array* uc = findUniqueColors(vertexColor, n);
accessUniqueColors(g, uc, vertexColor, 0, uc->length);
free(uc->arr);
free(uc);
}
free(vertexColor);
return sccCounter;
}