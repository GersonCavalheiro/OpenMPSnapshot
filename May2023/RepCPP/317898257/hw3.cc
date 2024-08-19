#include <fstream>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#define INF 1073741823
int main(int argc, char** argv) {
int i, j, *result, len;
struct stat buf;
int fd = open(argv[1], O_RDONLY);
fstat(fd,&buf);
len = (int)buf.st_size;
result = (int*)mmap(0,len,PROT_READ,MAP_FILE|MAP_PRIVATE,fd,0);
int V = result[0];
int E = result[1];
int **dist = (int**)calloc(V, sizeof(int*));
for(int i=0;i<V;i++)
dist[i] = (int*)calloc(V, sizeof(int));
for (i = 0; i < V; ++i)
for (j = 0; j < V; ++j)
if(i!=j)
dist[i][j]=INF;
for (int i = 2; i < 3*E+2; i+=3)
dist[result[i]][result[i+1]] = result[i+2];
for(int k=0; k<V; k++){
#pragma omp parallel for num_threads(omp_get_max_threads())
for(int i=0;i<V;i++)
for(int j=0;j<V;j++)
if(dist[i][k] + dist[k][j] < dist[i][j])
dist[i][j] = dist[i][k] + dist[k][j];
}
FILE *fp = fopen( argv[2] , "wb" );
for (int i = 0; i < V; ++i)
fwrite(dist[i], sizeof(int), V, fp);
}
