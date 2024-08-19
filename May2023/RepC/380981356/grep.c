#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/wait.h>
#include <string.h>
#include <fcntl.h>
#define FILESIZE 4096
#define BLOCK 256
int NPROC;
int NTHRD;
typedef int filed;
typedef struct tNode* ptr;
typedef struct tNode {
filed info;    
ptr next;   
} Node ;
typedef ptr List;
ptr AlokNode(filed info) {
ptr P = (ptr) malloc (sizeof(Node));
P->info = info;
P->next = NULL;
return P;
}
ptr last(List L) {
ptr P = L;
while (P->next) P = P->next;
return P;
}
List listing(char * path) {
List L = NULL;
DIR * d = opendir(path);
if (d==NULL) return L;
struct dirent * dir; ptr P = AlokNode(0), Pinit = P; 
char d_path[512];
while ((dir = readdir(d)) != NULL) {
sprintf(d_path, "%s/%s", path, dir->d_name);
if (dir->d_type != DT_DIR) 
{
P->next = AlokNode(fileno(fopen(d_path, "r")));
P = P->next;
if (!L) L = P;
}
else if ((dir->d_type == DT_DIR) 
&& (strcmp(dir->d_name,".") != 0) 
&& (strcmp(dir->d_name,"..") != 0))
{
P->next = listing(d_path); 
if (P->next) {
P = last(P->next);
if (!L) L = P->next;
}
}
}
free(Pinit);
closedir(d);
return L;
}
int searchFile(char * text, char * pattern, int length) {
int i,j;
for(i = 0; i < length; i++) {
for(j = 0; pattern[j] && text[i+j]; j++) {
if (text[i+j] != pattern[j]) break;
}
if (pattern[j] == '\0') return 1; 
}
return 0; 
}
int printFile(int fd) {
char filePath[BLOCK];
char result[BLOCK];
sprintf(filePath, "/proc/self/fd/%d", fd);
memset(result, 0, sizeof(result));
readlink(filePath, result, sizeof(result));
printf("%s\n", result);
return fd;
}
int main(int argc, char ** args) {
char * STR;
DIR * dir;
char buffer[FILESIZE];
char * filename;
ptr P; List L;
int child, i;
filed fd; 
if (argc < 4) {
printf("Usage : ./grep n-process n-thread string\n");
exit(0);
}
NPROC = atoi(args[1]);
NTHRD = atoi(args[2]);
STR = args[3];
L = listing(".");
int start = omp_get_wtime();
P = L; child = 0;
while (P) {
pid_t cid;
if ((cid = fork()) == 0) {
goto child;
}
child++;
if (child == NPROC) {
int stat;
cid = wait(&stat);
child--;
}
P = P->next;
}
while(child > 0) {
int stat; pid_t cid;
cid = wait(&stat);
child--;
}
printf("Program ended in %f seconds.\n", omp_get_wtime()-start);
exit(0);
child:
fd = P->info;
FILE *fp = fdopen(fd, "rb");
fread(buffer, 1, FILESIZE, fp);
int len = strlen(STR);
int found = 0;
i = 0;
omp_set_num_threads(NTHRD);
#pragma omp parallel default(none) shared(STR, buffer, found, fd, len, i) 
#pragma omp single
while (buffer[i] && !found) {
#pragma omp task 
if (searchFile(&buffer[i], STR, BLOCK)) {
if (!found) {
found++;
printFile(fd);
}
}
i += BLOCK;
}
return 0;
}