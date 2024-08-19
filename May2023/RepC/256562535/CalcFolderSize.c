#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
int isFile(char *path);
int isDir(char *path);
long long fileSize(char *path);
void concatPath(char *dest, char *p, char *fn);
long long newDir(char *path);
long long sum = 4096; 
int main(int argc, char *argv[])
{
if (argc != 2)
{
fprintf(stderr, "Directory argments neeed\n");
return EXIT_FAILURE;
}
#pragma omp parallel
{
#pragma omp single
{
printf("Threads %d\n", omp_get_num_threads());
newDir(argv[1]);
}
}
printf("Total %lld bytes\n", sum);
}
long long newDir(char *path)
{
long long localsum = 0;
DIR *dir;
struct dirent *dirzeiger;
if ((dir = opendir(path)) == NULL)
fprintf(stderr, "Fehler bei opendir ...\n");
while ((dirzeiger = readdir(dir)) != NULL)
{                                                                                       
if (strcmp((*dirzeiger).d_name, ".") == 0 || strcmp((*dirzeiger).d_name, "..") == 0) 
continue;
char nextPath[1024];
concatPath(nextPath, path, (*dirzeiger).d_name);
if (isDir(nextPath))
#pragma omp task
{
newDir(nextPath); 
}
localsum += fileSize(nextPath); 
}
#pragma omp atomic
sum += localsum;
if (closedir(dir) == -1)
printf("Fehler beim Schließen von %s\n", path);
}
int isFile(char *path)
{
struct stat sb;
if (lstat(path, &sb) == -1)
perror("stat");
if ((sb.st_mode & S_IFMT) == S_IFREG)
return 1;
return 0;
}
int isDir(char *path)
{
struct stat sb;
if (lstat(path, &sb) == -1)
perror("stat");
if ((sb.st_mode & S_IFMT) == S_IFDIR)
return 1;
return 0;
}
long long fileSize(char *path)
{
struct stat sb;
if (lstat(path, &sb) == -1)
perror("stat");
return (long long)sb.st_size;
}
void concatPath(char *dest, char *p, char *fn)
{ 
*dest = 0;
strcat(dest, p);
strcat(dest, "/");
strcat(dest, fn);
}
