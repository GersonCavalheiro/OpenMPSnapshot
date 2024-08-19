#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>
#include<dirent.h>
#include<time.h>
#define MAX_FILES 10
#define MAX_CHAR 2000
#define MAX_KEYS 200
#define MAX_KEY_LENGTH 200
struct node_t
{
char *data;
struct node_t* next;
};
typedef struct node_t node_t;
int getfile(FILE** files, char *dir_name)
{
DIR *dir = opendir(dir_name);
struct dirent *inside;
char filename[100];
int count = 0;
if(dir == NULL)
{
fprintf(stderr, "Can't open the directory.\n");
exit(EXIT_FAILURE);
}
while((inside = readdir(dir)) != NULL)
{
if(inside->d_type == DT_REG)
{
strcpy(filename, dir_name);
strcat(filename, inside->d_name);
files[count++] = fopen(filename, "r");
}
}
closedir(dir);
return count;
}
void tokenize(char *data, char* keywords[], int total_word, int key_count[])
{
int count;
char *saveptr = NULL;
char* line = NULL;
line = strtok_r(data, " ", &saveptr);
while(line != NULL)
{
for(count=0; count<total_word; count++)
{
if(strcmp(line, keywords[count]) == 0)
{
key_count[count]++;
break;
}
}
line = strtok_r(NULL, " ", &saveptr);
}
return;
}
void enqueue(char* line, node_t** head, node_t** tail)
{
node_t* new = malloc(sizeof(node_t));
new->data = line;
new->next = NULL;
if((*head) == NULL)
(*head) = (*tail) = new;
else
{
(*tail)->next = new;
(*tail) = new;
}
return;
}
void readfile(FILE* file, node_t** head, node_t** tail)
{
char buffer[MAX_CHAR];
char *line = NULL;
while(fgets(buffer, MAX_CHAR, file)!=NULL)
{
line = malloc(sizeof(char)*strlen(buffer)+1);
strncpy(line, buffer, strlen(buffer));
line[strlen(buffer)] = '\0';
enqueue(line, head, tail);
}
fclose(file);
return;
}
node_t* dequeue(node_t** head, node_t** tail)
{
node_t* temp = NULL;
if((*head) == NULL)
return NULL;
temp = (*head);
if((*head) == (*tail))
(*head) = (*tail) = NULL; 
else
(*head) = (*head)->next;
return temp;
}
void prod_con(int total_threads, FILE* file[], int file_count, char* keywords[], int word_count, int key_count[])
{
int producer = total_threads/2 +total_threads%2;
int producer_done = 0;
node_t* q_head = NULL;
node_t* q_tail = NULL;
#pragma omp parallel num_threads(total_threads)
{
int my_id = omp_get_thread_num();
int i;
if(my_id < producer)
{
for(i = my_id; i<file_count; i+=producer)
readfile(file[i], &q_head, &q_tail);
#pragma omp atomic
producer_done++;
}
else
{
node_t* temp = NULL;
while((producer_done < producer) || (q_head!=NULL))
{
temp = dequeue(&q_head, &q_tail);
if(temp != NULL)
tokenize(temp->data, keywords, word_count, key_count);
free(temp);
}
}
}
return;
}
int main(int argc, char* argv[])
{
int thread_sz;
FILE *file[MAX_FILES];
int total_file = 0;
int i;
int total_key = 0;
char* keys[MAX_KEYS];
int keys_count[MAX_KEYS];
char buffer[MAX_KEY_LENGTH];
char *temp;
clock_t start, end;
thread_sz = strtol(argv[1], NULL, 10);
start = clock();
FILE *fp = fopen("keys.txt", "r");
if(fp == NULL)
{
fprintf(stderr, "Unable to open file\n");
exit(EXIT_FAILURE);
}
while(fgets(buffer, MAX_KEY_LENGTH, fp) != NULL)
{
temp = malloc(sizeof(char)*strlen(buffer));
strncpy(temp, buffer, strlen(buffer)-1);
temp[strlen(buffer)-1] = '\0';
keys[total_key] = temp;
keys_count[total_key] = 0;
total_key++;
}
fclose(fp);
total_file = getfile(&file, "words/");
prod_con(thread_sz, file, total_file, keys, total_key, keys_count);
for(i=0; i<total_key; i++)
printf("%s: %d times\n", keys[i], keys_count[i]);
end = clock();
double diff = end-start;
printf("Total time is: %f s\n", diff/CLOCKS_PER_SEC);
return 0;
}