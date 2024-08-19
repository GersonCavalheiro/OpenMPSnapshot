#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>


struct node {
int idata;          
char *sdata;        
struct node *next;
};


typedef struct queue {
struct node *head;
struct node *tail;
} queue;


queue *newqueue()
{
queue *q = (queue*)malloc(sizeof(queue));
q->head = NULL;
q->tail = NULL;
return q;
}


void increase(queue *q, char *data)
{
struct node *ptr;
for (ptr = q->head; ptr != NULL; ptr = ptr->next) {
if (strcmp(ptr->sdata, data) == 0) {
#pragma omp critical
ptr->idata++;
return;
}
}
}


void enqueue(queue *q, char *data)
{
struct node *_new = (struct node*)malloc(sizeof(struct node));
_new->idata = 0;
_new->sdata = strdup(data);
_new->next = NULL;
#pragma omp critical
{
if (q->tail == NULL) {
q->head = _new;
q->tail = _new;
} else {
q->tail->next = _new;
q->tail = _new;
}
}
}


char *dequeue(queue *q)
{
char *data = NULL;
#pragma omp critical
{
if (q->head != NULL) {
data = q->head->sdata;
struct node *del = q->head;
q->head = q->head->next;
free(del);
}
if (q->head == NULL) {
q->tail = NULL;
}
}
return data;
}


void produce(queue *lines, char *filename)
{
FILE *fp = fopen(filename, "r");
char line[1000];
while (fgets(line, sizeof(line), fp)) {
if (line[strlen(line) - 1] == '\n') {
line[strlen(line) - 1] = '\0';
}
enqueue(lines, line);
}
}


void try_consume(queue *lines, queue *dict)
{
char *line = dequeue(lines);
if (line == NULL) {
return;
}

int index = 0, len = 0;
char buffer[40];
while (line[index] != '\0') {
if (line[index] == ' ') {
buffer[len] = '\0';
increase(dict, buffer);
len = 0;
} else {
buffer[len] = line[index];
len++;
}
index++;
}
buffer[len] = '\0';
increase(dict, buffer);
free(line);
}


int done(queue *lines)
{
int val = 0;
#pragma omp critical
{
if (lines->head == NULL) {
val = 1;
}
}
return val;
}


int main(int argc, char *argv[])
{
long n_thread = 4;
long n_producer = 2;


if (argc >= 2) {
n_thread = strtol(argv[1], NULL, 10);
if (n_thread < 1) {
printf("n_thread must > 0! abort.\n");
return 0;
}
}


char *keyword_dir = "./keywords.txt";
if (argc >= 3) {
keyword_dir = argv[2];
}


char *text_dir = "./words";
if (argc >= 4) {
text_dir = argv[3];
}


FILE *fp = fopen(keyword_dir, "r");
if (!fp) {
puts("Cannot open keyword file.");
return 0;
}
fseek(fp, 0, SEEK_END);
long fsize = ftell(fp);
rewind(fp);
char *str = (char*)malloc(sizeof(char) * (fsize + 1));
fread(str, sizeof(char), fsize, fp);
str[fsize] = '\0';
fclose(fp);

queue *dict = newqueue();
char *keyword = strtok(str, " \n");
while (keyword != NULL) {
enqueue(dict, keyword);
keyword = strtok(NULL, " ");
}


int n_file = 0;
struct dirent *dir;
DIR *d = opendir(text_dir);
if (!d) {
puts("Error reading directory.\n");
return 0;
}
while (1) {
dir = readdir(d);
if (dir == NULL)
break;
if (dir->d_type != DT_REG)
continue;
n_file++;
}
closedir(d);
d = opendir(text_dir);


omp_set_num_threads(n_thread);
queue *lines = newqueue();
int i;
#pragma omp parallel
{
#pragma omp for
for (i = 0; i < n_file; i++) {

do {
dir = readdir(d);
} while (dir->d_type != DT_REG);
char filename[40];
strcpy(filename, text_dir);
strcat(filename, "/");
strcat(filename, dir->d_name);


produce(lines, filename);


try_consume(lines, dict);
}


while (!done(lines))
try_consume(lines, dict);
}




closedir(d);
}