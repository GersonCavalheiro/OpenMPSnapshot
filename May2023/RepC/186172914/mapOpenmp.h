#ifndef MAPTABLELIBRARY_H_INCLUDED
#define MAPTABLELIBRARY_H_INCLUDED

#include "queue_omp.h"
#include "standardHeaders.h"
#define mapTableSize 4

unsigned long hashCode(unsigned char *mapStr)
{

unsigned long hashKey = 5381;
int c;

while (c = *mapStr++)
hashKey = ((hashKey << 5) + hashKey) + c; 

return (hashKey);
}

struct mapNode{
char* mapStr; 
int wordCount;
struct mapNode* next; 
};

struct mapChain {
struct mapNode* head;
struct mapNode* tail;
};

struct mapChain* initMapTable(int capacity){

struct mapChain* mapTable;
mapTable = (struct mapChain*) malloc(capacity*sizeof(struct mapChain));

#pragma omp parallel for
for(int i = 0; i < capacity; i++){
mapTable[i].head = NULL;
mapTable[i].tail = NULL;
}

return(mapTable);
}

void mapWord(struct mapChain* mapTable, char* readStr, int capacity, int addWordCount)
{    

int key = (hashCode(readStr)%capacity);
struct mapNode *chain = (struct mapNode*) mapTable[key].head;

struct mapNode *word = (struct mapNode*) malloc(sizeof(struct mapNode));
word->mapStr = malloc(strlen(readStr)+1);
strcpy(word->mapStr, readStr);
if(addWordCount == 0)
{
word->wordCount = 1;
} else {
word->wordCount = addWordCount;
}
word->next = NULL;

if (chain == NULL){
mapTable[key].head = word;
mapTable[key].tail = word;
} else if (strcmp(readStr,chain->mapStr) == 0){
mapTable[key].head->wordCount+=(word->wordCount); 
} else {
chain = chain->next;
int stopIndex = 1;
int foundIndex = -1;
while(chain != NULL){
if(strcmp(readStr,chain->mapStr) == 0){
chain->wordCount+=(word->wordCount);  
foundIndex = stopIndex;
}
stopIndex++;
chain = chain->next;
}

if(foundIndex == -1){
mapTable[key].tail->next = word;
mapTable[key].tail = word;
}
}
}


void combineWords(struct mapChain* mapTable, char* readStr, int tableKey, int addWordCount)
{    

int key = tableKey;
struct mapNode *chain = (struct mapNode*) mapTable[key].head;

struct mapNode *word = (struct mapNode*) malloc(sizeof(struct mapNode));
word->mapStr = malloc(strlen(readStr)+1);
strcpy(word->mapStr, readStr);
if(addWordCount == 0)
{
word->wordCount = 1;
} else {
word->wordCount = addWordCount;
}
word->next = NULL;

if (chain == NULL){
mapTable[key].head = word;
mapTable[key].tail = word;
} else if (strcmp(readStr,chain->mapStr) == 0){
mapTable[key].head->wordCount+=(word->wordCount); 
} else {
chain = chain->next;
int stopIndex = 1;
int foundIndex = -1;
while(chain != NULL){
if(strcmp(readStr,chain->mapStr) == 0){
chain->wordCount+=(word->wordCount);  
foundIndex = stopIndex;
}
stopIndex++;
chain = chain->next;
}

if(foundIndex == -1){
mapTable[key].tail->next = word;
mapTable[key].tail = word;
}
}
}





void saveMapToFile(struct mapChain* mapTable, int capacity, int thread, int writers){

FILE *fp;
char fileName[50] = "mapOutput";
char threadStr[3];
sprintf(threadStr,"%d",thread);
strcat(fileName, threadStr);
strcat(fileName, ".txt");

fp=fopen(fileName,"w");

for(int i = thread; i<capacity; i+=writers){

struct mapNode* curChain = (struct mapNode*) mapTable[i].head;
if (curChain == NULL) {
; 
} else {
while (curChain != NULL){
fprintf(fp, "< %s, %d >\n", curChain->mapStr, curChain->wordCount);
curChain = curChain->next;
}
}
}
fclose(fp);
}


char* quickstrcat(char* dest, char* src)
{


while(*src != '\0')
{
*dest = *src;
src++;
dest++;
}

*dest = *src;

return(dest);

}

void convertMap(struct mapChain* mapTable, char* mapString, int* mapCount, int capacity, int tableIter){

int entr = 0;
char* locP = mapString;

for(int i = 0; i<capacity; i++){

struct mapNode* curChain = (struct mapNode*) mapTable[i].head;
if (curChain == NULL) {
; 
} else {
while (curChain != NULL){
mapCount[entr] = curChain->wordCount;
locP = quickstrcat(locP, "*");
locP = quickstrcat(locP,curChain->mapStr);
entr++;
curChain = curChain->next;
}
}
}

return;

}



struct mapNode* getWord(struct mapChain* mapTable, char* word, int capacity){

int key = hashCode(word)%capacity;
struct mapNode *chain = (struct mapNode*) mapTable[key].head;

while (chain != NULL){

if(strcmp(word,chain->mapStr) == 0){
return(chain);
}
chain = chain->next;
}

return(NULL);
}

char* strtok_a(char *s1, const char *s2, char **lasts)
{



char *ret;

if (s1 == NULL)
{
s1 = *lasts;
}

while(*s1 && strchr(s2, *s1))
{
++s1;
}

if(*s1 == '\0')
{
return NULL;
}

ret = s1;
while(*s1 && !strchr(s2, *s1))
{
++s1;
}

if(*s1)
{
*s1++ = '\0';
}

*lasts = s1;
return ret;
}

void mapper(struct mapChain* mapTable , char* strPtr, int capacity){

char delim[65] = " ,\n'.;'<>-_\"\"!=+[]{}0123456789@#$%^&*():\\
char* word;
char* sentence;

sentence = (char*) malloc((strlen(strPtr)+1)*sizeof(char));
strcpy(sentence,strPtr);


token = strtok_a(sentence,delim, &word);    
while(token != NULL){
mapWord(mapTable, token, capacity, 0);
token = strtok_a(NULL, delim, &word);

}

return;

}





void mergeTable(struct mapChain* mapTable,char* mapString, int* mapCount, int capacity)
{

char delim[2] = "*";
char* next;
char* cptr  = strtok_a(mapString,delim, &next);
int entry = 0;

while(cptr != NULL){
mapWord(mapTable, cptr, capacity, mapCount[entry]);
entry++;
cptr = strtok_a(NULL, delim, &next);
}

return;


}

#endif 
