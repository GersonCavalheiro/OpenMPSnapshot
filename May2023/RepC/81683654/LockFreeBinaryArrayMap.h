#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#include<omp.h>
#include<unistd.h>
#ifndef LockFreeBinaryArrayMap
#define LockFreeBinaryArrayMap
struct _node;
typedef struct _data_map
{
struct _node *value;
int size;
}_data_map;
typedef struct _data_list
{
float ranking_1;
float ranking_2;
int frequency;
}_data_list;
typedef union _private_data
{
struct _data_map data_map;
struct _data_list data_list;
}private_data;
typedef struct _node{
char *key;
private_data data;
struct _node *next;
}node_t;
typedef struct _Array_Pointers{
node_t** node;
int size_instant;	
int size;
int umbral;
}Array_Pointers_t;
typedef struct _map{
node_t head;
int size;
int umbral;		
}map_t;
int IS_SAME_KEY(char *key, char *value){
if (strcmp(key,value)!=0)
return 0;
return 1;
}
void print_node(map_t *map, char *key){
node_t *node=map->head.next;
int i,key_found=0;
while(!IS_SAME_KEY(node->key,"")){
if(IS_SAME_KEY(node->key,key)){
key_found=1;
break;
}
node=node->next;
}
if (key_found==1)
{
node_t *sub_node=node->data.data_map.value->next;
printf("\n==========================List Pointers==========================\n");
sub_node=node->data.data_map.value->next;
while(strcmp(sub_node->key,"")!=0){
printf("\"%10s\"(%f)\n",sub_node->key,sub_node->data.data_list.ranking_1);
if (strcmp(sub_node->next->key,"")!=0)
printf("           |\n           v\n");
sub_node=sub_node->next;
}
printf("==========================%s size:%d==================================\n     ", key,node->data.data_map.size);		
printf("\n");
}
}
void print_map(map_t *map){
int i;
node_t*node;
printf("\n==========================List Pointers==========================\n");
node=map->head.next;
while(strcmp(node->key,"")!=0){
printf("\"%10s\"",node->key);
if (strcmp(node->next->key,"")!=0){
printf("->");
}
node=node->next;
}
printf("\n======================MAP size:%d Umbral:%d======================\n", map->size,map->umbral);
printf("\n");
}
void free_map(map_t *map){
int i;
node_t*node;
node_t*free_node;
node=map->head.next;
while(strcmp(node->key,"")!=0){
free_node=node;
node_t *sub_node=node->data.data_map.value->next;
node_t *free_sub_node;
while(sub_node->data.data_list.ranking_1!=1){
free_sub_node=sub_node;
sub_node = sub_node->next;
free(free_sub_node->key);
free(free_sub_node);
}
free(sub_node->key);
free(sub_node);
node=node->next;
free(free_node->key);
free(free_node);
}
free(map->head.key);
free(map);
}
void free_array_pointers(Array_Pointers_t *array_pointers){
int i;
node_t*node;
free(array_pointers->node);
free(array_pointers);
}
int check_input(char*key, char*value){
int error=0;
if(strcmp(key,"")==0)
error=1;
if(strcmp(value,"")==0)
error=1;
if(strstr(key,",")!=NULL)
error=1;
if(strstr(value,",")!=NULL)
error=1;
if(error==1)
printf("Input Error:\"%s\" \"%s\" \n", key, value);
return error;
}
int check_error(map_t *map){
int error=0, index=1, sub_index;
node_t*node=map->head.next;
node_t*node_next;
node_t*sub_node;
node_t*sub_node_next;
while(strcmp(node->key,"")!=0){
node_next=node->next;
if ((strcmp(node_next->key,node->key)<0)&&(!IS_SAME_KEY(node_next->key,""))){
error=1;
break;
}
sub_node=node->data.data_map.value->next;
sub_index=1;
while(strcmp(sub_node->key,"")!=0){
sub_node_next=sub_node->next;
if (sub_node_next->data.data_list.ranking_1 > sub_node->data.data_list.ranking_1
&&sub_node_next->data.data_list.ranking_1 != 1){
error=2;
break;
}	
sub_node=sub_node_next;
sub_index++;
}
if (error==2)
break;
node=node_next;
index++;
}
if(error==1){
print_map(map);
printf("Had a error at index[%d]: %s.\n",index,node->next->key);
}
if(error==2){
print_node(map,node->key);
printf("Had a error at index[%d]: %s >> sub_node[%d]: %s\n",index,node->key,sub_index,sub_node->next->key);
}
return error;
}
void CopyString(char *strCopy, char *strOriginal){
strcpy(strCopy,strOriginal);
}
map_t *map_init(map_t *map, int umbral)
{
int i;
map=(map_t*)malloc(sizeof(map_t));
map->head.key = (char *)malloc(sizeof(char));
map->head.key[0] = '\0';
map->head.next = &map->head;
map->size = 0;
map->umbral = umbral;
return map;
}
void array_pointers_update(map_t *map, int map_size, Array_Pointers_t *array_pointers, int tid, int all_t){
node_t *current_node;
int i,j,index=0;
int size_instant=map_size/array_pointers->umbral;
current_node = &map->head;
for (i = tid; i < size_instant; i+=all_t){
for (j = index; j < array_pointers->umbral*i; ++j)
current_node=current_node->next;
index=j;
array_pointers->node[i]=current_node;
}
}
void map_insert(
map_t *map, 
char *key, 
char *value, 
float ranking_1, 
float ranking_2, 
int frequency,
float reduction,
int all_num_input,
int num_input,
int thread_ending[],
char thread_using[][189819],
int *prev_map_size,
int node_inserted[],
Array_Pointers_t *array_pointers)
{
node_t *current_node;
node_t *repeat_node;
int tid=omp_get_thread_num();
int all_t=omp_get_num_threads();
int i, key_repeat;
int again=1;
while(again==1||thread_ending[tid]==1){
if((thread_ending[tid])!=1){
again=0;
key_repeat=0;		
current_node = &map->head;	
if(array_pointers->size_instant>0){
int start=0, end=array_pointers->size_instant-1, mid=0;
while(1){
if(IS_SAME_KEY(array_pointers->node[mid]->key,key)){
current_node=array_pointers->node[mid];
repeat_node=current_node;
key_repeat=1;
break;
}
if(strcmp(array_pointers->node[mid]->key,key)<0&&mid+1==array_pointers->size_instant)
break;
if(strcmp(array_pointers->node[mid]->key,key)<0&&mid+1<array_pointers->size_instant&&strcmp(array_pointers->node[mid+1]->key,key)>0)
break;
if(strcmp(array_pointers->node[mid]->key,key)<0){
start=mid+1;
}
else if(strcmp(array_pointers->node[mid]->key,key)>0){
end=mid-1;
}
mid=start+(end-start)/2;
}
current_node = array_pointers->node[mid];
}
if(key_repeat==0){
while(strcmp(current_node->next->key,key)<=0){
if(IS_SAME_KEY(current_node->next->key,key)){
repeat_node=current_node->next;
current_node=current_node->next;
key_repeat=1;
break;
}
if(IS_SAME_KEY(current_node->next->key,""))
break;
current_node=current_node->next;
}
}
CopyString(thread_using[tid],current_node->key);
}
#pragma omp barrier
if(thread_ending[tid]!=1){
for(i=0;i<all_t;i++)
if(	i!=tid
&&	IS_SAME_KEY(thread_using[i],thread_using[tid])
&&	i<tid)
again=1;
if(again==0){
if (key_repeat==1){
key_repeat=0;
node_t *node=repeat_node;
node_t *sub_node=node->data.data_map.value;
node_t *ranking_node=node->data.data_map.value;
while(1){
if(	sub_node->data.data_list.ranking_1 >= ranking_1
&&	(ranking_1 >= sub_node->next->data.data_list.ranking_1
||	sub_node->next->data.data_list.ranking_1 == 1))
ranking_node=sub_node;
if (IS_SAME_KEY(sub_node->key,value)){
key_repeat=1;
break;
}
if(IS_SAME_KEY(sub_node->next->key,""))
break;
sub_node=sub_node->next;
}
if (key_repeat==0)
{
sub_node=ranking_node;						
node_t *new_sub_node=(node_t*)malloc(sizeof(node_t));
new_sub_node->key=(char *)malloc(sizeof(char)*(strlen(value)+1));
CopyString(new_sub_node->key, value);
new_sub_node->data.data_list.ranking_1=ranking_1;
new_sub_node->data.data_list.ranking_2=ranking_2;
new_sub_node->data.data_list.frequency=frequency;						
new_sub_node->next=sub_node->next;
sub_node->next=new_sub_node;
node->data.data_map.size++;
}
}
else{
node_t *insert_node;
insert_node = (node_t *)malloc(sizeof(node_t));
insert_node->key=(char *)malloc(sizeof(char)*(strlen(key)+1));
CopyString(insert_node->key, key);
node_t *sub_node;
sub_node=(node_t*)malloc(sizeof(node_t));
sub_node->key=(char *)malloc(sizeof(char)*(strlen(value)+1));
sub_node->data.data_list.ranking_1=ranking_1;
sub_node->data.data_list.ranking_2=ranking_2;
sub_node->data.data_list.frequency=frequency;						
CopyString(sub_node->key, value);
node_t *header;
header=(node_t*)malloc(sizeof(node_t));
header->key=(char *)malloc(sizeof(char));
CopyString(header->key, "");
header->data.data_list.ranking_1=1;
header->data.data_list.ranking_2=1;
header->data.data_list.frequency=1;
header->next=sub_node;
sub_node->next=header;
insert_node->data.data_map.value=header;
insert_node->data.data_map.size=1;
insert_node->next=current_node->next;
current_node->next=insert_node;
node_inserted[tid]++;	
}
}
}
if((num_input+all_t)>=all_num_input&&again==0)
thread_ending[tid]=1;
#pragma omp barrier
int map_size=0;
for(i=0;i<all_t;i++)
map_size+=node_inserted[i];
if(	map_size <= (*prev_map_size) + map->umbral + all_t
&&	map_size >= map->umbral
&&	map_size >= (*prev_map_size) + map->umbral - all_t){
array_pointers_update( map, map_size, array_pointers, tid, all_t);
if(tid==0)
array_pointers->size_instant = map_size/array_pointers->umbral;
#pragma omp barrier
if(tid==0){
(*prev_map_size)=map_size;
map->umbral=(int)((map->umbral)*reduction);
}
}
CopyString(thread_using[tid],"*");
if(thread_ending[tid]==1){
int all_thread_ending=1;
for(i=0;i<all_t;i++)
if(thread_ending[i]==0)
all_thread_ending=0;
if(all_thread_ending==1)
break;
}
}
}
#endif
