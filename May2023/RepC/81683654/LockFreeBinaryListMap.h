#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<time.h>
#include<omp.h>
#include<unistd.h>
#ifndef LockFreeBinaryListMap
#define LockFreeBinaryListMap
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
struct _node *next[4];
}node_t;
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
node_t *node=map->head.next[1];
int i,key_found=0;
while(!IS_SAME_KEY(node->key,"")){
if(IS_SAME_KEY(node->key,key)){
key_found=1;
break;
}
node=node->next[1];
}
if (key_found==1)
{
node_t *sub_node=node->data.data_map.value;
printf("\n==========================List Pointers==========================\n");
printf("\"%10s\"(%f)<-\"%10s\"(%f)<-\"%10s\"(%f)->\"%10s\"(%f)->\"%10s\"(%f)\n",
sub_node->next[3]->key,sub_node->next[3]->data.data_list.ranking_1,
sub_node->next[2]->key,sub_node->next[2]->data.data_list.ranking_1,
sub_node->key,sub_node->data.data_list.ranking_1,
sub_node->next[1]->key,sub_node->next[1]->data.data_list.ranking_1,
sub_node->next[0]->key,sub_node->next[0]->data.data_list.ranking_1);
printf("                              |\n                              v\n");
sub_node=node->data.data_map.value->next[1];
while(strcmp(sub_node->key,"")!=0){
printf("\"%10s\"(%f)<-\"%10s\"(%f)<-\"%10s\"(%f)->\"%10s\"(%f)->\"%10s\"(%f)\n",
sub_node->next[3]->key,sub_node->next[3]->data.data_list.ranking_1,
sub_node->next[2]->key,sub_node->next[2]->data.data_list.ranking_1,
sub_node->key,sub_node->data.data_list.ranking_1,
sub_node->next[1]->key,sub_node->next[1]->data.data_list.ranking_1,
sub_node->next[0]->key,sub_node->next[0]->data.data_list.ranking_1);
if (strcmp(sub_node->next[1]->key,"")!=0)
printf("                              |\n                              v\n");
sub_node=sub_node->next[1];
}
printf("==========================%s size:%d==================================\n     ", key,node->data.data_map.size);		
printf("\n");
}
}
void print_map(map_t *map){
int i;
node_t*node;
printf("\n==========================List Pointers==========================\n");
node=map->head.next[1];
printf("\"%10s\"<-\"%10s\"<-\"%10s\"->\"%10s\"->\"%10s\"\n",map->head.next[3]->key,map->head.next[2]->key,map->head.key,map->head.next[1]->key,map->head.next[0]->key);
printf("                                 |\n                                 v\n");
while(strcmp(node->key,"")!=0){
printf("\"%10s\"<-\"%10s\"<-\"%10s\"->\"%10s\"->\"%10s\"\n",node->next[3]->key,node->next[2]->key,node->key,node->next[1]->key,node->next[0]->key);
if (strcmp(node->next[1]->key,"")!=0)
printf("                                 |\n                                 v\n");
node=node->next[1];
}
printf("\n======================MAP size:%d Umbral:%d======================\n", map->size,map->umbral);
printf("\n");
}
void free_map(map_t *map){
int i;
node_t*node;
node_t*free_node;
node=map->head.next[1];
while(strcmp(node->key,"")!=0){
free_node=node;
node_t *sub_node=node->data.data_map.value->next[1];
node_t *free_sub_node;
while(sub_node->data.data_list.ranking_1!=1){
free_sub_node=sub_node;
sub_node = sub_node->next[1];
free(free_sub_node->key);
free(free_sub_node);
}
free(sub_node->key);
free(sub_node);
node=node->next[1];
free(free_node->key);
free(free_node);
}
free(map->head.key);
free(map);
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
node_t*node=map->head.next[1];
node_t*node_next;
node_t*sub_node;
node_t*sub_node_next;
while(strcmp(node->key,"")!=0){
node_next=node->next[1];
if ((strcmp(node_next->key,node->key)<0)&&(!IS_SAME_KEY(node_next->key,""))){
error=1;
break;
}
sub_node=node->data.data_map.value->next[1];
sub_index=1;
while(strcmp(sub_node->key,"")!=0){
sub_node_next=sub_node->next[1];
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
printf("Had a error at index[%d]: %s.\n",index,node->next[1]->key);
}
if(error==2){
print_node(map,node->key);
printf("Had a error at index[%d]: %s >> sub_node[%d]: %s\n",index,node->key,sub_index,sub_node->next[1]->key);
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
for (i = 0; i < 4; i++)
map->head.next[i] = &map->head;
map->size = 0;
map->umbral = umbral;
return map;
}
void Update_pointers(node_t* node, node_t* node_pred, int num_jump, int num_jump_backward){
int i;
node_t* node_forward=node;
node_t* node_backward=node;
if (num_jump/2>0)
{
for (i = 0; i < num_jump/2; ++i)
node_forward=node_forward->next[1];
node->next[0]=node_forward;
Update_pointers(node_forward, node, num_jump-num_jump/2, num_jump/2);
if(node_pred!=NULL&&num_jump_backward/2>0){
for (i = 0; i < num_jump_backward/2; ++i)
node_backward=node_backward->next[2];
node->next[3]=node_backward;
Update_pointers(node_backward, node, num_jump_backward/2,num_jump_backward-num_jump_backward/2);
}
}
}
void Simply_pointers(node_t* node, int size){
while(size>0){
if(!IS_SAME_KEY(node->next[1]->key,node->next[0]->key))
node->next[0]=node->next[1];
if(!IS_SAME_KEY(node->next[2]->key,node->next[3]->key))
node->next[3]=node->next[2];
node=node->next[1];
size--;
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
int node_inserted[])
{
node_t *current_node;
node_t *repeat_node;
int tid=omp_get_thread_num();
int all_t=omp_get_num_threads();
int i, key_repeat;
int again=1;
current_node = &map->head;
while(again==1||thread_ending[tid]==1){
if((thread_ending[tid])!=1){			
again=0;
key_repeat=0;
while(1){
if(IS_SAME_KEY(current_node->key,key)){
repeat_node=current_node;
key_repeat=1;
break;
}
if(strcmp(current_node->key,key)<0&&IS_SAME_KEY(current_node->next[1]->key,""))
break;
if(strcmp(current_node->key,key)<0&&strcmp(current_node->next[1]->key,key)>0)
break;
if(strcmp(current_node->key,key)<=0&&!IS_SAME_KEY(current_node->next[0]->key,""))
current_node=current_node->next[0];
else if(strcmp(current_node->key,key)>=0&&!IS_SAME_KEY(current_node->next[3]->key,""))
current_node=current_node->next[3];
else if(strcmp(current_node->key,key)<=0)
current_node=current_node->next[1];
else
current_node=current_node->next[2];	
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
&&	(ranking_1 >= sub_node->next[1]->data.data_list.ranking_1
||	sub_node->next[1]->data.data_list.ranking_1 == 1))
ranking_node=sub_node;
if (IS_SAME_KEY(sub_node->key,value)){
key_repeat=1;
break;
}
if(IS_SAME_KEY(sub_node->next[1]->key,""))
break;
sub_node=sub_node->next[1];	
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
new_sub_node->next[0]=sub_node->next[1];
new_sub_node->next[1]=sub_node->next[1];
new_sub_node->next[2]=sub_node;
new_sub_node->next[3]=sub_node;
sub_node->next[0]=new_sub_node;
sub_node->next[1]->next[3]=new_sub_node;
sub_node->next[1]->next[2]=new_sub_node;
sub_node->next[1]=new_sub_node;
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
for(i=0;i<4;i++)
header->next[i]=sub_node;
for(i=0;i<4;i++)
sub_node->next[i]=header;
insert_node->data.data_map.value=header;
insert_node->data.data_map.size=1;
insert_node->next[0]=current_node->next[1];
insert_node->next[1]=current_node->next[1];
insert_node->next[2]=current_node;
insert_node->next[3]=current_node;
current_node->next[0]=insert_node;
current_node->next[1]->next[3]=insert_node;
current_node->next[1]->next[2]=insert_node;
current_node->next[1]=insert_node;
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
&&	map_size >= (*prev_map_size) +map->umbral - all_t){
if(tid==0){
Simply_pointers(map->head.next[1], map_size);
Update_pointers(map->head.next[1], NULL, map_size, -1);
}
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
