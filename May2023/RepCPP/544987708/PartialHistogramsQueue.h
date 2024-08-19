
#ifndef N_GRAM_PARTIALHISTOGRAMSQUEUE_H
#define N_GRAM_PARTIALHISTOGRAMSQUEUE_H

#include <omp.h>
#include <iostream>
#include <map>
#include <fstream>

class PartialHistogram{
public:
std::map<std::string, int> histogram;
PartialHistogram* next;
};

class PartialHistogramsQueue{
public:

PartialHistogramsQueue(int num_threads){
PartialHistogram* tmp1 = new PartialHistogram(); 
PartialHistogram* tmp2 = new PartialHistogram();
head = tmp1;
tmp1->next = tmp2;
tmp2->next = nullptr;
tail = tmp2;
this->num_threads = num_threads;
num_tasks = 0;
omp_init_lock(&headLock);
omp_init_lock(&tailLock);
}

~PartialHistogramsQueue(){
PartialHistogram* current = head;
PartialHistogram* tmp;

while(current != nullptr){
tmp = current;
current = current->next;
delete(tmp);
}

head = tail = nullptr;
omp_destroy_lock(&headLock);
omp_destroy_lock(&tailLock);
}

void enqueue(std::map<std::string, int>& partialHistogram){

PartialHistogram* tmp = new PartialHistogram();
tmp->histogram = partialHistogram;
tmp->next = nullptr;

omp_set_lock(&tailLock); 
tail->next = tmp;
tail = tmp;
omp_unset_lock(&tailLock);
}

bool dequeue(std::map<std::string, int>& hist1, std::map<std::string, int>& hist2){

omp_set_lock(&headLock);

PartialHistogram* placeHolder1 = head;
PartialHistogram* placeHolder2 = head->next;
PartialHistogram* node1 = placeHolder2->next;
if (node1 == nullptr) {
omp_unset_lock(&headLock);
return false;
}

PartialHistogram* node2 = node1->next;
if (node2 == nullptr) {
omp_unset_lock(&headLock);
return false;
}

hist1 = node1->histogram;
hist2 = node2->histogram;
head = node1;
num_tasks += 1;
omp_unset_lock(&headLock);
delete placeHolder1;
delete placeHolder2;
return true;
}

bool done(){
#pragma omp flush(num_tasks)
if(num_tasks == (num_threads - 1))
return true;
else
return false;
}

void writeHistogramToFile(std::string path){
PartialHistogram* partialHistogram = head->next->next;

std::ofstream outputFile;
outputFile.open(path);

for(auto& kv : partialHistogram->histogram)
outputFile << kv.first << "\t" << kv.second << std::endl;

outputFile.close();
}



private:
PartialHistogram* head;
PartialHistogram* tail;
omp_lock_t headLock;
omp_lock_t tailLock;
int num_threads;
int num_tasks;
};

#endif 
