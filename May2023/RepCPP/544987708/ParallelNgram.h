
#ifndef N_GRAM_PARALLELNGRAM_H
#define N_GRAM_PARALLELNGRAM_H

#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <filesystem>

#include "JobsQueue.h"
#include "Utils.h"
#include "SharedHistogram.h"
#include "Other Solutions/HistogramCollector.h"
#include "Other Solutions/PartialHistogramsQueue.h"

void parallelNgramWords(std::string& file_name, std::string& out_folder_parallel, int n, int chunk_size, int num_threads);

void parallelNgramWords(std::string& file_name, std::string& out_folder_parallel, int n, int chunk_size, int num_threads){

std::ifstream infile(file_name);
JobsQueue jobsQueue; 
SharedHistogram sharedHistogram; 


#pragma omp parallel num_threads(num_threads) default(none) shared(infile, out_folder_parallel, jobsQueue, n, chunk_size, std::cout, sharedHistogram)
{
#pragma omp single nowait
{


std::vector<std::string> wordsLoaded;
std::string border[n-1];

std::string line;
std::string processedLine;
std::string tmp; 
int counter;
size_t pos;

counter = 0;

while (std::getline(infile, line)) {


std::remove_copy_if(
line.begin(),
line.end(),
std::back_inserter(processedLine),
std::ptr_fun<char&,bool>(&processInputChar));

pos = processedLine.find(' '); 

while (pos != std::string::npos) {
if(pos > 0) { 
wordsLoaded.push_back(processedLine.substr(0, pos));
counter += 1;
}

processedLine.erase(0, pos + 1);
pos = processedLine.find(' ');

if (counter >= chunk_size) { 
jobsQueue.enqueue(wordsLoaded);

for(int i=0; i < n-1; i++) 
border[i] = wordsLoaded[chunk_size - n + 1 + i];

wordsLoaded.clear(); 

for(int i=0; i < n-1; i++) 
wordsLoaded.push_back(border[i]);
counter = n-1;
}
}

if(!processedLine.empty()) {
tmp = processedLine.substr(0, processedLine.length() - 1);
if (!tmp.empty()) {
wordsLoaded.push_back(tmp);
counter += 1;
}
}

processedLine.clear();
}

if(counter > 0){ 
jobsQueue.enqueue(wordsLoaded);
}

jobsQueue.producerEnd(); 
}


std::vector<std::string> wordsChunk;

std::map<std::string, int> partialHistogram; 
std::map<std::string, int>::iterator it;

std::string ngram;
size_t pos;

while(!jobsQueue.done()){ 
if(jobsQueue.dequeue(wordsChunk)){ 


ngram = "";


for(int j=0; j < n; j++)
ngram += wordsChunk[j] + " ";

it = partialHistogram.find(ngram);
if(it != partialHistogram.end()) 
it->second += 1;
else
partialHistogram.insert(std::make_pair(ngram, 1));

pos = ngram.find(' '); 


for(int i=n; i < wordsChunk.size(); i++){
ngram.erase(0, pos + 1); 
ngram += wordsChunk[i] + " "; 

it = partialHistogram.find(ngram);
if(it != partialHistogram.end()) 
it->second += 1;
else
partialHistogram.insert(std::make_pair(ngram, 1));

pos =  ngram.find(' ');
}
}
}

sharedHistogram.mergeHistogram(partialHistogram); 


} 


sharedHistogram.writeHistogramToFile(out_folder_parallel + std::to_string(n) + "gram_outputParallelVersion.txt");
}

#endif 
