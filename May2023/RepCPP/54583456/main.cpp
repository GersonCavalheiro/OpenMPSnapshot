#include <algorithm>
#include <iostream>
#include <cstdio>
#include <vector>
#include <string>

#include "StringPair.h"
#include "StringOperations.h"
#include "Bucket.h"

using namespace std;

int main(int argc, char **argv){

int segmentsPerBucket = 9;
int totalSegments = 0;
string line;
vector<string> tempSegments;
vector<vector<string> > tempBuckets;
cout << "vars created\n";
while(getline(cin, line)){
if(line[line.size()-1] == '\r'){
line = line.substr(0, line.size()-1);

}
tempSegments.push_back(line);
totalSegments++;
if(tempSegments.size() == segmentsPerBucket){
tempBuckets.push_back(tempSegments);
tempSegments.clear();
}
}
if(!tempSegments.empty()){
tempBuckets.push_back(tempSegments);
tempSegments.clear();
}
cout << "temp buckets instantiated\n";
int nBuckets = tempBuckets.size();
Bucket* buckets = new Bucket[nBuckets];

for(int i = 0; i < nBuckets; i++){
buckets[i].nSegments = tempBuckets[i].size();
buckets[i].segments = new string[buckets[i].nSegments];
for(int j = 0; j < buckets[i].nSegments; j++){
buckets[i].segments[j] = tempBuckets[i][j];
}
tempBuckets[i].clear();
}
tempBuckets.clear();
cout << "buckets instantiated\n";
#pragma omp parallel
{
#pragma omp for
for(int actualBucket = 0; actualBucket < nBuckets; actualBucket++){
cout << "Bucket " << actualBucket <<":\n";
cout << "the bucket " << actualBucket << " have " << buckets[actualBucket].nSegments << "." << endl;
buckets[actualBucket].process(false);
cout << "Final processing of the bucket[outside]\n";
for (int i = 0; i < buckets[actualBucket].nSegments; i++){
cout << "segments[" << i << "] = \"" << buckets[actualBucket].segments[i] << "\"\n";
}
cout << "the bucket " << actualBucket << " now have " << buckets[actualBucket].nSegments << "." << endl;
}

}

int segmentsLeft = 0;
for(int i = 0; i < nBuckets; i++){
segmentsLeft += buckets[i].nSegments;
}
cout << segmentsLeft << " segments left in total." << endl;
Bucket finalBucket;
finalBucket.segments = new string[segmentsLeft];
finalBucket.nSegments = segmentsLeft;
int index = 0;

for(int i = 0; i < nBuckets; i++){
for(int j = 0; j < buckets[i].nSegments; j++){
finalBucket.segments[index] = buckets[i].segments[j];
index++;
}
}
cout << "Bucket final:\n";
finalBucket.process(true);
string textoSaida = finalBucket.segments[0];


for(int i = 0; i < textoSaida.size()-1; i++){
if((textoSaida[i] == '%') && (textoSaida[i+1] == '%')){
textoSaida.replace(i,2,"\n");
}
}

cout << textoSaida;

return 0;
}
