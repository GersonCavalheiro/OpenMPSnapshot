#include "mpi.h"
#include <omp.h>
#include <cstdio>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iterator>
#include <ctype.h> 
#include <stdio.h>

using namespace std;

int main (int argc, char *argv[]) {
MPI_Init (&argc, &argv); 

int rank, size, namelen;
char name[MPI_MAX_PROCESSOR_NAME];

MPI_Comm_rank (MPI_COMM_WORLD, &rank); 
MPI_Get_processor_name (name, &namelen); 
MPI_Comm_size (MPI_COMM_WORLD, &size); 

string word = "brussels";
for (int i = 0; i < word.length(); i++) {
word[i] = tolower(word[i]);
}

if (rank == 0) {
string line;
vector<vector<string> > lines;
multimap<int, string, greater<int> > dictionary;
ifstream file("Particles1.csv");
if (file) {
while (getline(file, line)) {
size_t n = lines.size();
lines.resize(n + 1);
istringstream ss(line);
string field, push_field("");
bool no_quotes = true;
while (getline(ss, field, ',')) {
if (static_cast<size_t>(count(field.begin(), field.end(), '"')) % 2 != 0) {
no_quotes = !no_quotes;
}
push_field += field + (no_quotes ? "" : ",");
if (no_quotes) {
lines[n].push_back(push_field);
push_field.clear();
}
}
}
}
int totalCount = 0;
# pragma omp parallel for
for (int i=0;i<lines.size();i++) {
vector<string> line = lines[i];
int length = line.size();
if (length > 1)
{
stringstream ss(line[length - 1]);
string token;
int count = 0;
while (getline(ss, token, ' '))
{
if(token.find(word) != string::npos){
count++;
totalCount++;
}
}
if (count > 0)
{
dictionary.insert(pair<int, string>(count, line[1] + "\t" + line[2])); 
}
}
}
multimap<int, string>::iterator itr;
int i = 0;
for (itr = dictionary.begin(); itr != dictionary.end(); ++itr) { 
cout << itr->first << '\t' << itr->second << '\n';
if(i == 9) break;
i++;
}
cout << word + " is " << totalCount << " times in this article" << endl;
printf("Hello World from rank %d running on %s!\n", rank, name);
}
if (rank == 1) {
string line;
vector<vector<string> > lines;
multimap<int, string, greater<int> > dictionary;
ifstream file("Particles2.csv");
if (file) {
while (getline(file, line)) {
size_t n = lines.size();
lines.resize(n + 1);
istringstream ss(line);
string field, push_field("");
bool no_quotes = true;
while (getline(ss, field, ',')) {
if (static_cast<size_t>(count(field.begin(), field.end(), '"')) % 2 != 0) {
no_quotes = !no_quotes;
}
push_field += field + (no_quotes ? "" : ",");
if (no_quotes) {
lines[n].push_back(push_field);
push_field.clear();
}
}
}
}
int totalCount = 0;
# pragma omp parallel for
for (int i=0;i<lines.size();i++) {
vector<string> line = lines[i];
int length = line.size();
if (length > 1)
{
stringstream ss(line[length - 1]);
string token;
int count = 0;
while (getline(ss, token, ' '))
{
if(token.find(word) != string::npos){
count++;
totalCount++;
}
}
if (count > 0)
{
dictionary.insert(pair<int, string>(count, line[1] + "\t" + line[2])); 
}
}
}
multimap<int, string>::iterator itr;
int i = 0;
for (itr = dictionary.begin(); itr != dictionary.end(); ++itr) {
cout << itr->first << '\t' << itr->second << '\n';
if(i == 9) break;
i++;
}
cout << word + " is " << totalCount << " times in this article" << endl;
printf ("Hello World from rank %d running on %s!\n", rank, name);
}
if (rank == 2) {
string line;
vector<vector<string> > lines;
multimap<int, string, greater<int> > dictionary;
ifstream file("Particles3.csv");
if (file) {
while (getline(file, line)) {
size_t n = lines.size();
lines.resize(n + 1);
istringstream ss(line);
string field, push_field("");
bool no_quotes = true;
while (getline(ss, field, ',')) {
if (static_cast<size_t>(count(field.begin(), field.end(), '"')) % 2 != 0) {
no_quotes = !no_quotes;
}
push_field += field + (no_quotes ? "" : ",");
if (no_quotes) {
lines[n].push_back(push_field);
push_field.clear();
}
}
}
}
int totalCount = 0;
# pragma omp parallel for
for (int i=0;i<lines.size();i++) {
vector<string> line = lines[i];
int length = line.size();
if (length > 1)
{
stringstream ss(line[length - 1]);
string token;
int count = 0;
while (getline(ss, token, ' '))
{
if(token.find(word) != string::npos){
count++;
totalCount++;
}
}
if (count > 0)
{
dictionary.insert(pair<int, string>(count, line[1] + "\t" + line[2])); 
}
}
}
multimap<int, string>::iterator itr;
int i = 0;
for (itr = dictionary.begin(); itr != dictionary.end(); ++itr) {
cout << itr->first << '\t' << itr->second << '\n';
if(i == 9) break;
i++;
}
cout << word + " is " << totalCount << " times in this article" << endl;

printf ("Hello World from rank %d running on %s!\n", rank, name);
}

MPI_Finalize (); 
}
