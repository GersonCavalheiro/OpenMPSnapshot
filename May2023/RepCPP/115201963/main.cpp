

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <sstream>
#include <map>
#include <omp.h>
#include <list> 

using namespace std;

int nodesExplored;
int nodesGenerated;
string optrs[] = {"UP", "DOWN", "LEFT", "RIGHT"};
int goal[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0};

class Node{

public:
int puzzle[16];
Node *parent;
string optr;
int static id;
Node(){
parent = NULL;
optr = "";
id++;
}
Node(int board[]){
optr = "";
parent = NULL;

for(int i=0;i<16;i++)
this->puzzle[i] = board[i];
id++;
}
Node(Node *temp){
this->parent = temp->parent;
optr = temp->optr;

for(int i=0;i<16;i++)
this->puzzle[i] = temp->puzzle[i];
id++;
}
string ToString(){
string retStr="";
for(int i=0;i<16;i++){
stringstream ss;
ss << puzzle[i];
string str = ss.str();
retStr = retStr + str;	
}
return retStr;
}
};
int Node::id=0;
void Print(int puzzle[], int num){

if(num == 0) cout<<"Solution Found: ";
else 		 cout<<"Step#"<<num<<":";

cout<<endl;

for(int i=0;i<16;i++){
cout<<puzzle[i]<<" ";
if((i+1)%4 == 0)
cout<<endl;   
}
}
void SolPath(Node *head){

Node *p = head;

if(p==NULL)
return;

int i = 0;

while(p!=NULL){
Print(p->puzzle,i);
p = p->parent;
i++;
}

cout<<endl;
}
bool GoalTest(int board[]){

int count = 0;
for(int i=0;i<16;i++){
if(board[i] == goal[i])
count++;
}

if(count == 16)
return true;

return false;
}
bool validate(int board[], string o){

if(o == "UP"){
if(board[0] == 0 || board[1] == 0 || board[2]== 0 || board[3]== 0)
return false;
else 
return true;
}

else if(o == "DOWN"){
if(board[15] == 0 || board[14] == 0 || board[13]== 0 || board[12]== 0)
return false;
else 
return true;         
}

else if(o == "LEFT"){
if(board[0] == 0 || board[4] == 0 || board[8]== 0 || board[12]== 0)
return false;
else 
return true;
}

else{
if(board[3] == 0 || board[7] == 0 || board[11]== 0 || board[15]== 0)
return false;
else 
return true;
}
}
int SearchBlank(int board[]){
for(int i=0;i<16;i++)
if(board[i] == 0)
return i;
}

int main() {

ifstream in;
in.open("input.txt");

int puzzle[16];

for(int i=0;i<16;i++)
in >> puzzle[i];

in.close();

Node node = new Node(puzzle);
static bool flag = true;

queue<Node> BFS_Q;

BFS_Q.push(node);

double start_time = omp_get_wtime();

if(GoalTest(node.puzzle)){
Print(node.puzzle,0);
}
else{

list<string> hashSet;

#pragma omp parallel num_threads(8) shared(BFS_Q, hashSet, nodesExplored, nodesGenerated)
{	
while(flag){

#pragma omp critical
{ if(!BFS_Q.empty()){
}
Node* current =  new Node (BFS_Q.front());

BFS_Q.pop();

hashSet.push_back(current->ToString());

nodesExplored++;

for(int i=0;i<4 && flag;i++){

if(validate(current->puzzle,optrs[i]) == true){

int board[16];
for(int j=0;j<16;j++)
board[j] = current->puzzle[j];

int blankIndex = SearchBlank(board);

if(optrs[i] == "UP"){
board[blankIndex] = board[blankIndex-4];
board[blankIndex-4] = 0;
}
else if(optrs[i] == "DOWN"){
board[blankIndex] = board[blankIndex+4];
board[blankIndex+4] = 0;
}
else if(optrs[i] == "LEFT"){
board[blankIndex] = board[blankIndex-1];
board[blankIndex-1] = 0;
}
else{
board[blankIndex] = board[blankIndex+1];
board[blankIndex+1] = 0;
}

Node* child = new Node(board);

child->parent = current;

child->optr = optrs[i];

nodesGenerated++;

if(!(find(hashSet.begin(),hashSet.end(),child->ToString())
!= hashSet.end())){	

if(GoalTest(child->puzzle) == true){
SolPath(child);
flag = false;

}

BFS_Q.push(child);

}
}
}
}
}
}
}

double time = omp_get_wtime() - start_time;

cout<<"Breath First Search Completed"<<endl;
cout<<"Time taken: "<<time<<endl;
cout<<"Nodes Explored: "<<nodesExplored<<endl;
cout<<"Nodes Generated: "<<nodesGenerated<<endl;	

return 0;
}

