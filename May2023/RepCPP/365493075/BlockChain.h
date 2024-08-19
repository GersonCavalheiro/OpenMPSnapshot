#pragma once
#include <string>
#include <iostream>
using namespace std;


struct Block {
size_t blockNum;
std::string data;
std::string initializeHash;							
Block* next;
};

struct BlockChain
{
Block* head;										
Block* current;										
size_t blockCounter = 0;
BlockChain();
~BlockChain();
bool Empty() const;									
void AddBack(std::string data, std::string hash);	
void RemoveFront();									
string GetText(size_t& privateNonce);
bool Control(string& hash);
};

BlockChain::BlockChain()								
{
head = NULL;
current = NULL;
}

BlockChain::~BlockChain()								
{
while (!Empty()) RemoveFront();
}

bool BlockChain::Empty() const							
{
return head == NULL;
}

void BlockChain::RemoveFront()							
{
Block* temp = head;									
head = head->next;									
delete temp;										
}

void BlockChain::AddBack(std::string data, std::string hash)		
{
blockCounter++;
Block* v = new Block;
v->blockNum = blockCounter;
v->data = data;
v->initializeHash = hash;
v->next = NULL;
current = v;

if (head == NULL) head = v;
else
{
Block* first = head;
while (first->next != NULL) first = first->next;
first->next = v;
}
}

string BlockChain::GetText(size_t& privateNonce)					
{
return to_string(current->blockNum) + current->data + current->initializeHash + to_string(privateNonce);
}

bool BlockChain::Control(string& hash)								
{
for (size_t i = 0; i < current->blockNum; i++)
if (hash[i] != *"0")
return false;

if (hash[current->blockNum] == *"0") return false;

return true;
}