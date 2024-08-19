#pragma once


#include <stdio.h>
#include <string.h>
#include <cctype>
#include <iostream>

using namespace std;

class KeyValue {
public :
string key;
string value;
KeyValue * left;
KeyValue * right;

KeyValue ();

KeyValue (string k, string v);

void operator = (string k);
};

class AssociativeArray {

int amount;

public:

AssociativeArray ();
KeyValue * root;

KeyValue* insert_string(KeyValue * x, string z);


void output(KeyValue * x);
bool checkExist(KeyValue *x, string k);

KeyValue* search (KeyValue * x, string k);

string& operator[] (string k);
void operator << (string k);

void operator << (KeyValue second);
void operator >> (string k);
KeyValue* minimum (KeyValue* x);

KeyValue* takeOut(KeyValue * root, string z);

KeyValue* concat (KeyValue *first, KeyValue *second);

KeyValue* operator+=(const AssociativeArray A);
KeyValue* operator+(const AssociativeArray A);

void operator = (KeyValue* R);


KeyValue* insert_node(KeyValue * x, KeyValue y);
};
