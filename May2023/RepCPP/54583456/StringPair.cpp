#include <string>
#include <omp.h>
#include "StringPair.h"

using namespace std;

StringPair::StringPair(){
x = "";
y = "";
compared = false;
}

StringPair::StringPair(string a, string b){
x = a;
y = b;
compared = false;
}

string StringPair::mergePair(StringPair a, int module, bool initial){

string result;
string cuttedY;
if(initial){
cuttedY = a.y.substr(0, a.y.size() - module);
result = cuttedY;
result.append(a.x);
}else{
cuttedY =  a.y.substr(module,  a.y.size() - module);
result = a.x;
result.append(cuttedY);
}

return result;
}

string StringPair::printPair(){
return x + ", " + y;
}

int StringPair::calcularInitialTrue(string s, string t){
int length = 0;

string busca;
for(int i = 0; i < t.size(); i++){
busca = t.substr(t.size()-1-i);


if(s.substr(0,i+1) == busca){
length = busca.size();
}
}
return length;
}

int StringPair::calcularInitialFalse(string s, string t){
int length = 0;

string busca;
for(int i = 0; i < s.size(); i++){
busca = s.substr(s.size()-1-i);


if(t.substr(0,i+1) == busca){
length = busca.size();
}
}

return length;
}

void StringPair::calcularCompareResult(bool forceMerge){

CompareResult comp(0,false,"");

string first  = x;
string second = y;
int minModule = (int)((first.size()+second.size())/2 * 0.05);
if(minModule <= 0){ minModule = 1;}

int initialTrue;
int initialFalse;

#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{initialTrue = calcularInitialTrue(first,second);}
#pragma omp section
{initialFalse = calcularInitialFalse(first,second);}
}
}

if(initialTrue >= initialFalse){
comp.module  = initialTrue;
comp.initial = true;
} else {
comp.module  = initialFalse;
comp.initial = false;
}

comp.result = (comp.module < minModule && !forceMerge) ? "" : mergePair(StringPair(first, second),comp.module,comp.initial);


result = comp;
}

void StringPair::calcResult(bool force){
if(!compared){
calcularCompareResult(force);
compared = true;
}
}