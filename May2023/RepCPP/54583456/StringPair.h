#ifndef _STRING_PAIR_
#define _STRING_PAIR_
#include <string>

#define FIRST 0
#define LAST  size()-1


using namespace std;

struct CompareResult{
CompareResult(){
module = -1;
initial = false;
}
CompareResult(int m, bool i, string a){
module = m;
initial = i;
result = a;
}
int module;
bool initial;
string result;
};

class StringPair{
public:
string x;
string y;
bool compared;
CompareResult result;
StringPair(){
x = "";
y = "";
compared = false;
}
StringPair(string a, string b){
x = a;
y = b;
compared = false;
}

string mergePair(StringPair a, int module, bool initial){

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

string printPair(){
return x + ", " + y;
}

int calcularInitialTrue(string s, string t){
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

int calcularInitialFalse(string s, string t){
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

void calcularCompareResult(){
CompareResult comp(0,false,"");

string first  = x;
string second = y;

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

comp.result = mergePair(StringPair(first, second),comp.module,comp.initial);

result = comp;
}

void calcResult(){
if(!compared){
calcularCompareResult();
compared = true;
}
}
};

inline bool operator==(const StringPair& first, const StringPair& second){
if((first.x.compare(second.x) == 0) && (first.y.compare(second.y) == 0)){
return true;
}else if((first.x.compare(second.y) == 0) && (first.y.compare(second.x) == 0)){
return true;
}
return false;
}

inline bool operator!=(const StringPair& first, const StringPair& second){
return (!(first == second));
}



#endif
