#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;
#include <bits/stdc++.h>

const int MAXS = 550000;

const int MAXC = 256;

int** out;

int f[MAXS];

int g[MAXS][MAXC];

int buildMatchingMachine(vector<string> arr,  int  k)
{



int states = 1;

out = new int*[MAXS];
for(int i=0;i<MAXS;i++)
out[i] = new int[k+1]();

DEBUG2("MAX = %d, k=%d, Size of out= %d",MAXS, k, sizeof(out));

memset(g, -1, sizeof(g));

for (int i = 0; i < k; ++i)
{
const string &word = arr[i];
int currentState = 0;

for (int j = 0; j < word.size(); ++j)
{
int ch = word[j];



if (g[currentState][ch] == -1)
g[currentState][ch] = states++;

currentState = g[currentState][ch];
DEBUG2("pattern=%d CurrentState = %d",i, g[currentState][ch]);
}
DEBUG2("CurrentState = %d", currentState);
if(currentState!=0) {
int outSize = out[currentState][0];
out[currentState][outSize+1] = i;
out[currentState][0]++;
DEBUG2("Out currentState=%d patIndex=%d",currentState,out[currentState][outSize+1]);
}
}

for (int ch = 0; ch < MAXC; ++ch)
if (g[0][ch] == -1)
g[0][ch] = 0;


memset(f, -1, sizeof f);

queue<int> q;


for (int ch = 0; ch < MAXC; ++ch)
{
if (g[0][ch] != 0)
{
f[g[0][ch]] = 0;
q.push(g[0][ch]);
}
}

for(int i=0;i<q.size();i++)
{
if (!q.empty()) {
int state = 0;

state = q.front();
q.pop();

for (int ch = 0; ch < MAXC; ++ch)
{
if (g[state][ch] != -1)
{
int failure = f[state];

while (g[failure][ch] == -1)
failure = f[failure];

failure = g[failure][ch];
f[g[state][ch]] = failure;


if(failure!=0)
{
int outSize = out[g[state][ch]][0];
int failureOutSize = out[failure][0];

for(int i=1;i<=failureOutSize;i++)
{
out[g[state][ch]][outSize+1] = out[failure][i];
out[g[state][ch]][0]++;
outSize++;
}
}
q.push(g[state][ch]);

}
}
}
}

return states;
}

int findNextState(int currentState, char nextInput)
{
int answer = currentState;
int ch = nextInput;

while (g[answer][ch] == -1)
answer = f[answer];

return g[answer][ch];
}
