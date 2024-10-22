#include <stdlib.h>		
#include <iostream>

#include "Sha.h"
#include "BlockChain.h"
#include <omp.h>
#include <iomanip>

using namespace std;


int main()
{
string inputText = "Bize her yer Trabzon! Bolumun en yakisikli hocasi Ibrahim Hoca'dir";		
string initializeHash = "0000000000000000000000000000000000000000000000000000000000000000";

size_t threadNums;
cout << "THREAD NUMS: ";
cin >> threadNums;

size_t nonce = 0;
bool flag = false;
size_t acceptNonce;			
size_t acceptCounter;		

WORD* sha256K = InitializeK();
BlockChain blockChain;
blockChain.AddBack(inputText, initializeHash);
cout << "\n# BLOCK: " << blockChain.current->blockNum << "\t tid: " << omp_get_thread_num() << endl;
cout << "Hash initialized: " << blockChain.current->initializeHash << "\t nonce: " << nonce << "\t \t tid: " << omp_get_thread_num() << endl;

size_t entryCounter;
omp_lock_t lock;
omp_init_lock(&lock);
double start = omp_get_wtime(), end;
double globalTime = start;
int sec, h, m;

#pragma omp parallel num_threads(threadNums)
{
string privateText, privateHash;
size_t privateNonce;

#pragma omp critical 
{
privateNonce = nonce;
nonce++;
}
#pragma omp barrier

while (true)
{
privateText = blockChain.GetText(privateNonce);		
privateHash = Sha256(privateText, sha256K);			

if (blockChain.Control(privateHash))
{
#pragma omp single nowait
{
acceptCounter = 0;
entryCounter = 0;
acceptNonce = privateNonce;
flag = true;
while (acceptCounter < threadNums / 2) {}		
while (entryCounter < threadNums - 1) {}		

end = omp_get_wtime();
cout << "Hash found: " << privateHash << "\t \t nonce: " << acceptNonce << "\t \t tid: " << omp_get_thread_num() << endl << "Text: " << privateText << endl << "Block Run-time(s): " << std::setprecision(3) << fixed << (end - start);
start = omp_get_wtime();

sec = int(end - globalTime);
h = int(sec / 3600);
m = int((sec - (3600 * h)) / 60);
cout << "\t Total Run-time: " << h << ":" << m << ":" << (sec - (3600 * h) - (m * 60)) << endl;

blockChain.AddBack(privateText, privateHash);			
nonce = 0;
privateNonce = nonce;
nonce++;

cout << endl << endl << "# BLOCK: " << blockChain.blockCounter << "\t tid: " << omp_get_thread_num() << endl;
cout << "Hash initialized: " << blockChain.current->initializeHash << "\t nonce: " << privateNonce << "\t \t tid: " << omp_get_thread_num() << endl;

flag = false;
}
}
else {
omp_set_lock(&lock);
privateNonce = nonce;
nonce++;
omp_unset_lock(&lock);
}

if (flag) 
{
privateText = blockChain.GetText(acceptNonce);
privateHash = Sha256(privateText, sha256K);

#pragma omp critical
{
if (acceptCounter < threadNums / 2)			
{
if (blockChain.Control(privateHash)) {
cout << "Hash accepted: " << privateHash << "\t \t nonce: " << acceptNonce << "\t \t tid: " << omp_get_thread_num() << endl;
++acceptCounter;
}
else cout << "Error tid: " << omp_get_thread_num() << endl;
}
++entryCounter;
}
while (flag) {}

omp_set_lock(&lock);
privateNonce = nonce;
nonce++;
omp_unset_lock(&lock);
}
}
}

omp_destroy_lock(&lock);
return 0;
}
