#include <map>
#include <omp.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <crypt.h>
#include <fstream>
#include <string.h>
#include <unistd.h>
#include <iostream>

#include "crack.cpp"

using namespace std;

int main(int argc, char** argv)
{

int rank, nprocs;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == 0) {
cout << "master: there are " << nprocs - 1 << " slave processes" << endl;

string username = "project";

string salt_hash = get_salt_hash("/mirror/shadow.txt", username);	

if (salt_hash.empty()) {
cout << "File was not opened, terminating program." << endl;
MPI_Abort(MPI_COMM_WORLD, 0); 
exit(0);	
}

map<int, string> distrib = divide_alphabet(nprocs-1);	

for (int i = 1; i < nprocs; i++) {
MPI_Send(salt_hash.c_str(), 200, MPI_CHAR, i, 100, MPI_COMM_WORLD);	
MPI_Send(distrib[i].c_str(), 30, MPI_CHAR, i, 101, MPI_COMM_WORLD);	
}

if (distrib[0].length() > 0) {	
string letters = distrib[0];
cout << "master: " << letters << endl;

#pragma omp parallel num_threads(2)
{
if (omp_get_thread_num() == 0) {
for (int i = 0; i < letters.length(); i++) {
password_cracker(letters[i], salt_hash);
}
}

else {
char password[10];

MPI_Status status;
MPI_Recv(password, 10, MPI_CHAR, MPI_ANY_SOURCE, 200, MPI_COMM_WORLD, &status);	

cout << "Process " << status.MPI_SOURCE << " has cracked the password: " << password << endl;
cout << "Terminating all processes" << endl;

MPI_Abort(MPI_COMM_WORLD, 0); 
}
}
}

else {
char password[10];

MPI_Status status;
MPI_Recv(password, 10, MPI_CHAR, MPI_ANY_SOURCE, 200, MPI_COMM_WORLD, &status);	

cout << "Process " << status.MPI_SOURCE << " has cracked the password: " << password << endl;
cout << "Terminating all processes" << endl;

MPI_Abort(MPI_COMM_WORLD, 0);	
}
}

else {
char salt_hash[200], letters[30];

MPI_Recv(salt_hash, 200, MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
MPI_Recv(letters, 200, MPI_CHAR, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	

cout << "slave " << rank << ": " << letters << endl;

sleep(2);	

for (int i = 0; i < charToString(letters).length(); i++) {
password_cracker(letters[i], charToString(salt_hash));
}	
}

MPI_Finalize();

return 0;
}
