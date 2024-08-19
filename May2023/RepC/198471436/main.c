#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <openssl/md5.h>
char response[8];
char* secret;
struct timeval start, stop;
unsigned long long int combinations;
unsigned int found = 0, length, threads;
const char CHARSET[] = "@$#?!=+%abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const unsigned int CHARSET_LENGTH = strlen(CHARSET);
void compareMD5(char* input) {
unsigned char digest[MD5_DIGEST_LENGTH];
MD5_CTX context;
MD5_Init(&context);
MD5_Update(&context, input, strlen(input));
MD5_Final(digest, &context);
char output[33];
for (unsigned int i = 0; i < 16; i++)
sprintf(&output[i * 2], "%02x", (unsigned int)digest[i]);
if (strcmp(output, secret) == 0) {
found = 1;
strcpy(response, input);
}
}
void run(unsigned long long int i) {
char* input = malloc(length + 1);
input[length] = '\0';
for (unsigned int j = 0; j < length; j++) {
input[j] = CHARSET[(unsigned long long int)floor(i / pow(CHARSET_LENGTH, j)) % CHARSET_LENGTH];
}
compareMD5(input);
free(input);
}
void sequential() {
for (unsigned long long int i = 0; i < combinations; i++) {
if (found == 0) run(i);
}
}
void parallel() {
#pragma omp parallel for shared(found) schedule(dynamic)
for (unsigned long long int i = 0; i < combinations; i++) {
if (found == 0) run(i);
}
}
int main(unsigned int argc, char* argv[]) {
if (argc < 3) {
printf("Usage: %s <hash> <length> <threads>", argv[0]);
return EXIT_FAILURE;
}
secret = argv[1];
length = strtol(argv[2], NULL, 10);
threads = strtol(argv[3], NULL, 10);
combinations = pow(CHARSET_LENGTH, length);
gettimeofday(&start, NULL);
if (threads == 0) {
printf("Sequential\n");
sequential();
} else {
printf("Parallel with %i threads\n", threads);
omp_set_num_threads(threads);
parallel();
}
gettimeofday(&stop, NULL);
uint seconds = stop.tv_sec - start.tv_sec;
uint milliseconds = (stop.tv_usec - start.tv_usec) >= 0
? (stop.tv_usec - start.tv_usec)
: 1000000 - stop.tv_usec;
printf("Secret: %s\nElapsed Time: %d.%d\n", response, seconds, milliseconds);
return EXIT_SUCCESS;
}
