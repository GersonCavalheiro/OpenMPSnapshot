#ifndef PBWTZIP_HH
#define PBWTZIP_HH

#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>

#include "bwtzip_common.hh"
#include "bwtzip_mtf.hh"
#include "bwtzip_zle.hh"
#include "bwtzip_arith.hh"
#include "bwtzip_file.hh"
#include "clock.hh"
#include "log.hh"
#include "wclock.hh"

#define NUM_THREAD_STAGE_1 4
#define NUM_THREAD_STAGE_2 1
#define NUM_THREAD_STAGE_3 1

namespace pbwtzip {
using namespace std;
using namespace bwtzip;

typedef struct cnk_t {
int id;
vector<unsigned char> v;
} cnk_t;

extern unsigned int num_thread_stage_1;
extern unsigned int num_thread_stage_2;
extern unsigned int num_thread_stage_3;

extern unsigned int buffer_size;

extern cnk_t ***bufferR_1;
extern cnk_t ***buffer1_2;
extern cnk_t ***buffer2_3;
extern cnk_t ***buffer3_W;

extern bool read_completed;
extern bool ongoing_file_processing;

extern double stats_stages_time[];
extern int stats_lasted_longer_count[];
extern double stats_time_averages[];



void read_file(int bs, bwtzip::InputFile &infile, const unsigned long max_chunk_size);


template<typename T>
void stage_1(int bs, T bwtfxn) {
Log::pbwtzip::stage::started("stage_1", bs);
auto clk = new wClock();

#pragma omp parallel for num_threads(num_thread_stage_1) firstprivate(bs)
for (unsigned int i = 0; i < buffer_size; i++) {
cnk_t *c;

c = bufferR_1[bs][i];
Log::pbwtzip::stage::chunk_read("stage_1", "bufferR_1", bs, i, c);

if (c != nullptr && c->id == 1) {
bwtfxn(c->v);
}


buffer1_2[!bs][i] = c;
Log::pbwtzip::stage::chunk_written("stage_1", "buffer1_2", bs, i, c);
}

stats_stages_time[1] = clk->report();
Log::pbwtzip::stage::ended("read_file", stats_stages_time[1]);
delete clk;
}



void stage_2(int bs);


void stage_3(int bs);


void write_file(int bs, bwtzip::OutputFile &outfile);



void stats_update(int iter);


template<typename T>
void pbwtzipMain(int argc, char *argv[], std::string executableName, std::string algorithmName, T bwtfxn) {
using namespace std;

if (argc < 3 || argc > 5) {
cout << "USAGE: " << executableName << " infile outfile" << endl;
cout << "USAGE: " << executableName << " chunksize infile outfile" << endl;
cout << "USAGE: " << executableName << " threads_conf chunksize infile outfile" << endl;
cout << "\n\tthreads_conf eg: 5.1.1" << endl;
exit(EXIT_SUCCESS);
}


char threads_conf[20];
if (argc == 5) {
strcpy(threads_conf, argv[argc - 4]);
sscanf(threads_conf, "%u.%u.%u", &num_thread_stage_1, &num_thread_stage_2, &num_thread_stage_3);
} else {
num_thread_stage_1 = NUM_THREAD_STAGE_1;
num_thread_stage_2 = NUM_THREAD_STAGE_2;
num_thread_stage_3 = NUM_THREAD_STAGE_3;

sprintf(threads_conf, "%u.%u.%u", num_thread_stage_1, num_thread_stage_2, num_thread_stage_3);
}
buffer_size = max(num_thread_stage_1, num_thread_stage_2);
buffer_size = max(buffer_size, num_thread_stage_3);



const unsigned long max_chunk_size = (argc > 3) ? atol(argv[argc - 3]) : MAX_CHUNK_SIZE;
if (max_chunk_size < MIN_LENGTH_TO_COMPRESS) {
cout << "Chunk size too small!" << endl;
exit(EXIT_FAILURE);
}

remove(argv[argc - 1]);

InputFile infile(argv[argc - 2]);
OutputFile outfile(argv[argc - 1]);
outfile.append(encodeULL(BWTZIP_SIG));
for (unsigned int j = 0; j < 2; ++j) {
bufferR_1[j] = new cnk_t *[buffer_size];
buffer1_2[j] = new cnk_t *[buffer_size];
buffer2_3[j] = new cnk_t *[buffer_size];
buffer3_W[j] = new cnk_t *[buffer_size];

for (unsigned int i = 0; i < buffer_size; i++) {
bufferR_1[j][i] = nullptr;
buffer1_2[j][i] = nullptr;
buffer2_3[j][i] = nullptr;
buffer3_W[j][i] = nullptr;
}
}

if (LOG_PBWTZIP) {
cout << "[PBWTZIP] BWT Algorithm:         " << algorithmName << endl;
cout << "[PBWTZIP] Threads configuration: " << threads_conf << endl;
cout << "[PBWTZIP] Chunk Size:            " << max_chunk_size << endl;
}

if (LOG_STATISTICS_CSV) {
string filename = argv[argc - 1];
Log::pbwtzip::csv::start_new_line(filename, threads_conf, max_chunk_size);
}


auto clk = new wClock();

int bs = 0;
omp_set_nested(1);

int iter = 0;
while (ongoing_file_processing) {
Log::pbwtzip::started(iter, bs);

#pragma omp parallel sections firstprivate(bs)
{
#pragma omp section
read_file(bs, infile, max_chunk_size);

#pragma omp section
stage_1(bs, bwtfxn);

#pragma omp section
stage_2(bs);

#pragma omp section
stage_3(bs);

#pragma omp section
write_file(bs, outfile);
}
if (LOG_STATISTICS || LOG_STATISTICS_CSV) stats_update(iter);

Log::pbwtzip::ended(iter, bs);

bs = !bs;
iter++;
}
Log::pbwtzip::stats_print_summary(--iter, stats_lasted_longer_count, stats_time_averages);

Log::pbwtzip::print_total_exec_time(clk->report());

if (LOG_STATISTICS_CSV) Log::pbwtzip::csv::print_statistics_line();
delete clk;


for (unsigned int j = 0; j < 2; ++j) {
delete[] bufferR_1[j];
delete[] buffer1_2[j];
delete[] buffer2_3[j];
delete[] buffer3_W[j];
}
delete[] bufferR_1;
delete[] buffer1_2;
delete[] buffer2_3;
delete[] buffer3_W;
}
}

#endif 
