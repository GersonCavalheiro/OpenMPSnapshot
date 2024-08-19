#include "pbwtzip.hh"

namespace pbwtzip {
unsigned int num_thread_stage_1;
unsigned int num_thread_stage_2;
unsigned int num_thread_stage_3;

unsigned int buffer_size;

cnk_t ***bufferR_1 = new cnk_t **[2];
cnk_t ***buffer1_2 = new cnk_t **[2];
cnk_t ***buffer2_3 = new cnk_t **[2];
cnk_t ***buffer3_W = new cnk_t **[2];


bool read_completed = false;
bool ongoing_file_processing = true;

double stats_stages_time[5];
int stats_lasted_longer_count[5];
double stats_time_averages[5];
}


void pbwtzip::read_file(int bs, bwtzip::InputFile &infile, const unsigned long max_chunk_size) {
Log::pbwtzip::stage::started("read_file", bs);
auto clk = new wClock();

for (unsigned int i = 0; i < buffer_size; i++) {
cnk_t *c;
auto v = infile.extractAtMost(max_chunk_size);

if (v.empty()) {
if (LOG_STAGE) printf("[PBWTZIP][read_file] Read COMPLETED.");
c = nullptr;
pbwtzip::read_completed = true;
} else {
c = new cnk_t();

c->v = v;

c->id = 1;
if (c->v.size() < bwtzip::MIN_LENGTH_TO_COMPRESS) {
c->id = 0;
}
}

bufferR_1[!bs][i] = c;
Log::pbwtzip::stage::chunk_written("read_file", "bufferR_1", !bs, i, c);
}
stats_stages_time[0] = clk->report();
Log::pbwtzip::stage::ended("read_file", stats_stages_time[0]);
delete clk;
}


void pbwtzip::stage_2(int bs) {
Log::pbwtzip::stage::started("stage_2", bs);
auto clk = new wClock();

#pragma omp parallel for num_threads(num_thread_stage_2) firstprivate(bs)
for (unsigned int i = 0; i < buffer_size; i++) {
cnk_t *c;

c = buffer1_2[bs][i];
Log::pbwtzip::stage::chunk_read("stage_2", "buffer1_2", bs, i, c);

if (c != nullptr && c->id == 1) {
bwtzip::mtf2(c->v);
bwtzip::zleWheeler(c->v);
}

buffer2_3[!bs][i] = c;
Log::pbwtzip::stage::chunk_written("stage_2", "buffer2_3", !bs, i, c);
}

stats_stages_time[2] = clk->report();
Log::pbwtzip::stage::ended("stage_2", stats_stages_time[2]);
delete clk;
}



void pbwtzip::stage_3(int bs) {
Log::pbwtzip::stage::started("stage_3", bs);
auto clk = new wClock();

#pragma omp parallel for num_threads(num_thread_stage_3) firstprivate(bs)
for (unsigned int i = 0; i < buffer_size; i++) {
cnk_t *c;

c = buffer2_3[bs][i];
Log::pbwtzip::stage::chunk_read("stage_3", "buffer2_3", bs, i, c);

if (c != nullptr && c->id == 1) {
bwtzip::arith(c->v);
}

buffer3_W[!bs][i] = c;
Log::pbwtzip::stage::chunk_written("stage_3", "buffer3_W", !bs, i, c);
}

stats_stages_time[3] = clk->report();
Log::pbwtzip::stage::ended("stage_3", stats_stages_time[3]);
delete clk;
}


void pbwtzip::write_file(int bs, bwtzip::OutputFile &outfile) {
Log::pbwtzip::stage::started("write_file", bs);
auto clk = new wClock();

for (unsigned int i = 0; i < buffer_size; i++) {
cnk_t *c;

c = buffer3_W[bs][i];
Log::pbwtzip::stage::chunk_read("write_file", "buffer3_W", bs, i, c);

if (c != nullptr) {
vector<unsigned char> id;
id.push_back((unsigned char) c->id);

outfile.append(bwtzip::encodeUL(c->v.size()));
outfile.append(id);
outfile.append(c->v);
delete c;
} else if (read_completed) {
pbwtzip::ongoing_file_processing = false;
if (LOG_STAGE) printf("[PBWTZIP][write_file] Write COMPLETED.");
}
}

stats_stages_time[4] = clk->report();
Log::pbwtzip::stage::ended("write_file", stats_stages_time[4]);
delete clk;
}



void pbwtzip::stats_update(int iter) {
double max_time = 0;
int max_stage = 0;

for (int i = 0; i < 5; i++) {
if (stats_stages_time[i] > max_time) {
max_time = stats_stages_time[i];
max_stage = i;
}
}
stats_lasted_longer_count[max_stage]++;

if (iter == 0) {
for (int i = 0; i < 5; i++)
stats_time_averages[i] = stats_stages_time[i];
} else {
for (int i = 0; i < 5; i++)
stats_time_averages[i] = (stats_stages_time[i] + (iter * stats_time_averages[i])) / (iter + 1);
}
}
