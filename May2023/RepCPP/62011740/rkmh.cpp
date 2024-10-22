#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <vector>
#include <set>
#include <unordered_set>
#include <cstdint>
#include <string>
#include <list>
#include <sstream>
#include <zlib.h>
#include <omp.h>
#include <getopt.h>
#include <map>
#include <unordered_map>
#include "mkmh.hpp"
#include "equiv.hpp"
#include "json.hpp"
#include "HASHTCounter.hpp"
#include "kseq_reader.hpp"

using json = nlohmann::json;


using namespace std;
using namespace mkmh;
using namespace KSR;





vector<string> split(string s, char delim){
vector<string> ret;
stringstream sstream(s);
string temp;
while(getline(sstream, temp, delim)){
ret.push_back(temp);
}
return ret;

}

string join(vector<string> splits, string glue){                                              
string ret = "";
for (int i = 0; i < splits.size(); i++){
if (i != 0){
ret += glue;
}
ret += splits[i];
}

return ret;
}

void print_help(char** argv){
cerr << "Usage: " << argv[0] << " { classify | call | hash | stream } [options]" << endl
<< "    classify: match each read to the reference it most closely resembles using MinHash sketches." << endl
<< "    call: determine the SNPs and 1-2bp INDELs that differ between a set of reads and their closest reference." << endl
<< "    hash: compute the MinHash sketches of a set of reads and/or references (for interop with Mash/sourmash)." << endl
<< "    stream: classify reads or sequences from STDIN. Low memory, real time, but possibly lower precision." << endl
<< "    filter: spit out reads that meet thresholds for match to ref, uniqueness, etc." << endl
<< endl;
}

void help_classify(char** argv){
cerr << "Usage: " << argv[0] << " classify [options]" << endl
<< "Options:" << endl
<< "--reference/-r   <REF>" << endl
<< "--fasta/-f   <FASTAFILE>" << endl
<< "--kmer/-k    <KMERSIZE>" << endl
<< "--sketch-size/-s <SKETCHSIZE>" << endl
<< "--threads/-t <THREADS>" << endl
<< "--min-kmer-occurence/-M <MINOCCURENCE>" << endl
<< "--min-matches/-N <MINMATCHES>" << endl
<< "--min-diff/-D    <MINDIFFERENCE>" << endl
<< "--min-informative/-I <MAXSAMPLES> only use kmers present in fewer than MAXSAMPLES" << endl
<< endl;
}

void help_call(char** argv){
cerr << "Usage: " << argv[0] << " call [options]" << endl
<< "Options:" << endl
<< "--reference/-r <REF>      reference genomes in fasta format." << endl
<< "--fasta/-f <FASTA>        a fasta file to call mutations in relative to the reference." << endl
<< "--threads/-t <THREADS>    the number of OpenMP threads to utilize." << endl
<< "--window-len/-w <WINLEN>  the width of the sliding window to use for calculating average depth." << endl
<< "--depth/-d                output tab-separated values for position, avg depth, instantaneous depth, and rescued depth." << endl
<< endl;
}

void help_hash(char** argv){
cerr << "Usage: " << argv[0] << " hash [options]" << endl
<< "Options:" << endl
<< "--fasta/-f  <FASTA>          fasta file to hash." << endl
<< "--reference/-r   <REF>       reference file to hash." << endl
<< "--sketch-size/-s <SKTCHSZ>   sketch size." << endl
<< "--kmer/-k <KMER>             kmer size to hash." << endl
<< "--min-kmer-occurrence <M>    Minimum kmer occurrence. Failing kmers are removed from sketch." << endl
<< "--min-informative/-I  <I>    Maximum number of samples a kmer can occur in before it is removed" << endl
<< "--threads/-t <THREADS>       number of OpenMP threads to utilize." << endl
<< "--wabbitize /-w              output Vowpal Wabbit compatible vectors" << endl;
}

void help_stream(char** argv){
cerr << "Usage: " << argv[0] << " stream [options]" << endl
<< "Options:" << endl
<< "--input-stream/-i   classify reads coming from STDIN" << endl
<< "--output-reads/-z   output reads that pass the filters set, rather than classifications." << endl 
<< "--reference/-r   <REF>" << endl
<< "--fasta/-f   <FASTAFILE>" << endl
<< "--kmer/-k    <KMERSIZE>" << endl
<< "--sketch-size/-s <SKETCHSIZE>" << endl
<< "--ref-sketch / -S <REFSKTCHSZ>" << endl
<< "--threads/-t <THREADS>" << endl
<< "--min-kmer-occurence/-M <MINOCCURENCE>" << endl
<< "--min-matches/-N <MINMATCHES>" << endl
<< "--min-diff/-D    <MINDIFFERENCE>" << endl
<< "--min-informative/-I <MAXSAMPLES> only use kmers present in fewer than MAXSAMPLES" << endl
<< "--kmer-depth-map / -p <mapfile> the kmer depth map to use for min_kmer_occurence" << endl
<< "--ref-sample-map / -q <mapfile> the sample depth map for reference sample filtering." << endl
<< "--pre-fasta / -F  a file containing sketches in JSON format for reads." << endl
<< "--pre-reference / -R a file containing pre-hashed reference genomes in JSON format." << endl
<< endl;

}

void help_filter(char** argv){
cerr << "Usage: " << argv[0] << " filter [options]" << endl
<< "Options: " << endl
<< "--input-stream/-i   classify reads coming from STDIN" << endl
<< "--output-reads/-z   output reads that pass the filters set, rather than classifications." << endl 
<< "--reference/-r   <REF>" << endl
<< "--fasta/-f   <FASTAFILE>" << endl
<< "--kmer/-k    <KMERSIZE>" << endl
<< "--sketch-size/-s <SKETCHSIZE>" << endl
<< "--ref-sketch / -S <REFSKTCHSZ>" << endl
<< "--threads/-t <THREADS>" << endl
<< "--min-kmer-occurence/-M <MINOCCURENCE>" << endl
<< "--min-matches/-N <MINMATCHES>" << endl
<< "--min-diff/-D    <MINDIFFERENCE>" << endl
<< "--min-informative/-I <MAXSAMPLES> only use kmers present in fewer than MAXSAMPLES" << endl
<< "--kmer-depth-map / -p <mapfile> the kmer depth map to use for min_kmer_occurence" << endl
<< "--ref-sample-map / -q <mapfile> the sample depth map for reference sample filtering." << endl
<< "--pre-fasta / -F  a file containing sketches in JSON format for reads." << endl
<< "--pre-reference / -R a file containing pre-hashed reference genomes in JSON format." << endl
<< endl;
}

void parse_fastas(vector<char*>& files,
unordered_map<string, char*>& ret_to_seq,
unordered_map<string, int>& ret_to_len){



for (auto f : files){
gzFile fp;
kseq_t *seq;
int l;
fp = gzopen(f, "r");
seq = kseq_init(fp);
while ((l = kseq_read(seq)) >= 0) {
to_upper(seq->seq.s, seq->seq.l);

char * x = new char[seq->seq.l];
memcpy(x, seq->seq.s, seq->seq.l);
ret_to_seq[string(seq->name.s)] = x; 
ret_to_len[seq->name.s] = seq->seq.l;
} 
gzclose(fp);
}
}

void parse_fastas(vector<char*>& files,
vector<string>& seq_keys,
vector<char*>& seq_seqs,
vector<int>& seq_lens){

kseq_t *seq;
for (int i = 0; i < files.size(); i++){
char* f = files[i];
gzFile fp;
int l;
fp = gzopen(f, "r");
seq = kseq_init(fp);
while ((l = kseq_read(seq)) >= 0) {
to_upper(seq->seq.s, seq->seq.l);

char * x = new char[seq->seq.l];
memcpy(x, seq->seq.s, seq->seq.l);
seq_keys.push_back( string(seq->name.s) );
seq_seqs.push_back(x);
seq_lens.push_back(seq->seq.l);
} 
gzclose(fp);
}   
kseq_destroy(seq);
}

void parse_fastas(vector<char*>& files,
vector<string>& seq_keys,
vector<char*>& seq_seqs,
vector<int>& seq_lens,
vector<string>& seq_quals){

kseq_t *seq;
for (int i = 0; i < files.size(); i++){
char* f = files[i];
gzFile fp;
int l;
fp = gzopen(f, "r");
seq = kseq_init(fp);
while ((l = kseq_read(seq)) >= 0) {
to_upper(seq->seq.s, seq->seq.l);

char * x = new char[seq->seq.l];
memcpy(x, seq->seq.s, seq->seq.l);
seq_keys.push_back(seq->name.s);
seq_seqs.push_back(x);
seq_lens.push_back(seq->seq.l);
seq_quals.emplace_back(seq->qual.s);
} 
gzclose(fp);
}   
kseq_destroy(seq);
}
void hash_sequences(vector<string>& keys,
unordered_map<string, char*>& name_to_seq,
unordered_map<string, int>& name_to_length,
vector<int>& kmer,
unordered_map<string, hash_t*>& ret_to_hashes,
unordered_map<string, int>& ret_to_hash_num){

#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){

vector<hash_t> r = calc_hashes(name_to_seq[keys[i]], name_to_length[keys[i]], kmer);
ret_to_hashes[keys[i]] = &(*r.begin());
ret_to_hash_num[keys[i]] = r.size();
}

}
void hash_sequences(vector<string>& keys,
vector<char*>& seqs,
vector<int>& lengths,
vector<hash_t*>& hashes,
vector<int>& hash_lengths,
vector<int>& kmer,
HASHTCounter& read_hash_counter,
HASHTCounter& ref_hash_counter,
bool doReadDepth,
bool doReferenceDepth){


if (doReadDepth){
#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){
vector<hash_t> r = calc_hashes(seqs[i], lengths[i], kmer);
hashes[i] = &(*r.begin());
hash_lengths[i] = r.size();
{
for (int j = 0; j < hash_lengths[i]; j++){

read_hash_counter.increment( (uint64_t) hashes[i][j] );
}
}
}
}
else if (doReferenceDepth){
#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){
vector<hash_t> r = calc_hashes(seqs[i], lengths[i], kmer);
hashes[i] = &(*r.begin());
hash_lengths[i] = r.size();

set<hash_t> sample_set (hashes[i], hashes[i] + hash_lengths[i]);
{
for (auto x : sample_set){
ref_hash_counter.increment( x );
}
}
}
}

else{
#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){
vector<hash_t> r = calc_hashes(seqs[i], lengths[i], kmer);
hashes[i] =&(*r.begin());
hash_lengths[i] = r.size();
}

}

}




void hash_sequences(vector<string>& keys,
vector<char*>& seqs,
vector<int>& lengths,
vector<hash_t*>& hashes,
vector<int>& hash_lengths,
vector<int>& kmer,
unordered_map<hash_t, int>& read_hash_to_depth,
unordered_map<hash_t, int>& ref_to_sample_depth,
bool doReadDepth,
bool doReferenceDepth){

if (doReadDepth){
#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){
vector<hash_t> r = calc_hashes(seqs[i], lengths[i], kmer);
hashes[i] = &(*r.begin());
hash_lengths[i] = r.size();
{
for (int j = 0; j < hash_lengths[i]; j++){
#pragma omp critical 
++read_hash_to_depth[ hashes[i][j] ];
}
}
}
}
else if (doReferenceDepth){
#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){
vector<hash_t> r = calc_hashes(seqs[i], lengths[i], kmer);
hashes[i] = &(*r.begin());
hash_lengths[i] = r.size();

set<hash_t> sample_set (hashes[i], hashes[i] + hash_lengths[i]);
{
for (auto x : sample_set){
#pragma omp critical 
++ref_to_sample_depth[x];
}
}
}
}

else{
#pragma omp parallel for
for (int i = 0; i < keys.size(); i++){
vector<hash_t> r = calc_hashes(seqs[i], lengths[i], kmer);
hashes[i] = &(*r.begin());
hash_lengths[i] = r.size();
}

}

}

string sketch_to_json(string key,
vector<hash_t> mins,
int sketchlen,
vector<int> kmer,
int sketch_size){

}
void sketches_to_jsons(vector<string>& keys,
vector<vector<hash_t> >& mins,
vector<int>& sketchlens,
vector<int>& kmer,
int sketch_size){
cout << "[" << endl;
for (int i = 0; i < keys.size(); ++i){
cout << sketch_to_json(keys[i], mins[i], sketchlens[i], kmer, sketch_size);
}
cout << "]" << endl;
}


void rkmh_binary_output(vector<string> keys,
vector<vector<hash_t> >& mins,
vector<int> sketchlens,
vector<int>& kmer,
int sketch_size){

}

void print_wabbit(string key,
vector<hash_t>& mins,
int sketch_len,
int sketch_size,
vector<int> counts,
vector<int> kmers,
string label = "XYX",
string nspace = "vir"){

key = join( split(key, '|'), "_");
cout << label << " 1.0 " << "`" << key << "|" << nspace;
if (!counts.empty()){
for (int i = 0; i < sketch_len; ++i){
cout << " " << mins[i] << ":" << counts[i];
}
}
else{
for (int i = 0; i < sketch_len; ++i){
cout << " " << mins[i] << ":1";
}

}
cout << " |sketch k:" << kmers[0] << " " << "s:" << sketch_size;
cout << endl;
}

json dump_hash_json(string key, int seqLen,
vector<hash_t> mins,
vector<int> kmer, 
int sketchLen,
string alphabet = "ATGC",
string hash_type = "MurmurHash3_x64_128",
bool canonical = true,
int hash_bits = 64,
int hash_seed = 42
){
json j;
j["name"] = key;

stringstream kstr;
for (int i = 0; i < kmer.size(); i++){
kstr << kmer[i];
if (i < kmer.size() - 1){
kstr << " ";
}
}
j["kmer"] = kstr.str();
j["alphabet"] = alphabet;
j["preserveCase"] = "false";
j["canonical"] = (canonical ? "true" : "false");
j["hashType"] = hash_type;
j["hashBits"] = hash_bits;
j["hashSeed"] = hash_seed;
j["seqLen"] = seqLen;
j["sketches"] = {
{"name", key},
{"length", sketchLen},
{"comment", ""},
{"hashes", mins}
};

return j;
}

json dump_hashes(vector<string> keys,
vector<int> seqlens,
vector<vector<hash_t>> hashes,
vector<int> kmer,
int sketch_size){

json j;
for (int i = 0; i < keys.size(); i++){
j.push_back({
{"name", keys[i]},
{"alphabet", "ATGC"},
{"canonical", "false"},
{"hashBits", 64},
{"hash_type", "MurmurHash3_x64_128"},
{"hash_seed", 42},
{"sketches", hashes[i]},
{"length", sketch_size},
{"kmer", kmer},
{"preserveCase", "false"}
});
}

return j;
}

std::tuple<vector<string> ,
vector<int> ,
vector<vector<hash_t> > ,
vector<int> ,
int> load_hashes(json jj){
cout << jj << endl;
cerr << "Loading not implemented" << endl;
exit(1);

}

std::tuple<vector<string>,
vector<int> ,
vector<vector<hash_t> >,
vector<int> ,
int> load_hashes(istream ifi){
json jj;
ifi >> jj;
return load_hashes(jj);
}




std::tuple<vector<string> ,
vector<int> ,
vector<vector<hash_t> >,
vector<int>,
int> load_hashes(string filename){

}

int main_stream(int argc, char** argv){
vector<char*> ref_files;
vector<char*> read_files;
vector<char*> pre_read_files;
vector<char*> pre_ref_files;

vector<int> kmer;

int sketch_size = 1000;
int threads = 1;
int min_kmer_occ = -1;
int min_matches = -1;
int min_diff = 0;
int max_samples = 100000;

string read_kmer_map_file = "";
string ref_kmer_map_file = "";

bool doReadDepth = false;
bool doReferenceDepth = false;

bool useHASHTs = false;
int ref_sketch_size = 0;

bool streamify_me_capn = false;
bool output_reads = false;
bool merge_sketch = false;


int c;
int optind = 2;

if (argc <= 2){
help_stream(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"kmer", no_argument, 0, 'k'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"sketch-size", required_argument, 0, 's'},
{"ref-sketch", required_argument, 0, 'S'},
{"threads", required_argument, 0, 't'},
{"min-kmer-occurence", required_argument, 0, 'M'},
{"min-matches", required_argument, 0, 'N'},
{"min-diff", required_argument, 0, 'D'},
{"max-samples", required_argument, 0, 'I'},
{"pre-reads", required_argument, 0, 'F'},
{"pre-references", required_argument, 0, 'R'},
{"read-kmer-map-file", required_argument, 0, 'p'},
{"ref-kmer-map-file", required_argument, 0, 'q'},
{"in-stream", no_argument, 0, 'i'},
{"output-reads", no_argument, 0, 'z'},
{"merge-sketch", no_argument, 0, 'm'},
{0,0,0,0}
};

int option_index = 0;
c = getopt_long(argc, argv, "zmhdk:f:r:s:S:t:M:N:I:R:F:p:q:iD:", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 'm':
merge_sketch = true;
break;
case 'F':
pre_read_files.push_back(optarg);
break;
case 'R':
pre_ref_files.push_back(optarg);
break;
case 'p':
read_kmer_map_file = optarg;
break;
case 'q':
ref_kmer_map_file = optarg;
break;
case 't':
threads = atoi(optarg);
break;
case 'r':
ref_files.push_back(optarg);
break;
case 'f':
read_files.push_back(optarg);
break;
case 'k':
kmer.push_back(atoi(optarg));
break;
case 'N':
min_matches = atoi(optarg);
break;
case 'D':
min_diff = atoi(optarg);
break;
case '?':
case 'h':
print_help(argv);
exit(1);
break;
case 's':
sketch_size = atoi(optarg);
break;
case 'S':
useHASHTs = true;
ref_sketch_size = 3 * atoi(optarg);
break;
case 'M':
min_kmer_occ = atoi(optarg);
doReadDepth = true;
break;
case 'I':
max_samples = atoi(optarg);
doReferenceDepth = true;
break;
case 'i':
streamify_me_capn = true;
break;
case 'z':
output_reads = true;
break;
default:
print_help(argv);
abort();

}
}

if (sketch_size == -1){
cerr << "Sketch size unset." << endl
<< "Will use the default sketch size of s = 1000" << endl;
sketch_size = 1000;
}

if (kmer.size() == 0){
cerr << "No kmer size(s) provided. Will use a default kmer size of 16." << endl;
kmer.push_back(16);
}


omp_set_num_threads(threads);
HASHTCounter* read_hash_counter;
HASHTCounter* ref_hash_counter;
if (doReadDepth){
read_hash_counter  = new HASHTCounter(200000000);
}
if (doReferenceDepth){
ref_hash_counter = new HASHTCounter(200000000);
}


vector<string> ref_keys;
vector<char*> ref_seqs;
vector<int> ref_lens;

vector<string> read_keys;
vector<char*> read_seqs;
vector<int> read_lens;



bool stream_files = false;

if (!ref_files.empty()){
parse_fastas(ref_files, ref_keys, ref_seqs, ref_lens);
}
if (!read_files.empty() && !stream_files){
parse_fastas(read_files, read_keys, read_seqs, read_lens);
}

char** rseqs = new char*[read_seqs.size()];

for (int i = 0; i < read_seqs.size(); ++i){
rseqs[i] = read_seqs[i];
}

hash_t** read_hashes = new hash_t*[read_keys.size()];
vector<int> read_hash_lens(read_keys.size());

hash_t** read_mins = new hash_t*[read_keys.size()];
vector<int> read_min_lens(read_keys.size());


hash_t** ref_hashes = new hash_t*[ref_keys.size()];
vector<int> ref_hash_lens(ref_keys.size());

hash_t** ref_minhashes = new hash_t*[ref_keys.size()];
vector<int> ref_min_lens(ref_keys.size());


int numrefs = ref_keys.size();
int numreads = read_keys.size();

#pragma omp parallel
{
if (!doReferenceDepth){
#pragma omp for
for (int i = 0; i < numrefs; ++i){
to_upper(ref_seqs[i], ref_lens[i]);
int num;
calc_hashes(ref_seqs[i], ref_lens[i], kmer, ref_hashes[i], num);
minhashes(ref_hashes[i], num, sketch_size, ref_minhashes[i], ref_min_lens[i]);
delete [] ref_hashes[i];
delete [] ref_seqs[i];
}

}
else{
#pragma omp for
for (int i = 0; i < numrefs; ++i){
calc_hashes(ref_seqs[i], ref_lens[i], kmer, ref_hashes[i], ref_hash_lens[i], ref_hash_counter);
}
#pragma omp for
for (int i = 0; i < numrefs; ++i){
minhashes_frequency_filter(ref_hashes[i], ref_hash_lens[i], sketch_size,
ref_minhashes[i], ref_min_lens[i], ref_hash_counter, 0, max_samples);
}
}



if (!doReadDepth && !stream_files){
{
#pragma omp for
for (int i = 0; i < numreads; ++i){

int shared_arr [numrefs];
hash_t* h;
int num;
hash_t* mins;
int min_num;

{
to_upper(rseqs[i], read_lens[i]);
}

calc_hashes(rseqs[i], read_lens[i], kmer, h, num);

minhashes(h, num, sketch_size, mins, min_num);
delete [] h;
delete [] rseqs[i];

for (int j = 0; j < numrefs; ++j){
hash_intersection_size(mins, min_num, ref_minhashes[j], ref_min_lens[j], shared_arr[j]);
}
{

int max_shared = -1;
int max_id = 0;
int diff = 0;
for (int j = 0; j < numrefs; ++j){
if (shared_arr[j] > max_shared){
diff = shared_arr[j] - max_shared;
max_shared = shared_arr[j];
max_id = j;
}
}

bool diff_filter = diff > min_diff;
bool depth_filter = min_num <= min_matches;
bool match_filter = max_shared < min_matches;

stringstream outre;
outre << ref_keys[max_id] << "\t" << read_keys[i]  <<  "\t" << max_shared << "\t" << sketch_size << (depth_filter ? "FAIL:DEPTH" : "") << "\t" << (match_filter ? "FAIL:MATCHES" : "") << "\t" << (diff_filter ? "" : "FAIL:DIFF") << endl;
cout << outre.str();
outre.str("");
delete [] mins;

}
}

}
}
else if (doReadDepth && !stream_files){
#pragma omp for
for (int i = 0; i < numreads; ++i){
int shared_arr[numrefs];

to_upper(rseqs[i], read_lens[i]);
calc_hashes(rseqs[i], read_lens[i], kmer, read_hashes[i], read_hash_lens[i], read_hash_counter);
}
#pragma omp for
for (int i = 0; i < numreads; ++i){
hash_t* mins;
int num_mins;
int shared_arr[numrefs];
mask_by_frequency(read_hashes[i], read_hash_lens[i], read_hash_counter, min_kmer_occ);
minhashes(read_hashes[i], read_hash_lens[i], sketch_size, mins, num_mins);
delete [] read_hashes[i];


for (int j = 0; j < numrefs; ++j){
hash_intersection_size(mins, num_mins, ref_minhashes[j], ref_min_lens[j], shared_arr[j]);
}

int max_shared = -1;
int max_id = 0;
int diff = 0;
for (int j = 0; j < numrefs; ++j){
if (shared_arr[j] > max_shared){
diff = shared_arr[j] - max_shared;
max_shared = shared_arr[j];
max_id = j;
}
}


bool diff_filter = diff > min_diff;
bool depth_filter = num_mins <= min_matches;
bool match_filter = max_shared < min_matches;


stringstream outre;
outre << ref_keys[max_id] << "\t" << read_keys[i]  <<  "\t" << max_shared << "\t" << sketch_size << (depth_filter ? "FAIL:DEPTH" : "") << "\t" << (match_filter ? "FAIL:MATCHES" : "") << "\t" << (diff_filter ? "" : "FAIL:DIFF") << endl;
cout << outre.str();
outre.str("");
delete [] mins;
}
}
else if (!doReadDepth && stream_files){
KSEQ_Reader ksq;
int bufsz = 1000;
ksq.open(read_files[0]);
ksq.buffer_size(bufsz);
int l = 0;
while (l == 0){
ksequence_t* kt;
int rnum = 0;
l = ksq.get_next_buffer(kt, rnum);
#pragma omp for
for (int i = 0; i < rnum; ++i){
hash_t* h;
int hashnum;
hash_t* mins;
int minnum;
calc_hashes(rseqs[i], read_lens[i], kmer, h, hashnum);
minhashes(h, hashnum, sketch_size, mins, minnum);
}
}
}
else{

}
}

delete [] rseqs;
delete [] ref_hashes;
delete [] ref_minhashes;
return 0;



}



int main_filter(int argc, char** argv){
vector<char*> ref_files;
vector<char*> read_files;
vector<char*> pre_read_files;
vector<char*> pre_ref_files;

vector<int> kmer;

int sketch_size = 1000;
int threads = 1;
int min_kmer_occ = -1;
int min_matches = -1;
int min_diff = 0;
int max_samples = 100000;

string read_kmer_map_file = "";
string ref_kmer_map_file = "";

bool doReadDepth = false;
bool doReferenceDepth = false;

bool useHASHTs = false;
int ref_sketch_size = 0;

bool streamify_me_capn = false;
bool output_reads = false;


int c;
int optind = 2;

if (argc <= 2){
help_filter(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"kmer", no_argument, 0, 'k'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"sketch-size", required_argument, 0, 's'},
{"ref-sketch", required_argument, 0, 'S'},
{"threads", required_argument, 0, 't'},
{"min-kmer-occurence", required_argument, 0, 'M'},
{"min-matches", required_argument, 0, 'N'},
{"min-diff", required_argument, 0, 'D'},
{"max-samples", required_argument, 0, 'I'},
{"pre-reads", required_argument, 0, 'F'},
{"pre-references", required_argument, 0, 'R'},
{"read-kmer-map-file", required_argument, 0, 'p'},
{"ref-kmer-map-file", required_argument, 0, 'q'},
{"in-stream", no_argument, 0, 'i'},
{0,0,0,0}
};

int option_index = 0;
c = getopt_long(argc, argv, "hdk:f:r:s:S:t:M:N:I:R:F:p:q:iD:", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 'F':
pre_read_files.push_back(optarg);
break;
case 'R':
pre_ref_files.push_back(optarg);
break;
case 'p':
read_kmer_map_file = optarg;
break;
case 'q':
ref_kmer_map_file = optarg;
break;
case 't':
threads = atoi(optarg);
break;
case 'r':
ref_files.push_back(optarg);
break;
case 'f':
read_files.push_back(optarg);
break;
case 'k':
kmer.push_back(atoi(optarg));
break;
case 'N':
min_matches = atoi(optarg);
break;
case 'D':
min_diff = atoi(optarg);
break;
case '?':
case 'h':
print_help(argv);
exit(1);
break;
case 's':
sketch_size = atoi(optarg);
break;
case 'S':
useHASHTs = true;
ref_sketch_size = 3 * atoi(optarg);
break;
case 'M':
min_kmer_occ = atoi(optarg);
doReadDepth = true;
break;
case 'I':
max_samples = atoi(optarg);
doReferenceDepth = true;
break;
case 'i':
streamify_me_capn = true;
break;
default:
print_help(argv);
abort();

}
}

if (sketch_size == -1){
cerr << "Sketch size unset." << endl
<< "Will use the default sketch size of s = 10000" << endl;
sketch_size = 1000;
}

if (kmer.size() == 0){
cerr << "No kmer size(s) provided. Will use a default kmer size of 16." << endl;
kmer.push_back(16);
}


omp_set_num_threads(threads);
if (!read_kmer_map_file.empty()){

}
if (!ref_kmer_map_file.empty()){

}
if (!pre_read_files.empty()){

}
if (!pre_ref_files.empty()){

}

vector<string> ref_keys;
vector<char*> ref_seqs;
vector<int> ref_lens;

vector<string> read_keys;
vector<char*> read_seqs;
vector<string> read_quals;
vector<int> read_lens;



if (!ref_files.empty()){
parse_fastas(ref_files, ref_keys, ref_seqs, ref_lens);
}
if (!read_files.empty()){
parse_fastas(read_files, read_keys, read_seqs, read_lens, read_quals);
}

vector<hash_t*> ref_hashes(ref_keys.size());
vector<int> ref_hash_lens(ref_keys.size());

vector<hash_t*> read_hashes(read_keys.size());
vector<int> read_hash_lens(read_keys.size());

vector<hash_t*> ref_mins(ref_keys.size());
int* ref_min_lens = new int [ref_keys.size()];
int* ref_min_starts = new int [ref_keys.size()];

vector<hash_t*> read_mins(read_keys.size());
int* read_min_starts = new int [ read_keys.size() ];
int* read_min_lens = new int [read_keys.size() ];


HASHTCounter read_hash_counter(10000000);
HASHTCounter ref_hash_counter(10000000);

vector<vector<string> > results(threads);

if (!ref_files.empty()){
hash_sequences(ref_keys, ref_seqs, ref_lens, ref_hashes, ref_hash_lens, kmer, read_hash_counter, ref_hash_counter, false, doReferenceDepth);
}


if (!read_files.empty()){
hash_sequences(read_keys, read_seqs, read_lens, read_hashes, read_hash_lens, kmer, read_hash_counter, ref_hash_counter, doReadDepth, false);
}

#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < ref_keys.size(); i++){
ref_min_starts[i] = 0;
ref_min_lens[i] = 0;
ref_mins[i] = new hash_t[ sketch_size ];
std::sort(ref_hashes[i], ref_hashes[i] + ref_hash_lens[i]);
if (max_samples < 100000){
for (int j = 0; j < ref_hash_lens[i], ref_min_lens[i] < sketch_size; ++j){
hash_t curr = *(ref_hashes[i] + j);
if (curr != 0 && ref_hash_counter.get(curr) <= max_samples){
ref_mins[i][ref_min_lens[i]] = curr;
++ref_min_lens[i];
if (ref_min_lens[i] == sketch_size){
break;
}
}
else{
continue;
}
}

}
else{
while (ref_hashes[i][ref_min_starts[i]] == 0 && ref_min_starts[i] < ref_hash_lens[i]){
++ref_min_starts[i];
}
for (int j = ref_min_starts[i]; j < ref_hash_lens[i], ref_min_lens[i] < sketch_size; ++j){
*(ref_mins[i] +ref_min_lens[i]) = *(ref_hashes[i] + j);
++ref_min_lens[i];
}

}
ref_min_starts[i] = 0;

delete [] ref_hashes[i];

}

#pragma omp for
for (int i = 0; i < read_keys.size(); i++){
stringstream outre;
read_mins[i] = new hash_t[ sketch_size ];
read_min_lens[i] = 0;
read_min_starts[i] = 0;
std::sort(read_hashes[i], read_hashes[i] + read_hash_lens[i]);
if (doReadDepth){
for (int j = 0; j < read_hash_lens[i]; ++j){

if (read_hashes[i][j] != 0 && read_hash_counter.get(read_hashes[i][j]) >= min_kmer_occ){
read_mins[i][read_min_lens[i]] = *(read_hashes[i] + j);

++(read_min_lens[i]);
if (read_min_lens[i] == sketch_size){
break;
}
}
else{
continue;
}
}
}
else{
while (read_hashes[i][ read_min_starts[i] ] == 0 && read_min_starts[i] < read_hash_lens[i]){
++read_min_starts[i];
}
for (int j = read_min_starts[i]; j < read_hash_lens[i]; ++j){
read_mins[i][read_min_lens[i]] = *(read_hashes[i] + j);
++(read_min_lens[i]);
if (read_min_lens[i] == sketch_size){
break;
}
}
}
read_min_starts[i] = 0;
delete [] read_hashes[i];

tuple<string, int, int, bool> result;
result = classify_and_count_diff_filter(ref_keys, ref_mins, read_mins[i], ref_min_starts, read_min_starts[i], ref_min_lens, read_min_lens[i], sketch_size, min_diff);


bool depth_filter = read_min_lens[i] <= 0; 
bool match_filter = std::get<1>(result) < min_matches;


if (!depth_filter && !match_filter && std::get<3>(result)){
outre << ">" << string(read_keys[i]) << endl
<< string(read_seqs[i], read_lens[i]) << endl
<< "+" << endl
<< string(read_quals[i]) << endl;

#pragma omp critical
{
cout << outre.str();
outre.str("");
}
}
delete [] read_mins[i];


}
}



if (streamify_me_capn){
#pragma omp parallel
{
#pragma omp single nowait
{
FILE *instream = NULL;
instream = stdin;

gzFile fp = gzdopen(fileno(instream), "r");
kseq_t *seq = kseq_init(fp);


while (kseq_read(seq) >= 0){

to_upper(seq->seq.s, seq->seq.l);

string name = string(seq->name.s);
int len = seq->seq.l;


#pragma omp task
{
vector<hash_t> r = calc_hashes(seq->seq.s, len, kmer);
hash_t* hashes = &(*r.begin());
int hashlen = r.size();

stringstream outre;

std::sort(hashes, hashes + hashlen);
int sketch_start = 0;
int sketch_len = 0;
hash_t* mins = new hash_t[sketch_size];
if (min_kmer_occ > 0){
for (int i = 0; i < hashlen; ++i){
hash_t curr = *(hashes + i);
if (read_hash_counter.get(curr) >= min_kmer_occ && curr != 0){
mins[sketch_len] = curr;
++sketch_len;
}
if (sketch_len == sketch_size){
break;
}
}
}
else{
while (hashes[sketch_start] == 0 && sketch_start < hashlen){
++sketch_start;
}
for (int i = sketch_start; i < hashlen; ++i){
mins[sketch_len++] = *(hashes + i);
if (sketch_len == sketch_size){
break;
}
}
}
sketch_start = 0;
tuple<string, int, int, bool> result;
result = classify_and_count_diff_filter(ref_keys, ref_mins, mins, ref_min_starts, sketch_start, ref_min_lens, sketch_len, sketch_size, min_diff);

bool depth_filter = sketch_len <= 0; 
bool match_filter = std::get<1>(result) < min_matches;

outre  << "Sample: " << name << "\t" << "Result: " << 
std::get<0>(result) << "\t" << std::get<1>(result) << "\t" << std::get<2>(result) << "\t" <<
(depth_filter ? "FAIL:DEPTH" : "") << "\t" << (match_filter ? "FAIL:MATCHES" : "") << "\t" << (std::get<3>(result) ? "" : "FAIL:DIFF") << endl;

cout << outre.str();
outre.str("");

delete [] hashes;
delete [] mins;
}
}

kseq_destroy(seq);
gzclose(fp);
}
}
}


delete [] ref_min_lens;
delete [] read_min_lens;
delete [] read_min_starts;
delete [] ref_min_starts;



}




int main_call(int argc, char** argv){
vector<char*> ref_files;
vector<char*> read_files;

bool useHASHTCounter = true;

vector<int> kmer;

int sketch_size = 1000;
int threads = 1;
int window_len = 100;

bool show_depth = false;
bool output_vcf = true;

int c;
int optind = 2;

if (argc <= 2){
help_call(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"kmer", no_argument, 0, 'k'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"sketch", required_argument, 0, 's'},
{"threads", required_argument, 0, 't'},
{"show-depth", required_argument, 0, 'd'},
{"window-len", required_argument, 0, 'w'},
{0,0,0,0}
};

int option_index = 0;
c = getopt_long(argc, argv, "hdk:f:r:s:t:w:", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 't':
threads = atoi(optarg);
break;
case 'r':
ref_files.push_back(optarg);
break;
case 'f':
read_files.push_back(optarg);
break;
case 'k':
kmer.push_back(atoi(optarg));
break;
case '?':
case 'h':
print_help(argv);
exit(1);
break;
case 's':
sketch_size = atoi(optarg);
break;
case 'w':
window_len = atoi(optarg);
break;
case 'd':
show_depth = true;
output_vcf = false;
break;
default:
print_help(argv);
abort();

}
}

if (sketch_size == -1){
cerr << "Sketch size unset." << endl
<< "Will use the default sketch size of s = 1000" << endl;
sketch_size = 10000;
}

if (kmer.size() == 0){
cerr << "No kmer size(s) provided. Will use a default kmer size of 16." << endl;
kmer.push_back(16);
}
else if (kmer.size() > 1){
cerr << "Only a single kmer size may be used for calling." << endl
<< "Sizes provided: ";
for (auto k : kmer){
cerr << k << " ";
}
cerr << endl;
cerr << "Please choose a single kmer size." << endl;
exit(1);
}

omp_set_num_threads(threads);


vector<string> ref_keys;
vector<char*> ref_seqs;
vector<int> ref_lens;

vector<string> read_keys;
vector<char*> read_seqs;
vector<int> read_lens;


unordered_map<hash_t, int> read_hash_to_depth;
read_hash_to_depth.reserve(1000000);
unordered_map<hash_t, int> ref_hash_to_num_samples;
ref_hash_to_num_samples.reserve(1000000);


#pragma omp master
cerr << "Parsing sequences..." << endl;

if (ref_files.size() >= 1){
parse_fastas(ref_files, ref_keys, ref_seqs, ref_lens);
}
else{
cerr << "No references were provided. Please provide at least one reference file in fasta/fastq format." << endl;
help_call(argv);
exit(1);
}

if (read_files.size() >= 1){
parse_fastas(read_files, read_keys, read_seqs, read_lens);
}
else{
cerr << "No reads were provided. Please provide at least one read file in fasta/fastq format." << endl;
help_call(argv);
exit(1);
}

HASHTCounter* ref_htc = new HASHTCounter(10000000);
vector<hash_t*> ref_hashes(ref_keys.size());
vector<int> ref_hash_lens(ref_keys.size());
int num_refs = ref_seqs.size();

HASHTCounter* read_htc = new HASHTCounter(10000000);
vector<hash_t*> read_hashes(read_keys.size());
vector<int> read_hash_lens(read_keys.size());
int num_reads = read_seqs.size();     
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < num_refs; ++i){
to_upper(ref_seqs[i], ref_lens[i]);
calc_hashes(ref_seqs[i], ref_lens[i], kmer, ref_hashes[i], ref_hash_lens[i], ref_htc);
} 
#pragma omp for
for (int i = 0; i < num_reads; ++i){
to_upper(read_seqs[i], read_lens[i]);
calc_hashes(read_seqs[i], read_lens[i], kmer, read_hashes[i], read_hash_lens[i], read_htc);
#pragma omp critical
{
for (int j = 0; j < read_hash_lens[i]; ++j){
read_hash_to_depth[ read_hashes[i][j]] += 1;
}
}
}
}


std::function<double(vector<int>)> avg = [](vector<int> n_list){
int ret = 0;
for (int x = 0; x < n_list.size(); x++){
ret += n_list.at(x);
}
return (double) ret / (double) n_list.size();
};
vector<char> a_ret = {'C', 'T', 'G'};
vector<char> c_ret = {'T', 'G', 'A'};
vector<char> t_ret = {'C', 'G', 'A'};
vector<char> g_ret = {'A', 'C', 'T'};


std::function<vector<char>(char)> rotate_snps = [&a_ret, &c_ret, &g_ret, &t_ret](const char& c){

if ( c == 'A' || c == 'a'){
return a_ret;
}
else if (c == 'T' || c == 't'){
return t_ret;
}
else if (c == 'C' || c == 'c'){
return c_ret;
}
else if (c == 'G' || c == 'g'){
return g_ret; 
}
};

std::function<vector<string>(string)> permute = [&](string x){

vector<string> ret;
for (int i = 0; i < x.size(); i++){
char orig = x[i];
vector<char> other_chars = rotate_snps(x[i]);
for (int j = 0; j < other_chars.size(); j++){
x[i] = other_chars[j];
ret.push_back(x);
}
x[i] = orig;
}

for (int i = 0; i < x.size() - 1; i++){
char orig = x[i];
stringstream tmp;
for (int strpos = 0; strpos < x.size(); strpos++){
if (strpos != i){
tmp << x[strpos];
}
}
ret.push_back(tmp.str());
}

for (int i = 0; i < x.size() - 2; i++){
char orig = x[i];
stringstream tmp;
for (int strpos = 0; strpos < x.size(); strpos++){
if (strpos != i & strpos != i + 1){
tmp << x[strpos];
}
}
ret.push_back(tmp.str());
}

for (int i = 0; i < x.size(); i++){
char orig = x[i];
stringstream tmp;
for (int strpos = 0; strpos < x.size(); strpos++){
tmp << x[strpos]; 
}
}

for (int i = 0; i < x.size(); i++){
char orig = x[i];
stringstream tmp;
for (int strpos = 0; strpos < x.size(); strpos++){

}
}


return ret;
};




vector<int*> depth_arrs(ref_keys.size());

if (ref_keys.size() > 1){
cerr << "WARNING: more than one ref provided. VCF will not be correct" << endl;

}
if (output_vcf){
cout << "##fileformat=VCF4.2\n##source=rkmh\n##reference=" << ref_files[0] << endl <<
"##INFO=<ID=KD,Number=1,Type=Integer,Description=\"Number of times call for specific kmer appears\">" << endl
<< "##INFO=<ID=MD,Number=1,Type=Integer,Description=\"Maximum depth found for the rescue kmer.\">" << endl
<< "##INFO=<ID=RD,Number=1,Type=Integer,Description=\"Average depth in region\">"
<< "##INFO=<ID=OD,Number=1,Type=Integer,Description=\"Depth of original kmer at site before modification.\">"
<< endl;
}

std::function<string(vector<string>)> join = [](vector<string> strs){
stringstream strstream;
for (int i = 0; i < strs.size() - 1; i++){
strstream << strs[i] << "_";
}
strstream << strs[strs.size() - 1];
return strstream.str();
};

map<string, int> call_count;
map<string, int> call_max_depth;
map<string, int> call_avg_depth;
map<string, int> call_orig_depth;
vector<string> outbuf;
outbuf.reserve(1000);


#pragma omp parallel
{

list<int> d_window;


#pragma omp for
for (int i = 0; i < num_refs; i++){

stringstream outre;

for (int j = 0; j < ref_hash_lens[i]; j++){

int depth = read_hash_to_depth[ref_hashes[i][j]];
d_window.push_back(depth);
if (d_window.size() > window_len){
d_window.pop_front();
}

int avg_d = avg(vector<int>(d_window.begin(), d_window.end()));
int max_rescue = 0;


if (show_depth){
outre << j << "\t" << avg_d << "\t" <<  depth;

}
if (depth < .5 * avg_d){
string ref = string(ref_seqs[i] + j, kmer[0]);
string alt(ref);
string d_alt = (j > 0) ? string(ref_seqs[i] + j - 1, kmer[0] + 1) : "";

for (int alt_pos = 0; alt_pos < alt.size(); alt_pos++){
char orig = alt[alt_pos];
for (auto x : rotate_snps(orig)){
alt[alt_pos] = x;
int alt_depth = read_hash_to_depth[calc_hash(alt)];
max_rescue = max_rescue > alt_depth ? max_rescue : alt_depth;

if ( !show_depth && alt_depth >= .1 * avg_d & alt_depth > depth){
int pos = j + alt_pos + 1;
if (output_vcf){
#pragma omp critical
{
stringstream sstream;
sstream << ref_keys[i] << "\t" << pos << "\t" <<
"." << "\t" << orig << "\t" << x;
string s = sstream.str();
call_count[s] += 1;
call_avg_depth[s] = max(avg_d, call_avg_depth[s]);
call_orig_depth[s] = max(call_orig_depth[s], depth);
if (alt_depth > call_max_depth[s]){
call_max_depth[s] = alt_depth;
}
}
}
else{
outre << "CALL: " << orig << "->" << x << "\t" << "POS: " << pos << "\tRESCUE_DEPTH: " << alt_depth << endl;
outre << "\t" << "old: " << ref << endl << "\t" << "new: " << alt;
}
}
alt[alt_pos] = orig;
}

}

char atgc[4] = {'A', 'T', 'G', 'C'};

if (j > 0){
for (int alt_pos = 1; alt_pos < d_alt.size(); alt_pos++){
stringstream mod;
char orig = d_alt[alt_pos];
mod << d_alt.substr(0, alt_pos) << d_alt.substr(alt_pos + 1, d_alt.length() - alt_pos);
int alt_depth = read_hash_to_depth[calc_hash(mod.str())];
if (output_vcf && alt_depth > 0.9 * avg_d){
int pos = j + alt_pos + 1;
stringstream sstream;
sstream << ref_keys[i] << "\t" << pos << "\t" << "." << "\t" << orig << "\t" << "-";
string s = sstream.str();
call_count[s] += 1;
call_avg_depth[s] = max(call_avg_depth[s], avg_d);
call_orig_depth[s] = max(call_orig_depth[s], depth);
if (alt_depth > call_max_depth[s]){
call_max_depth[s] = alt_depth;
}
}
}

}



}

if (show_depth){
outre << "\t" << (max_rescue > 0 ? max_rescue : depth);
}

}

}

}

for (auto x : call_count){
cout << x.first << "\t" << "99" << "\t" << "PASS" << "\t" << "KC=" << x.second << ";" <<
"MD=" << call_max_depth[x.first] << ";" << "RD=" << call_avg_depth[x.first] << ";OD=" << call_orig_depth[x.first] << endl;
}

for (auto x : read_hashes){
delete [] x;;
}
for (auto x : read_seqs){
delete [] x;
}
for (auto y : ref_hashes){
delete [] y;
}
for (auto y : ref_seqs){
delete [] y;
}

return 0;
}




int main_hash(int argc, char** argv){
vector<char*> input_files;


vector<int> kmer;

int sketch_size = 0;
int threads = 1;
int min_kmer_occ = 0;
int min_matches = -1;
int min_diff = 0;
int max_samples = 100000;

bool doReadDepth = false;
bool doReferenceDepth = false;
bool wabbitize = false;
bool merge_sketch = false;

bool traditional_minhash = false;
bool output_kmers = false;
bool output_counts = false;
string outname = "";

int c;
int optind = 2;

if (argc <= 2){
help_hash(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"count", no_argument, 0, 'c'},
{"kmer", no_argument, 0, 'k'},
{"output=kmers", no_argument, 0, 'K'},
{"merge-sample", no_argument, 0, 'm'},
{"wabbitize", no_argument, 0, 'w'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"sketch-size", required_argument, 0, 's'},
{"threads", required_argument, 0, 't'},
{"min-kmer-occurence", required_argument, 0, 'M'},
{"max-samples", required_argument, 0, 'I'},
{"out-prefix", required_argument, 0, 'o'},
{0,0,0,0}
};

int option_index = 0;

c = getopt_long(argc, argv, "ThcwKk:f:r:s:t:mM:I:o:", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 'T':
traditional_minhash = true;
break;
case 'c':
output_counts = true;
break;
case 'm':
merge_sketch = true;
break;
case 'w':
wabbitize = true;
break;
case 't':
threads = atoi(optarg);
break;
case 'f':
input_files.push_back(optarg);
break;
case 'k':
kmer.push_back(atoi(optarg));
break;
case 'K':
output_kmers = true;
break;
case '?':
case 'h':
print_help(argv);
exit(1);
break;
case 's':
sketch_size = atoi(optarg);
break;
case 'M':
min_kmer_occ = atoi(optarg);
doReadDepth = true;
break;
case 'I':
max_samples = atoi(optarg);
doReferenceDepth = true;
break;
case 'o':
outname = string(optarg);
break;
default:
print_help(argv);
abort();

}
}

if (kmer.size() != 0){
cerr << "Using a kmer size of " << kmer[0] << endl; 
}
else {
cerr << "Using default kmer size of 16." << endl;
kmer.push_back(16);
}

bool use_freqs = (doReferenceDepth || doReadDepth);

omp_set_num_threads(threads);

int bufsz = 1000;


#pragma omp parallel
{

#pragma omp single
{
if (output_kmers){
char* f = input_files[0];
KSEQ_Reader ksr;
ksr.buffer_size(bufsz);
ksr.open(f);

int l = 0;
ksequence_t* kt;
int num;
while (l == 0){
l = ksr.get_next_buffer(kt, num);
for (int i = 0; i < num; ++i){
#pragma omp task 
{
print_kmers(kt[i].sequence, kt[i].length, kmer[0], kt[i].name); 
}
}
}
#pragma omp taskwait
}
else if (!use_freqs){
char* f = input_files[0];
KSEQ_Reader ksr;
ksr.buffer_size(bufsz);
ksr.open(f);

int l = 0;
ksequence_t* kt;
int num;
while (l == 0){
l = ksr.get_next_buffer(kt, num);
for (int i = 0; i < num; ++i){
#pragma omp task 
{
hash_t* h;
int num;
calc_hashes(kt[i].sequence, kt[i].length, kmer, h, num);
print_hashes(h, num, kt[i].name);
delete [] h;
}
}
}
#pragma omp taskwait
}
else{

}
}
}

return 0;
}


int main_search(int argc, char** argv){
vector<char*> ref_files;
vector<char*> read_files;

int bufsz = 1000;

string summary_file_suffix = "rkmh.summary.txt";

int threads = 1;
vector<int> kmer;

int c;
int optind = 2;

if (argc <= 2){
help_hash(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"threads", required_argument, 0, 't'},
{"kmer", required_argument, 0, 'k'},
{0,0,0,0}
};

int option_index = 0;

c = getopt_long(argc, argv, "k:f:r:h", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 't':
threads = atoi(optarg);
break;
case 'k':
kmer.push_back(stoi(optarg));
break;
case 'f':
read_files.push_back(optarg);
break;
case 'r':
ref_files.push_back(optarg);
break;
case '?':
case 'h':
exit(1);
break;
default:
abort();

}
}

std::unordered_set<string> refs;

HASHTCounter htc;
for (auto r : ref_files){
ifstream infile(r);
string line;
while(getline(infile, line)){
vector<string> tokens = split(line, ' ');
hash_t h = calc_hash(tokens[0]);
htc.increment(h);
}
}


#pragma omp parallel
{
#pragma omp single
{
for (auto f : read_files){
KSEQ_Reader kh;
kh.open(f);
kh.buffer_size(bufsz);
ksequence_t* buf;
int buflen;
int l = 0;
stringstream seqstr;
while (l == 0){
l= kh.get_next_buffer(buf, buflen);
for (int i = 0; i < buflen; ++i){
#pragma omp task shared(buf, buflen, seqstr, refs)
{
vector<char*> foundmers;
seqstr << (buf + i)->name << "\t";
int seqlen = (buf + i)->length;
to_upper((buf + i)->sequence, seqlen);

mkmh_kmer_list_t kmers = kmerize((buf + i)->sequence, seqlen, kmer[0]);
if (kmers.length > 0){
for (int j = 0; j < kmers.length; ++j){
if (htc.get(kmers.kmers[j] > 0)){
foundmers.push_back(kmers.kmers[j]);
}
}
}
for (int j = 0; j < foundmers.size(); ++j){
seqstr << foundmers[j];
if (j < foundmers.size() - 1){
seqstr << ",";
}
}
seqstr << "\n";
cout << seqstr.str();
seqstr.str("");
}
}
}
}

}
}

return 0;
}




int main_count(int argc, char** argv){
vector<char*> read_files;
vector<int> kmer;
int threads = 1;

int bz = 1000;

int c;
int optind = 2;

if (argc <= 2){
help_hash(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"fasta", required_argument, 0, 'f'},
{"threads", required_argument, 0, 't'},
{0,0,0,0}
};

int option_index = 0;

c = getopt_long(argc, argv, "k:t:f:h", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 't':
threads = atoi(optarg);
break;
case 'k':
kmer.push_back(stoi(optarg));
break;
case 'f':
read_files.push_back(optarg);
break;
case '?':
case 'h':
exit(1);
break;
default:
abort();

}
}

omp_set_num_threads(threads);
HASHTCounter htc(640000);
KSEQ_Reader* kt = new KSEQ_Reader();
kt->buffer_size(bz);

#pragma omp parallel shared(kt, htc)
{
#pragma omp single
{
for (int fi_ind = 0; fi_ind < read_files.size(); fi_ind++){
int l = 0;
kt->open(read_files[fi_ind]);
while (l == 0){
{
ksequence_t* kst;
int num;
l = kt->get_next_buffer(kst, num);
for (int i = 0; i < num; ++i){
char* s = (kst + i)->sequence;
int length = (kst + i)->length;
#pragma omp task
{
vector<hash_t> hashes = calc_hashes((const char*) s, length, kmer);
for (int h_ind = 0; h_ind < hashes.size(); ++h_ind){
htc.increment(hashes[h_ind]);
}
}

}

}
}
}
}
}

delete kt;

return 0;
}

int main_hpv16(int argc, char** argv){
vector<char*> read_files;
vector<char*> ref_files;
string refpath = "data";

int sketch_size = 4000;
int threads = 1;
int min_kmer_occ = 0;
int min_matches = -1;
int min_diff = 0;

bool do_read_depth = false;
bool do_ref_depth = false;

int default_kmer_size = 16;
vector<int> kmer_sizes;

int c;
int optind = 2;

if (argc <= 2){
help_classify(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"kmer", no_argument, 0, 'k'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"sketch", required_argument, 0, 's'},
{"threads", required_argument, 0, 't'},
{"min-kmer-occurence", required_argument, 0, 'M'},
{"min-matches", required_argument, 0, 'N'},
{"min-diff", required_argument, 0, 'D'},
{"max-samples", required_argument, 0, 'I'},
{0,0,0,0}
};

int option_index = 0;
c = getopt_long(argc, argv, "hk:f:R:s:t:M:N:D:", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 't':
threads = atoi(optarg);
break;
case 'f':
read_files.push_back(optarg);
break;
case 'R':
refpath = optarg;
break;
case 'k':
kmer_sizes.push_back(atoi(optarg));
break;
case '?':
case 'h':
print_help(argv);
exit(1);
break;
case 's':
sketch_size = atoi(optarg);
break;
case 'M':
min_kmer_occ = atoi(optarg);
do_read_depth = true;
break;
case 'N':
min_matches = atoi(optarg);
break;
case 'D':
min_diff = atoi(optarg);
break;
default:
print_help(argv);
abort();

}
}

if (kmer_sizes.size() < 1){
cerr << "NO KMER SIZE PROVIDED. USING A DEFAULT KMER SIZE OF " << default_kmer_size << endl;
kmer_sizes.push_back(default_kmer_size);
}

string hpv_type_ref_file = refpath + "/" + "all_pave_ref.fa";
ref_files.push_back( (char*) hpv_type_ref_file.c_str());
string sublin_type_refs = refpath +  "/" + "new_refs.fa";
char* sublin_type_ref_file = (char*) sublin_type_refs.c_str();

omp_set_num_threads(threads);

vector<string> type_keys;
vector<char*> type_seqs;
vector<int> type_lens;

vector<string> subtype_keys;
vector<char*> subtype_seqs;
vector<int> subtype_lens;

vector<string> read_keys;
vector<char*> read_seqs;
vector<int> read_lens;



if (!ref_files.empty()){
parse_fastas(ref_files, type_keys, type_seqs, type_lens);
}
if (!read_files.empty()){
parse_fastas(read_files, read_keys, read_seqs, read_lens);
}

vector<char*> rfi;
rfi.push_back(sublin_type_ref_file);
parse_fastas(rfi, subtype_keys, subtype_seqs, subtype_lens);
int n_subtypes = subtype_keys.size();
hash_t** subtype_hashes = new hash_t*[subtype_keys.size()];
vector<int> subtype_hash_lens(subtype_keys.size());


int nrefs = type_keys.size();

char** rseqs = new char*[type_seqs.size()];
for (int i = 0; i < type_seqs.size(); ++i){
rseqs[i] = type_seqs[i];
}


hash_t** type_hashes = new hash_t*[type_keys.size()];
vector<int> type_hash_lens(type_keys.size());

hash_t** type_minhashes = new hash_t*[type_keys.size()];
vector<int> type_min_lens(type_keys.size());



map<char, set<hash_t>> lin_to_hashes;
map<char, set<hash_t>> lin_to_uniqs;
map<string, set<hash_t>> sublin_to_hashes;
map<string, unordered_set<hash_t>> sublin_to_uniqs;

HASHTCounter* readhtc;
if (do_read_depth)
{
readhtc = new HASHTCounter(800000000);

#pragma omp parallel for
for (int i = 0; i < read_keys.size(); ++i)
{
hash_t *h;
int hashnum;
{
calc_hashes(read_seqs[i], read_lens[i], kmer_sizes, h, hashnum, readhtc);
delete[] h;
}
}

}

vector<string> lineage_names;
vector<string> sublineage_names;
vector<hash_t*> lineage_hashes;
vector<hash_t*> sublineage_hashes;
vector<int> lineage_hash_lens;
vector<int> sublineage_hash_lens;

#pragma omp parallel
{

#pragma omp for
for (int i = 0; i < nrefs; ++i){
calc_hashes(type_seqs[i], type_lens[i], kmer_sizes[0], type_hashes[i], type_hash_lens[i]);
minhashes(type_hashes[i], type_hash_lens[i], sketch_size, type_minhashes[i], type_min_lens[i]);
}

#pragma omp for
for (int i = 0; i < n_subtypes; ++i){
calc_hashes(subtype_seqs[i], subtype_lens[i], kmer_sizes[0], subtype_hashes[i], subtype_hash_lens[i]);
}


#pragma omp single
{
for (int i = 0; i < n_subtypes; ++i){
char lin = subtype_keys[i][0];
for (int j = 0; j < subtype_hash_lens[i]; ++j ){
lin_to_hashes[lin].insert(subtype_hashes[i][j]);
}
}


for (auto x : lin_to_hashes){
vector<hash_t> diff(x.second.size() + 1000);
vector<hash_t> xdiff(x.second.begin(), x.second.end());
unordered_set<hash_t> uniqs;
std::vector<hash_t>::iterator it;

for (auto y : lin_to_hashes){
if (x.first == y.first){
continue;
}
else{
it=std::set_difference(xdiff.begin(), xdiff.end(), y.second.begin(), y.second.end(), diff.begin());
diff.resize(it-diff.begin());
xdiff = diff; 
}
}

lin_to_uniqs[x.first].insert(xdiff.begin(), xdiff.end());
lineage_names.push_back(string(1, x.first));
hash_t* n = new hash_t[xdiff.size()];
int count = 0;
for (auto x : xdiff){
n[count++] = x;
}
mkmh::sort(n, xdiff.size());
lineage_hashes.push_back(n);
lineage_hash_lens.push_back(xdiff.size());
}

ofstream ofi;
ofi.open("lineage_specific_hashes." + to_string(kmer_sizes[0]) + ".tst");

cerr << "Lineage specific kmer table created:" << endl;
for (auto t : lin_to_uniqs){
cerr << "\t" << t.first << "\t" << t.second.size() << endl;
ofi << t.first << "\t";
for (auto x : t.second){
ofi << x << "\t";
}

ofi << endl;
}


for (int i = 0; i < n_subtypes; ++i){
string sublin = string(subtype_keys[i]).substr(0, 2);
for (int j = 0; j < subtype_hash_lens[i]; ++j ){
sublin_to_hashes[sublin].insert(subtype_hashes[i][j]);
}
}
for (auto x : sublin_to_hashes){

vector<hash_t> diff(x.second.size() + 1000);
vector<hash_t> xdiff(x.second.begin(), x.second.end());
std::vector<hash_t>::iterator it;
for (auto y : sublin_to_hashes){
if (x.first == y.first){
continue;
}
else{
it=std::set_difference(xdiff.begin(), xdiff.end(), y.second.begin(), y.second.end(), diff.begin());
diff.resize(it-diff.begin());
xdiff = diff;
}
}
sublin_to_uniqs[x.first].insert(xdiff.begin(), xdiff.end());

sublineage_names.push_back( x.first);
hash_t* n = new hash_t[xdiff.size()];
int count = 0;
for (auto x : xdiff){
n[count++] = x;
}
mkmh::sort(n, xdiff.size());
sublineage_hashes.push_back(n);
sublineage_hash_lens.push_back(xdiff.size());
}

cerr << "Sublineage specific kmer table created:" << endl;
for (auto t : sublin_to_uniqs){
cerr << "\t" << t.first << "\t" << t.second.size() << endl;
}
}


#pragma omp for
for (int i = 0; i < read_keys.size(); ++i){
hash_t* h;
int hashnum;
calc_hashes(read_seqs[i], read_lens[i], kmer_sizes, h, hashnum);
if (do_read_depth){
mask_by_frequency(h, hashnum, readhtc, min_kmer_occ);
}

mkmh::sort(h, hashnum);

int max_shared = -1;
int max_id = 0;
for (int j = 0; j < nrefs; ++j){
int shared = 0;
hash_set_intersection_size(h, hashnum, type_hashes[j], type_hash_lens[j], shared);
if (shared > max_shared){
max_shared = shared;
max_id = j;
}
}
string type_name( type_keys[max_id]);
stringstream st;
st << read_keys[i] << "\t";
st << type_name << "\t" << max_shared << "/" << hashnum << "\t";

vector<string> lin_names;
vector<double> lin_sims;
vector<int> lin_intersections;
sort_by_similarity(h, hashnum, lineage_names, lineage_names.size(),
lineage_hashes, lineage_hash_lens, lin_names, lin_sims, lin_intersections);

for (int x = 0; x < lin_names.size(); ++x){
st << lin_names[x] << ":" << lin_sims[x] << ";";
}

st << "\t";

vector<string> sublin_names;
vector<double> sublin_sims;
vector<int> sublin_intersections;
sort_by_similarity(h, hashnum, sublineage_names, sublineage_names.size(),
sublineage_hashes, sublineage_hash_lens, sublin_names, sublin_sims, sublin_intersections);
for (int x = 0; x < sublin_names.size(); ++x){
st << sublin_names[x] << ":" << sublin_sims[x] << ";";
}

st << "\t";
for (int x = 0; x < lin_names.size(); ++x){
st << lin_intersections[x] << ";";
}
st << "\t";
for (int x = 0; x < sublin_names.size(); ++x){
st << sublin_intersections[x] << ";";
}
st << endl;
cout << st.str();
st.str("");
delete [] h;
}


}
return 0;
}




int main_classify(int argc, char** argv){

cerr << "CLASSIFY COMMAND IS TEMPORARILY UNAVAILABLE: TRY rkmh stream INSTEAD." << endl;
return main_stream(argc, argv);

return main_stream(argc, argv);

vector<char*> ref_files;
vector<char*> read_files;

vector<int> kmer;

int sketch_size = -1;
int threads = 1;
int min_kmer_occ = 0;
int min_matches = -1;
int min_diff = 0;
int max_samples = 1000000;

int c;
int optind = 2;

if (argc <= 2){
help_classify(argv);
exit(1);
}

while (true){
static struct option long_options[] =
{
{"help", no_argument, 0, 'h'},
{"kmer", no_argument, 0, 'k'},
{"fasta", required_argument, 0, 'f'},
{"reference", required_argument, 0, 'r'},
{"sketch", required_argument, 0, 's'},
{"threads", required_argument, 0, 't'},
{"min-kmer-occurence", required_argument, 0, 'M'},
{"min-matches", required_argument, 0, 'N'},
{"min-diff", required_argument, 0, 'D'},
{"max-samples", required_argument, 0, 'I'},
{0,0,0,0}
};

int option_index = 0;
c = getopt_long(argc, argv, "hk:f:r:s:t:M:N:D:I:", long_options, &option_index);
if (c == -1){
break;
}

switch (c){
case 't':
threads = atoi(optarg);
break;
case 'r':
ref_files.push_back(optarg);
break;
case 'f':
read_files.push_back(optarg);
break;
case 'k':
kmer.push_back(atoi(optarg));
break;
case '?':
case 'h':
print_help(argv);
exit(1);
break;
case 's':
sketch_size = atoi(optarg);
break;
case 'M':
min_kmer_occ = atoi(optarg);
break;
case 'N':
min_matches = atoi(optarg);
break;
case 'D':
min_diff = atoi(optarg);
break;
case 'I':
max_samples = atoi(optarg);
break;
default:
print_help(argv);
abort();

}
}

if (sketch_size == -1){
cerr << "Sketch size unset." << endl
<< "Will use the default sketch size of s = 1000" << endl;
sketch_size = 1000;
}

if (kmer.size() == 0){
cerr << "No kmer size(s) provided. Will use a default kmer size of 16." << endl;
kmer.push_back(16);
}

omp_set_num_threads(threads);


vector<string> ref_keys;
vector<char*> ref_seqs;
vector<int> ref_lens;

vector<string> read_keys;
vector<char*> read_seqs;
vector<int> read_lens;

unordered_map<hash_t, int> read_hash_to_depth;
read_hash_to_depth.reserve(10000);
unordered_map<hash_t, int> ref_hash_to_num_samples;
ref_hash_to_num_samples.reserve(10000);


#pragma omp master
cerr << "Parsing sequences..." << endl;

if (ref_files.size() >= 1){
parse_fastas(ref_files, ref_keys, ref_seqs, ref_lens);
}
else{
cerr << "No references were provided. Please provide at least one reference file in fasta/fastq format." << endl;
help_classify(argv);
exit(1);
}

if (read_files.size() >= 1){
parse_fastas(read_files, read_keys, read_seqs, read_lens);
}
else{
cerr << "No reads were provided. Please provide at least one read file in fasta/fastq format." << endl;
help_classify(argv);
exit(1);
}



#pragma omp master
cerr << " Done." << endl <<
ref_keys.size() << " references and " << read_keys.size() << " reads parsed." << endl;

int refnums = ref_keys.size();
vector<hash_t*> ref_hashes(ref_keys.size());
vector<int> ref_hash_nums(ref_keys.size());

vector<hash_t*> read_hashes(read_keys.size());
vector<int> read_hash_nums(read_keys.size());

#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < refnums; ++i){

mkmh::calc_hashes(ref_seqs[i], ref_lens[i], kmer, ref_hashes[i], ref_hash_nums[i]);
mkmh::sort(ref_hashes[i], ref_hash_nums[i]);
hash_t* t_mins;
int t_num;
mkmh::minhashes(ref_hashes[i], ref_hash_nums[i], sketch_size, t_mins, t_num);
delete [] ref_hashes[i];
ref_hashes[i] = t_mins;
ref_hash_nums[i] = t_num;
}
}


return 0;
}


int main(int argc, char** argv){

if (argc <= 1){
print_help(argv);
exit(1);
}
string cmd = argv[1];
if (cmd == "classify"){
return main_classify(argc, argv);
}
else if (cmd == "hash"){
return main_hash(argc, argv);
}
else if (cmd == "call"){
return main_call(argc, argv);
}
else if (cmd == "count"){
return main_count(argc, argv);
}
else if (cmd == "search"){
return main_search(argc, argv);
}
else if (cmd == "stream"){
return main_stream(argc, argv);
}
else if (cmd == "filter"){
return main_filter(argc, argv);
}
else if (cmd == "hpv16"){
return main_hpv16(argc, argv);
}
else{
print_help(argv);
exit(1);
}

}
