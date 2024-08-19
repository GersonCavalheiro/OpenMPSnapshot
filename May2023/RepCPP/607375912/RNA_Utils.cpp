

#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include "RNA_Utils.hpp"
#include "smithlab_utils.hpp"
#include "Part_Func.hpp"

using std::string;
using std::vector;
using std::stringstream;
using std::cerr;
using std::endl;

const size_t RNAUtils::SEGMENT_LENGTH;
const size_t RNAUtils::OVERLAP_LENGTH;

const int RNAUtils::RNAPair[4][4] = { { 0, 0, 0, 5 }, { 0, 0, 1, 0 }, { 0, 2, 0,
3 }, { 6, 0, 4, 0 } };
const double RNAUtils::DEFAULT_BACKGROUND[RNAUtils::RNA_ALPHABET_SIZE] =\
{0.3,  0.2, 0.2,  0.3};
const double RNAUtils::energy_pf = -1.0;


char
RNAUtils::sampleNucFromNucDist(const vector<double> &dist, const Runif &rng) {
double r = rng.runif(0.0, 1.0);
if (r < dist[RNAUtils::base2int('A')]) return 'A';
else if (r < dist[RNAUtils::base2int('A')] +\
dist[RNAUtils::base2int('C')]) return 'C';
else if (r < dist[RNAUtils::base2int('A')] +\
dist[RNAUtils::base2int('C')] +\
dist[RNAUtils::base2int('G')]) return 'G';
else return 'T';
}


char
RNAUtils::sampleNuc(const Runif &rng) {
double r = rng.runif(0.0, 1.0);
if (r < RNAUtils::DEFAULT_BACKGROUND[RNAUtils::base2int('A')])
return 'A';
else if (r < RNAUtils::DEFAULT_BACKGROUND[RNAUtils::base2int('A')] +\
RNAUtils::DEFAULT_BACKGROUND[RNAUtils::base2int('C')])
return 'C';
else if (r < RNAUtils::DEFAULT_BACKGROUND[RNAUtils::base2int('A')] +\
RNAUtils::DEFAULT_BACKGROUND[RNAUtils::base2int('C')] +\
RNAUtils::DEFAULT_BACKGROUND[RNAUtils::base2int('G')])
return 'G';
else return 'T';
}


string
RNAUtils::sampleSeqFromNucDist(const vector<double> &dist, const size_t length,
const Runif &rng) {
if (dist.size() != RNAUtils::RNA_ALPHABET_SIZE) {
stringstream ss;
ss << "Failed to generate sequence from nucleotide distribution, "
<< "distribution vector was malformed: found "
<< dist.size() << " entries; expected " << RNAUtils::RNA_ALPHABET_SIZE;
throw SMITHLABException(ss.str());
}

string res = "";
for (size_t i = 0; i < length; ++i) res += sampleNucFromNucDist(dist, rng);
return res;
}


string
RNAUtils::sampleSeq(const size_t length, const Runif &rng) {
string res = "";
for (size_t i = 0; i < length; ++i) res += RNAUtils::sampleNuc(rng);
return res;
}


string
RNAUtils::sampleSequenceFromPWM(const vector<vector<double> > pwm,
const Runif &rng) {
string res = "";
for (size_t j = 0; j < pwm.size(); ++j) {
if (pwm[j].size() != RNAUtils::RNA_ALPHABET_SIZE) {
stringstream ss;
ss << "Failed to generate sequence from PWM, PWM was malformed: found "
<< pwm[j].size() << " entries for position " << j
<< "; expected " << RNAUtils::RNA_ALPHABET_SIZE;
throw SMITHLABException(ss.str());
}
res += sampleNucFromNucDist(pwm[j], rng);
}
return res;
}


double
RNAUtils::get_minimum_free_energy(const string seq, const string cnstrnt) {

vector<char> seqc(seq.size() + 1);
copy(seq.begin(), seq.end(), seqc.begin());
vector<char> conc(seq.size() + 1);
copy(cnstrnt.begin(), cnstrnt.end(), conc.begin());

MC mc;
mc.init_pf_fold(seq.length(), energy_pf);
return mc.getMinimumFreeEnergy(&*seqc.begin(), &*conc.begin());
}


void
RNAUtils::get_base_pair_probability_vector(bool VERBOSE,
const vector<string> &seqs, vector<vector<double> > &bppvs) {
bppvs.clear();
bppvs.resize(seqs.size());
#pragma omp parallel for
for (size_t i = 0; i < seqs.size(); ++i) {
if (VERBOSE) {
const double done = i * 100 / seqs.size();
cerr << "\r" << "CALCULATING BASE PAIR PROBABILITIES ... (" << done
<< "% COMPLETE...)" << std::flush;
}
get_base_pair_probability_vector(seqs[i], bppvs[i]);
}
if (VERBOSE)
cerr << "\r" << "CALCULATING BASE PAIR PROBABILITIES ... DONE"
<< "              " << endl;
}


void
RNAUtils::get_base_pair_probability_vector(const string seq,
vector<double> &bppv) {
string constraint(seq.size(), '.');
get_base_pair_probability_vector(seq, constraint, bppv);
}


void
RNAUtils::get_base_pair_probability_vector(const string seq,
const string cnstrnt,
vector<double> &bppv) {

if (seq.size() != cnstrnt.size()) {
stringstream ss;
ss << "Calculating base pair prob. vector failed. Reason: sequence was "
<< seq.size() << " nucleotides long, but constraint string was "
<< cnstrnt.size() << " characters";
throw SMITHLABException(ss.str());
}

bppv.clear();
bppv.resize(seq.size());

size_t cpos = 0;
while(cpos < seq.length()) {
size_t sub_len = std::min(RNAUtils::SEGMENT_LENGTH, seq.length() - cpos);

vector<char> seqc(sub_len + 1);
copy(seq.begin() + cpos, seq.begin() + cpos + sub_len, seqc.begin());
vector<char> conc(sub_len + 1);
copy(cnstrnt.begin() + cpos, cnstrnt.begin() + cpos + sub_len, conc.begin());
assert(seqc.size() == conc.size());

vector<double> bppv_local;
MC mc;
mc.init_pf_fold(seqc.size(), energy_pf);
double q = mc.pf_fold(&*seqc.begin(), &*conc.begin());
if (q > std::numeric_limits<double>::min())
mc.getProbVector(bppv_local, sub_len);

assert(RNAUtils::SEGMENT_LENGTH > 2 * RNAUtils::OVERLAP_LENGTH);
for (size_t i = 0; i < bppv_local.size(); ++i) {
assert(cpos+i < bppv.size());
if ((i < RNAUtils::OVERLAP_LENGTH) && (cpos != 0)) {
double w = i / RNAUtils::OVERLAP_LENGTH;
bppv[cpos+i] = (w * bppv_local[i]) + ((1-w) * bppv[cpos+i]);
} else bppv[cpos+i] = bppv_local[i];
}

cpos = cpos + RNAUtils::SEGMENT_LENGTH - RNAUtils::OVERLAP_LENGTH;
}
}


void
RNAUtils::get_base_pair_probability_matrix(const string seq,
vector<vector<double> > &bppm) {
string constraint(seq.size(), '.');
get_base_pair_probability_matrix(seq, constraint, bppm);
}


double
RNAUtils::get_base_pair_probability_matrix(const string seq,
const string cnstrnt,
vector<vector<double> > &bppm) {

MC mc;

if (seq.size() != cnstrnt.size()) {
stringstream ss;
ss << "Calculating base pair prob. matrix failed. Reason: sequence was "
<< seq.size() << " nucleotides long, but constraint string was "
<< cnstrnt.size() << " characters";
throw SMITHLABException(ss.str());
}

bppm.clear();

vector<char> seqc(seq.size() + 1);
copy(seq.begin(), seq.end(), seqc.begin());
vector<char> conc(seq.size() + 1);
copy(cnstrnt.begin(), cnstrnt.end(), conc.begin());

mc.init_pf_fold(seq.size(), energy_pf);

double q = mc.pf_fold(&*seqc.begin(), &*conc.begin());
if (q > std::numeric_limits<double>::min())
mc.getProbMatrix(bppm, seq.length());
return q;
}

