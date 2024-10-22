

#include "DefaultKernel.h"

#include <string>

using std::string;
using std::to_string;

inline short max(short a, short b) {
return a > b ? a : b;
}


void DefaultKernel::compute_alignments(int const & opt, int const & aln_number, char const * const * const reads,
char const * const * const refs, Alignment * const alignments) {

int alignment_algorithm = opt & 0xF;

fp_alignment_call alignment_call = 0;

switch(alignment_algorithm) {
case 0:
alignment_call = &DefaultKernel::calc_alignment_smith_waterman;
break;
case 1:
alignment_call = &DefaultKernel::calc_alignment_needleman_wunsch;
break;
default:
break;
}

if (alignment_call != 0) {

Logger.log(0, KERNEL, string("Running DefaultKernel align with " + to_string(Parameters.param_int("num_threads")) + " threads.").c_str());

#pragma omp parallel for num_threads(Parameters.param_int("num_threads"))
for (int i = 0; i < aln_number; ++i) {
(this->*alignment_call)(reads[i],refs[i],&alignments[i]);
}
}
}

void DefaultKernel::score_alignments(int const & opt, int const & aln_number, char const * const * const reads,
char const * const * const refs, short * const scores) {

int alignment_algorithm = opt & 0xF;

fp_scoring_call scoring_call = 0;

switch(alignment_algorithm) {
case 0:
scoring_call = &DefaultKernel::score_alignment_smith_waterman;
break;
case 1:
scoring_call = &DefaultKernel::score_alignment_needleman_wunsch;
break;
default:
break;
}

if (scoring_call != 0) {

Logger.log(0, KERNEL, string("Running DefaultKernel score with " + to_string(Parameters.param_int("num_threads")) + " threads.").c_str());

#pragma omp parallel for num_threads(Parameters.param_int("num_threads"))
for (int i = 0; i < aln_number; ++i) {
(this->*scoring_call)(reads[i],refs[i],&scores[i]);
}
}
}

void DefaultKernel::score_alignment_smith_waterman(char const * const read,
char const * const ref, short * const scores) {

short max_score = 0;

short * matrix = new short[(refLength + 1) * 2]();

int prev_row = 0;
int current_row = 1;

for (int read_pos = 0; read_pos < readLength; ++read_pos) {

#ifndef NDEBUG

string matrix_line ("");
#endif

for (int ref_pos = 0; ref_pos < refLength; ++ref_pos) {

short up = matrix[prev_row * (refLength + 1) + ref_pos + 1];
short diag = matrix[prev_row * (refLength + 1) + ref_pos];
short left = matrix[current_row * (refLength + 1) + ref_pos];

diag += base_score[char_to_score[read[read_pos]]][char_to_score[ref[ref_pos]]];

short cur = max(up + scoreGapRef, max(left + scoreGapRead, max(diag, 0)));

matrix[current_row * (refLength + 1) + ref_pos + 1] = cur;

#ifndef NDEBUG

matrix_line += to_string(cur) + " ";

#endif
max_score = max(max_score, cur);

}

#ifndef NDEBUG

Logger.log(0, KERNEL, matrix_line.c_str());

#endif
prev_row = current_row;
(++current_row) &= 1;
}
delete [] matrix; matrix = 0;

#ifndef NDEBUG

Logger.log(0, KERNEL, string("Max score:\t" + to_string(max_score)).c_str());

#endif

memset(scores, max_score, 1);
}

void DefaultKernel::score_alignment_needleman_wunsch(char const * const read,
char const * const ref, short * const scores) {

short * matrix = new short[(refLength + 1) * 2]();

int prev_row = 0;
int current_row = 1;

short globalMax = 0;

for (int read_pos = 0; read_pos < readLength; ++read_pos) {

#ifndef NDEBUG

string matrix_line ("");

#endif

for (int ref_pos = 0; ref_pos < refLength; ++ref_pos) {

short up = matrix[prev_row * (refLength + 1) + ref_pos + 1];
short diag = matrix[prev_row * (refLength + 1) + ref_pos];
short left = matrix[current_row * (refLength + 1) + ref_pos];

diag += base_score[char_to_score[read[read_pos]]][char_to_score[ref[ref_pos]]];

short cur = max(up + scoreGapRef, max(left + scoreGapRead, diag));

matrix[current_row * (refLength + 1) + ref_pos + 1] = cur;

#ifndef NDEBUG

matrix_line += to_string(cur) + " ";

#endif
}

globalMax = max(globalMax, matrix[current_row * (refLength + 1) + refLength]);

#ifndef NDEBUG

Logger.log(0, KERNEL, matrix_line.c_str());

#endif

prev_row = current_row;
(++current_row) &= 1;
}

for (int ref_pos = 0; ref_pos < refLength + 1; ++ref_pos) {
globalMax = max(globalMax, matrix[prev_row * (refLength + 1) + ref_pos]);
}

#ifndef NDEBUG

Logger.log(0, KERNEL, string("Max score:\t" + to_string(globalMax)).c_str());

#endif

memset(scores, globalMax, 1);

delete [] matrix; matrix = 0;
}

void DefaultKernel::calculate_alignment_matrix_smith_waterman(char const * const read,
char const * const ref, alnMat const matrix, short * const best_coordinates) {

short best_read_pos = 0;
short best_ref_pos = 0;
short max_score = 0;

short prev_row_score = 0;
short current_row_score = 1;

short current_row_aln = 1;

short * scoreMat = new short[(refLength + 1) * 2]();

for (int read_pos = 0; read_pos < readLength; ++read_pos) {

#ifndef NDEBUG

string matrix_line ("");

#endif

for (int ref_pos = 0; ref_pos < refLength; ++ref_pos) {

short up = scoreMat[prev_row_score * (refLength + 1) + ref_pos + 1];
short diag = scoreMat[prev_row_score * (refLength + 1) + ref_pos];
short left = scoreMat[current_row_score * (refLength + 1) + ref_pos];

diag += base_score[char_to_score[read[read_pos]]][char_to_score[ref[ref_pos]]];

short cur = max(up + scoreGapRef, max(left + scoreGapRead, max(diag, 0)));

scoreMat[current_row_score * (refLength + 1) + ref_pos + 1] = cur;

char pointer = START;

if (cur == 0) {
pointer = START;
} else if (cur == diag) {
pointer = DIAG;
} else if (cur == up + scoreGapRef) {
pointer = UP;
} else if (cur == left + scoreGapRead) {
pointer = LEFT;
}

matrix[current_row_aln * (refLength + 1) + ref_pos + 1] = pointer;

if (cur > max_score) {
best_read_pos = read_pos;
best_ref_pos = ref_pos;
max_score = cur;
}

#ifndef NDEBUG

matrix_line += pointer + " ";

#endif
}

#ifndef NDEBUG

Logger.log(0, KERNEL, matrix_line.c_str());

#endif
prev_row_score = current_row_score;
(++current_row_score) &= 1;

++current_row_aln;
}

delete []scoreMat; scoreMat = 0;

best_coordinates[0] = best_read_pos;
best_coordinates[1] = best_ref_pos;
}

void DefaultKernel::calculate_alignment_matrix_needleman_wunsch(char const * const read,
char const * const ref, alnMat const matrix, short * const best_coordinates) {

short max_read_pos = readLength - 1;
short max_ref_pos = refLength - 1;

short prev_row_score = 0;
short current_row_score = 1;

short current_row_aln = 1;

short * scoreMat = new short[(refLength + 1) * 2]();

short rowMax = SHRT_MIN;

short globalRowMaxIndex = -1;

short rowMaxIndex = 0;


for (int read_pos = 0; read_pos < readLength; ++read_pos) {

matrix[current_row_aln * (refLength + 1)] = UP;
scoreMat[current_row_score * (refLength + 1)] = (read_pos + 1) * scoreGapRef;

if (max_read_pos == readLength - 1 && char_to_score[read[read_pos]] == 0) {
max_read_pos = read_pos - 1;
}

if (max_read_pos + 1 == read_pos) {
globalRowMaxIndex = rowMaxIndex;
}

rowMax = scoreMat[current_row_score * (refLength + 1)];
rowMaxIndex = 0;

#ifndef NDEBUG

string matrix_line ("");

#endif

for (int ref_pos = 0; ref_pos < refLength; ++ref_pos) {

short up = scoreMat[prev_row_score * (refLength + 1) + ref_pos + 1];
short diag = scoreMat[prev_row_score * (refLength + 1) + ref_pos];
short left = scoreMat[current_row_score * (refLength + 1) + ref_pos];

diag += base_score[char_to_score[read[read_pos]]][char_to_score[ref[ref_pos]]];

short cur = max(up + scoreGapRef, max(left + scoreGapRead, diag));

scoreMat[current_row_score * (refLength + 1) + ref_pos + 1] = cur;

char pointer = START;

if (cur == diag) {
pointer = DIAG;
} else if (cur == up + scoreGapRef) {
pointer = UP;
} else if (cur == left + scoreGapRead) {
pointer = LEFT;
}

if (max_ref_pos == refLength - 1 && char_to_score[ref[ref_pos]] == 0) {
max_ref_pos = ref_pos - 1;
}

if (cur > rowMax) {
rowMax = cur;
rowMaxIndex = ref_pos;
}

matrix[current_row_aln * (refLength + 1) + ref_pos + 1] = pointer;

#ifndef NDEBUG

matrix_line += pointer + " ";

#endif

}

#ifndef NDEBUG

Logger.log(0, KERNEL, matrix_line.c_str());

#endif

prev_row_score = current_row_score;
(++current_row_score) &= 1;

++current_row_aln;
}

delete []scoreMat; scoreMat = 0;

best_coordinates[0] = max_read_pos;

if (globalRowMaxIndex < 0 ) {
globalRowMaxIndex = rowMaxIndex;
}

best_coordinates[1] = std::min(max_ref_pos, globalRowMaxIndex);

}

void DefaultKernel::calc_alignment_smith_waterman(char const * const read,
char const * const ref, Alignment * const alignment) {

alnMat matrix = new char [(refLength + 1) * (readLength + 1)];
memset(matrix, START, (refLength + 1) * (readLength + 1) * sizeof(char));

short * best_coordinates = new short[2];

calculate_alignment_matrix_smith_waterman(read, ref, matrix, best_coordinates);

char * alignments = new char[alnLength * 2];

int read_pos = best_coordinates[0];
int ref_pos = best_coordinates[1];

#ifndef NDEBUG

Logger.log(0, KERNEL, string("Best read pos:\t" + to_string(read_pos)).c_str());
Logger.log(0, KERNEL, string("Best ref pos:\t" + to_string(ref_pos)).c_str());

#endif

int aln_pos = alnLength - 2;

char backtrack = matrix[(read_pos + 1) * (refLength + 1) + ref_pos + 1];

while(backtrack != START) {

#ifndef NDEBUG

Logger.log(0, KERNEL, string("Current backtrack:\t" + backtrack).c_str());

#endif

if (backtrack == UP) {
alignments[alnLength + aln_pos] = '-';
alignments[aln_pos] = read[read_pos--];
}
if (backtrack == LEFT) {
alignments[aln_pos] = '-';
alignments[alnLength + aln_pos] = ref[ref_pos--];
}
if (backtrack == DIAG) {
alignments[aln_pos] = read[read_pos--];
alignments[alnLength + aln_pos] = ref[ref_pos--];
}
backtrack = matrix[(read_pos + 1) * (refLength + 1) + ref_pos + 1];
--aln_pos;
}

alignment->read = new char[alnLength];
alignment->ref = new char[alnLength];

memcpy(alignment->read, alignments, alnLength * sizeof(char));
memcpy(alignment->ref, alignments + alnLength, alnLength * sizeof(char));

alignment->readStart = aln_pos + 1;
alignment->refStart = aln_pos + 1;

alignment->readEnd = alnLength - 1;
alignment->refEnd = alnLength - 1;

delete [] alignments; alignments = 0;
delete [] best_coordinates; best_coordinates = 0;
delete [] matrix; matrix = 0;
}

void DefaultKernel::calc_alignment_needleman_wunsch(char const * const read,
char const * const ref, Alignment * const alignment) {

alnMat matrix = new char [(refLength + 1) * (readLength + 1)];
memset(matrix, START, (refLength + 1) * (readLength + 1) * sizeof(char));

short * max_coordinates = new short[2];

calculate_alignment_matrix_needleman_wunsch(read, ref, matrix, max_coordinates);

char * alignments = new char[alnLength * 2];

int read_pos = max_coordinates[0];
int ref_pos = max_coordinates[1];

#ifndef NDEBUG

Logger.log(0, KERNEL, string("Best read pos:\t" + to_string(read_pos)).c_str());
Logger.log(0, KERNEL, string("Best ref pos:\t" + to_string(ref_pos)).c_str());

#endif

int aln_pos = alnLength - 2;

char backtrack = matrix[(read_pos + 1) * (refLength + 1) + ref_pos + 1];

alignments[alnLength - 1] = '\0';
alignments[2 * alnLength - 1] = '\0';

while(backtrack != START) {

#ifndef NDEBUG

Logger.log(0, KERNEL, string("Current backtrack:\t" + backtrack).c_str());

#endif
if (backtrack == UP) {
alignments[alnLength + aln_pos] = '-';
alignments[aln_pos] = read[read_pos--];
}
if (backtrack == LEFT) {
alignments[aln_pos] = '-';
alignments[alnLength + aln_pos] = ref[ref_pos--];
}
if (backtrack == DIAG) {
alignments[aln_pos] = read[read_pos--];
alignments[alnLength + aln_pos] = ref[ref_pos--];
}
backtrack = matrix[(read_pos + 1) * (refLength + 1) + ref_pos + 1];
--aln_pos;
}

alignment->read = new char[alnLength];
alignment->ref = new char[alnLength];

memcpy(alignment->read, alignments, alnLength * sizeof(char));
memcpy(alignment->ref, alignments + alnLength, alnLength * sizeof(char));

alignment->readStart = aln_pos + 1;
alignment->refStart = aln_pos + 1;

alignment->readEnd = alnLength - 1;
alignment->refEnd = alnLength - 1;

delete [] alignments; alignments = 0;
delete [] max_coordinates; max_coordinates = 0;
delete [] matrix; matrix = 0;
}

#undef KERNEL
