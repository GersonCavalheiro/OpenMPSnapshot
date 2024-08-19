

#include <string>
#include <vector>
#include <iterator>
#include <numeric>
#include <limits>

#include "smithlab_utils.hpp"
#include "Model.hpp"

#include "Util.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using std::stringstream;
using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::max;
using std::min;
using std::accumulate;

using smithlab::alphabet_size;

const double Model::pseudocount = 0.1;
const double Model::tolerance = 1e-10;
const double Model::zoops_threshold = 0;
const double Model::DEFAULT_GEO_P = 0.135;
const double Model::MAX_GEO_P = 0.99;
const double Model::MIN_GEO_P = 0.01;




static void
checkAndThrow_consistent(const vector<string> &seqs,
const vector<vector<double> > &struc,
const vector<vector<double> > &diagnostic_events,
const string &msg) {
if (diagnostic_events.size() != seqs.size()) {
stringstream ss;
ss << msg << " Expected diagnostic events vector to have same dimension "
<< "as sequences vector (" << seqs.size() << " elements), but it "
<< "doesn't (has " << diagnostic_events.size() << ")";
throw SMITHLABException(ss.str());
}
if (struc.size() != seqs.size()) {
stringstream ss;
ss << msg << " Expected structure vector to have same dimension "
<< "as sequences vector (" << seqs.size() << " elements), but it "
<< "doesn't (has " << struc.size() << ")";
throw SMITHLABException(ss.str());
}
for (size_t i = 0; i < seqs.size(); ++i) {
if (seqs[i].size() != struc[i].size()) {
stringstream ss;
ss << msg << " Expected structure vector for sequence " << i << " to "
<< "have " << seqs[i].size() << " elements, but it doesn't (has "
<< struc[i].size() << ")";
throw SMITHLABException(ss.str());
}
}
}




static string
vecToString(const vector<double> &v) {
stringstream ss;
for (size_t i = 0; i < v.size(); ++i) {
ss << v[i];
if (i != v.size() - 1) ss << ", ";
}
return ss.str();
}


static string
matrixToString(const vector<vector<double> > &matrix) {
stringstream ss;
for (size_t i = 0; i < matrix.size(); ++i) {
ss << vecToString(matrix[i]);
if (i != matrix.size() - 1) ss << endl;
}
return ss.str();
}




static void
calculate_number_of_bases_fg_bg(const vector<string> &seqs,
const vector<vector<double> > &site_indic,
const size_t motif_width,
vector<vector<double> > &nb_fg,
vector<double> &nb_bg,
const double pseudoCount=1) {
nb_fg.clear();
nb_bg.clear();
nb_fg.resize(motif_width, vector<double>(alphabet_size, pseudoCount));
nb_bg.resize(alphabet_size, pseudoCount);

#ifdef _OPENMP
vector<vector<double> > nb_fg_l;
vector<double> nb_bg_l;
nb_fg_l.resize(motif_width, vector<double>(alphabet_size, 0));
nb_bg_l.resize(alphabet_size, 0);

for (size_t i = 0; i < site_indic.size(); ++i) {
#pragma omp parallel num_threads(4) firstprivate(nb_fg_l, nb_bg_l)
{
#pragma omp for collapse(2)
for (size_t j = 0; j < site_indic[i].size(); ++j) {
for (size_t k = 0; k < motif_width; ++k) {
if ((seqs[i][j + k] == 'N') || (seqs[i][j + k] == 'n')) continue;

const size_t base = base2int(seqs[i][j + k]);
assert(base < alphabet_size);
nb_fg_l[k][base] += site_indic[i][j];
nb_bg_l[base] -= site_indic[i][j];
}
}

#pragma omp critical
{
for (size_t j = 0; j < alphabet_size; ++j) {
for (size_t k = 0; k < motif_width; ++k)
nb_fg[k][j] += nb_fg_l[k][j];
nb_bg[j] += nb_bg_l[j];
}
}
}
}

#else
for (size_t i = 0; i < site_indic.size(); ++i) {
for (size_t j = 0; j < site_indic[i].size(); ++j) {
for (size_t k = 0; k < motif_width; ++k) {
if ((seqs[i][j + k] == 'N') || (seqs[i][j + k] == 'n')) continue;

const size_t base = base2int(seqs[i][j + k]);
assert(base < alphabet_size);
nb_fg[k][base] += site_indic[i][j];
}
}
}
#endif

for (size_t i = 0; i < seqs.size(); ++i) {
for (size_t j = 0; j < seqs[i].length(); ++j) {
if ((seqs[i][j] == 'N') || (seqs[i][j] == 'n')) continue;

const size_t base = base2int(seqs[i][j]);
assert(base < alphabet_size);
nb_bg[base] += motif_width;
}
}

#ifndef _OPENMP
for (size_t i = 0; i < site_indic.size(); ++i) {
for (size_t j = 0; j < site_indic[i].size(); ++j) {
for (size_t k = 0; k < motif_width; ++k) {
if ((seqs[i][j + k] == 'N') || (seqs[i][j + k] == 'n')) continue;

const size_t base = base2int(seqs[i][j + k]);
assert(base < alphabet_size);
nb_bg[base] -= site_indic[i][j];
}
}
}
#endif
}


static void
calculate_number_of_bases_fg_bg_str(const vector<string> &seqs,
const vector<vector<double> > &secStr,
const vector<vector<double> > &site_indic,
const size_t motif_width,
vector<vector<double> > &nb_fg_ss,
vector<double> &nb_bg_ss,
vector<vector<double> > &nb_fg_ds,
vector<double> &nb_bg_ds,
const double pseudoCount=1) {

nb_fg_ss.clear();
nb_fg_ss.resize(motif_width, vector<double>(alphabet_size, pseudoCount));
nb_fg_ds.clear();
nb_fg_ds.resize(motif_width, vector<double>(alphabet_size, pseudoCount));
nb_bg_ss.clear();
nb_bg_ss.resize(alphabet_size, pseudoCount);
nb_bg_ds.clear();
nb_bg_ds.resize(alphabet_size, pseudoCount);

#ifdef _OPENMP
vector<vector<double> > nb_fg_ss_l, nb_fg_ds_l;
vector<double> nb_bg_ss_l, nb_bg_ds_l;

nb_fg_ss_l.resize(motif_width, vector<double>(alphabet_size, 0));
nb_fg_ds_l.resize(motif_width, vector<double>(alphabet_size, 0));
nb_bg_ss_l.resize(alphabet_size, 0);
nb_bg_ds_l.resize(alphabet_size, 0);

for (size_t i = 0; i < site_indic.size(); ++i) {
#pragma omp parallel num_threads(4) firstprivate(nb_fg_ds_l, nb_fg_ss_l, nb_bg_ds_l, nb_bg_ss_l)
{
for (size_t j = 0; j < site_indic[i].size(); ++j) {
for (size_t k = 0; k < motif_width; ++k) {
if ((seqs[i][j + k] == 'N') || (seqs[i][j + k] == 'n')) continue;

const size_t base_idx = base2int(seqs[i][j + k]);
assert(base_idx < alphabet_size);

const double curr_sec_str = secStr[i][j + k];
const double curr_site_indic = site_indic[i][j];
nb_fg_ss_l[k][base_idx] += (1.0 - curr_sec_str) * curr_site_indic;
nb_fg_ds_l[k][base_idx] += (curr_sec_str * curr_site_indic);
nb_bg_ss_l[base_idx] -= ((1.0 - secStr[i][j + k]) * site_indic[i][j]);
nb_bg_ds_l[base_idx] -= (secStr[i][j + k] * site_indic[i][j]);
}
}

#pragma omp critical
{
for (size_t j = 0; j < alphabet_size; ++j) {
for (size_t k = 0; k < motif_width; ++k) {
nb_fg_ss[k][j] += nb_fg_ss_l[k][j];
nb_fg_ds[k][j] += nb_fg_ds_l[k][j];
}
nb_bg_ss[j] += nb_bg_ss_l[j];
nb_bg_ds[j] += nb_bg_ds_l[j];
}
}
}
}
#else
for (size_t i = 0; i < site_indic.size(); ++i) {
for (size_t j = 0; j < site_indic[i].size(); ++j) {
for (size_t k = 0; k < motif_width; ++k) {
if ((seqs[i][j + k] == 'N') || (seqs[i][j + k] == 'n')) continue;

const size_t base_idx = base2int(seqs[i][j + k]);
assert(base_idx < alphabet_size);

const double curr_sec_str = secStr[i][j + k];
const double curr_site_indic = site_indic[i][j];
nb_fg_ss[k][base_idx] += (1.0 - curr_sec_str) * curr_site_indic;
nb_fg_ds[k][base_idx] += (curr_sec_str * curr_site_indic);
}
}
}
#endif

for (size_t i = 0; i < seqs.size(); ++i) {
for (size_t j = 0; j < seqs[i].length(); ++j) {
if ((seqs[i][j] == 'N') || (seqs[i][j] == 'n')) continue;

const size_t base_idx = base2int(seqs[i][j]);
assert(base_idx < alphabet_size);
nb_bg_ss[base_idx] += ((1.0 - secStr[i][j]) * motif_width);
nb_bg_ds[base_idx] += ((secStr[i][j] * motif_width));
}
}

#ifndef _OPENMP
for (size_t i = 0; i < site_indic.size(); ++i) {
for (size_t j = 0; j < site_indic[i].size(); ++j) {
for (size_t k = 0; k < motif_width; ++k) {
if ((seqs[i][j + k] == 'N') || (seqs[i][j + k] == 'n')) continue;

const size_t base_idx = base2int(seqs[i][j + k]);
assert(base_idx < alphabet_size);
nb_bg_ss[base_idx] -= ((1.0 - secStr[i][j + k]) * site_indic[i][j]);
nb_bg_ds[base_idx] -= (secStr[i][j + k] * site_indic[i][j]);
}
}
}
#endif
}





static void
maximization_seq(const vector<string> &seqs,
const vector<vector<double> > &siteInd,
vector<double> &seqIndic,
vector<vector<double> > &matrix,
vector<double> &freq,
double &gamma) {
static const double TINY = 1e-100;

vector<vector<double> > nb_fg;
vector<double> nb_bg;
calculate_number_of_bases_fg_bg(seqs, siteInd, matrix.size(), nb_fg, nb_bg);

for (size_t i = 0; i < matrix.size(); ++i) {
const double total = accumulate(nb_fg[i].begin(), nb_fg[i].end(), 0.0);
assert(std::isfinite(total));
transform(nb_fg[i].begin(), nb_fg[i].end(), matrix[i].begin(),
std::bind2nd(std::divides<double>(), total));

for (size_t j = 0; j < matrix[i].size(); ++j) {
if (matrix[i][j] < TINY)
matrix[i][j] = TINY;
assert(std::isfinite(matrix[i][j]));
}
}

const double total = accumulate(nb_bg.begin(), nb_bg.end(), 0.0);
transform(nb_bg.begin(), nb_bg.end(), freq.begin(),
std::bind2nd(std::divides<double>(), total));

gamma = accumulate(seqIndic.begin(), seqIndic.end(), 0.0) / seqs.size();
gamma = std::max(std::min(gamma, 1.0), TINY);
}


static void
maximization_str(const vector<vector<double> > &secondary_structure,
const vector<vector<double> > &site_indic,
const vector<double> &seq_indic,
vector<vector<double> > &matrix,
vector<double> &motif_sec_str,
double &f_sec_str) {
motif_sec_str.clear();
motif_sec_str.resize(matrix.size(), 0.0);
for (size_t i = 0; i < matrix.size(); ++i) {
for (size_t j = 0; j < site_indic.size(); ++j) {
for (size_t site = 0; site < site_indic[j].size(); ++site) {
motif_sec_str[i] += (seq_indic[j] * site_indic[j][site]
* secondary_structure[j][site + i]);
}
}

motif_sec_str[i] = motif_sec_str[i]
/ accumulate(seq_indic.begin(), seq_indic.end(), 0.0);
}

f_sec_str = 0.5;
}

static double
newtonRaphson (const vector<vector<double> > &diagnostic_events,
const vector<vector<double> > &siteInd,
const double &geoP,
const int &geoDelta)
{
double numerator = 0.0;
double denominator = 0.0;
for (size_t i = 0; i < siteInd.size(); i++)
for (size_t j = 0; j < siteInd[i].size(); j++) {
vector<double> sum_1;
vector<double> sum_2;
vector<double> sum_3;
for (size_t l = 0; l < diagnostic_events[i].size(); l++) {
double term_1 = 0.0;
double term_2 = 0.0;
double term_3 = 0.0;
double term_4 = 0.0;
double term_5 = 0.0;

if (abs(l - (j + geoDelta)) > 0)
term_1 = log(abs(l - (j + geoDelta)));

if ((abs(l - (j + geoDelta)) - 1) > 0)
term_2 = log(abs(l - (j + geoDelta)) - 1.0);

if ((abs(l - (j + geoDelta)) - 2) > 0)
term_3 = ((abs(l - (j + geoDelta)) - 2) * log(1.0 - geoP));

if (abs(l - (j + geoDelta)) > 0)
term_4 = (abs(l - (j + geoDelta)) * log(1.0 - geoP));

if ((abs(l - (j + geoDelta)) - 1) > 0)
term_5 = ((abs(l - (j + geoDelta)) - 1) * log(1.0 - geoP));

sum_1.push_back(log(diagnostic_events[i][l]) + term_1 + term_2 + term_3);
sum_2.push_back(log(diagnostic_events[i][l]) + term_4);
sum_3.push_back(log(diagnostic_events[i][l]) + term_1 + term_5);
}
const double log_A = smithlab::log_sum_log_vec(sum_1, sum_1.size());
const double log_B = smithlab::log_sum_log_vec(sum_2, sum_2.size());
const double log_C = smithlab::log_sum_log_vec(sum_3, sum_3.size());
const double num_ij = (1.0/geoP) - (exp(log_C - log_B));
const double denom_ij = (-1.0/(geoP*geoP) - (exp(log_A - log_B) - exp(2.0*(log_C - log_B))));
numerator += (siteInd[i][j] * num_ij);
denominator += (siteInd[i][j] * denom_ij);
}
assert(std::isfinite(numerator));
assert(std::isfinite(denominator));
return std::min(geoP - (numerator / denominator), Model::MAX_GEO_P);
}





static double
de_log_like(const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &siteInd,
const vector<double> &seqInd,
double &geoP,
int &geoDelta,
const bool optGeo) {
double res = 0;
double res_l = 0;
if (optGeo) {
for(size_t i = 0; i < diagnostic_events.size(); ++i) {
res_l = 0;
#pragma omp parallel for reduction(+:res_l)
for (size_t j = 0; j < siteInd[i].size(); ++j) {
vector<double> sum;
for (size_t l = 0; l < diagnostic_events[i].size(); ++l) {
double prior_log_prob = log(diagnostic_events[i][l]);
double geo_log_prob = log(geoP) + (abs(l - (j + geoDelta)) * log(1.0 - geoP));
double term = prior_log_prob + geo_log_prob;
assert(std::isfinite(term));
sum.push_back(term);
}
res_l += (smithlab::log_sum_log_vec(sum, sum.size()) * siteInd[i][j]);
}
res += res_l;
}
} else {
res_l = 0;
#pragma omp parallel for reduction(+:res_l)
for(size_t i = 0; i < diagnostic_events.size(); ++i) {
#pragma omp simd reduction(+:res_l)
for (size_t j = 0; j < siteInd[i].size(); ++j) {
res_l += (diag_values[i][j][geoDelta+8] * siteInd[i][j]);
}
}

res += res_l;
}
assert(std::isfinite(res));
return res;
}


static void
maximization_geoP(const vector<vector<double> > &diagnostic_events,
const vector<vector<double> > &siteInd,
const vector<double> &seqInd,
vector<vector<double> > &matrix,
double &geoP,
const int geoDelta) {
const double tol = 0.001;    
const size_t max_iters = 10;
size_t num_iters = 0;
double old;
do {
old = geoP;
geoP = max(min(newtonRaphson(diagnostic_events, siteInd, geoP, geoDelta),
Model::MAX_GEO_P), Model::MIN_GEO_P);
if (Model::DEBUG_LEVEL >= 2)
cerr << "\t\t\tOLD GEO_P: " << old << " NEW GEO_P: "
<< geoP << " diff is " << fabs(old - geoP) << endl;
num_iters += 1;
} while ((fabs(old - geoP) > tol) && (num_iters < max_iters));
if (Model::DEBUG_LEVEL >= 2)
cerr << "\t\t\tFINISHED GEO_P OPTIM.;" << endl;
}



static void
maximization_de(const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &siteInd,
const vector<double> &seqInd,
vector<vector<double> > &matrix,
double &geoP,
int &geoDelta,
const bool max_geo,
const bool max_delta) {
if ((!max_geo) && (!max_delta)) return;
if ((max_geo) && (!max_delta)) {
maximization_geoP(diagnostic_events, siteInd, seqInd, matrix, geoP, geoDelta);
}
if ((!max_geo) && (max_delta)) {
bool first = true;
int best_delta = 0;
double best_delta_ll = 0.0;
for (geoDelta = Model::MIN_DELTA; geoDelta <= Model::MAX_DELTA; ++geoDelta) {
if (Model::DEBUG_LEVEL >= 1)
cerr << "\t\tTRYING DELTA = " << geoDelta << endl;
double ll_delta_param = de_log_like(diagnostic_events, diag_values, siteInd, seqInd,
geoP, geoDelta,max_geo);
if (Model::DEBUG_LEVEL >= 1)
cerr << "\t\t\tLOGLIKE. FOR DELTA PARAM: " <<  ll_delta_param << endl;
if ((ll_delta_param > best_delta_ll) || (first)) {
best_delta_ll = ll_delta_param;
best_delta = geoDelta;
first = false;
if (Model::DEBUG_LEVEL >= 1)
cerr << "\t\t\tUPDATED BEST DELTA TO: " << best_delta
<< " with LL " << best_delta_ll << endl;
}
}
geoDelta = best_delta;
}
if ((max_geo) && (max_delta)) {
bool first = true;
int best_delta = 0;
double best_delta_ll = 0.0;
double best_geoP = 0.0;

for (geoDelta = Model::MIN_DELTA; geoDelta <= Model::MAX_DELTA; ++geoDelta) {
if (Model::DEBUG_LEVEL >= 1)
cerr << "\t\tTRYING DELTA = " << geoDelta << endl;
maximization_geoP(diagnostic_events, siteInd, seqInd, matrix, geoP, geoDelta);
double ll_delta_param = de_log_like(diagnostic_events, diag_values, siteInd, seqInd,
geoP, geoDelta,max_geo);
if (Model::DEBUG_LEVEL >= 1)
cerr << "\t\t\tLOGLIKE. FOR DELTA PARAM: " <<  ll_delta_param << endl;
if ((ll_delta_param > best_delta_ll) || (first)) {
best_delta_ll = ll_delta_param;
best_delta = geoDelta;
best_geoP = geoP;
first = false;
if (Model::DEBUG_LEVEL >= 1)
cerr << "\t\t\tUPDATED BEST DELTA TO: " << best_delta
<< " with LL " << best_delta_ll << endl;
}
}

geoDelta = best_delta;
geoP = best_geoP;
if (Model::DEBUG_LEVEL >= 1) {
cerr << "\t\tSET DELTA TO " << geoDelta << endl;
cerr << "\t\tSET GEO_P TO " << geoP << endl;
}
}
}







static void
get_numerator_seq_de_for_site(const string &seq,
const vector<double> &diagnostic_events,
const double diag_value,
const vector<vector<double> > &matrix,
const vector<double> &freqs,
const double geo_p,
const int geo_delta,
const double gamma,
const size_t site,
double &num,
const double de_weight,
const bool optGeo) {
const double N_LOG_PROB = -10000;

vector<double> f_powers(alphabet_size, 0.0);
for (size_t i = 0; i < seq.length(); ++i) {
const size_t base = base2int(seq[i]);
if (i >= site && i < site + matrix.size()) {
if ((seq[i] == 'N') || (seq[i] == 'n')) {
num += N_LOG_PROB;
} else {
assert(base < alphabet_size);
num += log(matrix[i - site][base]);
if (!std::isfinite(num)) {
stringstream ss;
ss << "failed expectation calculation; numerator non-finite. Matrix "
<< "entry was: " << matrix[i - site][base];
throw SMITHLABException(ss.str());
}
}
} else {
if ((seq[i] != 'N') && (seq[i] != 'n')) {
assert(base < alphabet_size);
f_powers[base]++;
}
if (!std::isfinite(f_powers[base]))
throw SMITHLABException("failed expectation calc; f_powers non-finite");
}
}

for (size_t b = 0; b < alphabet_size; b++) {
num += (f_powers[b] * log(freqs[b]));
assert(std::isfinite(num));
}

if (diagnostic_events.size() > 0) {
if (optGeo) {
vector<double> powers;
for (size_t j = 0; j < seq.length(); j++)
powers.push_back(log(diagnostic_events[j]) + de_weight*log(geo_p) + de_weight*(abs(j - (site + geo_delta)) * log(1.0-geo_p)));
num += smithlab::log_sum_log_vec(powers, powers.size());
} else {
num += diag_value;
}
}
num += log(gamma);
}


static void
get_numerator_seq_str_de_for_site(const string &seq,
const vector<double> &secondary_structure,
const vector<double> &diagnostic_events,
const double diag_value,
const vector<vector<double> > &matrix,
const vector<double> &motif_sec_str,
const vector<double> &freqs,
const double f_sec_str,
const double geo_p,
const int geo_delta,
const double gamma,
const size_t site,
double &num,
const double de_weight,
const bool optGeo) {
const double N_LOG_PROB = -10000;

vector<double> f_powers_ss(alphabet_size, 0.0);
vector<double> f_powers_ds(alphabet_size, 0.0);
for (size_t i = 0; i < seq.length(); ++i) {
const size_t base = base2int(seq[i]);
if (i >= site && i < site + matrix.size()) {
if ((seq[i] == 'N') || (seq[i] == 'n')) {
num += N_LOG_PROB;
}
else {
assert(base < alphabet_size);
num += (secondary_structure[i]
* log(matrix[i - site][base] * motif_sec_str[i - site]));
num += ((1.0 - secondary_structure[i])
* log(matrix[i - site][base] * (1.0 - motif_sec_str[i - site])));
if (!std::isfinite(num)) {
stringstream ss;
ss << "failed expectation calculation; numerator non-finite. BPP was: "
<< secondary_structure[i] << " matrix entry was: "
<< matrix[i - site][base] << " motif secondary structure was: "
<< motif_sec_str[i - site];
throw SMITHLABException(ss.str());
}
}
} else {
if ((seq[i] != 'N') && (seq[i] != 'n')) {
assert(base < alphabet_size);
f_powers_ss[base] += (1.0 - secondary_structure[i]);
f_powers_ds[base] += (secondary_structure[i]);
}
}
if (!std::isfinite(f_powers_ss[base]))
throw SMITHLABException("failed expectation calc; f_powers_ss non-finite");
if (!std::isfinite(f_powers_ds[base]))
throw SMITHLABException("failed expectation calc; f_powers_ds non-finite");
}

for (size_t b = 0; b < alphabet_size; b++) {
num += (f_powers_ss[b] * log(freqs[b] * (1.0 - f_sec_str)));
num += (f_powers_ds[b] * log(freqs[b] * (f_sec_str)));
assert(std::isfinite(num));
}

num += log(gamma);


double oldnum = num;
if (Model::DEBUG_LEVEL >= 4)
cerr << "\tlog prob before des: " << num << "; ";

if (diagnostic_events.size() > 0) {
if (optGeo) {
vector<double> powers;
for (size_t j = 0; j < seq.length(); j++)
powers.push_back(log(diagnostic_events[j]) + de_weight*log(geo_p) + de_weight*(abs(j - (site + geo_delta)) * log(1.0-geo_p)));
num += smithlab::log_sum_log_vec(powers, powers.size());
} else {
num += diag_value;
}
}

if (Model::DEBUG_LEVEL >= 4)
cerr << "\tlog prob after des: " << num << " diff = " << (num-oldnum) << endl;

}


static void
expectation_seq_str_de_for_single_seq(const string &seq,
const vector<double> &secondary_structure,
const vector<double> &diagnostic_events,
const vector<vector<double> > &diag_values,
const vector<vector<double> > &matrix,
const vector<double> &motif_sec_str,
const vector<double> &freqs,
const double f_sec_str,
const double geo_p,
const int geo_delta,
const double gamma,
vector<double> &site_indic,
double &seq_indic,
const double de_weight,
const bool optGeo) {
const double TINY = 1e-100;

if ((gamma > 1.0 ) || (gamma < 0.0)) {
stringstream ss;
ss << "failed expectation step: gamma (faction of sequences containing "
<< "the motif) was outside the expected bounds: " << gamma;
throw SMITHLABException(ss.str());
}

vector<double> numerator(site_indic.size(), 0.0);
#pragma omp parallel for
for (size_t i = 0; i < site_indic.size(); ++i) {
if (Model::DEBUG_LEVEL >= 4)
cerr << "loc  " << i << " = " << seq.substr(i,6) << endl;
double diag_value = 1.0;
if (diagnostic_events.size() > 0)
diag_value = diag_values[i][geo_delta + 8];

get_numerator_seq_str_de_for_site(seq, secondary_structure,
diagnostic_events, diag_value, matrix, motif_sec_str,
freqs, f_sec_str, geo_p, geo_delta, gamma,
i, numerator[i], de_weight, optGeo);
assert(std::isfinite(numerator[i]));
}

double no_motif = 0.0;
#pragma omp parallel for reduction(+:no_motif)
for (size_t i = 0; i < seq.length(); i++) {
if ((seq[i] == 'N') || (seq[i] == 'n')) continue;
const size_t base = base2int(seq[i]);
assert(base < alphabet_size);

no_motif += (secondary_structure[i]
* log(freqs[base] * f_sec_str));
no_motif += ((1.0 - secondary_structure[i])
* log(freqs[base] * (1.0 - f_sec_str)));
}

const double fracSeqsWithoutMotif = std::min((1.0 - gamma) + TINY, 1.0);
numerator.push_back(no_motif + log(fracSeqsWithoutMotif));

const double denominator = smithlab::log_sum_log_vec(numerator,
numerator.size());
assert(std::isfinite(denominator));
#pragma omp parallel for
for (size_t i = 0; i < site_indic.size(); ++i) {
site_indic[i] = std::max(std::numeric_limits<double>::min(), exp(numerator[i] - denominator));
}

seq_indic = accumulate(site_indic.begin(), site_indic.end(), 0.0);
assert(std::isfinite(seq_indic));
}


static void
expectation_seq_de_for_single_seq(const string &seq,
const vector<double> &diagnostic_events,
const vector<vector<double> > &diag_values,
const vector<vector<double> > &matrix,
const vector<double> &freqs,
const double geo_p,
const int geo_delta,
const double gamma,
vector<double> &site_indic,
double &seq_indic,
const double de_weight,
const bool optGeo) {
const double TINY = 1e-100;

if ((gamma > 1.0 ) || (gamma < 0.0)) {
stringstream ss;
ss << "failed expectation step: gamma (faction of sequences containing "
<< "the motif) was outside the expected bounds: " << gamma;
throw SMITHLABException(ss.str());
}

vector<double> numerator(site_indic.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < site_indic.size(); ++i) {
double diag_value = 1.0;
if (diagnostic_events.size() > 0)
diag_value = diag_values[i][geo_delta + 8];
get_numerator_seq_de_for_site(seq, diagnostic_events, diag_value, matrix, freqs, geo_p,
geo_delta, gamma, i, numerator[i], de_weight, optGeo);
assert(std::isfinite(numerator[i]));
}

double no_motif = 0.0;
#pragma omp parallel for reduction(+:no_motif)
for (size_t i = 0; i < seq.length(); i++) {
if ((seq[i] == 'N') || (seq[i] == 'n')) continue;
const size_t base = base2int(seq[i]);
assert(base < alphabet_size);
no_motif += log(freqs[base]);
}

const double fracSeqsWithoutMotif = std::min((1.0 - gamma) + TINY, 1.0);
numerator.push_back(no_motif + log(fracSeqsWithoutMotif));

const double denominator = smithlab::log_sum_log_vec(numerator, numerator.size());

#pragma omp parallel for
for (size_t i = 0; i < site_indic.size(); ++i) {
site_indic[i] = std::max(std::numeric_limits<double>::min(), exp(numerator[i] - denominator));
assert(std::isfinite(site_indic[i]));
}

seq_indic = accumulate(site_indic.begin(), site_indic.end(), 0.0);
}


static void
expectation_for_single_seq(const string &seq,
const vector<vector<double> > &matrix,
const vector<double> &freqs,
const double gamma,
vector<double> &site_indic,
double &seq_indic,
const bool optGeo) {
vector<double> diagnostic_events;
vector<vector<double> > diag_values;
const double geoP = 1;
const int geoDelta = 0;
expectation_seq_de_for_single_seq(seq, diagnostic_events, diag_values, matrix, freqs, geoP,
geoDelta, gamma, site_indic, seq_indic, 1, optGeo);

}


static void
expectation_seq(const vector<string> &sequences,
const vector<vector<double> > &matrix,
const vector<double> &freqs,
const double gamma,
vector<vector<double> > &site_indic,
vector<double> &seq_indic,
const bool optGeo) {

for (size_t i = 0; i < sequences.size(); i++)
expectation_for_single_seq(sequences[i], matrix, freqs, gamma,
site_indic[i], seq_indic[i], optGeo);
}


static void
expectation_seq_de(const vector<string> &sequences,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &matrix,
const vector<double> &freqs,
const double geo_p,
const int geo_delta,
const double gamma,
vector<vector<double> > &site_indic,
vector<double> &seq_indic,
const double de_weight,
const bool optGeo) {
if (Model::DEBUG_LEVEL >= 2)
cerr << "performing expectation step with matrix " << endl
<< matrixToString(matrix) << endl;

for (size_t i = 0; i < sequences.size(); i++) {
expectation_seq_de_for_single_seq(sequences[i], diagnostic_events[i], diag_values[i],
matrix, freqs, geo_p, geo_delta, gamma,
site_indic[i], seq_indic[i], de_weight, optGeo);
}

if (Model::DEBUG_LEVEL >= 2)
cerr << "finished expectation step" << endl;
}


static void
expectation_seq_str_de(const vector<string> &sequences,
const vector<vector<double> > &secondary_structure,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &matrix,
const vector<double> &motif_sec_str,
const vector<double> &freqs,
const double f_sec_str,
const double geo_p,
const int geo_delta,
const double gamma,
vector<vector<double> > &site_indic,
vector<double> &seq_indic,
const double de_weight,
const bool optGeo) {
if (Model::DEBUG_LEVEL >= 3)
cerr << "performing expectation step with matrix " << endl
<< matrixToString(matrix) << endl;
for (size_t i = 0; i < sequences.size(); i++) {
if (Model::DEBUG_LEVEL >= 3) {
cerr << "for seq " << i << endl;
cerr << "des: ";
for (size_t j = 0; j < diagnostic_events[i].size(); ++j)
cerr << diagnostic_events[i][j] << ", ";
cerr << endl;
}

expectation_seq_str_de_for_single_seq(sequences[i], secondary_structure[i],
diagnostic_events[i], diag_values[i], matrix,
motif_sec_str, freqs, f_sec_str,
geo_p, geo_delta, gamma,
site_indic[i], seq_indic[i], de_weight, optGeo);

if (Model::DEBUG_LEVEL >= 3) {
cerr << "gamma is " << gamma << endl;
cerr << "site indicators after exp step: ";
for (size_t j = 0; j < site_indic[i].size(); ++j)
cerr << site_indic[i][j] << ", ";
size_t best_loc = std::distance(site_indic[i].begin(),
max_element(site_indic[i].begin(),
site_indic[i].end()));
cerr << endl;
cerr << "sum of indicators: "
<< std::accumulate(site_indic[i].begin(), site_indic[i].end(), 0.0)
<< endl;
cerr << "most likely occurrence at location " << best_loc << " with prob "
<< site_indic[i][best_loc] << " has seq "
<< sequences[i].substr(best_loc,6) << endl;
size_t best_de_l = std::distance(diagnostic_events[i].begin(),
max_element(diagnostic_events[i].begin(),
diagnostic_events[i].end()));
if (!diagnostic_events[0].empty()) {
cerr << "best DE loc is at " << best_de_l << " with "
<< diagnostic_events[i][best_de_l] << endl;
}
}
}
}


string
Model::toString_pwm() const {
return matrixToString(this->matrix);
}




double
Model::calculate_oops_log_l(const vector<string> &sequences,
const vector<vector<double> > &site_indic) const {

vector<vector<double> > nb_fg;
vector<double> nb_bg;
calculate_number_of_bases_fg_bg(sequences, site_indic, matrix.size(), nb_fg,
nb_bg);

double ret = 0.0;
for (size_t i = 0; i < alphabet_size; ++i) {
ret += (nb_bg[i] * log(f[i]));
for (size_t j = 0; j < matrix.size(); ++j)
ret += (nb_fg[j][i] * log(matrix[j][i]));
}

for (size_t i = 0; i < site_indic.size(); ++i)
ret -= log(site_indic[i].size());

return ret;
}


double
Model::calculate_zoops_log_l(const vector<string> &seqs,
const vector<vector<double> > &site_indic,
const vector<double> &seq_indic) const {
vector<vector<double> > diagnostic_events (seqs.size());
vector<vector<vector<double> > > diag_values (seqs.size());
return calculate_zoops_log_l(seqs, diagnostic_events, diag_values, site_indic, seq_indic);
}


double
Model::calculate_zoops_log_l(const vector<string> &sequences,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &site_indic,
const vector<double> &seq_indic) const {
const double TINY = 1e-100;

vector<vector<double> > nb_fg;
vector<double> nb_bg;
calculate_number_of_bases_fg_bg(sequences, site_indic, matrix.size(), nb_fg, nb_bg);

double ret = 0.0;
for (size_t i = 0; i < alphabet_size; ++i) {
ret += (nb_bg[i] * log(f[i]));
if (!std::isfinite(ret)) {
stringstream ss;
ss << "log-likelihood calculation failed; result not finite. background "
<< "count for base was " << nb_bg[i] << "; f for base was " << f[i];
throw SMITHLABException(ss.str());
}

for (size_t j = 0; j < matrix.size(); ++j) {
ret += (nb_fg[j][i] * log(matrix[j][i]));
if (!std::isfinite(ret)) {
stringstream ss;
ss << "log-likelihood calculation failed; result not finite. "
<< "foreground count for base was " << nb_fg[j][i] << "; matrix "
<< "entry for base was " << matrix[j][i];
throw SMITHLABException(ss.str());
}
}
}

if (diagnostic_events.size() > 0) {
if (opt_geo) {
for (size_t i = 0; i < sequences.size(); i++)
if (diagnostic_events[i].size() > 0)
for (size_t k = 0; k < site_indic[i].size(); k++) {
vector<double> powers;
for (size_t j = 0; j < sequences[i].length(); j++)
powers.push_back(log(diagnostic_events[i][j]) + de_weight*log(p) + de_weight * (abs(j - (k + delta)) * log(1.0-p)));
ret += (site_indic[i][k] * smithlab::log_sum_log_vec(powers, powers.size()));
}
} else {
for (size_t i = 0; i < sequences.size(); i++)
if (diagnostic_events[i].size() > 0)
for (size_t k = 0; k < site_indic[i].size(); k++)
ret += (site_indic[i][k] * diag_values[i][k][delta+8]);
}
}

double ret_l = 0.0; 
#pragma omp parallel for reduction(+:ret_l)
for (size_t i = 0; i < sequences.size(); i++) {
double has_no_motif = 0.0;
for (size_t j = 0; j < sequences[i].length(); j++) {
if ((sequences[i][j] == 'N') || (sequences[i][j] == 'n')) continue;
const size_t base = base2int(sequences[i][j]);
assert(base < alphabet_size);
has_no_motif += log(f[base]);
}

ret_l += ((1.0 - seq_indic[i]) * has_no_motif);
ret_l += ((1.0 - seq_indic[i]) * log(std::min(1.0 - gamma + TINY, 1.0)));
ret_l += (seq_indic[i] * log(gamma));
}

ret += ret_l;

return ret;
}


double
Model::calculate_zoops_log_l(const vector<string> &sequences,
const vector<vector<double> > &secondary_structure,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &site_indic,
const vector<double> &seq_indic) const {
const double TINY = 1e-100;

checkAndThrow_consistent(sequences, secondary_structure,
diagnostic_events, "Likelihood calculation failed.");

vector<vector<double> > nb_fg_ss;
vector<vector<double> > nb_fg_ds;
vector<double> nb_bg_ss;
vector<double> nb_bg_ds;
calculate_number_of_bases_fg_bg_str(sequences, secondary_structure,
site_indic, matrix.size(), nb_fg_ss,
nb_bg_ss, nb_fg_ds, nb_bg_ds);

double ret = 0.0;
for (size_t i = 0; i < alphabet_size; ++i) {
ret += (nb_bg_ss[i] * log(f[i] * (1.0 - f_sec_str)));
ret += (nb_bg_ds[i] * log(f[i] * f_sec_str));
for (size_t j = 0; j < matrix.size(); ++j) {
ret += (nb_fg_ss[j][i] * log(matrix[j][i] * (1.0 - motif_sec_str[j])));
ret += (nb_fg_ds[j][i] * log(matrix[j][i] * motif_sec_str[j]));
}
}

if (diagnostic_events.size() > 0) {
if (opt_geo) {
for (size_t i = 0; i < sequences.size(); i++)
if (diagnostic_events[i].size() > 0)
for (size_t k = 0; k < site_indic[i].size(); k++) {
vector<double> powers;
for (size_t j = 0; j < sequences[i].length(); j++)
powers.push_back(log(diagnostic_events[i][j]) + de_weight*log(p) + de_weight * (abs(j - (k + delta)) * log(1.0-p)));
ret += (site_indic[i][k] * smithlab::log_sum_log_vec(powers, powers.size()));
}
} else {
for (size_t i = 0; i < sequences.size(); i++)
if (diagnostic_events[i].size() > 0) 
for (size_t k = 0; k < site_indic[i].size(); k++) 
ret += (site_indic[i][k] * diag_values[i][k][delta+8]);      
}
}

double ret_l = 0.0; 
#pragma omp parallel for reduction(+:ret_l)
for (size_t i = 0; i < sequences.size(); i++) {
double has_no_motif = 0.0;
for (size_t j = 0; j < sequences[i].length(); j++) {
if ((sequences[i][j] == 'N') || (sequences[i][j] == 'n')) continue;
const size_t base = base2int(sequences[i][j]);
assert(base < alphabet_size);
has_no_motif += log(f[base]);
}

ret_l += ((1.0 - seq_indic[i]) * has_no_motif);
ret_l += ((1.0 - seq_indic[i]) * log(std::min(1.0 - gamma + TINY, 1.0)));
ret_l += (seq_indic[i] * log(gamma));
}

ret += ret_l;
return ret;
}





void
Model::expectation_maximization_seq_str(const vector<string> &sequences,
const vector<vector<double> > &secStr,
vector<vector<double> > &site_indic,
vector<double> &seq_indic) {
vector<vector<double> > diagnostic_events(sequences.size());
vector<vector<vector<double> > > diag_values(sequences.size());
expectationMax_SeqStrDE(sequences, secStr, diagnostic_events, diag_values,
site_indic, seq_indic);
}


void
Model::expectation_maximization_seq(const vector<string> &seqs,
vector<vector<double> > &site_indic,
vector<double> &seq_indic) {
bool first = true;
double prev_score = std::numeric_limits<double>::max();
for (size_t i = 0; i < max_iterations; ++i) {
expectation_seq(seqs, matrix, f, gamma, site_indic, seq_indic, opt_geo);
maximization_seq(seqs, site_indic, seq_indic, matrix, f, gamma);
const double score = calculate_zoops_log_l(seqs, site_indic, seq_indic);
if (!first) {
const double delta = fabs(prev_score - score);
const double deltaProp = delta / fabs(prev_score);
if (Model::DEBUG_LEVEL >= 2)
cerr << "new ll: " << score << " old ll: " << prev_score << " "
<< "absolute change is " << delta << " fraction change "
<< deltaProp << endl;
if (deltaProp < tolerance) break;
} else {
first = false;
}
prev_score = score;
}
}


void
Model::expectation_maximization_seq_de(const vector<string> &seqs,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
vector<vector<double> > &site_indic,
vector<double> &seq_indic,
const bool holdDelta = false) {
this->delta = Model::DEFAULT_DELTA;
double prev_score = std::numeric_limits<double>::max();
bool first = true;
double score = 0.0;

for (size_t i = 0; i < max_iterations; ++i) {  
if (Model::DEBUG_LEVEL >= 1) {
cerr << "EM, SEQ. AND DE, ITER NUM " << i << endl
<< "\tEXPECTATION STEP" << endl;
}

expectation_seq_de(seqs, diagnostic_events, diag_values, matrix, f, p, delta, gamma,
site_indic, seq_indic, de_weight, opt_geo);

if (Model::DEBUG_LEVEL >= 1) 
cerr << "\tSEQUENCE MAX. STEP" << endl;

maximization_seq(seqs, site_indic, seq_indic, matrix, f, gamma);

if (diagnostic_events.front().size() > 0) {
if (Model::DEBUG_LEVEL >= 1) 
cerr << "\tDE MAX. STEP" << endl;
maximization_de(diagnostic_events, diag_values, site_indic, seq_indic, matrix, p, delta,
this->opt_geo, this->opt_delta);
}

if (Model::DEBUG_LEVEL >= 1) 
cerr << "\tCALC. LOG-LIKE" << endl;

score = calculate_zoops_log_l(seqs, diagnostic_events, diag_values, site_indic, seq_indic);
if (!first) {
const double delta = fabs(prev_score - score);
const double deltaProp = delta / fabs(prev_score);
if (Model::DEBUG_LEVEL >= 1)
cerr << "new ll: " << score << " old ll: " << prev_score << " "
<< "absolute change is " << delta << " fraction change "
<< deltaProp << endl;
if (deltaProp < tolerance) break;
} else {
first = false;
}
prev_score = score;
}
}



void
Model::expectationMax_SeqStrDE(const vector<string> &seqs,
const vector<vector<double> > &secStr,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
vector<vector<double> > &siteInd,
vector<double> &seqInd) {
this->delta = Model::DEFAULT_DELTA;
double prev_score = std::numeric_limits<double>::max();
double score = 0.0;
bool first = true;
for (size_t i = 0; i < max_iterations; ++i) {
expectation_seq_str_de(seqs, secStr, diagnostic_events, diag_values, matrix, motif_sec_str,
f, f_sec_str, p, delta, gamma, siteInd, seqInd, de_weight, opt_geo);
maximization_seq(seqs, siteInd, seqInd, matrix, f, gamma);
maximization_str(secStr, siteInd, seqInd, matrix, motif_sec_str, f_sec_str);

if (diagnostic_events.front().size() > 0) {
maximization_de(diagnostic_events, diag_values, siteInd, seqInd, matrix, p, delta,
this->opt_geo, this->opt_delta);
}

score = calculate_zoops_log_l(seqs, secStr, diagnostic_events, diag_values, siteInd, seqInd);
if (!first) {
const double delta = fabs(prev_score - score);
const double deltaProp = delta / fabs(prev_score);
if (Model::DEBUG_LEVEL >= 1)
cerr << "new ll: " << score << " old ll: " << prev_score << " "
<< "absolute change is " << delta << " fraction change "
<< deltaProp << endl;
if (deltaProp < tolerance) break;
} else {
first = false;
}
prev_score = score;
}
}



void
Model::expectationMax(const vector<string> &seqs,
const vector<vector<double> > &diagnostic_events,
const vector<vector<vector<double> > > &diag_values,
const vector<vector<double> > &secStruct,
vector<vector<double> > &site_indic,
vector<double> &seq_indic) {
if (secStruct.empty()) {
expectation_maximization_seq_de(seqs, diagnostic_events, diag_values, site_indic, seq_indic);
}
else {
expectationMax_SeqStrDE(seqs, secStruct, diagnostic_events, diag_values, site_indic, seq_indic);
}
}





void
Model::set_model_uniform(const size_t width,
Model &model) {
model.matrix.clear();
model.matrix.resize(width,
vector<double>(alphabet_size, 1.0 / alphabet_size));
model.motif_sec_str = vector<double>(width, 0.5);
model.f = vector<double>(alphabet_size, 1.0 / alphabet_size);
model.f_sec_str = 0.5;
model.p = 0.5;
model.delta = 0;
model.gamma = 0.5;
}


void
Model::set_model_by_word(const double pseudocount,
const string &kmer,
Model &model) {

const size_t len = kmer.length();
model.matrix.clear();
model.matrix.resize(len, vector<double>(alphabet_size, pseudocount));

for (size_t i = 0; i < len; ++i) {
const size_t base = base2int(kmer[i]);
assert(base < alphabet_size);
model.matrix[i][base] += 1.0;
}

for (size_t i = 0; i < len; ++i)
for (size_t j = 0; j < alphabet_size; ++j) {
const double tot = accumulate(model.matrix[i].begin(),
model.matrix[i].end(), 0.0);
transform(model.matrix[i].begin(), model.matrix[i].end(),
model.matrix[i].begin(),
std::bind2nd(std::divides<double>(), tot));
}
}
