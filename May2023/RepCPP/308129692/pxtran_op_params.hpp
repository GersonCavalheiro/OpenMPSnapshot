#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <costa/scalapack.hpp>

namespace costa {
template <typename T>
struct pxtran_op_params {
int ma; 
int na; 

int mc; 
int nc; 

int bma; 
int bna; 

int bmc; 
int bnc; 

int ia = 1; 
int ja = 1; 

int ic = 1; 
int jc = 1; 

int m;
int n;

T alpha = T{1};
T beta = T{0};

int lld_a;
int lld_c;

int p_rows; 
int p_cols; 
int P;
char order = 'R';

int src_ma = 0; 
int src_na = 0; 

int src_mc = 0; 
int src_nc = 0; 

char op = 'T'; 

pxtran_op_params() = default;

void initialize(int mm, int nn,
int block_a1, int block_a2,
int block_c1, int block_c2,
int prows, int pcols,
T a, T b) {
m = mm;
n = nn;

ma = n;
na = m;
mc = m;
nc = n;

bma = block_a1;
bna = block_a2;

bmc = block_c1;
bnc = block_c2;

ia = 1; ja = 1;
ic = 1; jc = 1;

alpha = a;
beta = b;

order = 'R';
p_rows = prows;
p_cols = pcols;
P = p_rows * p_cols;

lld_a = scalapack::max_leading_dimension(ma, bma, p_rows);
lld_c = scalapack::max_leading_dimension(mc, bmc, p_rows);

src_ma = 0; src_na = 0;
src_mc = 0; src_nc = 0;
}

pxtran_op_params(int m, int n,
int bm, int bn,
int prows, int pcols,
T a, T b) {
bma = bm;
bna = bn;

bmc = bm;
bnc = bn;

initialize(m, n,
bma, bna,
bmc, bnc,
prows, pcols,
a, b);
std::string info;
if (!valid(info)) {
std::runtime_error("WRONG PXTRAN(U/C) PARAMETER: " + info);
}
}


pxtran_op_params(int m, int n,
int block_a1, int block_a2,
int block_c1, int block_c2,
int prows, int pcols,
T a, T b) {
initialize(m, n,
block_a1, block_a2,
block_c1, block_c2,
prows, pcols,
a, b);
std::string info;
if (!valid(info)) {
std::runtime_error("WRONG PXTRAN(U/C) PARAMETER: " + info);
}
}

pxtran_op_params(
int ma, int na, 
int mc, int nc, 

int bma, int bna, 
int bmc, int bnc, 

int ia, int ja, 
int ic, int jc, 

int m, int n,

T alpha, T beta,

int lld_a, int lld_c,

int p_rows, int p_cols,
char order,

int src_ma, int src_na, 
int src_mc, int src_nc, 

char op 
) :
ma(ma), na(na),
mc(mc), nc(nc),

bma(bma), bna(bna),
bmc(bmc), bnc(bnc),

ia(ia), ja(ja),
ic(ic), jc(jc),

m(m), n(n),

alpha(alpha), beta(beta),

lld_a(lld_a), lld_c(lld_c),

order(std::toupper(order)),
p_rows(p_rows), p_cols(p_cols),
P(p_rows * p_cols),

src_ma(src_ma), src_na(src_na),
src_mc(src_mc), src_nc(src_nc),

op(std::toupper(op))
{
std::string info;
if (!valid(info)) {
std::runtime_error("WRONG PXTRAN(U/C) PARAMETER: " + info);
}
}

bool valid(std::string& info) {
info = "";
if (order != 'R' && order != 'C') {
info = "oder = " + std::to_string(order);
return false;
}

std::vector<int> positive = {
ma, na, mc, nc,
bma, bna, bmc, bnc,
m, n,
lld_a, lld_c,
p_rows, p_cols, P,
};
std::vector<std::string> positive_labels = {
"ma", "na", "mc", "nc",
"bma", "bna", "bmc", "bnc",
"m", "n",
"lld_a", "lld_c",
"p_rows", "p_cols", "P"
};
for (int i = 0; i < positive.size(); ++i) {
if (positive[i] < 0) {
info = positive_labels[i] + " = " + std::to_string(positive[i]);
return false;
}
}

if (ia < 1 || ia > ma) {
info = "ia = " + std::to_string(ia);
return false;
}
if (ja < 1 || ja > na) {
info = "ja = " + std::to_string(ja);
return false;
}

if (ic < 1 || ic > mc) {
info = "ic = " + std::to_string(ic);
return false;
}
if (jc < 1 || jc > nc) {
info = "jc = " + std::to_string(jc);
return false;
}

int ma_sub = n;
int ma_sub_end = ia - 1 + ma_sub;
if (ma_sub_end >= ma) {
info = "ia - 1 + (m or k) = " + std::to_string(ma_sub_end);
return false;
}
int na_sub = m;
int na_sub_end = ja - 1 + na_sub;
if (na_sub_end >= na) {
info = "ja - 1 + (k or m) = " + std::to_string(na_sub_end);
return false;
}

int mc_sub = m;
int mc_sub_end = ic - 1 + mc_sub;
if (mc_sub_end >= mc) {
info = "ic - 1 + m = " + std::to_string(mc_sub_end);
return false;
}
int nc_sub = n;
int nc_sub_end = jc - 1 + nc_sub;
if (nc_sub_end >= nc) {
info = "jc - 1 + n = " + std::to_string(nc_sub_end);
return false;
}

if (src_ma < 0 || src_ma >= ma) {
info = "src_ma = " + std::to_string(src_ma);
return false;
}
if (src_na < 0 || src_na >= na) {
info = "src_na = " + std::to_string(src_na);
return false;
}

if (src_mc < 0 || src_mc >= mc) {
info = "src_mc = " + std::to_string(src_mc);
return false;
}
if (src_nc < 0 || src_nc >= nc) {
info = "src_nc = " + std::to_string(src_nc);
return false;
}

int min_lld_a = scalapack::min_leading_dimension(ma, bma, p_rows);
int min_lld_c = scalapack::min_leading_dimension(mc, bmc, p_rows);

if (lld_a < min_lld_a) {
info = "lld_a = " + std::to_string(min_lld_a);
return false;
}
if (lld_c < min_lld_c) {
info = "lld_c = " + std::to_string(min_lld_c);
return false;
}

if (op != 'T' && op != 'C') {
info = "op = " + std::to_string(op);
return false;
}

return true;
}

friend std::ostream &operator<<(std::ostream &os,
const pxtran_op_params &obj) {
os << "=============================" << std::endl;
os << "      GLOBAL MAT. SIZES" << std::endl;
os << "=============================" << std::endl;
os << "A = " << obj.ma << " x " << obj.na << std::endl;
os << "C = " << obj.mc << " x " << obj.nc << std::endl;
os << "=============================" << std::endl;
os << "        SUBMATRICES" << std::endl;
os << "=============================" << std::endl;
os << "(ia, ja) = (" << obj.ia << ", " << obj.ja << ")" << std::endl;
os << "(ic, jc) = (" << obj.ic << ", " << obj.jc << ")" << std::endl;
os << "=============================" << std::endl;
os << "      SUBMATRIX SIZES" << std::endl;
os << "=============================" << std::endl;
os << "m = " << obj.m << std::endl;
os << "n = " << obj.n << std::endl;
os << "=============================" << std::endl;
os << "      ADDITIONAL OPTIONS" << std::endl;
os << "=============================" << std::endl;
os << "alpha = " << obj.alpha << std::endl;
os << "beta = " << obj.beta << std::endl;
os << "=============================" << std::endl;
os << "         PROC GRID" << std::endl;
os << "=============================" << std::endl;
os << "grid = " << obj.p_rows << " x " << obj.p_cols << std::endl;
os << "grid order = " << obj.order << std::endl;
os << "=============================" << std::endl;
os << "         PROC SRCS" << std::endl;
os << "=============================" << std::endl;
os << "P_SRC(A) = (" << obj.src_ma << ", " << obj.src_na << ")" << std::endl;
os << "P_SRC(C) = (" << obj.src_mc << ", " << obj.src_nc << ")" << std::endl;
os << "=============================" << std::endl;
os << "          BLOCK SIZES" << std::endl;
os << "=============================" << std::endl;
os << "Blocks(A) = (" << obj.bma << ", " << obj.bna << ")" << std::endl;
os << "Blocks(C) = (" << obj.bmc << ", " << obj.bnc << ")" << std::endl;
os << "=============================" << std::endl;
os << "          LEADING DIMS" << std::endl;
os << "=============================" << std::endl;
os << "lld_a = " << obj.lld_a << std::endl;
os << "lld_c = " << obj.lld_c << std::endl;
os << "=============================" << std::endl;
os << "            OPERATION" << std::endl;
os << "=============================" << std::endl;
os << "Operation = ";
if (obj.op == 'T')
os << "Transpose" << std::endl;
else if (obj.op == 'C')
os << "Conjugate-Transpose" << std::endl;
else
os << "Unknown operation" << std::endl;
os << "=============================" << std::endl;
return os;
}
};
}
