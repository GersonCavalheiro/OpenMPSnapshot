void kernel1(
nd_item<3> &item,
const int N0, 
const int N1,
const int N2,
const int ifirst, const int ilast,
const int jfirst, const int jlast,
const int kfirst, const int klast,
const float_sw4 a1, const float_sw4 sgn,
const float_sw4* __restrict__ a_u, 
const float_sw4* __restrict__ a_mu,
const float_sw4* __restrict__ a_lambda,
const float_sw4* __restrict__ a_met,
const float_sw4* __restrict__ a_jac,
float_sw4* __restrict__ a_lu, 
const float_sw4* __restrict__ a_acof, 
const float_sw4* __restrict__ a_bope,
const float_sw4* __restrict__ a_ghcof, 
const float_sw4* __restrict__ a_acof_no_gp,
const float_sw4* __restrict__ a_ghcof_no_gp, 
const float_sw4* __restrict__ a_strx,
const float_sw4* __restrict__ a_stry ) 
{
int i = item.get_global_id(2);
int j = item.get_global_id(1);
int k = item.get_global_id(0);
if ((i < N0) && (j < N1) && (k < N2)) {
float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
float_sw4 istry = 1 / (stry(j));
float_sw4 istrx = 1 / (strx(i));
float_sw4 istrxy = istry * istrx;

float_sw4 r1 = 0, r2 = 0, r3 = 0;

float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
met(1, i - 2, j, k) * met(1, i - 2, j, k) *
strx(i - 2);
float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
met(1, i - 1, j, k) * met(1, i - 1, j, k) *
strx(i - 1);
float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
met(1, i, j, k) * strx(i);
float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
met(1, i + 1, j, k) * met(1, i + 1, j, k) *
strx(i + 1);
float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
met(1, i + 2, j, k) * met(1, i + 2, j, k) *
strx(i + 2);

float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

r1 = r1 + i6 *
(mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
istry;

cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
met(1, i, j - 2, k) * stry(j - 2);
cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
met(1, i, j - 1, k) * stry(j - 1);
cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
met(1, i, j + 1, k) * stry(j + 1);
cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
met(1, i, j + 2, k) * stry(j + 2);

mux1 = cof2 - tf * (cof3 + cof1);
mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
mux4 = cof4 - tf * (cof3 + cof5);

r1 = r1 + i6 *
(mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
istrx;

cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
met(1, i - 2, j, k) * strx(i - 2);
cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
met(1, i - 1, j, k) * strx(i - 1);
cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
met(1, i + 1, j, k) * strx(i + 1);
cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
met(1, i + 2, j, k) * strx(i + 2);

mux1 = cof2 - tf * (cof3 + cof1);
mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
mux4 = cof4 - tf * (cof3 + cof5);

r2 = r2 + i6 *
(mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +
mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +
mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +
mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *
istry;

cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
met(1, i, j - 2, k) * met(1, i, j - 2, k) * stry(j - 2);
cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
met(1, i, j - 1, k) * met(1, i, j - 1, k) * stry(j - 1);
cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
met(1, i, j, k) * stry(j);
cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
met(1, i, j + 1, k) * met(1, i, j + 1, k) * stry(j + 1);
cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
met(1, i, j + 2, k) * met(1, i, j + 2, k) * stry(j + 2);
mux1 = cof2 - tf * (cof3 + cof1);
mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
mux4 = cof4 - tf * (cof3 + cof5);

r2 = r2 + i6 *
(mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +
mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +
mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +
mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *
istrx;

cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
met(1, i - 2, j, k) * strx(i - 2);
cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
met(1, i - 1, j, k) * strx(i - 1);
cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
met(1, i + 1, j, k) * strx(i + 1);
cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
met(1, i + 2, j, k) * strx(i + 2);

mux1 = cof2 - tf * (cof3 + cof1);
mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
mux4 = cof4 - tf * (cof3 + cof5);

r3 = r3 + i6 *
(mux1 * (u(3, i - 2, j, k) - u(3, i, j, k)) +
mux2 * (u(3, i - 1, j, k) - u(3, i, j, k)) +
mux3 * (u(3, i + 1, j, k) - u(3, i, j, k)) +
mux4 * (u(3, i + 2, j, k) - u(3, i, j, k))) *
istry;

cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
met(1, i, j - 2, k) * stry(j - 2);
cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
met(1, i, j - 1, k) * stry(j - 1);
cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
met(1, i, j + 1, k) * stry(j + 1);
cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
met(1, i, j + 2, k) * stry(j + 2);
mux1 = cof2 - tf * (cof3 + cof1);
mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
mux4 = cof4 - tf * (cof3 + cof5);

r3 = r3 + i6 *
(mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +
mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +
mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +
mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *
istrx;

float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;
for (int q = 1; q <= 8; q++) {
mucofu2 = 0;
mucofuv = 0;
mucofuw = 0;
mucofvw = 0;
mucofv2 = 0;
mucofw2 = 0;
for (int m = 1; m <= 8; m++) {
mucofu2 += acof(k, q, m) *
((2 * mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
strx(i) * met(2, i, j, m) * strx(i) +
mu(i, j, m) * (met(3, i, j, m) * stry(j) *
met(3, i, j, m) * stry(j) +
met(4, i, j, m) * met(4, i, j, m)));
mucofv2 += acof(k, q, m) *
((2 * mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *
stry(j) * met(3, i, j, m) * stry(j) +
mu(i, j, m) * (met(2, i, j, m) * strx(i) *
met(2, i, j, m) * strx(i) +
met(4, i, j, m) * met(4, i, j, m)));
mucofw2 += acof(k, q, m) *
((2 * mu(i, j, m) + la(i, j, m)) * met(4, i, j, m) *
met(4, i, j, m) +
mu(i, j, m) * (met(2, i, j, m) * strx(i) *
met(2, i, j, m) * strx(i) +
met(3, i, j, m) * stry(j) *
met(3, i, j, m) * stry(j)));
mucofuv += acof(k, q, m) * (mu(i, j, m) + la(i, j, m)) *
met(2, i, j, m) * met(3, i, j, m);
mucofuw += acof(k, q, m) * (mu(i, j, m) + la(i, j, m)) *
met(2, i, j, m) * met(4, i, j, m);
mucofvw += acof(k, q, m) * (mu(i, j, m) + la(i, j, m)) *
met(3, i, j, m) * met(4, i, j, m);
}

r1 += istrxy * mucofu2 * u(1, i, j, q) + mucofuv * u(2, i, j, q) +
istry * mucofuw * u(3, i, j, q);
r2 += mucofuv * u(1, i, j, q) + istrxy * mucofv2 * u(2, i, j, q) +
istrx * mucofvw * u(3, i, j, q);
r3 += istry * mucofuw * u(1, i, j, q) +
istrx * mucofvw * u(2, i, j, q) +
istrxy * mucofw2 * u(3, i, j, q);
}

mucofu2 =
ghcof(k) * ((2 * mu(i, j, 1) + la(i, j, 1)) * met(2, i, j, 1) *
strx(i) * met(2, i, j, 1) * strx(i) +
mu(i, j, 1) * (met(3, i, j, 1) * stry(j) *
met(3, i, j, 1) * stry(j) +
met(4, i, j, 1) * met(4, i, j, 1)));
mucofv2 =
ghcof(k) * ((2 * mu(i, j, 1) + la(i, j, 1)) * met(3, i, j, 1) *
stry(j) * met(3, i, j, 1) * stry(j) +
mu(i, j, 1) * (met(2, i, j, 1) * strx(i) *
met(2, i, j, 1) * strx(i) +
met(4, i, j, 1) * met(4, i, j, 1)));
mucofw2 =
ghcof(k) *
((2 * mu(i, j, 1) + la(i, j, 1)) * met(4, i, j, 1) *
met(4, i, j, 1) +
mu(i, j, 1) *
(met(2, i, j, 1) * strx(i) * met(2, i, j, 1) * strx(i) +
met(3, i, j, 1) * stry(j) * met(3, i, j, 1) * stry(j)));
mucofuv = ghcof(k) * (mu(i, j, 1) + la(i, j, 1)) * met(2, i, j, 1) *
met(3, i, j, 1);
mucofuw = ghcof(k) * (mu(i, j, 1) + la(i, j, 1)) * met(2, i, j, 1) *
met(4, i, j, 1);
mucofvw = ghcof(k) * (mu(i, j, 1) + la(i, j, 1)) * met(3, i, j, 1) *
met(4, i, j, 1);
r1 += istrxy * mucofu2 * u(1, i, j, 0) + mucofuv * u(2, i, j, 0) +
istry * mucofuw * u(3, i, j, 0);
r2 += mucofuv * u(1, i, j, 0) + istrxy * mucofv2 * u(2, i, j, 0) +
istrx * mucofvw * u(3, i, j, 0);
r3 += istry * mucofuw * u(1, i, j, 0) +
istrx * mucofvw * u(2, i, j, 0) +
istrxy * mucofw2 * u(3, i, j, 0);

r1 +=
c2 *
(mu(i, j + 2, k) * met(1, i, j + 2, k) *
met(1, i, j + 2, k) *
(c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
c1 *
(u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
mu(i, j - 2, k) * met(1, i, j - 2, k) *
met(1, i, j - 2, k) *
(c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
c1 * (u(2, i + 1, j - 2, k) -
u(2, i - 1, j - 2, k)))) +
c1 *
(mu(i, j + 1, k) * met(1, i, j + 1, k) *
met(1, i, j + 1, k) *
(c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
c1 *
(u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
mu(i, j - 1, k) * met(1, i, j - 1, k) *
met(1, i, j - 1, k) *
(c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
c1 *
(u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

r1 +=
c2 *
(la(i + 2, j, k) * met(1, i + 2, j, k) *
met(1, i + 2, j, k) *
(c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
c1 *
(u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
la(i - 2, j, k) * met(1, i - 2, j, k) *
met(1, i - 2, j, k) *
(c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
c1 * (u(2, i - 2, j + 1, k) -
u(2, i - 2, j - 1, k)))) +
c1 *
(la(i + 1, j, k) * met(1, i + 1, j, k) *
met(1, i + 1, j, k) *
(c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
c1 *
(u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
la(i - 1, j, k) * met(1, i - 1, j, k) *
met(1, i - 1, j, k) *
(c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
c1 *
(u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

r2 +=
c2 *
(la(i, j + 2, k) * met(1, i, j + 2, k) *
met(1, i, j + 2, k) *
(c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +
c1 *
(u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -
la(i, j - 2, k) * met(1, i, j - 2, k) *
met(1, i, j - 2, k) *
(c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +
c1 * (u(1, i + 1, j - 2, k) -
u(1, i - 1, j - 2, k)))) +
c1 *
(la(i, j + 1, k) * met(1, i, j + 1, k) *
met(1, i, j + 1, k) *
(c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +
c1 *
(u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -
la(i, j - 1, k) * met(1, i, j - 1, k) *
met(1, i, j - 1, k) *
(c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +
c1 *
(u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));

r2 +=
c2 *
(mu(i + 2, j, k) * met(1, i + 2, j, k) *
met(1, i + 2, j, k) *
(c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +
c1 *
(u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -
mu(i - 2, j, k) * met(1, i - 2, j, k) *
met(1, i - 2, j, k) *
(c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +
c1 * (u(1, i - 2, j + 1, k) -
u(1, i - 2, j - 1, k)))) +
c1 *
(mu(i + 1, j, k) * met(1, i + 1, j, k) *
met(1, i + 1, j, k) *
(c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +
c1 *
(u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -
mu(i - 1, j, k) * met(1, i - 1, j, k) *
met(1, i - 1, j, k) *
(c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +
c1 *
(u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));

float_sw4 dudrm2 = 0, dudrm1 = 0, dudrp1 = 0, dudrp2 = 0;
float_sw4 dvdrm2 = 0, dvdrm1 = 0, dvdrp1 = 0, dvdrp2 = 0;
float_sw4 dwdrm2 = 0, dwdrm1 = 0, dwdrp1 = 0, dwdrp2 = 0;
for (int q = 1; q <= 8; q++) {
dudrm2 += bope(k, q) * u(1, i - 2, j, q);
dvdrm2 += bope(k, q) * u(2, i - 2, j, q);
dwdrm2 += bope(k, q) * u(3, i - 2, j, q);
dudrm1 += bope(k, q) * u(1, i - 1, j, q);
dvdrm1 += bope(k, q) * u(2, i - 1, j, q);
dwdrm1 += bope(k, q) * u(3, i - 1, j, q);
dudrp2 += bope(k, q) * u(1, i + 2, j, q);
dvdrp2 += bope(k, q) * u(2, i + 2, j, q);
dwdrp2 += bope(k, q) * u(3, i + 2, j, q);
dudrp1 += bope(k, q) * u(1, i + 1, j, q);
dvdrp1 += bope(k, q) * u(2, i + 1, j, q);
dwdrp1 += bope(k, q) * u(3, i + 1, j, q);
}

r1 += (c2 * ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
met(2, i + 2, j, k) * met(1, i + 2, j, k) *
strx(i + 2) * dudrp2 +
la(i + 2, j, k) * met(3, i + 2, j, k) *
met(1, i + 2, j, k) * dvdrp2 * stry(j) +
la(i + 2, j, k) * met(4, i + 2, j, k) *
met(1, i + 2, j, k) * dwdrp2 -
((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
met(2, i - 2, j, k) * met(1, i - 2, j, k) *
strx(i - 2) * dudrm2 +
la(i - 2, j, k) * met(3, i - 2, j, k) *
met(1, i - 2, j, k) * dvdrm2 * stry(j) +
la(i - 2, j, k) * met(4, i - 2, j, k) *
met(1, i - 2, j, k) * dwdrm2)) +
c1 * ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
met(2, i + 1, j, k) * met(1, i + 1, j, k) *
strx(i + 1) * dudrp1 +
la(i + 1, j, k) * met(3, i + 1, j, k) *
met(1, i + 1, j, k) * dvdrp1 * stry(j) +
la(i + 1, j, k) * met(4, i + 1, j, k) *
met(1, i + 1, j, k) * dwdrp1 -
((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
met(2, i - 1, j, k) * met(1, i - 1, j, k) *
strx(i - 1) * dudrm1 +
la(i - 1, j, k) * met(3, i - 1, j, k) *
met(1, i - 1, j, k) * dvdrm1 * stry(j) +
la(i - 1, j, k) * met(4, i - 1, j, k) *
met(1, i - 1, j, k) * dwdrm1))) *
istry;

r2 +=
c2 *
(mu(i + 2, j, k) * met(3, i + 2, j, k) *
met(1, i + 2, j, k) * dudrp2 +
mu(i + 2, j, k) * met(2, i + 2, j, k) *
met(1, i + 2, j, k) * dvdrp2 * strx(i + 2) * istry -
(mu(i - 2, j, k) * met(3, i - 2, j, k) *
met(1, i - 2, j, k) * dudrm2 +
mu(i - 2, j, k) * met(2, i - 2, j, k) *
met(1, i - 2, j, k) * dvdrm2 * strx(i - 2) * istry)) +
c1 * (mu(i + 1, j, k) * met(3, i + 1, j, k) *
met(1, i + 1, j, k) * dudrp1 +
mu(i + 1, j, k) * met(2, i + 1, j, k) *
met(1, i + 1, j, k) * dvdrp1 * strx(i + 1) * istry -
(mu(i - 1, j, k) * met(3, i - 1, j, k) *
met(1, i - 1, j, k) * dudrm1 +
mu(i - 1, j, k) * met(2, i - 1, j, k) *
met(1, i - 1, j, k) * dvdrm1 * strx(i - 1) * istry));

r3 += istry *
(c2 * (mu(i + 2, j, k) * met(4, i + 2, j, k) *
met(1, i + 2, j, k) * dudrp2 +
mu(i + 2, j, k) * met(2, i + 2, j, k) *
met(1, i + 2, j, k) * dwdrp2 * strx(i + 2) -
(mu(i - 2, j, k) * met(4, i - 2, j, k) *
met(1, i - 2, j, k) * dudrm2 +
mu(i - 2, j, k) * met(2, i - 2, j, k) *
met(1, i - 2, j, k) * dwdrm2 * strx(i - 2))) +
c1 * (mu(i + 1, j, k) * met(4, i + 1, j, k) *
met(1, i + 1, j, k) * dudrp1 +
mu(i + 1, j, k) * met(2, i + 1, j, k) *
met(1, i + 1, j, k) * dwdrp1 * strx(i + 1) -
(mu(i - 1, j, k) * met(4, i - 1, j, k) *
met(1, i - 1, j, k) * dudrm1 +
mu(i - 1, j, k) * met(2, i - 1, j, k) *
met(1, i - 1, j, k) * dwdrm1 * strx(i - 1))));


dudrm2 = 0;
dudrm1 = 0;
dudrp1 = 0;
dudrp2 = 0;
dvdrm2 = 0;
dvdrm1 = 0;
dvdrp1 = 0;
dvdrp2 = 0;
dwdrm2 = 0;
dwdrm1 = 0;
dwdrp1 = 0;
dwdrp2 = 0;
for (int q = 1; q <= 8; q++) {
dudrm2 += bope(k, q) * u(1, i, j - 2, q);
dvdrm2 += bope(k, q) * u(2, i, j - 2, q);
dwdrm2 += bope(k, q) * u(3, i, j - 2, q);
dudrm1 += bope(k, q) * u(1, i, j - 1, q);
dvdrm1 += bope(k, q) * u(2, i, j - 1, q);
dwdrm1 += bope(k, q) * u(3, i, j - 1, q);
dudrp2 += bope(k, q) * u(1, i, j + 2, q);
dvdrp2 += bope(k, q) * u(2, i, j + 2, q);
dwdrp2 += bope(k, q) * u(3, i, j + 2, q);
dudrp1 += bope(k, q) * u(1, i, j + 1, q);
dvdrp1 += bope(k, q) * u(2, i, j + 1, q);
dwdrp1 += bope(k, q) * u(3, i, j + 1, q);
}

r1 +=
c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
met(1, i, j + 2, k) * dudrp2 * stry(j + 2) * istrx +
mu(i, j + 2, k) * met(2, i, j + 2, k) *
met(1, i, j + 2, k) * dvdrp2 -
(mu(i, j - 2, k) * met(3, i, j - 2, k) *
met(1, i, j - 2, k) * dudrm2 * stry(j - 2) * istrx +
mu(i, j - 2, k) * met(2, i, j - 2, k) *
met(1, i, j - 2, k) * dvdrm2)) +
c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
met(1, i, j + 1, k) * dudrp1 * stry(j + 1) * istrx +
mu(i, j + 1, k) * met(2, i, j + 1, k) *
met(1, i, j + 1, k) * dvdrp1 -
(mu(i, j - 1, k) * met(3, i, j - 1, k) *
met(1, i, j - 1, k) * dudrm1 * stry(j - 1) * istrx +
mu(i, j - 1, k) * met(2, i, j - 1, k) *
met(1, i, j - 1, k) * dvdrm1));

r2 += c2 * (la(i, j + 2, k) * met(2, i, j + 2, k) *
met(1, i, j + 2, k) * dudrp2 +
(2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
met(3, i, j + 2, k) * met(1, i, j + 2, k) * dvdrp2 *
stry(j + 2) * istrx +
la(i, j + 2, k) * met(4, i, j + 2, k) *
met(1, i, j + 2, k) * dwdrp2 * istrx -
(la(i, j - 2, k) * met(2, i, j - 2, k) *
met(1, i, j - 2, k) * dudrm2 +
(2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
met(3, i, j - 2, k) * met(1, i, j - 2, k) *
dvdrm2 * stry(j - 2) * istrx +
la(i, j - 2, k) * met(4, i, j - 2, k) *
met(1, i, j - 2, k) * dwdrm2 * istrx)) +
c1 * (la(i, j + 1, k) * met(2, i, j + 1, k) *
met(1, i, j + 1, k) * dudrp1 +
(2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
met(3, i, j + 1, k) * met(1, i, j + 1, k) * dvdrp1 *
stry(j + 1) * istrx +
la(i, j + 1, k) * met(4, i, j + 1, k) *
met(1, i, j + 1, k) * dwdrp1 * istrx -
(la(i, j - 1, k) * met(2, i, j - 1, k) *
met(1, i, j - 1, k) * dudrm1 +
(2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
met(3, i, j - 1, k) * met(1, i, j - 1, k) *
dvdrm1 * stry(j - 1) * istrx +
la(i, j - 1, k) * met(4, i, j - 1, k) *
met(1, i, j - 1, k) * dwdrm1 * istrx));

r3 += (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
met(1, i, j + 2, k) * dwdrp2 * stry(j + 2) +
mu(i, j + 2, k) * met(4, i, j + 2, k) *
met(1, i, j + 2, k) * dvdrp2 -
(mu(i, j - 2, k) * met(3, i, j - 2, k) *
met(1, i, j - 2, k) * dwdrm2 * stry(j - 2) +
mu(i, j - 2, k) * met(4, i, j - 2, k) *
met(1, i, j - 2, k) * dvdrm2)) +
c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
met(1, i, j + 1, k) * dwdrp1 * stry(j + 1) +
mu(i, j + 1, k) * met(4, i, j + 1, k) *
met(1, i, j + 1, k) * dvdrp1 -
(mu(i, j - 1, k) * met(3, i, j - 1, k) *
met(1, i, j - 1, k) * dwdrm1 * stry(j - 1) +
mu(i, j - 1, k) * met(4, i, j - 1, k) *
met(1, i, j - 1, k) * dvdrm1))) *
istrx;

for (int q = 1; q <= 8; q++) {
r1 += bope(k, q) *
(
(2 * mu(i, j, q) + la(i, j, q)) * met(2, i, j, q) *
met(1, i, j, q) *
(c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
strx(i) * istry +
mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
(c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) +
mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
(c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
istry
+ mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
(c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) *
stry(j) * istrx +
la(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
(c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))));

r2 += bope(k, q) *
(
la(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
(c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) +
mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
(c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) *
strx(i) * istry
+ mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
(c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) +
(2 * mu(i, j, q) + la(i, j, q)) * met(3, i, j, q) *
met(1, i, j, q) *
(c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
stry(j) * istrx +
mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
(c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
istrx);

r3 += bope(k, q) *
(
la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
(c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
istry +
mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
(c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
strx(i) * istry
+ mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
(c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
stry(j) * istrx +
la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
(c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
istrx);
}

lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;
lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;
}
}

