#include <cassert>
#include <memory>
#ifdef __INTEL_COMPILER
#include <cstdlib>
#include <iostream>
#endif
#include <omp.h>
#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"
static inline int getNumberOfDevices() { return omp_get_num_devices(); }
class CGMultiOpenMPTarget : public CG {
std::unique_ptr<floatType[]> p;
std::unique_ptr<floatType[]> q;
std::unique_ptr<floatType[]> r;
std::unique_ptr<floatType[]> z;
std::unique_ptr<floatType[]> vectorDotResults;
floatType *getVector(Vector v) {
switch (v) {
case VectorK:
return k;
case VectorX:
return x;
case VectorP:
return p.get();
case VectorQ:
return q.get();
case VectorR:
return r.get();
case VectorZ:
return z.get();
}
assert(0 && "Invalid value of v!");
return nullptr;
}
virtual bool supportsMatrixFormat(MatrixFormat format) override {
return format == MatrixFormatCRS || format == MatrixFormatELL;
}
virtual bool supportsPreconditioner(Preconditioner preconditioner) override {
return preconditioner == PreconditionerJacobi;
}
virtual int getNumberOfChunks() override { return getNumberOfDevices(); }
virtual bool supportsOverlappedGather() override { return true; }
virtual void init(const char *matrixFile) override;
virtual bool needsTransfer() override { return true; }
virtual void doTransferTo() override;
virtual void doTransferFrom() override;
virtual void cpy(Vector _dst, Vector _src) override;
void matvecGatherXViaHost(floatType *x);
template <bool roundup = false>
void matvecKernelCRS(MatrixDataCRS *matrices, floatType *x, floatType *y);
template <bool roundup = false>
void matvecKernelELL(MatrixDataELL *matrices, floatType *x, floatType *y);
virtual void matvecKernel(Vector _x, Vector _y) override;
virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
virtual floatType vectorDotKernel(Vector _a, Vector _b) override;
void applyPreconditionerKernelJacobi(floatType *x, floatType *y);
virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;
public:
CGMultiOpenMPTarget()
: CG(MatrixFormatCRS, PreconditionerJacobi,
true) {}
};
void CGMultiOpenMPTarget::init(const char *matrixFile) {
int devices = getNumberOfDevices();
#ifdef __INTEL_COMPILER
if (devices != 2) {
std::cerr << "Only supports exactly two devices!" << std::endl;
std::exit(1);
}
#endif
for (int d = 0; d < getNumberOfDevices(); d++) {
#pragma omp target device(d)
{ }
}
CG::init(matrixFile);
assert(workDistribution->numberOfChunks == devices);
p.reset(new floatType[N]);
q.reset(new floatType[N]);
r.reset(new floatType[N]);
if (preconditioner != PreconditionerNone) {
z.reset(new floatType[N]);
}
vectorDotResults.reset(new floatType[devices]);
}
static inline void enterMatrixCRS(const MatrixDataCRS &matrix, int N) {
int *ptr = matrix.ptr;
int nz = ptr[N];
int *index = matrix.index;
floatType *value = matrix.value;
#pragma omp target enter data nowait map(to: ptr[0:N+1], index[0:nz]) map(to: value[0:nz])
}
static inline void enterMatrixELL(const MatrixDataELL &matrix, int N) {
int elements = matrix.elements;
int *length = matrix.length;
int *index = matrix.index;
floatType *data = matrix.data;
#pragma omp target enter data nowait map(to: length[0:N], index[0:elements]) map(to: data[0:elements])
}
void CGMultiOpenMPTarget::doTransferTo() {
int N = this->N;
floatType *p = this->p.get();
floatType *q = this->q.get();
floatType *r = this->r.get();
floatType *x = this->x;
floatType *k = this->k;
floatType *vectorDotResults = this->vectorDotResults.get();
for (int d = 0; d < getNumberOfDevices(); d++) {
omp_set_default_device(d);
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target enter data nowait map(alloc: p[0:N], q[offset:length]) map(alloc: r[offset:length]) map(to: x[0:N], k[offset:length])
switch (matrixFormat) {
case MatrixFormatCRS:
if (!overlappedGather) {
enterMatrixCRS(splitMatrixCRS->data[d], length);
} else {
enterMatrixCRS(partitionedMatrixCRS->diag[d], length);
enterMatrixCRS(partitionedMatrixCRS->minor[d], length);
}
break;
case MatrixFormatELL:
if (!overlappedGather) {
enterMatrixELL(splitMatrixELL->data[d], length);
} else {
enterMatrixELL(partitionedMatrixELL->diag[d], length);
enterMatrixELL(partitionedMatrixELL->minor[d], length);
}
break;
default:
assert(0 && "Invalid matrix format!");
}
if (preconditioner != PreconditionerNone) {
floatType *z = this->z.get();
#pragma omp target enter data nowait map(alloc: z[offset:length])
switch (preconditioner) {
case PreconditionerJacobi: {
floatType *C = jacobi->C;
#pragma omp target enter data nowait map(to: C[offset:length])
break;
}
default:
assert(0 && "Invalid preconditioner!");
}
}
#pragma omp target enter data nowait map(alloc: vectorDotResults[d:1])
}
#pragma omp taskwait
}
static inline void exitMatrixCRS(const MatrixDataCRS &matrix, int N) {
int *ptr = matrix.ptr;
int nz = ptr[N];
int *index = matrix.index;
floatType *value = matrix.value;
#pragma omp target exit data nowait map(release: ptr[0:N+1], index[0:nz]) map(release: value[0:nz])
}
static inline void exitMatrixELL(const MatrixDataELL &matrix, int N) {
int elements = matrix.elements;
int *length = matrix.length;
int *index = matrix.index;
floatType *data = matrix.data;
#pragma omp target exit data nowait map(release: length[0:N]) map(release: index[0:elements]) map(release: data[0:elements])
}
void CGMultiOpenMPTarget::doTransferFrom() {
int N = this->N;
floatType *p = this->p.get();
floatType *q = this->q.get();
floatType *r = this->r.get();
floatType *x = this->x;
floatType *k = this->k;
floatType *vectorDotResults = this->vectorDotResults.get();
for (int d = 0; d < getNumberOfDevices(); d++) {
omp_set_default_device(d);
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target update nowait from(x[offset:length]) depend(out: x[offset:length])
#ifdef __INTEL_COMPILER
#pragma omp taskwait
#endif
#pragma omp target exit data nowait map(release: x[0:N]) depend(in: x[offset:length])
#pragma omp target exit data nowait map(release: p[0:N], q[offset:length]) map(release: r[offset:length]) map(release: k[offset:length])
switch (matrixFormat) {
case MatrixFormatCRS:
if (!overlappedGather) {
exitMatrixCRS(splitMatrixCRS->data[d], length);
} else {
exitMatrixCRS(partitionedMatrixCRS->diag[d], length);
exitMatrixCRS(partitionedMatrixCRS->minor[d], length);
}
break;
case MatrixFormatELL:
if (!overlappedGather) {
exitMatrixELL(splitMatrixELL->data[d], length);
} else {
exitMatrixELL(partitionedMatrixELL->diag[d], length);
exitMatrixELL(partitionedMatrixELL->minor[d], length);
}
break;
default:
assert(0 && "Invalid matrix format!");
}
if (preconditioner != PreconditionerNone) {
floatType *z = this->z.get();
#pragma omp target exit data nowait map(release: z[offset:length])
switch (preconditioner) {
case PreconditionerJacobi: {
floatType *C = jacobi->C;
#pragma omp target exit data nowait map(release: C[offset:length])
break;
}
default:
assert(0 && "Invalid preconditioner!");
}
}
#pragma omp target exit data nowait map(release: vectorDotResults[d:1])
}
#pragma omp taskwait
}
void CGMultiOpenMPTarget::cpy(Vector _dst, Vector _src) {
floatType *dst = getVector(_dst);
floatType *src = getVector(_src);
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target nowait device(d) map(dst[offset:length], src[offset:length])
#pragma omp teams distribute parallel for simd
for (int i = offset; i < offset + length; i++) {
dst[i] = src[i];
}
}
#pragma omp taskwait
}
void CGMultiOpenMPTarget::matvecGatherXViaHost(floatType *x) {
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target update nowait device(d) from(x[offset:length]) depend(out: x[offset:length])
}
#ifdef __INTEL_COMPILER
int d0 = 0;
int src0 = 1;
int offset0 = workDistribution->offsets[src0];
int length0 = workDistribution->lengths[src0];
#pragma omp target update nowait device(d0) to(x[offset0:length0]) depend(in: x[offset0:length0])
int d1 = 1;
int src1 = 0;
int offset1 = workDistribution->offsets[src1];
int length1 = workDistribution->lengths[src1];
#pragma omp target update nowait device(d1) to(x[offset1:length1]) depend(in: x[offset1:length1])
#else
for (int d = 0; d < getNumberOfDevices(); d++) {
for (int src = 0; src < getNumberOfDevices(); src++) {
if (src == d) {
continue;
}
int offset = workDistribution->offsets[src];
int length = workDistribution->lengths[src];
#pragma omp target update nowait device(d) to(x[offset:length]) depend(in: x[offset:length])
}
}
#endif
#pragma omp taskwait
}
template <bool roundup>
void CGMultiOpenMPTarget::matvecKernelCRS(MatrixDataCRS *matrices, floatType *x,
floatType *y) {
int N = this->N;
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
int *ptr = matrices[d].ptr;
int nz = ptr[length];
int *index = matrices[d].index;
floatType *value = matrices[d].value;
#pragma omp target nowait device(d) map(x[0:N], y[offset:length]) map(ptr[0:length+1], index[0:nz], value[0:nz])
#pragma omp teams distribute parallel for
for (int i = 0; i < length; i++) {
if (!roundup || ptr[i] != ptr[i + 1]) {
floatType tmp = (roundup ? y[offset + i] : 0);
#pragma omp simd reduction(+:tmp)
for (int j = ptr[i]; j < ptr[i + 1]; j++) {
tmp += value[j] * x[index[j]];
}
y[offset + i] = tmp;
}
}
}
if (!overlappedGather) {
#pragma omp taskwait
}
}
template <bool roundup>
void CGMultiOpenMPTarget::matvecKernelELL(MatrixDataELL *matrices, floatType *x,
floatType *y) {
int N = this->N;
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
int elements = matrices[d].elements;
int *lengthA = matrices[d].length;
int *index = matrices[d].index;
floatType *data = matrices[d].data;
#pragma omp target nowait device(d) map(x[0:N], y[offset:length]) map(lengthA[0:length], index[0:elements], data[0:elements])
#pragma omp teams distribute parallel for simd
for (int i = 0; i < length; i++) {
if (!roundup || lengthA[i] > 0) {
floatType tmp = (roundup ? y[offset + i] : 0);
#pragma unroll 1
for (int j = 0; j < lengthA[i]; j++) {
int k = j * length + i;
tmp += data[k] * x[index[k]];
}
y[offset + i] = tmp;
}
}
}
if (!overlappedGather) {
#pragma omp taskwait
}
}
void CGMultiOpenMPTarget::matvecKernel(Vector _x, Vector _y) {
floatType *x = getVector(_x);
floatType *y = getVector(_y);
if (overlappedGather) {
switch (matrixFormat) {
case MatrixFormatCRS:
matvecKernelCRS(partitionedMatrixCRS->diag.get(), x, y);
break;
case MatrixFormatELL:
matvecKernelELL(partitionedMatrixELL->diag.get(), x, y);
break;
default:
assert(0 && "Invalid matrix format!");
}
}
matvecGatherXViaHost(x);
switch (matrixFormat) {
case MatrixFormatCRS:
if (!overlappedGather) {
matvecKernelCRS(splitMatrixCRS->data.get(), x, y);
} else {
matvecKernelCRS<true>(partitionedMatrixCRS->minor.get(), x, y);
}
break;
case MatrixFormatELL:
if (!overlappedGather) {
matvecKernelELL(splitMatrixELL->data.get(), x, y);
} else {
matvecKernelELL<true>(partitionedMatrixELL->minor.get(), x, y);
}
break;
default:
assert(0 && "Invalid matrix format!");
}
if (overlappedGather) {
#pragma omp taskwait
}
}
void CGMultiOpenMPTarget::axpyKernel(floatType a, Vector _x, Vector _y) {
floatType *x = getVector(_x);
floatType *y = getVector(_y);
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target nowait device(d) map(x[offset:length], y[offset:length])
#pragma omp teams distribute parallel for simd
for (int i = offset; i < offset + length; i++) {
y[i] += a * x[i];
}
}
#pragma omp taskwait
}
void CGMultiOpenMPTarget::xpayKernel(Vector _x, floatType a, Vector _y) {
floatType *x = getVector(_x);
floatType *y = getVector(_y);
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target nowait device(d) map(x[offset:length], y[offset:length])
#pragma omp teams distribute parallel for simd
for (int i = offset; i < offset + length; i++) {
y[i] = x[i] + a * y[i];
}
}
#pragma omp taskwait
}
floatType CGMultiOpenMPTarget::vectorDotKernel(Vector _a, Vector _b) {
floatType res = 0;
floatType *a = getVector(_a);
floatType *b = getVector(_b);
floatType *vectorDotResults = this->vectorDotResults.get();
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#ifndef __INTEL_COMPILER
vectorDotResults[d] = 0;
#pragma omp target update device(d) to(vectorDotResults[d:1])
#endif
#pragma omp target nowait device(d) map(a[offset:length], b[offset:length]) map(vectorDotResults[d:1])
#ifndef __INTEL_COMPILER
#pragma omp teams distribute parallel for simd reduction(+:vectorDotResults[d:1])
for (int i = offset; i < offset + length; i++) {
vectorDotResults[d] += a[i] * b[i];
}
#else
{
floatType red = 0;
#pragma omp parallel for reduction(+:red)
for (int i = offset; i < offset + length; i++) {
red += a[i] * b[i];
}
vectorDotResults[d] = red;
}
#endif
}
#pragma omp taskwait
for (int d = 0; d < getNumberOfDevices(); d++) {
#pragma omp target update device(d) from(vectorDotResults[d:1])
res += vectorDotResults[d];
}
return res;
}
void CGMultiOpenMPTarget::applyPreconditionerKernelJacobi(floatType *x,
floatType *y) {
floatType *C = jacobi->C;
for (int d = 0; d < getNumberOfDevices(); d++) {
int offset = workDistribution->offsets[d];
int length = workDistribution->lengths[d];
#pragma omp target nowait device(d) map(x[offset:length], y[offset:length], C[offset:length])
#pragma omp teams distribute parallel for simd
for (int i = offset; i < offset + length; i++) {
y[i] = C[i] * x[i];
}
}
#pragma omp taskwait
}
void CGMultiOpenMPTarget::applyPreconditionerKernel(Vector _x, Vector _y) {
floatType *x = getVector(_x);
floatType *y = getVector(_y);
switch (preconditioner) {
case PreconditionerJacobi:
applyPreconditionerKernelJacobi(x, y);
break;
default:
assert(0 && "Invalid preconditioner!");
}
}
CG *CG::getInstance() { return new CGMultiOpenMPTarget; }
