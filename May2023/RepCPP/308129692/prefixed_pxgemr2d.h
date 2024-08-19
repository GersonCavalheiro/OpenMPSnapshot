#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void costa_psgemr2d(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pdgemr2d(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pcgemr2d(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pzgemr2d(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pigemr2d(const int *m, const int *n,
const int *a,
const int *ia, const int *ja,
const int *desca,
int *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_psgemr2d_(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pdgemr2d_(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pcgemr2d_(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pzgemr2d_(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pigemr2d_(const int *m, const int *n,
const int *a,
const int *ia, const int *ja,
const int *desca,
int *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_psgemr2d__(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pdgemr2d__(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pcgemr2d__(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pzgemr2d__(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void costa_pigemr2d__(const int *m, const int *n,
const int *a,
const int *ia, const int *ja,
const int *desca,
int *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void COSTA_PSGEMR2D(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void COSTA_PDGEMR2D(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void COSTA_PCGEMR2D(const int *m, const int *n,
const float *a,
const int *ia, const int *ja,
const int *desca,
float *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void COSTA_PZGEMR2D(const int *m, const int *n,
const double *a,
const int *ia, const int *ja,
const int *desca,
double *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

void COSTA_PIGEMR2D(const int *m, const int *n,
const int *a,
const int *ia, const int *ja,
const int *desca,
int *b,
const int *ib, const int *jb,
const int *descb,
const int *ictxt);

#ifdef __cplusplus
}
#endif
