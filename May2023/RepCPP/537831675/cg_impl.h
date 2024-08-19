#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"


int colidx[NZ];
int rowstr[NA + 1];
int iv[NA];
int arow[NA];
int acol[NAZ];


double aelt[NAZ];
double a[NZ];
double x[NA + 2];
double z[NA + 2];
double p[NA + 2];
double q[NA + 2];
double r[NA + 2];


int naa;
int nzz;
int firstrow;
int lastrow;
int firstcol;
int lastcol;


double amult;
double tran;


logical timeron;

void conj_grad(int colidx[],
int rowstr[],
double x[],
double z[],
double a[],
double p[],
double q[],
double r[],
double *rnorm);
void makea(int n,
int nz,
double a[],
int colidx[],
int rowstr[],
int firstrow,
int lastrow,
int firstcol,
int lastcol,
int arow[],
int acol[][NONZER + 1],
double aelt[][NONZER + 1],
int iv[]);
void sparse(double a[],
int colidx[],
int rowstr[],
int n,
int nz,
int nozer,
int arow[],
int acol[][NONZER + 1],
double aelt[][NONZER + 1],
int firstrow,
int lastrow,
int nzloc[],
double rcond,
double shift);
void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
int icnvrt(double x, int ipwr2);
void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
void init(double *zeta);
void iterate(double *zeta, int *it);