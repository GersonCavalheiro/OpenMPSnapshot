#include "npbparams.h"
#define	AA		0
#define BB		1
#define CC		2
#define	BLOCK_SIZE	5
static int grid_points[3];	
static double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3;
static double dx1, dx2, dx3, dx4, dx5;
static double dy1, dy2, dy3, dy4, dy5;
static double dz1, dz2, dz3, dz4, dz5;
static double dssp, dt;
static double ce[5][13];	
static double dxmax, dymax, dzmax;
static double xxcon1, xxcon2, xxcon3, xxcon4, xxcon5;
static double dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1;
static double yycon1, yycon2, yycon3, yycon4, yycon5;
static double dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1;
static double zzcon1, zzcon2, zzcon3, zzcon4, zzcon5;
static double dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1;
static double dnxm1, dnym1, dnzm1, c1c2, c1c5, c3c4, c1345;
static double conz1, c1, c2, c3, c4, c5, c4dssp, c5dssp, dtdssp;
static double dttx1, dttx2, dtty1, dtty2, dttz1, dttz2;
static double c2dttx1, c2dtty1, c2dttz1, comz1, comz4, comz5, comz6;
static double c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;
#define	IMAX	PROBLEM_SIZE
#define	JMAX	PROBLEM_SIZE
#define	KMAX	PROBLEM_SIZE
static double us[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1];
static double vs[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1];
static double ws[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1];
static double qs[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1];
static double rho_i[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1];
static double square[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1];
static double forcing[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1][5+1];
static double u[(IMAX+1)/2*2+1][(JMAX+1)/2*2+1][(KMAX+1)/2*2+1][5];
static double rhs[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1][5];
static double lhs[IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1][3][5][5];
static double cuf[PROBLEM_SIZE];
static double q[PROBLEM_SIZE];
static double ue[PROBLEM_SIZE][5];
static double buf[PROBLEM_SIZE][5];
#pragma omp threadprivate(cuf, q, ue, buf)
static double fjac[IMAX/2*2+1][JMAX/2*2+1][KMAX-1+1][5][5];
static double njac[IMAX/2*2+1][JMAX/2*2+1][KMAX-1+1][5][5];
static double tmp1, tmp2, tmp3;
