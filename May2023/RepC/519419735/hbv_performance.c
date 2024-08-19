#include <math.h>
#include <sys/param.h>
#include <omp.h>

int hbv_performance(int n,int rtot,int ttot,int ytot,int btot,int ntot,int tstart,float **qo,float *mqo,float *qotot,float **eto,float *meto,float *etotot,float *difq,float **qr,float *dq,float *dqh,float *dql,float *vqo,float *vqoh,float *vqol,float *difet,float **eta,float *deta,float *veto,int th,float **qrh,float **qoh,float *mqoh,int tl,float **qrl,float **qol,float rql,float rqh,float *dqha,float *dqla,float *mqol,float *tyear,float **qom,float **qrm,float *yotot,float *mqom,float *mqrm,float *sqom,float *sqmr,float **NS,float **NSET,float **NSH,float **NSL,float **RVE,float **RVEET,float **RMAEH,float **RMAEL,float **REVE,float KL,float KU,float *qodef,float *qrdef,int TY,float *qdo,float *qdr,float QDB,int TS,float **fc,float **mfc,float **mbeta,float **beta,float **mlp,float **lp,float **malpha,float **alpha,float **mkf,float **kf,float **mks,float **ks,float **mperc,float **perc,float **mcflux,float **cflux,float **MNS,float **MNSET,float **MNSH,float **MNSL,float **MRVE,float **MRVEET,float **MRMAEH,float **MRMAEL,float **MREVE,float *dqd,float *Mdqd,float *mtot,float *m){
int b, t, u;
#pragma omp parallel for default(shared) private (b,t,u)
for (b=0;b<rtot-1;b++){
for (t=tstart;t<ttot-1;t++){
if (qo[b][t] >= 0.){
mqo[b]	= mqo[b]+qo[b][t];
qotot[b] = qotot[b]+1;
}
if (eto[b][t] > 0.){
meto[b]	= meto[b]+eto[b][t];	
etotot[b] = etotot[b]+1;
}
}
mqo[b]	= mqo[b] / qotot[b];
meto[b]	= meto[b] / etotot[b];

for (t=tstart;t<ttot-1;t++){
if (qo[b][t] >= 0.){
difq[b]	= difq[b] + (qr[b][t]-qo[b][t]);
dq[b]	= dq[b] + pow(qo[b][t]-qr[b][t],2);
dqh[b]	= dqh[b] + qo[b][t] * pow(qo[b][t]-qr[b][t],2); 
dql[b]	= dql[b] + (1/(1+qo[b][t]))*pow(qo[b][t]-qr[b][t],2);
vqo[b]	= vqo[b] + pow(qo[b][t]-mqo[b],2);
vqoh[b]	= vqoh[b] + qo[b][t] * pow(qo[b][t]-mqo[b],2);
vqol[b]	= vqol[b] + (1/(1+qo[b][t]))*pow(qo[b][t]-mqo[b],2);
}
if (eto[b][t] > 0.){
difet[b] = difet[b] + ( eta[b][t] - eto[b][t] );
deta[b]	= deta[b] + pow(eto[b][t]-eta[b][t],2);
veto[b]	= veto[b] + pow(eto[b][t]-meto[b],2);
}
}
for (t=(tstart+th);t<(ttot-th);t++){
for (u=(t-th);u<(t+th-2);u++){
qrh[b][t] = qrh[b][t] + qr[b][u];
if (qo[b][u] >= 0.){
qoh[b][t] = qoh[b][t] + qo[b][u];
}
}
qrh[b][t] = qrh[b][t] / (2*th-1);
qoh[b][t] = qoh[b][t] / (2*th-1);
if (qoh[b][t] > (rqh*mqo[b] )){
dqha[b]	= dqha[b]+fabs(qrh[b][t]-qoh[b][t]);
mqoh[b]	= mqoh[b]+qoh[b][t];
}
}
for (t=(tstart+tl);t<(ttot-tl);t++){
for (u=(t-tl);u<(t+tl-2);u++){
qrl[b][t] = qrl[b][t] + qr[b][u];
if (qo[b][u] >= 0.){
qol[b][t] = qol[b][t] + qo[b][u];
}
}
qrl[b][t] = qrl[b][t] / (2*tl-1);
qol[b][t] = qol[b][t] / (2*tl-1);
if (qol[b][t] <= ( rql * mqo[b] ) ){
dqla[b]	= dqla[b]+fabs(qrl[b][t]-qol[b][t]);
mqol[b]	= mqol[b]+qol[b][t];
}
}
for (t=0;t<ttot-1;t++){
tyear[t] = ceilf( ( (float)t-0.75 ) /365.25 );
}
for (u=0;u<ytot-1;u++){
for (t=tstart;t<ttot-1;t++){
if (tyear[t] == u){
qom[b][u] = MAX( qom[b][u], qo[b][t] );
qrm[b][u] = MAX( qrm[b][u], qr[b][t] );
}
}
if(qom[b][u] > 0.){
yotot[b] = yotot[b] + 1;
}
mqom[b]	= mqom[b]+qom[b][u];
mqrm[b]	= mqrm[b]+qrm[b][u];
}
mqom[b]	= mqom[b] / yotot[b];
mqrm[b]	= mqrm[b] / ytot;
for (u=0;u<ytot-1;u++){
if ( qom[b][u] > 0.){
sqom[b]	= sqom[b] + pow(qom[b][u]-mqom[b],2);
}
sqmr[b]	= sqmr[b] + pow(qrm[b][u]-mqrm[b],2);
}
sqom[b]	= sqrt(sqom[b]/(MAX(yotot[b]-1,1)));
sqmr[b]	= sqrt(MAX(1.,sqmr[b])/(ytot-1));

NS[b][n]	= 1-dq[b]/vqo[b];
NSET[b][n]	= 1-deta[b]/veto[b];	
NSH[b][n]	= 1-dqh[b]/vqoh[b];
NSL[b][n]	= 1-dql[b]/vqol[b];
RVE[b][n]	= 100*difq[b]/(mqo[b]*qotot[b]);
RVEET[b][n]	= 100*difet[b]/(meto[b]*etotot[b]); 
RMAEH[b][n]	= dqha[b]/mqoh[b];
RMAEL[b][n]	= dqla[b]/mqol[b];
REVE[b][n]	= 100*(((mqrm[b]-mqom[b]+(sqmr[b]-sqom[b])*KL)/(mqom[b]+sqom[b]*KL))+((mqrm[b]-mqom[b]+(sqmr[b]-sqom[b])*KU)/(mqom[b]+sqom[b]*KU)))/2;
}
#pragma omp barrier
qodef[n]=0.;
qrdef[n]=0.;
for (t=0;t<ytot-1;t++){
float maxVal = MIN(ttot-1,TY*t);
float minVal = MIN(1+(t-1)*TY,ttot-1);
for (u=minVal;u<maxVal;u++){
if(minVal < maxVal && maxVal < ttot){ 
qdo[t]	= qdo[t] + MAX( QDB - qo[btot-1][u], 0. ) * TS;
qdr[t]	= qdr[t] + MAX( QDB - qr[btot-1][u], 0. ) * TS;
}
}
dqd[n]	= dqd[n]+(qdo[t]-qdr[t])/qdo[t];
qodef[n] = qodef[n]+qdo[t];
qrdef[n] = qrdef[n]+qdr[t];
}
dqd[n]	= 100*dqd[n]/ytot;
qodef[n] = qodef[n]/ytot;
qrdef[n] = qrdef[n]/ytot;
for (b=0;b<btot-1;b++){
mfc[b][(int)m[b]] 	= fc[b][n];
mbeta[b][(int)m[b]] 	= beta[b][n];
mlp[b][(int)m[b]] 	= lp[b][n];
malpha[b][(int)m[b]] = alpha[b][n];
mkf[b][(int)m[b]] 	= kf[b][n];
mks[b][(int)m[b]] 	= ks[b][n];
mperc[b][(int)m[b]] 	= perc[b][n];
mcflux[b][(int)m[b]] = cflux[b][n];
}
for (b=0;b<rtot-1;b++){
if (m[b] < ntot && n < ntot){
MNS[b][(int)m[b]]	= NS[b][n];
MNSH[b][(int)m[b]]	= NSH[b][n];
MNSL[b][(int)m[b]]	= NSL[b][n];
MNSET[b][(int)m[b]]	= NSET[b][n];
MRVE[b][(int)m[b]]	= RVE[b][n];
MRVEET[b][(int)m[b]]	= RVEET[b][n];
MRMAEH[b][(int)m[b]]	= RMAEH[b][n];
MRMAEL[b][(int)m[b]]	= RMAEL[b][n];
MREVE[b][(int)m[b]]	= REVE[b][n];
Mdqd[(int)m[b]]	= dqd[n];
mtot[b]		= m[b];
m[b]		= m[b] + 1;
}
}
return (1);
}