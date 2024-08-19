#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<sys/param.h> 
#include "arrays.h"
#include "readcsv.h"
#include "hbv.h"

int main(int argc, char *argv[]){
int b, n, t;
int ntot 	= 45000;
int ttot 	= 916;	
int dtot 	= 1910;	
int ytot 	= 5;	
int btot 	= 8;	
int rtot 	= 8;	
int th 		= 3;	
int tl	 	= 15;	
int TY 		= 365;	
int TS 		= 86400;
int tstart 	= 274;	


float tcon 	= 86.4;
float ecevpfo	= 1.15;
float ecalt 	= 0.1;
float pcalt 	= 0.2;
float tcalt 	= 0.6;
float sfcf 	= 1.0;
float foscf 	= 1.0;
float rfcf 	= 1.0;
float tt 	= -0.5;
float tti 	= 2.0;
float dttm 	= 0.54391;
float cfmax 	= 3.5;
float focfmax 	= 0.6;
float cfr 	= 0.05;
float whc 	= 0.1;
float sgwmax 	= 5.0;

float **ssm	= af2d(btot,ttot+1);
float **ssw	= af2d(btot,ttot+1);
float **sgw	= af2d(btot,ttot+1);
float **ssp	= af2d(btot,ttot+1);
float **smw	= af2d(btot,ttot+1);
float **sgwx	= af2d(btot,ttot+1);
float **p	= af2d(btot,ttot);
float **ta	= af2d(btot,ttot);
float **s	= af2d(btot,ttot);
float **r	= af2d(btot,ttot);
float **sm	= af2d(btot,ttot);
float **sr	= af2d(btot,ttot);
float **inp	= af2d(btot,ttot);
float **etp	= af2d(btot,ttot);
float **qd	= af2d(btot,ttot);
float **qin	= af2d(btot,ttot);
float **qs	= af2d(btot,ttot);
float **qf	= af2d(btot,ttot);
float **qt	= af2d(btot,ttot);
float **qc	= af2d(btot,ttot);
float **eta	= af2d(btot,ttot);

float **fc	= af2d(btot,ntot);
float **beta	= af2d(btot,ntot);
float **lp	= af2d(btot,ntot);
float **alpha	= af2d(btot,ntot);
float **kf	= af2d(btot,ntot);
float **ks	= af2d(btot,ntot);
float **perc	= af2d(btot,ntot);
float **cflux	= af2d(btot,ntot);


float **qr	= af2d(btot,ttot);



float rqh 	= 5.; 
float rql 	= 0.5;
float QDB 	= 100.;
float KL 	= 1.30456; 
float KU 	= 3.13668; 

float *m	= af1d(rtot);	
float *mtot	= af1d(rtot);	
float *qotot	= af1d(rtot);	
float *yotot	= af1d(rtot);	
float *etotot	= af1d(rtot);	

float **NS	= af2d(rtot,ntot);
float **NSET	= af2d(rtot,ntot);
float **NSH	= af2d(rtot,ntot);
float **NSL	= af2d(rtot,ntot);
float **RVE	= af2d(rtot,ntot);
float **RVEET	= af2d(rtot,ntot);
float **RMAEH	= af2d(rtot,ntot);
float **RMAEL	= af2d(rtot,ntot);
float **REVE	= af2d(rtot,ntot);
float *dqd	= af1d(ntot);

float **MNS	= af2d(rtot,ntot);
float **MNSET	= af2d(rtot,ntot);
float **MNSH	= af2d(rtot,ntot);
float **MNSL	= af2d(rtot,ntot);
float **MRVE	= af2d(rtot,ntot);
float **MRVEET	= af2d(rtot,ntot);
float **MRMAEH	= af2d(rtot,ntot);
float **MRMAEL	= af2d(rtot,ntot);
float **MREVE	= af2d(rtot,ntot);
float *Mdqd	= af1d(ntot);




float *difq	= af1d(rtot);
float *dq	= af1d(rtot);
float *dqh	= af1d(rtot);
float *dql	= af1d(rtot);
float *mqo	= af1d(rtot);
float *meto	= af1d(rtot);
float *difet	= af1d(rtot);
float *deta	= af1d(rtot);
float *veto	= af1d(rtot);
float *mqoh	= af1d(rtot);
float *mqol	= af1d(rtot);
float *vqo	= af1d(rtot);
float *vqoh	= af1d(rtot);
float *vqol	= af1d(rtot);
float **qrh	= af2d(rtot,ttot);
float **qoh	= af2d(rtot,ttot);
float **qrl	= af2d(rtot,ttot);
float **qol	= af2d(rtot,ttot);
float *dqha	= af1d(rtot);
float *dqla	= af1d(rtot);
float *qdo	= af1d(ytot);
float *qdr	= af1d(ytot);
float *qodef	= af1d(ntot);
float *qrdef	= af1d(ntot);
float **mfc	= af2d(rtot,ntot);
float **mbeta	= af2d(rtot,ntot);
float **mlp	= af2d(rtot,ntot);
float **malpha	= af2d(rtot,ntot);
float **mkf	= af2d(rtot,ntot);
float **mks	= af2d(rtot,ntot);
float **mperc	= af2d(rtot,ntot);
float **mcflux	= af2d(rtot,ntot);
float *tyear	= af1d(ttot);
float **qom	= af2d(rtot,ytot);
float **qrm	= af2d(rtot,ytot);
float *mqom	= af1d(rtot);
float *mqrm	= af1d(rtot);
float *sqom	= af1d(rtot);
float *sqmr	= af1d(rtot);




float **po	= af2d(btot,dtot);
float **etpo	= af2d(btot,dtot);
float **qo	= af2d(rtot,dtot);
float **tm	= af2d(btot,dtot);
float **eto	= af2d(btot,dtot);
float **f7p	= af2d(btot,22);

po 	= readcsv("precip.csv",btot,dtot);	
etpo 	= readcsv("evap.csv",btot,dtot);	
qo 	= readcsv("dischargeobs.csv",rtot,dtot);
tm 	= readcsv("temp.csv",btot,dtot);	
eto 	= readcsv("etobs.csv",btot,dtot);	
f7p 	= readcsv("param_sto.csv",btot,22);	

float *lfc 	= af1d(btot);
float *hfc 	= af1d(btot);
float *lbeta 	= af1d(btot);
float *hbeta 	= af1d(btot);
float *llp 	= af1d(btot);
float *hlp 	= af1d(btot);
float *lalpha 	= af1d(btot);
float *halpha	= af1d(btot);
float *lkf 	= af1d(btot);
float *hkf 	= af1d(btot);
float *lks 	= af1d(btot);
float *hks 	= af1d(btot);
float *lperc 	= af1d(btot);
float *hperc 	= af1d(btot);
float *lcflux 	= af1d(btot);
float *hcflux 	= af1d(btot);

float *area 	= af1d(btot);
float *ffo 	= af1d(btot);
float *ffi 	= af1d(btot);
float *dep 	= af1d(btot);
float *det 	= af1d(btot);
float *dee 	= af1d(btot);

for (b=0;b<btot;b++){
lfc[b]		= f7p[b][0];
hfc[b]		= f7p[b][1];
lbeta[b]	= f7p[b][2];
hbeta[b]	= f7p[b][3];
llp[b]		= f7p[b][4];
hlp[b]		= f7p[b][5];
lalpha[b]	= f7p[b][6];
halpha[b]	= f7p[b][7];
lkf[b]		= f7p[b][8];
hkf[b]		= f7p[b][9];
lks[b]		= f7p[b][10];
hks[b]		= f7p[b][11];
lperc[b]	= f7p[b][12];
hperc[b]	= f7p[b][13];
lcflux[b]	= f7p[b][14];
hcflux[b]	= f7p[b][15];
area[b]		= f7p[b][16];
ffo[b]		= f7p[b][17];
ffi[b]		= f7p[b][18];
dep[b]		= f7p[b][19];
det[b]		= f7p[b][20];
dee[b]		= f7p[b][21];
}



for (n=0;n<ntot;n++){
printf("Random Realization Loop = %i/%i\r",n,ntot);
for (b=0;b<btot;b++){
fc[b][n]	= lfc[b]+(hfc[b]-lfc[b])*(rand()%1000)/1000;
beta[b][n]	= lbeta[b]+(hbeta[b]-lbeta[b])*(rand()%1000)/1000;
lp[b][n]	= llp[b]+(hlp[b]-llp[b])*(rand()%1000)/1000;
alpha[b][n]	= lalpha[b]+(halpha[b]-lalpha[b])*(rand()%1000)/1000;
kf[b][n]	= lkf[b]+(hkf[b]-lkf[b])*(rand()%1000)/1000;
ks[b][n]	= lks[b]+(hks[b]-lks[b])*(rand()%1000)/1000;
perc[b][n]	= lperc[b]+(hperc[b]-lperc[b])*(rand()%1000)/1000;
cflux[b][n]	= lcflux[b]+(hcflux[b]-lcflux[b])*(rand()%1000)/1000;
ssp[b][0]	= 0.0;
smw[b][0]	= 0.0;
ssm[b][0]	= MIN(15.0,fc[b][n]);
ssw[b][0]	= 15.0;
sgw[b][0]	= 15.0;
ssp[b][1]	= ssp[b][0];
smw[b][1]	= smw[b][0];
ssm[b][1]	= ssm[b][0];
ssw[b][1]	= ssw[b][0];
sgw[b][1]	= sgw[b][0];
qf[b][0]	= kf[b][n] * pow(ssw[b][0],1+alpha[b][n]);
qs[b][0]	= ks[b][n] * sgw[b][0];
qt[b][0]	= (qf[b][0]+qs[b][0])*area[b]/tcon;
}


#pragma omp parallel for default(shared) private(t,b)



for (t=0;t<ttot;t++){
for (b=0;b<btot;b++){
hbv_model(n,t,b,p,po,pcalt,dep,ta,tm,tcalt,det,tt,tti,s,ffo,foscf,ffi,sfcf,r,rfcf,sm,cfmax,dttm,ssp,cfr,focfmax,smw,inp,sr,whc,qd,ssm,fc,qin,beta,etp,etpo,ecevpfo,ecalt,dee,lp,qc,cflux,qf,kf,ssw,alpha,qs,ks,sgw,qt,area,tcon,sgwx,sgwmax,perc,eta);
qr[b][t] = qt[b][t];
}
}
#pragma omp barrier
hbv_performance(n,rtot,ttot,ytot,btot,ntot,tstart,qo,mqo,qotot,eto,meto,etotot,difq,qr,dq,dqh,dql,vqo,vqoh,vqol,difet,eta,deta,veto,th,qrh,qoh,mqoh,tl,qrl,qol,rql,rqh,dqha,dqla,mqol,tyear,qom,qrm,yotot,mqom,mqrm,sqom,sqmr,NS,NSET,NSH,NSL,RVE,RVEET,RMAEH,RMAEL,REVE,KL,KU,qodef,qrdef,TY,qdo,qdr,QDB,TS,fc,mfc,mbeta,beta,mlp,lp,malpha,alpha,mkf,kf,mks,ks,mperc,perc,mcflux,cflux,MNS,MNSET,MNSH,MNSL,MRVE,MRVEET,MRMAEH,MRMAEL,MREVE,dqd,Mdqd,mtot,m);
}
hbv_report(btot,mtot,ttot,qo,qr,po,eta,eto,mfc,mbeta,mlp,malpha,mkf,mks,mperc,mcflux,MNSET,MRVEET,MNS,MNSH,MNSL,MRVE,MRMAEH,MRMAEL,MREVE);
return(EXIT_SUCCESS);
}
