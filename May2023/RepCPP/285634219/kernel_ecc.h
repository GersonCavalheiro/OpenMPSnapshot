#pragma omp declare target
void 
kernel_ecc(const  fp timeinst,
const fp* initvalu,
fp* finavalu,
const int valu_offset,
const fp* params){


fp cycleLength;

int offset_1;
int offset_2;
int offset_3;
int offset_4;
int offset_5;
int offset_6;
int offset_7;
int offset_8;
int offset_9;
int offset_10;
int offset_11;
int offset_12;
int offset_13;
int offset_14;
int offset_15;
int offset_16;
int offset_17;
int offset_18;
int offset_19;
int offset_20;
int offset_21;
int offset_22;
int offset_23;
int offset_24;
int offset_25;
int offset_26;
int offset_27;
int offset_28;
int offset_29;
int offset_30;
int offset_31;
int offset_32;
int offset_33;
int offset_34;
int offset_35;
int offset_36;
int offset_37;
int offset_38;
int offset_39;
int offset_40;
int offset_41;
int offset_42;
int offset_43;
int offset_44;
int offset_45;
int offset_46;

fp initvalu_1;
fp initvalu_2;
fp initvalu_3;
fp initvalu_4;
fp initvalu_5;
fp initvalu_6;
fp initvalu_7;
fp initvalu_8;
fp initvalu_9;
fp initvalu_10;
fp initvalu_11;
fp initvalu_12;
fp initvalu_13;
fp initvalu_14;
fp initvalu_15;
fp initvalu_16;
fp initvalu_17;
fp initvalu_18;
fp initvalu_19;
fp initvalu_20;
fp initvalu_21;
fp initvalu_23;
fp initvalu_24;
fp initvalu_25;
fp initvalu_26;
fp initvalu_27;
fp initvalu_28;
fp initvalu_29;
fp initvalu_30;
fp initvalu_31;
fp initvalu_32;
fp initvalu_33;
fp initvalu_34;
fp initvalu_35;
fp initvalu_36;
fp initvalu_37;
fp initvalu_38;
fp initvalu_39;
fp initvalu_40;

fp pi;

fp R;                                      
fp Frdy;                                    
fp Temp;                                    
fp FoRT;                                    
fp Cmem;                                    
fp Qpow;

fp cellLength;                                  
fp cellRadius;                                  
fp Vcell;                                    
fp Vmyo; 
fp Vsr; 
fp Vsl; 
fp Vjunc; 
fp J_ca_juncsl;                                  
fp J_ca_slmyo;                                  
fp J_na_juncsl;                                  
fp J_na_slmyo;                                  

fp Fjunc;   
fp Fsl;
fp Fjunc_CaL; 
fp Fsl_CaL;

fp Cli;                                      
fp Clo;                                      
fp Ko;                                      
fp Nao;                                      
fp Cao;                                      
fp Mgi;                                      

fp ena_junc;                                  
fp ena_sl;                                    
fp ek;                                      
fp eca_junc;                                  
fp eca_sl;                                    
fp ecl;                                      

fp GNa;                                      
fp GNaB;                                    
fp IbarNaK;                                    
fp KmNaip;                                    
fp KmKo;                                    

fp pNaK;      
fp GtoSlow;                                    
fp GtoFast;                                    
fp gkp;

fp GClCa;                                    
fp GClB;                                    
fp KdClCa;                                    

fp pNa;                                      
fp pCa;                                      
fp pK;                                      
fp Q10CaL;       

fp IbarNCX;                                    
fp KmCai;                                    
fp KmCao;                                    
fp KmNai;                                    
fp KmNao;                                    
fp ksat;                                      
fp nu;                                      
fp Kdact;                                    
fp Q10NCX;                                    
fp IbarSLCaP;                                  
fp KmPCa;                                    
fp GCaB;                                    
fp Q10SLCaP;                                  

fp Q10SRCaP;                                  
fp Vmax_SRCaP;                                  
fp Kmf;                                      
fp Kmr;                                      
fp hillSRCaP;                                  
fp ks;                                      
fp koCa;                                    
fp kom;                                      
fp kiCa;                                    
fp kim;                                      
fp ec50SR;                                    

fp Bmax_Naj;                                  
fp Bmax_Nasl;                                  
fp koff_na;                                    
fp kon_na;                                    
fp Bmax_TnClow;                                  
fp koff_tncl;                                  
fp kon_tncl;                                  
fp Bmax_TnChigh;                                
fp koff_tnchca;                                  
fp kon_tnchca;                                  
fp koff_tnchmg;                                  
fp kon_tnchmg;                                  
fp Bmax_myosin;                                  
fp koff_myoca;                                  
fp kon_myoca;                                  
fp koff_myomg;                                  
fp kon_myomg;                                  
fp Bmax_SR;                                    
fp koff_sr;                                    
fp kon_sr;                                    
fp Bmax_SLlowsl;                                
fp Bmax_SLlowj;                                  
fp koff_sll;                                  
fp kon_sll;                                    
fp Bmax_SLhighsl;                                
fp Bmax_SLhighj;                                
fp koff_slh;                                  
fp kon_slh;                                    
fp Bmax_Csqn;                                  
fp koff_csqn;                                  
fp kon_csqn;                                  

fp am;
fp bm;
fp ah;
fp bh;
fp aj;
fp bj;
fp I_Na_junc;
fp I_Na_sl;

fp I_nabk_junc;
fp I_nabk_sl;

fp sigma;
fp fnak;
fp I_nak_junc;
fp I_nak_sl;
fp I_nak;

fp gkr;
fp xrss;
fp tauxr;
fp rkr;
fp I_kr;

fp pcaks_junc; 
fp pcaks_sl;  
fp gks_junc;
fp gks_sl; 
fp eks;  
fp xsss;
fp tauxs; 
fp I_ks_junc;
fp I_ks_sl;
fp I_ks;

fp kp_kp;
fp I_kp_junc;
fp I_kp_sl;
fp I_kp;

fp xtoss;
fp ytoss;
fp rtoss;
fp tauxtos;
fp tauytos;
fp taurtos; 
fp I_tos;  

fp tauxtof;
fp tauytof;
fp I_tof;
fp I_to;

fp aki;
fp bki;
fp kiss;
fp I_ki;

fp I_ClCa_junc;
fp I_ClCa_sl;
fp I_ClCa;
fp I_Clbk;

fp dss;
fp taud;
fp fss;
fp tauf;

fp ibarca_j;
fp ibarca_sl;
fp ibark;
fp ibarna_j;
fp ibarna_sl;
fp I_Ca_junc;
fp I_Ca_sl;
fp I_Ca;
fp I_CaK;
fp I_CaNa_junc;
fp I_CaNa_sl;

fp Ka_junc;
fp Ka_sl;
fp s1_junc;
fp s1_sl;
fp s2_junc;
fp s3_junc;
fp s2_sl;
fp s3_sl;
fp I_ncx_junc;
fp I_ncx_sl;
fp I_ncx;

fp I_pca_junc;
fp I_pca_sl;
fp I_pca;

fp I_cabk_junc;
fp I_cabk_sl;
fp I_cabk;

fp MaxSR;
fp MinSR;
fp kCaSR;
fp koSRCa;
fp kiSRCa;
fp RI;
fp J_SRCarel;                                  
fp J_serca;
fp J_SRleak;                                    

fp J_CaB_cytosol;

fp J_CaB_junction;
fp J_CaB_sl;

fp oneovervsr;

fp I_Na_tot_junc;                                
fp I_Na_tot_sl;                                  
fp oneovervsl;

fp I_K_tot;

fp I_Ca_tot_junc;                                
fp I_Ca_tot_sl;                                  

int state;                                      
fp I_app;
fp V_hold;
fp V_test;
fp V_clamp;
fp R_clamp;

fp I_Na_tot;                                    
fp I_Cl_tot;                                    
fp I_Ca_tot;
fp I_tot;


cycleLength = params[15];

offset_1 = valu_offset;
offset_2 = valu_offset+1;
offset_3 = valu_offset+2;
offset_4 = valu_offset+3;
offset_5 = valu_offset+4;
offset_6 = valu_offset+5;
offset_7 = valu_offset+6;
offset_8 = valu_offset+7;
offset_9 = valu_offset+8;
offset_10 = valu_offset+9;
offset_11 = valu_offset+10;
offset_12 = valu_offset+11;
offset_13 = valu_offset+12;
offset_14 = valu_offset+13;
offset_15 = valu_offset+14;
offset_16 = valu_offset+15;
offset_17 = valu_offset+16;
offset_18 = valu_offset+17;
offset_19 = valu_offset+18;
offset_20 = valu_offset+19;
offset_21 = valu_offset+20;
offset_22 = valu_offset+21;
offset_23 = valu_offset+22;
offset_24 = valu_offset+23;
offset_25 = valu_offset+24;
offset_26 = valu_offset+25;
offset_27 = valu_offset+26;
offset_28 = valu_offset+27;
offset_29 = valu_offset+28;
offset_30 = valu_offset+29;
offset_31 = valu_offset+30;
offset_32 = valu_offset+31;
offset_33 = valu_offset+32;
offset_34 = valu_offset+33;
offset_35 = valu_offset+34;
offset_36 = valu_offset+35;
offset_37 = valu_offset+36;
offset_38 = valu_offset+37;
offset_39 = valu_offset+38;
offset_40 = valu_offset+39;
offset_41 = valu_offset+40;
offset_42 = valu_offset+41;
offset_43 = valu_offset+42;
offset_44 = valu_offset+43;
offset_45 = valu_offset+44;
offset_46 = valu_offset+45;

initvalu_1 = initvalu[offset_1];
initvalu_2 = initvalu[offset_2];
initvalu_3 = initvalu[offset_3];
initvalu_4 = initvalu[offset_4];
initvalu_5 = initvalu[offset_5];
initvalu_6 = initvalu[offset_6];
initvalu_7 = initvalu[offset_7];
initvalu_8 = initvalu[offset_8];
initvalu_9 = initvalu[offset_9];
initvalu_10 = initvalu[offset_10];
initvalu_11 = initvalu[offset_11];
initvalu_12 = initvalu[offset_12];
initvalu_13 = initvalu[offset_13];
initvalu_14 = initvalu[offset_14];
initvalu_15 = initvalu[offset_15];
initvalu_16 = initvalu[offset_16];
initvalu_17 = initvalu[offset_17];
initvalu_18 = initvalu[offset_18];
initvalu_19 = initvalu[offset_19];
initvalu_20 = initvalu[offset_20];
initvalu_21 = initvalu[offset_21];
initvalu_23 = initvalu[offset_23];
initvalu_24 = initvalu[offset_24];
initvalu_25 = initvalu[offset_25];
initvalu_26 = initvalu[offset_26];
initvalu_27 = initvalu[offset_27];
initvalu_28 = initvalu[offset_28];
initvalu_29 = initvalu[offset_29];
initvalu_30 = initvalu[offset_30];
initvalu_31 = initvalu[offset_31];
initvalu_32 = initvalu[offset_32];
initvalu_33 = initvalu[offset_33];
initvalu_34 = initvalu[offset_34];
initvalu_35 = initvalu[offset_35];
initvalu_36 = initvalu[offset_36];
initvalu_37 = initvalu[offset_37];
initvalu_38 = initvalu[offset_38];
initvalu_39 = initvalu[offset_39];
initvalu_40 = initvalu[offset_40];

pi = 3.1416;

R = 8314;                                      
Frdy = 96485;                                    
Temp = 310;                                      
FoRT = Frdy/R/Temp;                                  
Cmem = 1.3810e-10;                                  
Qpow = (Temp-310)/10;

cellLength = 100;                                  
cellRadius = 10.25;                                  
Vcell = pi*powf(cellRadius,(fp)2)*cellLength*1e-15;                      
Vmyo = 0.65*Vcell; 
Vsr = 0.035*Vcell; 
Vsl = 0.02*Vcell; 
Vjunc = 0.0539*0.01*Vcell; 
J_ca_juncsl = 1/1.2134e12;                              
J_ca_slmyo = 1/2.68510e11;                              
J_na_juncsl = 1/(1.6382e12/3*100);                          
J_na_slmyo = 1/(1.8308e10/3*100);                          

Fjunc = 0.11;   
Fsl = 1-Fjunc;
Fjunc_CaL = 0.9; 
Fsl_CaL = 1-Fjunc_CaL;

Cli = 15;                                      
Clo = 150;                                      
Ko = 5.4;                                      
Nao = 140;                                      
Cao = 1.8;                                      
Mgi = 1;                                      

ena_junc = (1/FoRT)*logf(Nao/initvalu_32);                          
ena_sl = (1/FoRT)*logf(Nao/initvalu_33);                          
ek = (1/FoRT)*logf(Ko/initvalu_35);                            
eca_junc = (1/FoRT/2)*logf(Cao/initvalu_36);                        
eca_sl = (1/FoRT/2)*logf(Cao/initvalu_37);                          
ecl = (1/FoRT)*logf(Cli/Clo);                            

GNa =  16.0;                                    
GNaB = 0.297e-3;                                  
IbarNaK = 1.90719;                                  
KmNaip = 11;                                    
KmKo = 1.5;                                      

pNaK = 0.01833;      
GtoSlow = 0.06;                                    
GtoFast = 0.02;                                    
gkp = 0.001;

GClCa = 0.109625;                                  
GClB = 9e-3;                                    
KdClCa = 100e-3;                                  

pNa = 1.5e-8;                                    
pCa = 5.4e-4;                                    
pK = 2.7e-7;                                    
Q10CaL = 1.8;       

IbarNCX = 9.0;                                    
KmCai = 3.59e-3;                                  
KmCao = 1.3;                                    
KmNai = 12.29;                                    
KmNao = 87.5;                                    
ksat = 0.27;                                    
nu = 0.35;                                      
Kdact = 0.256e-3;                                  
Q10NCX = 1.57;                                    
IbarSLCaP = 0.0673;                                  
KmPCa = 0.5e-3;                                    
GCaB = 2.513e-4;                                  
Q10SLCaP = 2.35;                                  

Q10SRCaP = 2.6;                                    
Vmax_SRCaP = 2.86e-4;                                
Kmf = 0.246e-3;                                    
Kmr = 1.7;                                      
hillSRCaP = 1.787;                                  
ks = 25;                                      
koCa = 10;                                      
kom = 0.06;                                      
kiCa = 0.5;                                      
kim = 0.005;                                    
ec50SR = 0.45;                                    

Bmax_Naj = 7.561;                                  
Bmax_Nasl = 1.65;                                  
koff_na = 1e-3;                                    
kon_na = 0.1e-3;                                  
Bmax_TnClow = 70e-3;                                
koff_tncl = 19.6e-3;                                
kon_tncl = 32.7;                                  
Bmax_TnChigh = 140e-3;                                
koff_tnchca = 0.032e-3;                                
kon_tnchca = 2.37;                                  
koff_tnchmg = 3.33e-3;                                
kon_tnchmg = 3e-3;                                  
Bmax_myosin = 140e-3;                                
koff_myoca = 0.46e-3;                                
kon_myoca = 13.8;                                  
koff_myomg = 0.057e-3;                                
kon_myomg = 0.0157;                                  
Bmax_SR = 19*0.9e-3;                                  
koff_sr = 60e-3;                                  
kon_sr = 100;                                    
Bmax_SLlowsl = 37.38e-3*Vmyo/Vsl;                          
Bmax_SLlowj = 4.62e-3*Vmyo/Vjunc*0.1;                        
koff_sll = 1300e-3;                                  
kon_sll = 100;                                    
Bmax_SLhighsl = 13.35e-3*Vmyo/Vsl;                          
Bmax_SLhighj = 1.65e-3*Vmyo/Vjunc*0.1;                        
koff_slh = 30e-3;                                  
kon_slh = 100;                                    
Bmax_Csqn = 2.7;                                  
koff_csqn = 65;                                    
kon_csqn = 100;                                    

am = 0.32*(initvalu_39+47.13)/(1-expf(-0.1*(initvalu_39+47.13)));
bm = 0.08*expf(-initvalu_39/11);
if(initvalu_39 >= -40){
ah = 0; aj = 0;
bh = 1/(0.13*(1+expf(-(initvalu_39+10.66)/11.1)));
bj = 0.3*expf(-2.535e-7*initvalu_39)/(1+expf(-0.1*(initvalu_39+32)));
}
else{
ah = 0.135*expf((80+initvalu_39)/-6.8);
bh = 3.56*expf(0.079*initvalu_39)+3.1e5*expf(0.35*initvalu_39);
aj = (-127140*expf(0.2444*initvalu_39)-3.474e-5*expf(-0.04391*initvalu_39))*(initvalu_39+37.78)/(1+expf(0.311*(initvalu_39+79.23)));
bj = 0.1212*expf(-0.01052*initvalu_39)/(1+expf(-0.1378*(initvalu_39+40.14)));
}
finavalu[offset_1] = am*(1-initvalu_1)-bm*initvalu_1;
finavalu[offset_2] = ah*(1-initvalu_2)-bh*initvalu_2;
finavalu[offset_3] = aj*(1-initvalu_3)-bj*initvalu_3;
I_Na_junc = Fjunc*GNa*powf(initvalu_1,(fp)3)*initvalu_2*initvalu_3*(initvalu_39-ena_junc);
I_Na_sl = Fsl*GNa*powf(initvalu_1,(fp)3)*initvalu_2*initvalu_3*(initvalu_39-ena_sl);

I_nabk_junc = Fjunc*GNaB*(initvalu_39-ena_junc);
I_nabk_sl = Fsl*GNaB*(initvalu_39-ena_sl);

sigma = (expf(Nao/67.3)-1)/7;
fnak = 1/(1+0.1245*expf(-0.1*initvalu_39*FoRT)+0.0365*sigma*expf(-initvalu_39*FoRT));
I_nak_junc = Fjunc*IbarNaK*fnak*Ko /(1+powf((KmNaip/initvalu_32),(fp)4)) /(Ko+KmKo);
I_nak_sl = Fsl*IbarNaK*fnak*Ko /(1+powf((KmNaip/initvalu_33),(fp)4)) /(Ko+KmKo);
I_nak = I_nak_junc+I_nak_sl;

gkr = 0.03*sqrtf(Ko/5.4);
xrss = 1/(1+expf(-(initvalu_39+50)/7.5));
tauxr = 1/(0.00138*(initvalu_39+7)/(1-expf(-0.123*(initvalu_39+7)))+6.1e-4*(initvalu_39+10)/(expf(0.145*(initvalu_39+10))-1));
finavalu[offset_12] = (xrss-initvalu_12)/tauxr;
rkr = 1/(1+expf((initvalu_39+33)/22.4));
I_kr = gkr*initvalu_12*rkr*(initvalu_39-ek);

pcaks_junc = -log10f(initvalu_36)+3.0; 
pcaks_sl = -log10f(initvalu_37)+3.0;  
gks_junc = 0.07*(0.057 +0.19/(1+ expf((-7.2+pcaks_junc)/0.6)));
gks_sl = 0.07*(0.057 +0.19/(1+ expf((-7.2+pcaks_sl)/0.6))); 
eks = (1/FoRT)*logf((Ko+pNaK*Nao)/(initvalu_35+pNaK*initvalu_34));  
xsss = 1/(1+expf(-(initvalu_39-1.5)/16.7));
tauxs = 1/(7.19e-5*(initvalu_39+30)/(1-expf(-0.148*(initvalu_39+30)))+1.31e-4*(initvalu_39+30)/(expf(0.0687*(initvalu_39+30))-1)); 
finavalu[offset_13] = (xsss-initvalu_13)/tauxs;
I_ks_junc = Fjunc*gks_junc*powf(initvalu_12,(fp)2)*(initvalu_39-eks);
I_ks_sl = Fsl*gks_sl*powf(initvalu_13,(fp)2)*(initvalu_39-eks);
I_ks = I_ks_junc+I_ks_sl;

kp_kp = 1/(1+expf(7.488-initvalu_39/5.98));
I_kp_junc = Fjunc*gkp*kp_kp*(initvalu_39-ek);
I_kp_sl = Fsl*gkp*kp_kp*(initvalu_39-ek);
I_kp = I_kp_junc+I_kp_sl;

xtoss = 1/(1+expf(-(initvalu_39+3.0)/15));
ytoss = 1/(1+expf((initvalu_39+33.5)/10));
rtoss = 1/(1+expf((initvalu_39+33.5)/10));
tauxtos = 9/(1+expf((initvalu_39+3.0)/15))+0.5;
tauytos = 3e3/(1+expf((initvalu_39+60.0)/10))+30;
taurtos = 2800/(1+expf((initvalu_39+60.0)/10))+220; 
finavalu[offset_8] = (xtoss-initvalu_8)/tauxtos;
finavalu[offset_9] = (ytoss-initvalu_9)/tauytos;
finavalu[offset_40]= (rtoss-initvalu_40)/taurtos; 
I_tos = GtoSlow*initvalu_8*(initvalu_9+0.5*initvalu_40)*(initvalu_39-ek);                  

tauxtof = 3.5*expf(-initvalu_39*initvalu_39/30/30)+1.5;
tauytof = 20.0/(1+expf((initvalu_39+33.5)/10))+20.0;
finavalu[offset_10] = (xtoss-initvalu_10)/tauxtof;
finavalu[offset_11] = (ytoss-initvalu_11)/tauytof;
I_tof = GtoFast*initvalu_10*initvalu_11*(initvalu_39-ek);
I_to = I_tos + I_tof;

aki = 1.02/(1+expf(0.2385*(initvalu_39-ek-59.215)));
bki =(0.49124*expf(0.08032*(initvalu_39+5.476-ek)) + expf(0.06175*(initvalu_39-ek-594.31))) /(1 + expf(-0.5143*(initvalu_39-ek+4.753)));
kiss = aki/(aki+bki);
I_ki = 0.9*sqrtf(Ko/5.4)*kiss*(initvalu_39-ek);

I_ClCa_junc = Fjunc*GClCa/(1+KdClCa/initvalu_36)*(initvalu_39-ecl);
I_ClCa_sl = Fsl*GClCa/(1+KdClCa/initvalu_37)*(initvalu_39-ecl);
I_ClCa = I_ClCa_junc+I_ClCa_sl;
I_Clbk = GClB*(initvalu_39-ecl);

dss = 1/(1+expf(-(initvalu_39+14.5)/6.0));
taud = dss*(1-expf(-(initvalu_39+14.5)/6.0))/(0.035*(initvalu_39+14.5));
fss = 1/(1+expf((initvalu_39+35.06)/3.6))+0.6/(1+expf((50-initvalu_39)/20));
tauf = 1/(0.0197*expf(-powf(0.0337*(initvalu_39+14.5),2.0))+0.02); 
finavalu[offset_4] = (dss-initvalu_4)/taud;
finavalu[offset_5] = (fss-initvalu_5)/tauf;
finavalu[offset_6] = 1.7*initvalu_36*(1-initvalu_6)-11.9e-3*initvalu_6;                      
finavalu[offset_7] = 1.7*initvalu_37*(1-initvalu_7)-11.9e-3*initvalu_7;                      

ibarca_j = pCa*4*(initvalu_39*Frdy*FoRT) * (0.341*initvalu_36*expf(2*initvalu_39*FoRT)-0.341*Cao) /(expf(2*initvalu_39*FoRT)-1);
ibarca_sl = pCa*4*(initvalu_39*Frdy*FoRT) * (0.341*initvalu_37*expf(2*initvalu_39*FoRT)-0.341*Cao) /(expf(2*initvalu_39*FoRT)-1);
ibark = pK*(initvalu_39*Frdy*FoRT)*(0.75*initvalu_35*expf(initvalu_39*FoRT)-0.75*Ko) /(expf(initvalu_39*FoRT)-1);
ibarna_j = pNa*(initvalu_39*Frdy*FoRT) *(0.75*initvalu_32*expf(initvalu_39*FoRT)-0.75*Nao)  /(expf(initvalu_39*FoRT)-1);
ibarna_sl = pNa*(initvalu_39*Frdy*FoRT) *(0.75*initvalu_33*expf(initvalu_39*FoRT)-0.75*Nao)  /(expf(initvalu_39*FoRT)-1);
I_Ca_junc = (Fjunc_CaL*ibarca_j*initvalu_4*initvalu_5*(1-initvalu_6)*powf(Q10CaL,Qpow))*0.45;
I_Ca_sl = (Fsl_CaL*ibarca_sl*initvalu_4*initvalu_5*(1-initvalu_7)*powf(Q10CaL,Qpow))*0.45;
I_Ca = I_Ca_junc+I_Ca_sl;
finavalu[offset_43]=-I_Ca*Cmem/(Vmyo*2*Frdy)*1e3;
I_CaK = (ibark*initvalu_4*initvalu_5*(Fjunc_CaL*(1-initvalu_6)+Fsl_CaL*(1-initvalu_7))*powf(Q10CaL,Qpow))*0.45;
I_CaNa_junc = (Fjunc_CaL*ibarna_j*initvalu_4*initvalu_5*(1-initvalu_6)*powf(Q10CaL,Qpow))*0.45;
I_CaNa_sl = (Fsl_CaL*ibarna_sl*initvalu_4*initvalu_5*(1-initvalu_7)*powf(Q10CaL,Qpow))*0.45;

Ka_junc = 1/(1+powf((Kdact/initvalu_36),(fp)3));
Ka_sl = 1/(1+powf((Kdact/initvalu_37),(fp)3));
s1_junc = expf(nu*initvalu_39*FoRT)*powf(initvalu_32,(fp)3)*Cao;
s1_sl = expf(nu*initvalu_39*FoRT)*powf(initvalu_33,(fp)3)*Cao;
s2_junc = expf((nu-1)*initvalu_39*FoRT)*powf(Nao,(fp)3)*initvalu_36;
s3_junc = (KmCai*powf(Nao,(fp)3)*(1+powf((initvalu_32/KmNai),(fp)3))+powf(KmNao,(fp)3)*initvalu_36+ powf(KmNai,(fp)3)*Cao*(1+initvalu_36/KmCai)+KmCao*powf(initvalu_32,(fp)3)+powf(initvalu_32,(fp)3)*Cao+powf(Nao,(fp)3)*initvalu_36)*(1+ksat*expf((nu-1)*initvalu_39*FoRT));
s2_sl = expf((nu-1)*initvalu_39*FoRT)*powf(Nao,(fp)3)*initvalu_37;
s3_sl = (KmCai*powf(Nao,(fp)3)*(1+powf((initvalu_33/KmNai),(fp)3)) + powf(KmNao,(fp)3)*initvalu_37+powf(KmNai,(fp)3)*Cao*(1+initvalu_37/KmCai)+KmCao*powf(initvalu_33,(fp)3)+powf(initvalu_33,(fp)3)*Cao+powf(Nao,(fp)3)*initvalu_37)*(1+ksat*expf((nu-1)*initvalu_39*FoRT));
I_ncx_junc = Fjunc*IbarNCX*powf(Q10NCX,Qpow)*Ka_junc*(s1_junc-s2_junc)/s3_junc;
I_ncx_sl = Fsl*IbarNCX*powf(Q10NCX,Qpow)*Ka_sl*(s1_sl-s2_sl)/s3_sl;
I_ncx = I_ncx_junc+I_ncx_sl;
finavalu[offset_45]=2*I_ncx*Cmem/(Vmyo*2*Frdy)*1e3;

I_pca_junc = Fjunc*powf(Q10SLCaP,Qpow)*IbarSLCaP*powf(initvalu_36,(fp)(1.6))/(powf(KmPCa,(fp)(1.6))+powf(initvalu_36,(fp)(1.6)));
I_pca_sl = Fsl*powf(Q10SLCaP,Qpow)*IbarSLCaP*powf(initvalu_37,(fp)(1.6))/(powf(KmPCa,(fp)(1.6))+powf(initvalu_37,(fp)(1.6)));
I_pca = I_pca_junc+I_pca_sl;
finavalu[offset_44]=-I_pca*Cmem/(Vmyo*2*Frdy)*1e3;

I_cabk_junc = Fjunc*GCaB*(initvalu_39-eca_junc);
I_cabk_sl = Fsl*GCaB*(initvalu_39-eca_sl);
I_cabk = I_cabk_junc+I_cabk_sl;
finavalu[offset_46]=-I_cabk*Cmem/(Vmyo*2*Frdy)*1e3;

MaxSR = 15; 
MinSR = 1;
kCaSR = MaxSR - (MaxSR-MinSR)/(1+powf(ec50SR/initvalu_31,(fp)(2.5)));
koSRCa = koCa/kCaSR;
kiSRCa = kiCa*kCaSR;
RI = 1-initvalu_14-initvalu_15-initvalu_16;
finavalu[offset_14] = (kim*RI-kiSRCa*initvalu_36*initvalu_14)-(koSRCa*powf(initvalu_36,(fp)2)*initvalu_14-kom*initvalu_15);      
finavalu[offset_15] = (koSRCa*powf(initvalu_36,(fp)2)*initvalu_14-kom*initvalu_15)-(kiSRCa*initvalu_36*initvalu_15-kim*initvalu_16);      
finavalu[offset_16] = (kiSRCa*initvalu_36*initvalu_15-kim*initvalu_16)-(kom*initvalu_16-koSRCa*powf(initvalu_36,(fp)2)*RI);      
J_SRCarel = ks*initvalu_15*(initvalu_31-initvalu_36);                          
J_serca = powf(Q10SRCaP,Qpow)*Vmax_SRCaP*(powf((initvalu_38/Kmf),hillSRCaP)-powf((initvalu_31/Kmr),hillSRCaP))
/(1+powf((initvalu_38/Kmf),hillSRCaP)+powf((initvalu_31/Kmr),hillSRCaP));
J_SRleak = 5.348e-6*(initvalu_31-initvalu_36);                          

finavalu[offset_17] = kon_na*initvalu_32*(Bmax_Naj-initvalu_17)-koff_na*initvalu_17;                
finavalu[offset_18] = kon_na*initvalu_33*(Bmax_Nasl-initvalu_18)-koff_na*initvalu_18;              

finavalu[offset_19] = kon_tncl*initvalu_38*(Bmax_TnClow-initvalu_19)-koff_tncl*initvalu_19;            
finavalu[offset_20] = kon_tnchca*initvalu_38*(Bmax_TnChigh-initvalu_20-initvalu_21)-koff_tnchca*initvalu_20;      
finavalu[offset_21] = kon_tnchmg*Mgi*(Bmax_TnChigh-initvalu_20-initvalu_21)-koff_tnchmg*initvalu_21;        
finavalu[offset_22] = 0;                                    
finavalu[offset_23] = kon_myoca*initvalu_38*(Bmax_myosin-initvalu_23-initvalu_24)-koff_myoca*initvalu_23;        
finavalu[offset_24] = kon_myomg*Mgi*(Bmax_myosin-initvalu_23-initvalu_24)-koff_myomg*initvalu_24;        
finavalu[offset_25] = kon_sr*initvalu_38*(Bmax_SR-initvalu_25)-koff_sr*initvalu_25;                
J_CaB_cytosol = finavalu[offset_19] + finavalu[offset_20] + finavalu[offset_21] + finavalu[offset_22] + finavalu[offset_23] + finavalu[offset_24] + finavalu[offset_25];

finavalu[offset_26] = kon_sll*initvalu_36*(Bmax_SLlowj-initvalu_26)-koff_sll*initvalu_26;            
finavalu[offset_27] = kon_sll*initvalu_37*(Bmax_SLlowsl-initvalu_27)-koff_sll*initvalu_27;            
finavalu[offset_28] = kon_slh*initvalu_36*(Bmax_SLhighj-initvalu_28)-koff_slh*initvalu_28;            
finavalu[offset_29] = kon_slh*initvalu_37*(Bmax_SLhighsl-initvalu_29)-koff_slh*initvalu_29;            
J_CaB_junction = finavalu[offset_26]+finavalu[offset_28];
J_CaB_sl = finavalu[offset_27]+finavalu[offset_29];

finavalu[offset_30] = kon_csqn*initvalu_31*(Bmax_Csqn-initvalu_30)-koff_csqn*initvalu_30;            
oneovervsr = 1/Vsr;
finavalu[offset_31] = J_serca*Vmyo*oneovervsr-(J_SRleak*Vmyo*oneovervsr+J_SRCarel)-finavalu[offset_30];   

I_Na_tot_junc = I_Na_junc+I_nabk_junc+3*I_ncx_junc+3*I_nak_junc+I_CaNa_junc;    
I_Na_tot_sl = I_Na_sl+I_nabk_sl+3*I_ncx_sl+3*I_nak_sl+I_CaNa_sl;          
finavalu[offset_32] = -I_Na_tot_junc*Cmem/(Vjunc*Frdy)+J_na_juncsl/Vjunc*(initvalu_33-initvalu_32)-finavalu[offset_17];
oneovervsl = 1/Vsl;
finavalu[offset_33] = -I_Na_tot_sl*Cmem*oneovervsl/Frdy+J_na_juncsl*oneovervsl*(initvalu_32-initvalu_33)+J_na_slmyo*oneovervsl*(initvalu_34-initvalu_33)-finavalu[offset_18];
finavalu[offset_34] = J_na_slmyo/Vmyo*(initvalu_33-initvalu_34);                      

I_K_tot = I_to+I_kr+I_ks+I_ki-2*I_nak+I_CaK+I_kp;                  
finavalu[offset_35] = 0;                              

I_Ca_tot_junc = I_Ca_junc+I_cabk_junc+I_pca_junc-2*I_ncx_junc;            
I_Ca_tot_sl = I_Ca_sl+I_cabk_sl+I_pca_sl-2*I_ncx_sl;                
finavalu[offset_36] = -I_Ca_tot_junc*Cmem/(Vjunc*2*Frdy)+J_ca_juncsl/Vjunc*(initvalu_37-initvalu_36)
- J_CaB_junction+(J_SRCarel)*Vsr/Vjunc+J_SRleak*Vmyo/Vjunc;        
finavalu[offset_37] = -I_Ca_tot_sl*Cmem/(Vsl*2*Frdy)+J_ca_juncsl/Vsl*(initvalu_36-initvalu_37)
+ J_ca_slmyo/Vsl*(initvalu_38-initvalu_37)-J_CaB_sl;                  
finavalu[offset_38] = -J_serca-J_CaB_cytosol +J_ca_slmyo/Vmyo*(initvalu_37-initvalu_38);

state = 1;                                      
switch(state){
case 0:
I_app = 0;
break;
case 1:                                      
if(fmod(timeinst,cycleLength) <= 5){
I_app = 9.5;
}
else{
I_app = 0.0;
}
break;
case 2:     
V_hold = -55;
V_test = 0;
if(timeinst>0.5 & timeinst<200.5){
V_clamp = V_test;
}
else{
V_clamp = V_hold;
}
R_clamp = 0.04;
I_app = (V_clamp-initvalu_39)/R_clamp;
break;
} 

I_Na_tot = I_Na_tot_junc + I_Na_tot_sl;                        
I_Cl_tot = I_ClCa+I_Clbk;                              
I_Ca_tot = I_Ca_tot_junc+I_Ca_tot_sl;
I_tot = I_Na_tot+I_Cl_tot+I_Ca_tot+I_K_tot;
finavalu[offset_39] = -(I_tot-I_app);

finavalu[offset_41] = 0;
finavalu[offset_42] = 0;

}

#pragma omp end declare target
