#pragma omp declare target
void 
kernel_cam(const fp timeinst,
const fp* initvalu,
fp *finavalu,
const int valu_offset,
const fp* params,
const int params_offset,
fp* com,
const int com_offset,
const fp Ca){


fp Btot;
fp CaMKIItot;
fp CaNtot;
fp PP1tot;
fp K;
fp Mg;

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

fp CaM;
fp Ca2CaM;
fp Ca4CaM;
fp CaMB;
fp Ca2CaMB;
fp Ca4CaMB;           
fp Pb2;
fp Pb;
fp Pt;
fp Pt2;
fp Pa;                            
fp Ca4CaN;
fp CaMCa4CaN;
fp Ca2CaMCa4CaN;
fp Ca4CaMCa4CaN;

fp Kd02;                                    
fp Kd24;                                    
fp k20;                                      
fp k02;                                      
fp k42;                                      
fp k24;                                      

fp k0Boff;                                    
fp k0Bon;                                    
fp k2Boff;                                    
fp k2Bon;                                    
fp k4Boff;                                    
fp k4Bon;                                    

fp k20B;                                    
fp k02B;                                    
fp k42B;                                    
fp k24B;                                    

fp kbi;                                      
fp kib;                                      
fp kpp1;                                    
fp Kmpp1;                                    
fp kib2;
fp kb2i;
fp kb24;
fp kb42;
fp kta;                                      
fp kat;                                      
fp kt42;
fp kt24;
fp kat2;
fp kt2a;

fp kcanCaoff;                                  
fp kcanCaon;                                  
fp kcanCaM4on;                                  
fp kcanCaM4off;                                  
fp kcanCaM2on;
fp kcanCaM2off;
fp kcanCaM0on;
fp kcanCaM0off;
fp k02can;
fp k20can;
fp k24can;
fp k42can;

fp rcn02;
fp rcn24;

fp B;
fp rcn02B;
fp rcn24B;
fp rcn0B;
fp rcn2B;
fp rcn4B;

fp Ca2CaN;
fp rcnCa4CaN;
fp rcn02CaN; 
fp rcn24CaN;
fp rcn0CaN;
fp rcn2CaN;
fp rcn4CaN;

fp Pix;
fp rcnCKib2;
fp rcnCKb2b;
fp rcnCKib;
fp T;
fp kbt;
fp rcnCKbt;
fp rcnCKtt2;
fp rcnCKta;
fp rcnCKt2a;
fp rcnCKt2b2;
fp rcnCKai;

fp dCaM;
fp dCa2CaM;
fp dCa4CaM;
fp dCaMB;
fp dCa2CaMB;
fp dCa4CaMB;

fp dPb2;                                          
fp dPb;                                          
fp dPt;                                          
fp dPt2;                                          
fp dPa;                                          

fp dCa4CaN;                                      
fp dCaMCa4CaN;                                  
fp dCa2CaMCa4CaN;                                
fp dCa4CaMCa4CaN;                                


Btot = params[params_offset+1];
CaMKIItot = params[params_offset+2];
CaNtot = params[params_offset+3];
PP1tot = params[params_offset+4];
K = params[16];
Mg = params[17];

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

CaM        = initvalu[offset_1];
Ca2CaM      = initvalu[offset_2];
Ca4CaM      = initvalu[offset_3];
CaMB      = initvalu[offset_4];
Ca2CaMB      = initvalu[offset_5];
Ca4CaMB      = initvalu[offset_6];           
Pb2        = initvalu[offset_7];
Pb        = initvalu[offset_8];
Pt        = initvalu[offset_9];
Pt2        = initvalu[offset_10];
Pa        = initvalu[offset_11];                            
Ca4CaN      = initvalu[offset_12];
CaMCa4CaN    = initvalu[offset_13];
Ca2CaMCa4CaN  = initvalu[offset_14];
Ca4CaMCa4CaN  = initvalu[offset_15];

if (Mg <= 1){
Kd02 = 0.0025*(1+K/0.94-Mg/0.012)*(1+K/8.1+Mg/0.022);              
Kd24 = 0.128*(1+K/0.64+Mg/0.0014)*(1+K/13.0-Mg/0.153);              
}
else{
Kd02 = 0.0025*(1+K/0.94-1/0.012+(Mg-1)/0.060)*(1+K/8.1+1/0.022+(Mg-1)/0.068);   
Kd24 = 0.128*(1+K/0.64+1/0.0014+(Mg-1)/0.005)*(1+K/13.0-1/0.153+(Mg-1)/0.150);  
}
k20 = 10;                                      
k02 = k20/Kd02;                                    
k42 = 500;                                      
k24 = k42/Kd24;                                    

k0Boff = 0.0014;                                  
k0Bon = k0Boff/0.2;                                  
k2Boff = k0Boff/100;                                
k2Bon = k0Bon;                                    
k4Boff = k2Boff;                                  
k4Bon = k0Bon;                                    

k20B = k20/100;                                    
k02B = k02;                                      
k42B = k42;                                      
k24B = k24;                                      

kbi = 2.2;                                      
kib = kbi/33.5e-3;                                  
kpp1 = 1.72;                                    
Kmpp1 = 11.5;                                    
kib2 = kib;
kb2i = kib2*5;
kb24 = k24;
kb42 = k42*33.5e-3/5;
kta = kbi/1000;                                    
kat = kib;                                      
kt42 = k42*33.5e-6/5;
kt24 = k24;
kat2 = kib;
kt2a = kib*5;

kcanCaoff = 1;                                    
kcanCaon = kcanCaoff/0.5;                              
kcanCaM4on = 46;                                  
kcanCaM4off = 0.0013;                                
kcanCaM2on = kcanCaM4on;
kcanCaM2off = 2508*kcanCaM4off;
kcanCaM0on = kcanCaM4on;
kcanCaM0off = 165*kcanCaM2off;
k02can = k02;
k20can = k20/165;
k24can = k24;
k42can = k20/2508;

rcn02 = k02*powf(Ca,(fp)2)*CaM - k20*Ca2CaM;
rcn24 = k24*powf(Ca,(fp)2)*Ca2CaM - k42*Ca4CaM;

B = Btot - CaMB - Ca2CaMB - Ca4CaMB;
rcn02B = k02B*powf(Ca,(fp)2)*CaMB - k20B*Ca2CaMB;
rcn24B = k24B*powf(Ca,(fp)2)*Ca2CaMB - k42B*Ca4CaMB;
rcn0B = k0Bon*CaM*B - k0Boff*CaMB;
rcn2B = k2Bon*Ca2CaM*B - k2Boff*Ca2CaMB;
rcn4B = k4Bon*Ca4CaM*B - k4Boff*Ca4CaMB;

Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
rcnCa4CaN = kcanCaon*powf(Ca,(fp)2)*Ca2CaN - kcanCaoff*Ca4CaN;
rcn02CaN = k02can*powf(Ca,(fp)2)*CaMCa4CaN - k20can*Ca2CaMCa4CaN; 
rcn24CaN = k24can*powf(Ca,(fp)2)*Ca2CaMCa4CaN - k42can*Ca4CaMCa4CaN;
rcn0CaN = kcanCaM0on*CaM*Ca4CaN - kcanCaM0off*CaMCa4CaN;
rcn2CaN = kcanCaM2on*Ca2CaM*Ca4CaN - kcanCaM2off*Ca2CaMCa4CaN;
rcn4CaN = kcanCaM4on*Ca4CaM*Ca4CaN - kcanCaM4off*Ca4CaMCa4CaN;

Pix = 1 - Pb2 - Pb - Pt - Pt2 - Pa;
rcnCKib2 = kib2*Ca2CaM*Pix - kb2i*Pb2;
rcnCKb2b = kb24*powf(Ca,(fp)2)*Pb2 - kb42*Pb;
rcnCKib = kib*Ca4CaM*Pix - kbi*Pb;
T = Pb + Pt + Pt2 + Pa;
kbt = 0.055*T + 0.0074*powf(T,(fp)2) + 0.015*powf(T,(fp)3);
rcnCKbt = kbt*Pb - kpp1*PP1tot*Pt/(Kmpp1+CaMKIItot*Pt);
rcnCKtt2 = kt42*Pt - kt24*powf(Ca,(fp)2)*Pt2;
rcnCKta = kta*Pt - kat*Ca4CaM*Pa;
rcnCKt2a = kt2a*Pt2 - kat2*Ca2CaM*Pa;
rcnCKt2b2 = kpp1*PP1tot*Pt2/(Kmpp1+CaMKIItot*Pt2);
rcnCKai = kpp1*PP1tot*Pa/(Kmpp1+CaMKIItot*Pa);

dCaM = 1e-3*(-rcn02 - rcn0B - rcn0CaN);
dCa2CaM = 1e-3*(rcn02 - rcn24 - rcn2B - rcn2CaN + CaMKIItot*(-rcnCKib2 + rcnCKt2a) );
dCa4CaM = 1e-3*(rcn24 - rcn4B - rcn4CaN + CaMKIItot*(-rcnCKib+rcnCKta) );
dCaMB = 1e-3*(rcn0B-rcn02B);
dCa2CaMB = 1e-3*(rcn02B + rcn2B - rcn24B);
dCa4CaMB = 1e-3*(rcn24B + rcn4B);

dPb2 = 1e-3*(rcnCKib2 - rcnCKb2b + rcnCKt2b2);                    
dPb = 1e-3*(rcnCKib + rcnCKb2b - rcnCKbt);                      
dPt = 1e-3*(rcnCKbt-rcnCKta-rcnCKtt2);                        
dPt2 = 1e-3*(rcnCKtt2-rcnCKt2a-rcnCKt2b2);                      
dPa = 1e-3*(rcnCKta+rcnCKt2a-rcnCKai);                        

dCa4CaN = 1e-3*(rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN);              
dCaMCa4CaN = 1e-3*(rcn0CaN - rcn02CaN);                        
dCa2CaMCa4CaN = 1e-3*(rcn2CaN+rcn02CaN-rcn24CaN);                  
dCa4CaMCa4CaN = 1e-3*(rcn4CaN+rcn24CaN);                      

finavalu[offset_1] = dCaM;
finavalu[offset_2] = dCa2CaM;
finavalu[offset_3] = dCa4CaM;
finavalu[offset_4] = dCaMB;
finavalu[offset_5] = dCa2CaMB;
finavalu[offset_6] = dCa4CaMB;
finavalu[offset_7] = dPb2;
finavalu[offset_8] = dPb;
finavalu[offset_9] = dPt;
finavalu[offset_10] = dPt2;
finavalu[offset_11] = dPa;
finavalu[offset_12] = dCa4CaN;
finavalu[offset_13] = dCaMCa4CaN;
finavalu[offset_14] = dCa2CaMCa4CaN;
finavalu[offset_15] = dCa4CaMCa4CaN;

finavalu[com_offset] = 1e-3*(2*CaMKIItot*(rcnCKtt2-rcnCKb2b) - 2*(rcn02+rcn24+rcn02B+rcn24B+rcnCa4CaN+rcn02CaN+rcn24CaN)); 

}

#pragma omp end declare target
