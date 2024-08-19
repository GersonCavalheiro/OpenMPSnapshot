#include <stdio.h>
#include <stdlib.h>
#include <math.h>   
#include <string.h> 
#include <time.h>   
#include <omp.h>
#include "rasterfile.h"
#define MAX(a,b) ((a>b) ? a : b)
typedef struct {
struct rasterfile file;  
unsigned char rouge[256],vert[256],bleu[256];  
unsigned char *data;    
} Raster;
double my_gettimeofday(){
struct timeval tmp_time;
gettimeofday(&tmp_time, NULL);
return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}
void swap(int *i) {
unsigned char s[4],*n;
memcpy(s,i,4);
n=(unsigned char *)i;
n[0]=s[3];
n[1]=s[2];
n[2]=s[1];
n[3]=s[0];
}
void lire_rasterfile(char *nom, Raster *r) {
FILE *f;
int i;
if( (f=fopen( nom, "r"))==NULL) {
fprintf(stderr,"erreur a la lecture du fichier %s\n", nom);
exit(1);
}
fread( &(r->file), sizeof(struct rasterfile), 1, f);    
swap(&(r->file.ras_magic));
swap(&(r->file.ras_width));
swap(&(r->file.ras_height));
swap(&(r->file.ras_depth));
swap(&(r->file.ras_length));
swap(&(r->file.ras_type));
swap(&(r->file.ras_maptype));
swap(&(r->file.ras_maplength));
if ((r->file.ras_depth != 8) ||  (r->file.ras_type != RT_STANDARD) ||
(r->file.ras_maptype != RMT_EQUAL_RGB)) {
fprintf(stderr,"palette non adaptee\n");
exit(1);
}
fread(&(r->rouge),r->file.ras_maplength/3,1,f);
fread(&(r->vert), r->file.ras_maplength/3,1,f);
fread(&(r->bleu), r->file.ras_maplength/3,1,f);
if ((r->data=malloc(r->file.ras_width*r->file.ras_height))==NULL){
fprintf(stderr,"erreur allocation memoire\n");
exit(1);
}
fread(r->data,r->file.ras_width*r->file.ras_height,1,f);
fclose(f);
}
void sauve_rasterfile(char *nom, Raster *r)     {
FILE *f;
int i;
if( (f=fopen( nom, "w"))==NULL) {
fprintf(stderr,"erreur a l'ecriture du fichier %s\n", nom);
exit(1);
}
swap(&(r->file.ras_magic));
swap(&(r->file.ras_width));
swap(&(r->file.ras_height));
swap(&(r->file.ras_depth));
swap(&(r->file.ras_length));
swap(&(r->file.ras_type));
swap(&(r->file.ras_maptype));
swap(&(r->file.ras_maplength));
fwrite(&(r->file),sizeof(struct rasterfile),1,f);
fwrite(&(r->rouge),256,1,f);
fwrite(&(r->vert),256,1,f);
fwrite(&(r->bleu),256,1,f);
swap(&(r->file.ras_width));
swap(&(r->file.ras_height));
fwrite(r->data,r->file.ras_width*r->file.ras_height,1,f); 
fclose(f);
}
unsigned char division(int numerateur,int denominateur) {
if (denominateur != 0)
return (unsigned char) rint((double)numerateur/(double)denominateur); 
else 
return 0;
}
static int ordre( unsigned char *a, unsigned char *b) {
return (*a-*b);
}
typedef enum {
CONVOL_MOYENNE1, 
CONVOL_MOYENNE2, 
CONVOL_CONTOUR1, 
CONVOL_CONTOUR2, 
CONVOL_MEDIAN    
} filtre_t;
unsigned char filtre( filtre_t choix, 
unsigned char NO, unsigned char N,unsigned char NE, 
unsigned char O,unsigned char CO, unsigned char E, 
unsigned char SO,unsigned char S,unsigned char SE) {
int numerateur,denominateur;
switch (choix) {
case CONVOL_MOYENNE1:
numerateur = (int)NO + (int)N + (int)NE + (int)O + (int)CO + 
(int)E + (int)SO + (int)S + (int)SE;
denominateur = 9;
return division(numerateur,denominateur); 
case CONVOL_MOYENNE2:
numerateur = (int)NO + (int)N + (int)NE + (int)O + 4*(int)CO +
(int)E + (int)SO + (int)S + (int)SE;
denominateur = 12;
return division(numerateur,denominateur);	
case CONVOL_CONTOUR1:
numerateur = -(int)N - (int)O + 4*(int)CO - (int)E - (int)S;
return ((4*abs(numerateur) > 255) ? 255 :  4*abs(numerateur));
case CONVOL_CONTOUR2:
numerateur = MAX(abs(CO-E),abs(CO-S));
return ((4*numerateur > 255) ? 255 :  4*numerateur);
case CONVOL_MEDIAN:{
unsigned char tab[] = {NO,N,NE,O,CO,E,SO,S,SE};
qsort( tab, 9, sizeof(unsigned char), (int (*) (const void *,const void *))ordre);
return tab[4];
}
default:
printf("\nERREUR : Filtre inconnu !\n\n");
exit(1);
}
}
int convolution( filtre_t choix, unsigned char tab[],int nbl,int nbc) {
int i,j;
unsigned char *tmp;
omp_set_num_threads(8);
tmp = (unsigned char*) malloc(sizeof(unsigned char) *nbc*nbl);
if (tmp == NULL) {
printf("Erreur dans l'allocation de tmp dans convolution \n");
return 1;
}
#pragma omp parallel {
#pragma omp for private(i) schedule (runtime)
for(i=1 ; i<nbl-1 ; i++){
for(j=1 ; j<nbc-1 ; j++){
tmp[i*nbc+j] = filtre(
choix,
tab[(i+1)*nbc+j-1],tab[(i+1)*nbc+j],tab[(i+1)*nbc+j+1],
tab[(i  )*nbc+j-1],tab[(i)*nbc+j],tab[(i)*nbc+j+1],
tab[(i-1)*nbc+j-1],tab[(i-1)*nbc+j],tab[(i-1)*nbc+j+1]);
} 
} 
}
for( i=1; i<nbl-1; i++){
memcpy( tab+nbc*i+1, tmp+nbc*i+1, (nbc-2)*sizeof(unsigned char));
} 
free(tmp);   
}
static char usage [] = "Usage : %s <nom image SunRaster> [0|1|2|3|4] <nbiter>\n";
int main(int argc, char *argv[]) {
Raster r;
int    w, h;	
int 	 filtre;		
int 	 nbiter;		
double debut, fin;
int 	i,j;
if (argc != 4) {
fprintf( stderr, usage, argv[0]);
return 1;
}
filtre = atoi(argv[2]);
nbiter = atoi(argv[3]);
lire_rasterfile( argv[1], &r);
h = r.file.ras_height;
w = r.file.ras_width;
debut = my_gettimeofday();            
for(i=0 ; i < nbiter ; i++){
convolution( filtre, r.data, h, w);
} 
fin = my_gettimeofday();
printf("Temps total de calcul : %g seconde(s) \n", fin - debut);
{ 
char nom_sortie[100] = "";
sprintf(nom_sortie, "post-convolution2_filtre%d_nbIter%d.ras", filtre, nbiter);
sauve_rasterfile(nom_sortie, &r);
}
return 0;
}
