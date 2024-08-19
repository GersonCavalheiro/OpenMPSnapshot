#include <stdlib.h>
#include <stdio.h>
#include <time.h>	
#include <string.h>     
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "rasterfile.h"
char info[] = "\
Usage:\n\
mandel dimx dimy xmin ymin xmax ymax prof\n\
\n\
dimx,dimy : dimensions de l'image a generer\n\
xmin,ymin,xmax,ymax : domaine a calculer dans le plan complexe\n\
prof : nombre maximale d'iteration\n\
\n\
Quelques exemples d'execution\n\
mandel 800 800 0.35 0.355 0.353 0.358 200\n\
mandel 800 800 -0.736 -0.184 -0.735 -0.183 500\n\
mandel 800 800 -0.736 -0.184 -0.735 -0.183 300\n\
mandel 800 800 -1.48478 0.00006 -1.48440 0.00044 100\n\
mandel 800 800 -1.5 -0.1 -1.3 0.1 10000\n\
";
double my_gettimeofday(){
struct timeval tmp_time;
gettimeofday(&tmp_time, NULL);
return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}
int swap(int i) {
int init = i; 
int conv;
unsigned char *o, *d;
o = ( (unsigned char *) &init) + 3; 
d = (unsigned char *) &conv;
*d++ = *o--;
*d++ = *o--;
*d++ = *o--;
*d++ = *o--;
return conv;
}
unsigned char power_composante(int i, int p) {
unsigned char o;
double iD=(double) i;
iD/=255.0;
iD=pow(iD,p);
iD*=255;
o=(unsigned char) iD;
return o;
}
unsigned char cos_composante(int i, double freq) {
unsigned char o;
double iD=(double) i;
iD=cos(iD/255.0*2*M_PI*freq);
iD+=1;
iD*=128;
o=(unsigned char) iD;
return o;
}
#define COS_COLOR 
#ifdef ORIGINAL_COLOR
#define COMPOSANTE_ROUGE(i)    ((i)/2)
#define COMPOSANTE_VERT(i)     ((i)%190)
#define COMPOSANTE_BLEU(i)     (((i)%120) * 2)
#endif 
#ifdef COS_COLOR
#define COMPOSANTE_ROUGE(i)    cos_composante(i,13.0)
#define COMPOSANTE_VERT(i)     cos_composante(i,5.0)
#define COMPOSANTE_BLEU(i)     cos_composante(i+10,7.0)
#endif 
void sauver_rasterfile( char *nom, int largeur, int hauteur, unsigned char *p) {
FILE *fd;
struct rasterfile file;
int i;
unsigned char o;
if ( (fd=fopen(nom, "w")) == NULL ) {
printf("erreur dans la creation du fichier %s \n",nom);
exit(1);
}
file.ras_magic  = swap(RAS_MAGIC);	
file.ras_width  = swap(largeur);	  
file.ras_height = swap(hauteur);         
file.ras_depth  = swap(8);	          
file.ras_length = swap(largeur*hauteur); 
file.ras_type    = swap(RT_STANDARD);	  
file.ras_maptype = swap(RMT_EQUAL_RGB);
file.ras_maplength = swap(256*3);
fwrite(&file, sizeof(struct rasterfile), 1, fd); 
i = 256;
while( i--) {
o = COMPOSANTE_ROUGE(i);
fwrite( &o, sizeof(unsigned char), 1, fd);
}
i = 256;
while( i--) {
o = COMPOSANTE_VERT(i);
fwrite( &o, sizeof(unsigned char), 1, fd);
}
i = 256;
while( i--) {
o = COMPOSANTE_BLEU(i);
fwrite( &o, sizeof(unsigned char), 1, fd);
}
fwrite( p, largeur*hauteur, sizeof(unsigned char), fd);
fclose( fd);
}
unsigned char xy2color(double a, double b, int prof) {
double x, y, temp, x2, y2;
int i;
x = y = 0.;
for( i=0; i<prof; i++) {
temp = x;
x2 = x*x;
y2 = y*y;
x = x2 - y2 + a;
y = 2*temp*y + b;
if( x2 + y2 >= 4.0) break;
}
return (i==prof)?255:(int)((i%255)); 
}
int main(int argc, char *argv[]) {
double xmin, ymin;
double xmax, ymax;
int w,h;
double xinc, yinc;
int prof;
unsigned char	*ima, *pima;
int  i, j;
double x, y;
double debut, fin;
int nb_proc;
debut = my_gettimeofday();
if( argc == 1) fprintf( stderr, "%s\n", info);
xmin = -2; ymin = -2;
xmax =  2; ymax =  2;
w = h = 800;
prof = 10000;
nb_proc = 1;
if( argc > 1) nb_proc    = atoi(argv[1]);
if( argc > 2) w    = atoi(argv[2]);
if( argc > 3) h    = atoi(argv[3]);
if( argc > 4) xmin = atof(argv[4]);
if( argc > 5) ymin = atof(argv[5]);
if( argc > 6) xmax = atof(argv[6]);
if( argc > 7) ymax = atof(argv[7]);
if( argc > 8) prof = atoi(argv[8]);
xinc = (xmax - xmin) / (w-1);
yinc = (ymax - ymin) / (h-1);
fprintf( stderr, "Domaine: {[%lg,%lg]x[%lg,%lg]}\n", xmin, ymin, xmax, ymax);
fprintf( stderr, "Increment : %lg %lg\n", xinc, yinc);
fprintf( stderr, "Prof: %d\n",  prof);
fprintf( stderr, "Dim image: %dx%d\n", w, h);
pima = ima = (unsigned char *)malloc( w*h*sizeof(unsigned char));
if( ima == NULL) {
fprintf( stderr, "Erreur allocation mmoire du tableau \n");
return 0;
}
omp_set_num_threads(nb_proc);
#pragma omp parallel {
y = ymin; 
#pragma omp for private(j,pima,x,y) schedule (runtime)
for (i = 0; i < h; i++) {
y = ymin + i * yinc;
pima = &ima[i*w];	
x = xmin;
for (j = 0; j < w; j++) {
*pima++ = xy2color( x, y, prof); 
x += xinc;
}
y += yinc; 
}
}
fin = my_gettimeofday();
fprintf( stderr, "Temps total de calcul : %g sec\n", 
fin - debut);
fprintf( stdout, "%g\n", fin - debut);
sauver_rasterfile( "mandel.ras", w, h, ima);
return 0;
}