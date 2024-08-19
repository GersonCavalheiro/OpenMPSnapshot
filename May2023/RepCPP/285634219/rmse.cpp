











#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

extern double wtime(void);



__inline
float euclid_dist_2(float *pt1,
float *pt2,
int    numdims)
{
int i;
float ans=0.0;

for (i=0; i<numdims; i++)
ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

return(ans);
}


__inline
int find_nearest_point(float  *pt,          
int     nfeatures,
float  **pts,         
int     npts)
{
int index, i;
float max_dist=FLT_MAX;


for (i=0; i<npts; i++) {
float dist;
dist = euclid_dist_2(pt, pts[i], nfeatures);  
if (dist < max_dist) {
max_dist = dist;
index    = i;
}
}
return(index);
}


float rms_err	(float **feature,         
int     nfeatures,
int     npoints,
float **cluster_centres, 
int     nclusters)
{
int    i;
int	   nearest_cluster_index;	
float  sum_euclid = 0.0;		
float  ret;						


#pragma omp parallel for \
shared(feature,cluster_centres) \
firstprivate(npoints,nfeatures,nclusters) \
private(i, nearest_cluster_index) \
schedule (static)	
for (i=0; i<npoints; i++) {
nearest_cluster_index = find_nearest_point(feature[i], 
nfeatures, 
cluster_centres, 
nclusters);

sum_euclid += euclid_dist_2(feature[i],
cluster_centres[nearest_cluster_index],
nfeatures);

}	

ret = sqrt(sum_euclid / npoints);

return(ret);
}

