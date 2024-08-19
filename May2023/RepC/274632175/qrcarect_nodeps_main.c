#include "qrca_kernels.h"
extern double **gl_Ah;
extern double **gl_Th;
extern double **gl_Sh;
void qrcarect_nodeps(int diagl, int tr, int tc, int bs, int mt, int nt) {
int rk=0;
int k;
for( k = 0; k < diagl; k++ ) {
#pragma omp taskwait
int row=rk / tr;
int rskip=rk % tr;
dgeqt2_task( bs, tr, tc, rskip,  
gl_Ah[ mt*k+row ], 
gl_Th[ mt*k+row ],
gl_Sh[ mt*k+row ]);
#pragma omp taskwait
int j;
for( j = k+1; j < nt; j++ ) {
dlarfb_task( bs, tr, tc, tc, rskip, 
gl_Ah[ mt*k+row ], 
gl_Sh[ mt*k+row ], 
gl_Ah[ mt*j+row ]);
}
int i;
for( i = row+1; i < mt; i++ ) {
NoFLA_Compute_td_QR_var31a( bs, tr, tc, rskip, 
gl_Ah[ mt*k+row ], 
gl_Ah[ mt*k+i ], 
gl_Th[ mt*k+i ],
gl_Sh[ mt*k+i ]  );
#pragma omp taskwait
int j;
for( j = k+1; j < nt; j++ ) {
NoFLA_Apply_td_QT_var31a( bs, tr, tc, rskip, 
gl_Ah[ mt*k+i ], 
gl_Sh[ mt*k+i ], 
gl_Ah[ mt*j+row ], 
gl_Ah[ mt*j+i ]);
}
}
rk+=tc;
}
}
