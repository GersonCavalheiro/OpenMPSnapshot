#include "config.h"
#include "ltb.h"
int tB_norm=1;
unsigned int chunk_size=1;
double Ricci_fun(double E, double E_r, double R, double R_r)
{
return -4.0*( E_r/(R*R_r) + E/(R*R) );
}
int ltb_model_Ricci(Ltb_model *ltb, gsl_vector *E_r, gsl_matrix *R_r,gsl_matrix *Ric)
{
CHECK_PTR(ltb,GSL_EFAULT);
CHECK_PTR(E_r,GSL_EFAULT);
CHECK_PTR(R_r,GSL_EFAULT);
CHECK_PTR(Ric,GSL_EFAULT);
CHECK_M_DIM(Ric,ltb->R_mat,"Ricci matrix and R matrix have diffrent dimensions");
CHECK_M_DIM(Ric,R_r,"Ricci matrix and R_r matrix have diffrent dimensions");
CHECK_SIZE(Ric->size2,E_r->size,"Ricci matrix and E_r vector have diffrent dimensions");
unsigned int i=0;
unsigned int j=0;
for(i=0;i<Ric->size1;++i)
{
for(j=0;j<Ric->size2;++j)
{
gsl_matrix_set(Ric,i,j,
Ricci_fun(ltb->E_vec->data[j],
E_r->data[j],
gsl_matrix_get(ltb->R_mat,i,j),
gsl_matrix_get(R_r,i,j))
);
}
}
return GSL_SUCCESS;
}
double rho_fun(double M_r, double R, double R_r)
{
return M_r/(4 * pi * R*R * R_r);
}
int ltb_model_rho(Ltb_model *ltb, gsl_vector *M_r, gsl_matrix *R_r, gsl_matrix *rho_mat)
{
CHECK_PTR(ltb,GSL_EFAULT);
CHECK_PTR(M_r,GSL_EFAULT);
CHECK_PTR(R_r,GSL_EFAULT);
CHECK_PTR(rho_mat,GSL_EFAULT);
CHECK_M_DIM(rho_mat,ltb->R_mat,"Rho matrix and R matrix have diffrent dimensions");
CHECK_M_DIM(rho_mat,R_r,"Ricci matrix and R_r matrix have diffrent dimensions");
CHECK_SIZE(rho_mat->size2,M_r->size,"Rho matrix and M_r vector have diffrent dimensions");
unsigned int i=0;
unsigned int j=0;
for(i=0;i<rho_mat->size1;++i)
{
for(j=0;j<rho_mat->size2;++j)
{
gsl_matrix_set(rho_mat,i,j,
rho_fun(gsl_vector_get(M_r,j),
gsl_matrix_get(ltb->R_mat,i,j),
gsl_matrix_get(R_r,i,j))
);
}
}
return GSL_SUCCESS;
}
double grr_sq_fun(double R_r, double E)
{
if (E < -0.5)
{
GSL_ERROR_VAL("E < -0.5 !",GSL_EINVAL,GSL_NAN);
}
return sqrt((R_r*R_r)/(1 + 2.0*E));
}
int ltb_model_grr_sq(gsl_vector *E, gsl_matrix *R_r, gsl_matrix *grr_sq)
{
CHECK_PTR(grr_sq,GSL_EFAULT);
CHECK_PTR(R_r,GSL_EFAULT);
CHECK_PTR(E,GSL_EFAULT);
CHECK_M_DIM(grr_sq,R_r,"grr_sq matrix and R_r matrix have diffrent dimensions");
CHECK_SIZE(grr_sq->size2,E->size,"grr_sq matrix and E vector have diffrent dimensions");
unsigned int i=0;
unsigned int j=0;
for(i=0;i<grr_sq->size1;++i)
{
for(j=0;j<grr_sq->size2;++j)
{
gsl_matrix_set(grr_sq,i,j,
grr_sq_fun(gsl_matrix_get(R_r,i,j),
E->data[j])
);
}
}
return GSL_SUCCESS;
}
double exp_fun(double R, double R_r, double R_t, double R_rt)
{
return ( (2.0*R_t/R) + (R_rt/R_r) )/3.0;
}
int ltb_model_exp(Ltb_model *ltb, gsl_matrix *R_r,
gsl_matrix *R_t, gsl_matrix *R_rt, gsl_matrix *exp)
{
CHECK_PTR(ltb,GSL_EFAULT);
CHECK_PTR(R_r,GSL_EFAULT);
CHECK_PTR(R_t,GSL_EFAULT);
CHECK_PTR(R_rt,GSL_EFAULT);
CHECK_PTR(exp,GSL_EFAULT);
CHECK_M_DIM(exp,ltb->R_mat,"exp matrix and R matrix have diffrent dimensions");
CHECK_M_DIM(exp,R_r,"exp matrix and R_r matrix have diffrent dimensions");
CHECK_M_DIM(exp,R_t,"exp matrix and R_t matrix have diffrent dimensions");
CHECK_M_DIM(exp,R_rt,"exp matrix and R_rt matrix have diffrent dimensions");
unsigned int i=0;
unsigned int j=0;
for(i=0;i<exp->size1;++i)
{
for(j=0;j<exp->size2;++j)
{
gsl_matrix_set(exp,i,j,
exp_fun(gsl_matrix_get(ltb->R_mat,i,j),
gsl_matrix_get(R_r,i,j),
gsl_matrix_get(R_t,i,j),
gsl_matrix_get(R_rt,i,j))
);
}
}
return GSL_SUCCESS;
}
int set_cur(const gsl_vector *E_vec, int *cur)
{
*cur = cur_fun(E_vec->data[0]);
unsigned int i = 0;
for(i=0;i<E_vec->size;++i)
{
if (cur_fun(E_vec->data[i]) != *cur)
{
*cur=-999;
return LTB_ECURV;
}
}
return GSL_SUCCESS;
}
int cur_fun(double E)
{
int cur = 0;
if (E>0)
{
cur = -1;
}
else if (E<0)
{
cur = 1;
}
return cur;
}
int get_chi(int cur, gsl_vector *E, gsl_vector *chi_vec)
{
CHECK_SIZE(chi_vec->size,E->size,"chi vector and E vector have diffrent dimensions");
if (cur == 0)
{
gsl_vector_set_all(chi_vec,1.0);
}
else
{
gsl_vector_memcpy(chi_vec,E);
if (cur == 1)
{
gsl_vector_scale(chi_vec,-2.0);
}
else if (cur == -1)
{
gsl_vector_scale(chi_vec,2.0);
}
}
return GSL_SUCCESS;
}
double xi_fun(double tB, double M, double E, double t, int cur)
{
if ( M == 0.0)
{
GSL_ERROR_VAL("M can't be 0.0 !",GSL_EINVAL,GSL_NAN);
}
double xi=0;
xi = (t - tB) / (G_grav*M);
if (cur == 1)
{
xi *= pow(-2*E,1.5);
}
else if (cur  == -1)
{
xi *= pow(2*E,1.5);
}
return xi;
}
int get_xi(gsl_vector *tB,gsl_vector *M, gsl_vector *t, gsl_vector *E,
int cur,gsl_matrix *xi_mat)
{
CHECK_PTR(tB,GSL_EFAULT);
CHECK_PTR(M,GSL_EFAULT);
CHECK_PTR(t,GSL_EFAULT);
CHECK_PTR(E,GSL_EFAULT);
CHECK_PTR(xi_mat,GSL_EFAULT);
if (xi_mat->size2 != tB->size || xi_mat->size2 != M->size ||
xi_mat->size2 != E->size || xi_mat->size1 != t->size)
{
return GSL_EBADLEN;
}
unsigned int i=0;
unsigned int j=0;
#pragma omp parallel default(shared) private(i,j)
{
#pragma omp for schedule(static, chunk_size)
for (i=0;i<t->size;i++)
{
for (j=0;j<tB->size;j++)
{
gsl_matrix_set(xi_mat,i,j,
xi_fun(gsl_vector_get(tB,j),
gsl_vector_get(M,j),
gsl_vector_get(E,j),
gsl_vector_get(t,i),
cur)
);
}
}
}
return GSL_SUCCESS;
}
double fiSpherical(double eta)
{
return 1.0 - cos(eta);
}
double fiFlat(double eta)
{
return eta*eta / 2.0;
}
double fiHyperbolic(double eta)
{
return cosh(eta) - 1.0;
}
double fi_fun_from_eta(double eta, int cur)
{
if (cur == 0)
{
return fiFlat(eta);
}
else if (cur == 1)
{
return fiSpherical(eta);
}
else if (cur == -1)
{
return fiHyperbolic(eta);
}
else
{
printf("Wrong value of cur=%d\n",cur);
return GSL_EINVAL;
}
}
double xi_fun_from_eta(double eta, int cur)
{
if (cur == 0)
{
return xiFlat(eta);
}
else if (cur == 1)
{
return xiSpherical(eta);
}
else if (cur == -1)
{
return xiHyperbolic(eta);
}
else
{
GSL_ERROR("Wrong value of cur",LTB_ECURV);
return GSL_NAN;
}
}
double xiFlat(double eta)
{
return eta*eta*eta / 6.0;
}
double xiHyperbolic(double eta)
{
return sinh(eta) - eta;
}
double xiSpherical(double eta)
{
return eta - sin(eta);
}
int get_min_max_for_interp(const Ltb_model *model, double *max, double *min)
{
unsigned int i,j;
double max_t = model->t_vec->data[0] - model->tB_vec->data[0];
double min_t = model->t_vec->data[0] - model->tB_vec->data[0];
double tmp=0;
for (i=0;i<model->t_vec->size;++i)
{
for (j=0;j<model->tB_vec->size;++j)
{
tmp = model->t_vec->data[i] - model->tB_vec->data[j];
if (tmp > max_t)
max_t = tmp;
if (tmp < min_t)
min_t = tmp;
}
}
double M_max = gsl_vector_max(model->M_vec);
double M_min = gsl_vector_min(model->M_vec);
double chi_max = 1;
double chi_min = 1;
if (model->cur == 0)
{
chi_max = 1;
chi_min = 1;
}
else
{
if (model->cur > 0)
{
chi_max = -2.0 * gsl_vector_min(model->E_vec);
chi_min = -2.0 * gsl_vector_max(model->E_vec);
}
else
{
chi_max = 2.0 * gsl_vector_max(model->E_vec);
chi_min = 2.0 * gsl_vector_min(model->E_vec);
}
chi_max = pow(chi_max,1.5);
chi_min = pow(chi_min,1.5);
}
*max = max_t*chi_max / (G_grav*M_min);
*min = min_t*chi_min / (G_grav*M_max);
return GSL_SUCCESS;
}
Interp_data* interp_data_alloc_for_ltb(const Ltb_model *model,  size_t n)
{
if (!model)
{
GSL_ERROR_VAL("model pointer is NULL",GSL_EFAULT,0);
}
Interp_data *idata=NULL;
double max,min;
get_min_max_for_interp(model,&max,&min);
max = set_eta_max(max);
min = set_eta_min(min);
idata = interp_data_alloc(max,min,n,model->cur,&xi_fun_from_eta);
if (!idata)
{
GSL_ERROR_VAL("idata is NULL",GSL_EINVAL,0);
}
return idata;
}
double eta_fun(double xi,int cur, Interp_data *eta_interp)
{
double eta=0;
if (cur == 0)
{
eta=cbrt(6.0*xi);
}
else
{
if (eta_interp == NULL)
{
GSL_ERROR_VAL("Interp_data is NULL",GSL_EFAULT,GSL_NAN);
}
else
{
gsl_interp_eval_e(eta_interp->spline,
eta_interp->y,
eta_interp->x,
xi,
eta_interp->acc,
&eta);
}
}
return eta;
}
double R_fun(double M, double E, double eta, int cur)
{
double R=0.0;
R = G_grav*M*fi_fun_from_eta(eta,cur);
if (cur == 1)
{
R = R / (-2.0*E);
}
else if (cur == -1)
{
R = R / (2.0*E);
}
return R;
}
int ltb_model_set_R(Ltb_model *model, Interp_data *eta_interp)
{
if (model->cur != 0 && eta_interp == NULL)
{
GSL_ERROR("eta_interp is NULL",GSL_EFAULT);
}
CHECK_PTR(model,GSL_EFAULT);
if (model->R_mat == NULL)
{
model->R_mat = gsl_matrix_alloc(model->t_size,model->r_size);
CHECK_PTR(model->R_mat,GSL_ENOMEM);
}
unsigned i,j;
double eta=0;
double xi=0;
#pragma omp parallel default(shared) private(i,j,eta,xi)
{
for (i=0;i<model->t_size;++i)
{
#pragma omp for schedule(dynamic, chunk_size)
for (j=0;j<model->r_size;++j)
{
xi = xi_fun(gsl_vector_get(model->tB_vec,j),
gsl_vector_get(model->M_vec,j),
gsl_vector_get(model->E_vec,j),
gsl_vector_get(model->t_vec,i),
model->cur);
eta = eta_fun(xi,model->cur,eta_interp);
gsl_matrix_set(model->R_mat,i,j,
R_fun(gsl_vector_get(model->M_vec,j),
gsl_vector_get(model->E_vec,j),
eta,
model->cur
)
);
}
}
}
return GSL_SUCCESS;
}
Ltb_model* ltb_model_alloc(size_t r_size, size_t t_size,
double (*M)(double),double (*E)(double),
double (*tB)(double))
{
if (r_size == 0 || t_size == 0)
{
GSL_ERROR_VAL("ltb_model r_size=0 or(and) t_size=0",GSL_EINVAL,0);
}
Ltb_model *ret = (Ltb_model*)malloc(sizeof(Ltb_model));
if (!ret)
{
GSL_ERROR_VAL("ltb_model not allocated",GSL_ENOMEM,0);
}
ret->M_fun=M;
ret->E_fun=E;
ret->tB_fun=tB;
ret->tB_max = GSL_NAN;
ret->cur = -999;
ret->M_vec = NULL;
ret->tB_vec = NULL;
ret->E_vec = NULL;
ret->R_mat = NULL;
ret->r_size = r_size;
ret->t_size = t_size;
ret->r_vec = gsl_vector_alloc(r_size);
if (!ret->r_vec)
{
ltb_model_free(ret);
GSL_ERROR_VAL("r_vec not allocated",GSL_ENOMEM,0);
}
ret->t_vec = gsl_vector_alloc(t_size);
if(!ret->t_vec)
{
ltb_model_free(ret);
GSL_ERROR_VAL("t_vec not allocated",GSL_ENOMEM,0);
}
ret->M_vec =  gsl_vector_alloc(r_size);
if(!ret->M_vec)
{
ltb_model_free(ret);
GSL_ERROR_VAL("M_vec not allocated",GSL_ENOMEM,0);
}
ret->tB_vec = gsl_vector_alloc(r_size);
if(!ret->tB_vec)
{
ltb_model_free(ret);
GSL_ERROR_VAL("tB_vec not allocated",GSL_ENOMEM,0);
}
ret->E_vec = gsl_vector_alloc(r_size);
if(!ret->E_vec)
{
ltb_model_free(ret);
GSL_ERROR_VAL("E_vec not allocated",GSL_ENOMEM,0);
}
ret->R_mat = gsl_matrix_alloc(t_size,r_size);
if(!ret->R_mat)
{
ltb_model_free(ret);
GSL_ERROR_VAL("R_mat not allocated",GSL_ENOMEM,0);
}
return ret;
}
Ltb_model* ltb_model_alloc_from_vec(gsl_vector *r_vec, gsl_vector *t_vec,
double (*M)(double),double (*E)(double),
double (*tB)(double))
{
int status = GSL_SUCCESS;
if (!r_vec)
{
GSL_ERROR_VAL("r_vec is null",GSL_EFAULT,0);
}
if (!t_vec)
{
GSL_ERROR_VAL("r_vec is null",GSL_EFAULT,0);
}
Ltb_model *ret = ltb_model_alloc(r_vec->size,t_vec->size,M,E,tB);
if (!ret)
{
GSL_ERROR_VAL("allocation error",GSL_EFAULT,0);
}
if (gsl_vector_memcpy(ret->r_vec,r_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy r_vec",GSL_EFAILED,0);
}
if (gsl_vector_memcpy(ret->t_vec,t_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy t_vec",GSL_EFAILED,0);
}
return ret;
}
Ltb_model* ltb_model_cpy(Ltb_model *src)
{
if(src == NULL)
{
GSL_ERROR_VAL("src pointer is null",GSL_ENOMEM,0);
}
Ltb_model *ret = ltb_model_alloc(src->r_size,src->t_size,src->M_fun,src->E_fun,src->tB_fun);
ret->R_mat = NULL;
ret->cur = src->cur;
ret->tB_max = src->tB_max;
if(gsl_vector_memcpy(ret->r_vec,src->r_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy src->r_vec",GSL_EFAILED,0);
}
if(gsl_vector_memcpy(ret->t_vec,src->t_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy src->t_vec",GSL_EFAILED,0);
}
if(gsl_vector_memcpy(ret->M_vec,src->M_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy src->M_vec",GSL_EFAILED,0);
}
if(gsl_vector_memcpy(ret->E_vec,src->E_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy src->E_vec",GSL_EFAILED,0);
}
if(gsl_vector_memcpy(ret->tB_vec,src->tB_vec))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy src->tB_vec",GSL_EFAILED,0);
}
if (src->R_mat)
{
if(gsl_matrix_memcpy(ret->R_mat,src->R_mat))
{
ltb_model_free(ret);
GSL_ERROR_VAL("Can't copy src->R_mat",GSL_EFAILED,0);
}
}
return ret;
}
void ltb_model_free(Ltb_model *model)
{
if (!model)
return;
if (model->r_vec != NULL)
{
gsl_vector_free(model->r_vec);
model->r_vec = NULL;
}
if (model->t_vec != NULL)
{
gsl_vector_free(model->t_vec);
model->t_vec = NULL;
}
if (model->M_vec != NULL)
{
gsl_vector_free(model->M_vec);
model->M_vec = NULL;
}
if (model->tB_vec != NULL)
{
gsl_vector_free(model->tB_vec);
model->tB_vec = NULL;
}
if (model->E_vec != NULL)
{
gsl_vector_free(model->E_vec);
model->E_vec = NULL;
}
if (model->R_mat != NULL)
{
gsl_matrix_free(model->R_mat);
model->R_mat = NULL;
}
model->M_fun = NULL;
model->tB_fun = NULL;
model->E_fun = NULL;
free(model);
}
int ltb_model_set_fun(Ltb_model *model)
{
if (!model->M_fun)
{
GSL_ERROR("Null pointer on M_fun",GSL_EFAULT);
}
if (!model->E_fun)
{
GSL_ERROR("Null pointer on E_fun",GSL_EFAULT);
}
if (!model->tB_fun)
{
GSL_ERROR("Null pointer on tB_fun",GSL_EFAULT);
}
int status=GSL_SUCCESS;
if (model->M_vec == NULL)
{
model->M_vec=gsl_vector_alloc(model->r_size);
CHECK_PTR(model->M_vec,GSL_ENOMEM);
}
if (model->tB_vec == NULL)
{
model->tB_vec=gsl_vector_alloc(model->r_size);
CHECK_PTR(model->tB_vec,GSL_ENOMEM);
}
if (model->E_vec == NULL)
{
model->E_vec=gsl_vector_alloc(model->r_size);
CHECK_PTR(model->E_vec,GSL_ENOMEM);
}
status = gsl_vector_gen_fun(model->M_vec,model->r_size,model->r_vec,model->M_fun);
CHECK_STAT(status);
status = gsl_vector_gen_fun(model->tB_vec,model->r_size,model->r_vec,model->tB_fun);
CHECK_STAT(status);
if (!status)
{
model->tB_max = gsl_vector_max(model->tB_vec);
if (tB_norm)
{
gsl_vector_add_constant(model->tB_vec,-model->tB_max);
}
}
status = gsl_vector_gen_fun(model->E_vec,model->r_size,model->r_vec,model->E_fun);
CHECK_STAT(status);
status = set_cur(model->E_vec,&(model->cur));
CHECK_STAT(status);
return status;
}
int ltb_model_write_fun(const Ltb_model *model, char* file_name)
{
int status=GSL_SUCCESS;
CHECK_PTR(model,GSL_EFAULT);
CHECK_PTR(model->r_vec,GSL_EFAULT);
CHECK_PTR(model->M_vec,GSL_EFAULT);
CHECK_PTR(model->tB_vec,GSL_EFAULT);
CHECK_PTR(model->E_vec,GSL_EFAULT);
gsl_matrix *mat = gsl_matrix_alloc(model->r_size,4);
CHECK_PTR(mat,GSL_ENOMEM);
gsl_matrix_set_col(mat,0,model->r_vec);
gsl_matrix_set_col(mat,1,model->M_vec);
gsl_matrix_set_col(mat,2,model->tB_vec);
gsl_matrix_set_col(mat,3,model->E_vec);
status=gsl_write_mat(mat,file_name,"# r M tB E\n");
CHECK_STAT(status);
gsl_matrix_free(mat);
return GSL_SUCCESS;
}
double R_t_analytic_fun(double R, double M, double E,int sgn)
{
return sgn*sqrt( (2*G_grav*M/R) + 2*E);
}
static int check_slope_R_t(double r, double t, double t_val, const Ltb_model *model, Interp_data *eta_interp)
{
double E = model->E_fun(r);
double M = model->M_fun(r);
double tB = model->tB_fun(r);
if (tB_norm)
{
tB = tB - model->tB_max;
}
int cur = 0;
if (E < 0)
{
cur = 1;
}
else if (E > 0)
{
cur = -1;
}
double xi = xi_fun(tB,M,E,t,cur);
double eta = eta_fun(xi,cur,eta_interp);
double slope_val = R_fun(M,E,eta,cur);
if (slope_val < t_val)
{
return -1;
}
else
{
return 1;
}
}
int ltb_model_R_t(const Ltb_model *model, Interp_data *eta_interp, gsl_matrix *R_t)
{
CHECK_PTR(model,GSL_EFAULT);
CHECK_PTR(R_t,GSL_EFAULT);
CHECK_M_DIM(model->R_mat,R_t,"R_mat matrix and R_t matrix have diffrent dimensions");
unsigned int i,j;
double slope=1e-5;
double t_val=0.0;
int sgn=1;
for (i=0;i<model->t_size;++i)
{
t_val = gsl_matrix_get(model->R_mat,i,0);
sgn=check_slope_R_t(model->r_vec->data[0],
model->t_vec->data[i]+slope,
t_val,
model,
eta_interp);
for (j=0;j<model->r_size;++j)
{
gsl_matrix_set(R_t,i,j,
R_t_analytic_fun(gsl_matrix_get(model->R_mat,i,j),
model->M_vec->data[j],
model->E_vec->data[j],
sgn)
);
}
}
return GSL_SUCCESS;
}
double R_r_analytic_fun(double R,double R_t,double E,double E_r,
double M,double M_r,double tB,double tB_r,double t)
{
double M_rM = M_r/M;
double E_rE;
if (E != 0.0)
{
E_rE = E_r/E;
}
else
{
E_rE = 0.0;
}
return (M_rM - E_rE)*R - (tB_r + (M_rM - 1.5*E_rE)*(t-tB))*R_t;
}
int ltb_model_R_r(const Ltb_model *model,const gsl_matrix *R_t,
const gsl_vector *E_r_vec, const gsl_vector *M_r_vec,
const gsl_vector *tB_r_vec, gsl_matrix *R_r)
{
CHECK_PTR(model,GSL_EFAULT);
CHECK_PTR(R_t,GSL_EFAULT);
CHECK_PTR(E_r_vec,GSL_EFAULT);
CHECK_PTR(M_r_vec,GSL_EFAULT);
CHECK_PTR(tB_r_vec,GSL_EFAULT);
CHECK_PTR(R_r,GSL_EFAULT);
CHECK_M_DIM(model->R_mat,R_r,"R_mat matrix and R_r matrix have diffrent dimensions");
CHECK_M_DIM(R_t,R_r,"R_t matrix and R_r matrix have diffrent dimensions");
CHECK_SIZE(E_r_vec->size,R_t->size2,"E_r vector and R_r matrix have diffrent dimensions");
CHECK_SIZE(M_r_vec->size,R_t->size2,"M_r vector and R_r matrix have diffrent dimensions");
CHECK_SIZE(tB_r_vec->size,R_t->size2,"tB_r vector and R_r matrix have diffrent dimensions");
unsigned i,j;
for (i=0;i<model->t_size;++i)
{
for (j=0;j<model->r_size;++j)
{
gsl_matrix_set(R_r,i,j,
R_r_analytic_fun(gsl_matrix_get(model->R_mat,i,j),
gsl_matrix_get(R_t,i,j),
model->E_vec->data[j],
E_r_vec->data[j],
model->M_vec->data[j],
M_r_vec->data[j],
model->tB_vec->data[j],
tB_r_vec->data[j],
model->t_vec->data[i])
);
}
}
return GSL_SUCCESS;
}
double R_rt_analytic_fun(double R, double R_r, double R_t, double E_r,double M, double M_r)
{
return ( G_grav*( M_r/R - M*R_r/(R*R) ) + E_r)/R_t;
}
int ltb_model_R_rt(const Ltb_model *model,
const gsl_matrix *R_r, const gsl_matrix *R_t,
const gsl_vector *E_r, const gsl_vector *M_r,
gsl_matrix *R_rt)
{
CHECK_PTR(model,GSL_EFAULT);
CHECK_PTR(R_r,GSL_EFAULT);
CHECK_PTR(R_t,GSL_EFAULT);
CHECK_PTR(R_rt,GSL_EFAULT);
CHECK_PTR(E_r,GSL_EFAULT);
CHECK_PTR(M_r,GSL_EFAULT);
CHECK_M_DIM(model->R_mat,R_rt,"R_mat matrix and R_rt matrix have diffrent dimensions");
CHECK_M_DIM(R_r,R_rt,"R_r matrix and R_rt matrix have diffrent dimensions");
CHECK_M_DIM(R_t,R_rt,"R_t matrix and R_rt matrix have diffrent dimensions");
CHECK_SIZE(E_r->size,R_t->size2,"E_r vector and R_rt matrix have diffrent dimensions");
CHECK_SIZE(M_r->size,R_t->size2,"M_r vector and R_rt matrix have diffrent dimensions");
unsigned i,j;
for (i=0;i<model->t_size;++i)
{
for (j=0;j<model->r_size;++j)
{
gsl_matrix_set(R_rt,i,j,
R_rt_analytic_fun(gsl_matrix_get(model->R_mat,i,j),
gsl_matrix_get(R_r,i,j),
gsl_matrix_get(R_t,i,j),
E_r->data[j],
model->M_vec->data[j],
M_r->data[j])
);
}
}
return GSL_SUCCESS;
}
