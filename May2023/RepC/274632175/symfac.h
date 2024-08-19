#ifndef __SYMFAC_H__
#define __SYMFAC_H__


#include "hb.h"
#include "vector.h"
#include "fptype.h"

#ifdef SINGLE_PRECISION

#define symbolic_csr_task			symbolic_csr_task_single
#define ereach_csr					ereach_csr_single
#define ereach_csr_p				ereach_csr_p_single
#define hb2hbh_sym_etree_csr		hb2hbh_sym_etree_csr_single
#define hb2hbh_sym_etree_csr_p		hb2hbh_sym_etree_csr_p_single
#define hyper_sym_csr_task0			hyper_sym_csr_task0_single
#define hyper_sym_csr_task1			hyper_sym_csr_task1_single
#define hyper_sym_csr_task2			hyper_sym_csr_task2_single
#define hb2hbh_sym_etree			hb2hbh_sym_etree_single
#define hbh2hb_sym					hbh2hb_sym_single

#else

#define symbolic_csr_task			symbolic_csr_task_double
#define ereach_csr					ereach_csr_double
#define ereach_csr_p				ereach_csr_p_double
#define hb2hbh_sym_etree_csr		hb2hbh_sym_etree_csr_double
#define hb2hbh_sym_etree_csr_p		hb2hbh_sym_etree_csr_p_double
#define hyper_sym_csr_task0			hyper_sym_csr_task0_double
#define hyper_sym_csr_task1			hyper_sym_csr_task1_double
#define hyper_sym_csr_task2			hyper_sym_csr_task2_double
#define hb2hbh_sym_etree			hb2hbh_sym_etree_double
#define hbh2hb_sym					hbh2hb_sym_double

#endif

#pragma omp task in([1]A) out([1]entry, [1]block)
void symbolic_csr_task_single(int I, int J, hbmat_t *A, int b, int *etree, int *entry, hbmat_t *block);

void ereach_csr_single(hbmat_t *A, int r, int *etree, vector_t* sub_row);
hbmat_t* hb2hbh_sym_etree_csr_single(hbmat_t *A, int b, int* etree);
void ereach_csr_p_single(hbmat_t *A, int r, int *etree, vector_t *sub_row, vector_t *sub_val);
hbmat_t* hb2hbh_sym_etree_csr_p_single(hbmat_t *A, int b, int *etree);
void hyper_sym_csr_task0_single(int I, int J, hbmat_t *A, int b, int *etree, int *entry);
void hyper_sym_csr_task1_single(hbmat_t *block);
void hyper_sym_csr_task2_single(hbmat_t *block);
hbmat_t* hb2hbh_sym_etree_single(hbmat_t *A, int b, int* etree);
hbmat_t *hbh2hb_sym_single(hbmat_t *A);

#pragma omp task in([1]A) out([1]entry, [1]block)
void symbolic_csr_task_double(int I, int J, hbmat_t *A, int b, int *etree, int *entry, hbmat_t *block);

void ereach_csr_double(hbmat_t *A, int r, int *etree, vector_t* sub_row);
hbmat_t* hb2hbh_sym_etree_csr_double(hbmat_t *A, int b, int* etree);
void ereach_csr_p_double(hbmat_t *A, int r, int *etree, vector_t *sub_row, vector_t *sub_val);
hbmat_t* hb2hbh_sym_etree_csr_p_double(hbmat_t *A, int b, int *etree);
void hyper_sym_csr_task0_double(int I, int J, hbmat_t *A, int b, int *etree, int *entry);
void hyper_sym_csr_task1_double(hbmat_t *block);
void hyper_sym_csr_task2_double(hbmat_t *block);
hbmat_t* hb2hbh_sym_etree_double(hbmat_t *A, int b, int* etree);
hbmat_t *hbh2hb_sym_double(hbmat_t *A);


#endif 
