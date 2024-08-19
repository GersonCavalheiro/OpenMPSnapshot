




















#include <stdio.h>
#include <nanos.h>
#include <alloca.h>

#define NUM_ITERS      100
#define VECTOR_SIZE    1000
#define USE_COMPOUND_WD


typedef struct { int *M; } main__section_1_data_t;
void main__section_1 ( void *p_args );
void main__section_1 ( void *p_args )
{
int i;
main__section_1_data_t *args = (main__section_1_data_t *) p_args;
for ( i = 0; i < VECTOR_SIZE; i++) args->M[i]++;
}
nanos_smp_args_t main__section_1_device_args = { main__section_1 };


typedef struct { int *M; } main__section_2_data_t;
void main__section_2 ( void *p_args );
void main__section_2 ( void *p_args )
{
int i;
main__section_2_data_t *args = (main__section_2_data_t *) p_args;
for ( i = 0; i < VECTOR_SIZE; i++) args->M[i]++;
}
nanos_smp_args_t main__section_2_device_args = { main__section_2 };


typedef struct { int *M; } main__section_3_data_t;
void main__section_3 ( void *p_args );
void main__section_3 ( void *p_args )
{
int i;
main__section_3_data_t *args = (main__section_3_data_t *) p_args;
for ( i = 0; i < VECTOR_SIZE; i++) args->M[i]++;
}
nanos_smp_args_t main__section_3_device_args = { main__section_3 };


typedef struct { int *M; } main__section_4_data_t;
void main__section_4 ( void *p_args );
void main__section_4 ( void *p_args )
{
int i;
main__section_4_data_t *args = (main__section_4_data_t *) p_args;
for ( i = 0; i < VECTOR_SIZE; i++) args->M[i]++;
}
nanos_smp_args_t main__section_4_device_args = { main__section_4 };


void main__sections ( void *p_args );
void main__sections ( void *p_args ) { fprintf(stderr,"es\n"); }




struct nanos_const_wd_definition_1
{
nanos_const_wd_definition_t base;
nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data1 = 
{
{{
.mandatory_creation = true,
.tied = false},
0,
0,
1, 0,NULL},
{
{
nanos_smp_factory,
&main__section_1_device_args
}
}
};

struct nanos_const_wd_definition_1 const_data2 = 
{
{
{ .mandatory_creation = true, .tied = false}, 
0, 
0, 
1, 
0, 
NULL 
},
{
{
nanos_smp_factory,
&main__section_2_device_args
}
}
};
struct nanos_const_wd_definition_1 const_data3 = 
{
{{
.mandatory_creation = true,
.tied = false},
0,
0,
1, 0, NULL},
{
{
nanos_smp_factory,
&main__section_3_device_args
}
}
};
struct nanos_const_wd_definition_1 const_data4 = 
{
{{
.mandatory_creation = true,
.tied = false},
0,
0,
1,0,NULL},
{
{
nanos_smp_factory,
&main__section_4_device_args
}
}
};

int main ( int argc, char **argv )
{
int i;
bool check = true; 
int *A, *B, *C, *D;

A = (int *) alloca(sizeof(int)*VECTOR_SIZE);
B = (int *) alloca(sizeof(int)*VECTOR_SIZE);
C = (int *) alloca(sizeof(int)*VECTOR_SIZE);
D = (int *) alloca(sizeof(int)*VECTOR_SIZE);

for (i = 0; i < VECTOR_SIZE; i++) {
A[i] = 0; B[i] = 0; C[i] = 0; D[i] = 0;
}

for ( i = 0; i < NUM_ITERS; i++ ) {


nanos_wd_t wd[4] = { NULL, NULL, NULL, NULL };


main__section_1_data_t *section_data_1 = NULL;
const_data1.base.data_alignment = __alignof__(section_data_1);
nanos_wd_dyn_props_t dyn_props = {0};
NANOS_SAFE( nanos_create_wd_compact ( &wd[0], &const_data1.base, &dyn_props, sizeof(section_data_1), (void **) &section_data_1,
nanos_current_wd(), NULL, NULL ) );

section_data_1->M = A;


main__section_2_data_t *section_data_2 = NULL;
const_data2.base.data_alignment = __alignof__(section_data_2);
NANOS_SAFE( nanos_create_wd_compact ( &wd[1], &const_data2.base, &dyn_props, sizeof(section_data_2), (void **) &section_data_2,
nanos_current_wd(), NULL, NULL ) );

section_data_2->M = B;


main__section_3_data_t *section_data_3 = NULL;
const_data3.base.data_alignment = __alignof__(section_data_3);
NANOS_SAFE( nanos_create_wd_compact ( &wd[2], &const_data3.base, &dyn_props, sizeof(section_data_3), (void **) &section_data_3,
nanos_current_wd(), NULL, NULL ) );

section_data_3->M = C;


main__section_4_data_t *section_data_4 = NULL;
const_data4.base.data_alignment = __alignof__(section_data_4);
NANOS_SAFE( nanos_create_wd_compact ( &wd[3], &const_data4.base, &dyn_props, sizeof(section_data_4), (void **) &section_data_4,
nanos_current_wd(), NULL, NULL ) );


section_data_4->M = D;

#ifdef USE_COMPOUND_WD
nanos_slicer_t slicer = nanos_find_slicer("compound_wd");

nanos_wd_t cwd = NULL;



void * compound_f;

nanos_slicer_get_specific_data ( slicer, &compound_f );
nanos_smp_args_t main__sections_device_args = { compound_f };
nanos_device_t main__sections_device[1] = { NANOS_SMP_DESC( main__sections_device_args ) };

nanos_compound_wd_data_t *list_of_wds = NULL;


NANOS_SAFE( nanos_create_sliced_wd ( &cwd, 1, main__sections_device,
sizeof(nanos_compound_wd_data_t) + (4) * sizeof(nanos_wd_t), __alignof__(nanos_compound_wd_data_t),
(void **) &list_of_wds, nanos_current_wd(), slicer, &const_data1.base.props , &dyn_props, 0, NULL, 0, NULL ) );


list_of_wds->nsect = 4;
list_of_wds->lwd[0] = wd[0];
list_of_wds->lwd[1] = wd[1];
list_of_wds->lwd[2] = wd[2];
list_of_wds->lwd[3] = wd[3];

NANOS_SAFE( nanos_submit( cwd,0,0,0 ) );

#else 
NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );
NANOS_SAFE( nanos_submit( wd[1],0,0,0 ) );
NANOS_SAFE( nanos_submit( wd[2],0,0,0 ) );
NANOS_SAFE( nanos_submit( wd[3],0,0,0 ) );
#endif

NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
}


for (i = 0; i < VECTOR_SIZE; i++) {
if ( A[i] != NUM_ITERS) check = false;
if ( B[i] != NUM_ITERS) check = false;
if ( C[i] != NUM_ITERS) check = false;
if ( D[i] != NUM_ITERS) check = false;
}

fprintf(stderr, "%s : %s\n", argv[0], check ? "  successful" : "unsuccessful");
if (check) { return 0; } else { return -1; }
}

