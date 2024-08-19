#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <memory.h>
#include <sys/time.h>
#include <libgen.h>
#include "bots_common.h"
#include "bots_main.h"
#include "bots.h"
#include "app-desc.h"
int bots_sequential_flag = FALSE;
int bots_check_flag = FALSE;
bots_verbose_mode_t bots_verbose_mode = BOTS_VERBOSE_DEFAULT;
int bots_result = BOTS_RESULT_NOT_REQUESTED;
int bots_output_format = 1;
int bots_print_header = FALSE;
char bots_name[BOTS_TMP_STR_SZ];
char bots_execname[BOTS_TMP_STR_SZ];
char bots_parameters[BOTS_TMP_STR_SZ];
char bots_model[BOTS_TMP_STR_SZ];
char bots_resources[BOTS_TMP_STR_SZ];
char bots_exec_date[BOTS_TMP_STR_SZ];
char bots_exec_message[BOTS_TMP_STR_SZ];
char bots_comp_date[BOTS_TMP_STR_SZ];
char bots_comp_message[BOTS_TMP_STR_SZ];
char bots_cc[BOTS_TMP_STR_SZ];
char bots_cflags[BOTS_TMP_STR_SZ];
char bots_ld[BOTS_TMP_STR_SZ];
char bots_ldflags[BOTS_TMP_STR_SZ];
char bots_cutoff[BOTS_TMP_STR_SZ];
double bots_time_program = 0.0;
double bots_time_sequential = 0.0;
unsigned long long bots_number_of_tasks = 0; 
#ifndef BOTS_APP_NAME
#error "Application name must be defined (#define BOTS_APP_NAME)"
#endif
#ifndef BOTS_APP_PARAMETERS_DESC
#define BOTS_APP_PARAMETERS_DESC ""
#endif
#ifndef BOTS_APP_PARAMETERS_LIST
#define BOTS_APP_PARAMETERS_LIST
#endif
#ifndef BOTS_APP_INIT
#define BOTS_APP_INIT
#endif
#ifndef BOTS_APP_FINI
#define BOTS_APP_FINI
#endif
#ifndef KERNEL_CALL
#error "Initial kernell call must be specified (#define KERNEL_CALL)"
#endif
#ifndef KERNEL_INIT
#define KERNEL_INIT
#endif
#ifndef KERNEL_FINI
#define KERNEL_FINI
#endif
#ifndef KERNEL_SEQ_INIT
#define KERNEL_SEQ_INIT
#endif
#ifndef KERNEL_SEQ_FINI
#define KERNEL_SEQ_FINI
#endif
#ifndef BOTS_MODEL_DESC
#define BOTS_MODEL_DESC "Unknown"
#endif
#ifdef BOTS_APP_USES_ARG_SIZE
#ifndef BOTS_APP_DEF_ARG_SIZE
#error "Default vaule for argument size must be specified (#define BOTS_APP_DEF_ARG_SIZE)"
#endif
#ifndef BOTS_APP_DESC_ARG_SIZE
#error "Help description for argument size must be specified (#define BOTS_APP_DESC_ARG_SIZE)"
#endif
int bots_arg_size = BOTS_APP_DEF_ARG_SIZE;
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_1
#ifndef BOTS_APP_DEF_ARG_SIZE_1
#error "Default vaule for argument size must be specified (#define BOTS_APP_DEF_ARG_SIZE_1)"
#endif
#ifndef BOTS_APP_DESC_ARG_SIZE_1
#error "Help description for argument size must be specified (#define BOTS_APP_DESC_ARG_SIZE_1)"
#endif
int bots_arg_size_1 = BOTS_APP_DEF_ARG_SIZE_1;
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_2
#ifndef BOTS_APP_DEF_ARG_SIZE_2
#error "Default vaule for argument size must be specified (#define BOTS_APP_DEF_ARG_SIZE_2)"
#endif
#ifndef BOTS_APP_DESC_ARG_SIZE_2
#error "Help description for argument size must be specified (#define BOTS_APP_DESC_ARG_SIZE_2)"
#endif
int bots_arg_size_2 = BOTS_APP_DEF_ARG_SIZE_2;
#endif
#ifdef BOTS_APP_USES_ARG_REPETITIONS
#ifndef BOTS_APP_DEF_ARG_REPETITIONS
#error "Default vaule for argument repetitions must be specified (#define BOTS_APP_DEF_ARG_REPETITIONS)"
#endif
#ifndef BOTS_APP_DESC_ARG_REPETITIONS
#error "Help description for argument repetitions must be specified (#define BOTS_APP_DESC_ARG_REPETITIONS)"
#endif
int bots_arg_repetitions = BOTS_APP_DEF_ARG_REPETITIONS;
#endif
#ifdef BOTS_APP_USES_ARG_FILE
#ifndef BOTS_APP_DESC_ARG_FILE
#error "Help description for argument file must be specified (#define BOTS_APP_DESC_ARG_FILE)"
#endif
char bots_arg_file[255]="";
#endif
#ifdef BOTS_APP_USES_ARG_BLOCK
#ifndef BOTS_APP_DEF_ARG_BLOCK
#error "Default value for argument block must be specified (#define BOTS_APP_DEF_ARG_BLOCK)"
#endif
#ifndef BOTS_APP_DESC_ARG_BLOCK
#error "Help description for argument block must be specified (#define BOTS_APP_DESC_ARG_BLOCK)"
#endif
int bots_arg_block = BOTS_APP_DEF_ARG_BLOCK;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF
#ifndef BOTS_APP_DEF_ARG_CUTOFF
#error "Default value for argument cutoff  must be specified (#define BOTS_APP_DEF_ARG_CUTOFF)"
#endif
#ifndef BOTS_APP_DESC_ARG_CUTOFF
#error "Help description for argument cutoff must be specified (#define BOTS_APP_DESC_ARG_CUTOFF)"
#endif
int bots_app_cutoff_value = BOTS_APP_DEF_ARG_CUTOFF;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_1
#ifndef BOTS_APP_DEF_ARG_CUTOFF_1
#error "Default value for argument cutoff  must be specified (#define BOTS_APP_DEF_ARG_CUTOFF_1)"
#endif
#ifndef BOTS_APP_DESC_ARG_CUTOFF_1
#error "Help description for argument cutoff must be specified (#define BOTS_APP_DESC_ARG_CUTOFF_1)"
#endif
int bots_app_cutoff_value_1 = BOTS_APP_DEF_ARG_CUTOFF_1;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_2
#ifndef BOTS_APP_DEF_ARG_CUTOFF_2
#error "Default value for argument cutoff  must be specified (#define BOTS_APP_DEF_ARG_CUTOFF_2)"
#endif
#ifndef BOTS_APP_DESC_ARG_CUTOFF_2
#error "Help description for argument cutoff must be specified (#define BOTS_APP_DESC_ARG_CUTOFF_2)"
#endif
int bots_app_cutoff_value_2 = BOTS_APP_DEF_ARG_CUTOFF_2;
#endif
#if defined(MANUAL_CUTOFF) || defined(IF_CUTOFF) || defined(FINAL_CUTOFF)
int  bots_cutoff_value = BOTS_CUTOFF_DEF_VALUE;
#endif
void bots_print_usage()
{
fprintf(stderr, "\n");
fprintf(stderr, "Usage: %s -[options]\n", bots_execname);
fprintf(stderr, "\n");
fprintf(stderr, "Where options are:\n");
#ifdef BOTS_APP_USES_REPETITIONS
fprintf(stderr, "  -r <value> : Set the number of repetitions (default = 1).\n");
#endif
#ifdef BOTS_APP_USES_ARG_SIZE
fprintf(stderr, "  -n <size>  : "BOTS_APP_DESC_ARG_SIZE"\n");
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_1
fprintf(stderr, "  -m <size>  : "BOTS_APP_DESC_ARG_SIZE_1"\n");
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_2
fprintf(stderr, "  -l <size>  : "BOTS_APP_DESC_ARG_SIZE_2"\n");
#endif
#ifdef BOTS_APP_USES_ARG_FILE
fprintf(stderr, "  -f <file>  : "BOTS_APP_DESC_ARG_FILE"\n");
#endif
#if defined(MANUAL_CUTOFF) || defined(IF_CUTOFF) || defined(FINAL_CUTOFF)
fprintf(stderr, "  -x <value> : OpenMP tasks cut-off value (default=%d)\n",BOTS_CUTOFF_DEF_VALUE);
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF
fprintf(stderr, "  -y <value> : "BOTS_APP_DESC_ARG_CUTOFF"(default=%d)\n", BOTS_APP_DEF_ARG_CUTOFF);
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_1
fprintf(stderr, "  -a <value> : "BOTS_APP_DESC_ARG_CUTOFF_1"(default=%d)\n", BOTS_APP_DEF_ARG_CUTOFF_1);
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_2
fprintf(stderr, "  -b <value> : "BOTS_APP_DESC_ARG_CUTOFF_2"(default=%d)\n", BOTS_APP_DEF_ARG_CUTOFF_2);
#endif
fprintf(stderr, "\n");
fprintf(stderr, "  -e <str>   : Include 'str' execution message.\n");
fprintf(stderr, "  -v <level> : Set verbose level (default = 1).\n");
fprintf(stderr, "               0 - none.\n");
fprintf(stderr, "               1 - default.\n");
fprintf(stderr, "               2 - debug.\n");
fprintf(stderr, "  -o <value> : Set output format mode (default = 1).\n");
fprintf(stderr, "               0 - no benchmark output.\n");
fprintf(stderr, "               1 - detailed list format.\n");
fprintf(stderr, "               2 - detailed row format.\n");
fprintf(stderr, "               3 - abridged list format.\n");
fprintf(stderr, "               4 - abridged row format.\n");
fprintf(stderr, "  -z         : Print row header (if output format is a row variant).\n");
fprintf(stderr, "\n");
#ifdef KERNEL_SEQ_CALL
fprintf(stderr, "  -s         : Run sequential version.\n");
#endif
#ifdef BOTS_APP_CHECK_USES_SEQ_RESULT
fprintf(stderr, "  -c         : Check mode ON (implies running sequential version).\n");
#else
fprintf(stderr, "  -c         : Check mode ON.\n");
#endif
fprintf(stderr, "\n");
fprintf(stderr, "  -h         : Print program's usage (this help).\n");
fprintf(stderr, "\n");
}
void
bots_get_params_common(int argc, char **argv)
{
int i;
strcpy(bots_execname, basename(argv[0]));
bots_get_date(bots_exec_date);
strcpy(bots_exec_message,"");
for (i=1; i<argc; i++) 
{
if (argv[i][0] == '-')
{
switch (argv[i][1])
{
#ifdef BOTS_APP_USES_ARG_CUTOFF_1
case 'a':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_app_cutoff_value_1 = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_2
case 'b':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_app_cutoff_value_2 = atoi(argv[i]);
break;
#endif
case 'c': 
argv[i][1] = '*';
bots_check_flag = TRUE;
break;
case 'e': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
strcpy(bots_exec_message, argv[i]);
break;
#ifdef BOTS_APP_USES_ARG_FILE
case 'f': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
strcpy(bots_arg_file,argv[i]);
break;
#endif
case 'h': 
argv[i][1] = '*';
bots_print_usage();
exit (100);
#ifdef BOTS_APP_USES_ARG_SIZE_2
case 'l': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_size_2 = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_1
case 'm': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_size_1 = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_SIZE
case 'n': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_size = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_BLOCK
#endif
case 'o': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_output_format = atoi(argv[i]);
break;
#ifdef BOTS_APP_USES_REPETITIONS
case 'r': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_repetition = atoi(argv[i]);
break;
#endif
#ifdef KERNEL_SEQ_CALL
case 's': 
argv[i][1] = '*';
bots_sequential_flag = TRUE;
break;
#endif
case 'v': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_verbose_mode = (bots_verbose_mode_t) atoi(argv[i]);
#ifndef BOTS_DEBUG
if ( bots_verbose_mode > 1 ) {
fprintf(stderr, "Error: Configure the suite using '--debug' option in order to use a verbose level greather than 1.\n");
exit(100);
}
#endif
break;
#if defined(MANUAL_CUTOFF) || defined(IF_CUTOFF) || defined(FINAL_CUTOFF)
case 'x':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_cutoff_value = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF
case 'y':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_app_cutoff_value = atoi(argv[i]);
break;
#endif
case 'z':
argv[i][1] = '*';
bots_print_header = TRUE;
break;
default:
fprintf(stderr, "Error: Unrecognized parameter.\n");
bots_print_usage();
exit (100);
}
}
else
{
fprintf(stderr, "Error: Unrecognized parameter.\n");
bots_print_usage();
exit (100);
}
}
}
void
bots_get_params(int argc, char **argv)
{
bots_get_params_common(argc, argv);
}
void bots_set_info ()
{
snprintf(bots_name, BOTS_TMP_STR_SZ, BOTS_APP_NAME);
snprintf(bots_parameters, BOTS_TMP_STR_SZ, BOTS_APP_PARAMETERS_DESC BOTS_APP_PARAMETERS_LIST);
snprintf(bots_model, BOTS_TMP_STR_SZ, BOTS_MODEL_DESC);
snprintf(bots_resources, BOTS_TMP_STR_SZ, "%d", omp_get_max_threads());
snprintf(bots_comp_date, BOTS_TMP_STR_SZ, CDATE);
snprintf(bots_comp_message, BOTS_TMP_STR_SZ, CMESSAGE);
snprintf(bots_cc, BOTS_TMP_STR_SZ, CC);
snprintf(bots_cflags, BOTS_TMP_STR_SZ, CFLAGS);
snprintf(bots_ld, BOTS_TMP_STR_SZ, LD);
snprintf(bots_ldflags, BOTS_TMP_STR_SZ, LDFLAGS);
#if defined(MANUAL_CUTOFF) 
snprintf(bots_cutoff, BOTS_TMP_STR_SZ, "manual (%d)",bots_cutoff_value);
#elif defined(IF_CUTOFF) 
snprintf(bots_cutoff, BOTS_TMP_STR_SZ, "pragma-if (%d)",bots_cutoff_value);
#elif defined(FINAL_CUTOFF)
snprintf(bots_cutoff, BOTS_TMP_STR_SZ, "final (%d)",bots_cutoff_value);
#else
strcpy(bots_cutoff,"none");
#endif
}
int
main(int argc, char* argv[])
{
#ifndef BOTS_APP_SELF_TIMING
long bots_t_start;
long bots_t_end;
#endif
bots_get_params(argc,argv);
BOTS_APP_INIT;
bots_set_info();
#ifdef KERNEL_SEQ_CALL
#ifdef BOTS_APP_CHECK_USES_SEQ_RESULT
if (bots_sequential_flag || bots_check_flag)
#else
if (bots_sequential_flag)
#endif
{
bots_sequential_flag = 1;
KERNEL_SEQ_INIT;
#ifdef BOTS_APP_SELF_TIMING
bots_time_sequential = KERNEL_SEQ_CALL;
#else
bots_t_start = bots_usecs();
KERNEL_SEQ_CALL;
bots_t_end = bots_usecs();
bots_time_sequential = ((double)(bots_t_end-bots_t_start))/1000000;
#endif
KERNEL_SEQ_FINI;
}
#endif
KERNEL_INIT;
#ifdef BOTS_APP_SELF_TIMING
bots_time_program = KERNEL_CALL;
#else
bots_t_start = bots_usecs();
KERNEL_CALL;
bots_t_end = bots_usecs();
bots_time_program = ((double)(bots_t_end-bots_t_start))/1000000;
#endif
KERNEL_FINI;
#ifdef KERNEL_CHECK
if (bots_check_flag) {
bots_result = KERNEL_CHECK;
}
#endif
BOTS_APP_FINI;
bots_print_results();
return (0);
}
