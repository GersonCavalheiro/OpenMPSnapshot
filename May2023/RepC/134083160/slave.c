#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <mpi.h>
#include "../../../rt_common.h"
#include "slave.h"
#include "memory.h"
static MPI_Comm communicator; 
static int cmd_params[4];     
static void dbg(char *format, ...)
{
va_list ap;
va_start(ap, format);
fprintf(stderr, "  DBG(slave@%d): ", getpid());
vfprintf(stderr, format, ap);
va_end(ap);
}
void cmdinit(), cmdget(), cmdput(), cmdalloc(), cmdfree(),
cmdexecute(), cmdshutdown();
void (*cmdfunc[])() = {
cmdinit, cmdget, cmdput, cmdalloc, cmdfree, cmdexecute, cmdshutdown
};
int read_command(void)
{
MPI_Recv(cmd_params, 4, MPI_INT, 0, MPI_ANY_TAG, communicator, MPI_STATUS_IGNORE);
return cmd_params[0]; 
}
void wait_and_execute(MPI_Comm merged_comm)
{
int cmd;
communicator = merged_comm;
for (cmd = read_command(); cmd >= 0 && cmd <= MPI_CMD_LAST; cmd= read_command())
(cmdfunc[cmd])();
dbg("illegal command %d received\n", cmd);
exit(2);
}
#pragma weak actual_devpart_med2dev_addr
char *(*actual_devpart_med2dev_addr)(void *, unsigned long);
#pragma weak actual_omp_is_initial_device
int (*actual_omp_is_initial_device)(void);
char *mpinode_devpart_med2dev_addr(void *medaddr, unsigned long size)
{
void *devaddr;
int rank;
devaddr = alloc_items_get((size_t)medaddr);
#ifdef DEBUG
dbg("  devpart_med2dev_addr %p --> %p\n", medaddr, devaddr);
#endif
return devaddr;
}
void override_devpart_med2dev_addr(void)
{
extern char *(*actual_devpart_med2dev_addr)(void *, unsigned long);
actual_devpart_med2dev_addr = mpinode_devpart_med2dev_addr;
}
int mpinode_omp_is_initial_device(void)
{
return 0;
}
void override_omp_is_initial_device(void)
{
extern int (*actual_omp_is_initial_device)(void);
actual_omp_is_initial_device = mpinode_omp_is_initial_device;
}
void cmdinit()
{
int devid = cmd_params[1];
alloc_items_init(1); 
alloc_items_init_global_vars(devid, 0);
}
void cmdget()
{
size_t maddr = cmd_params[1];
int offset = cmd_params[2], nbytes = cmd_params[3];
void *devaddr; 
#ifdef DEBUG
dbg(" GET cmd from %d (offset:%d, size:%d)\n", maddr, offset, nbytes);
#endif
devaddr = alloc_items_get(maddr);
MPI_Send(devaddr + offset, nbytes, MPI_BYTE, 0, 0, communicator);
}
void cmdput()
{
size_t maddr = cmd_params[1];
int offset = cmd_params[2], nbytes = cmd_params[3];
void *devaddr; 
#ifdef DEBUG
dbg(" PUT cmd to %p (offset:%d, size:%d)\n", maddr, offset, nbytes);
#endif
devaddr = alloc_items_get(maddr);
MPI_Recv(devaddr + offset, nbytes, MPI_BYTE, 0, MPI_ANY_TAG, communicator,
MPI_STATUS_IGNORE);
}
void cmdalloc()
{
int nbytes = cmd_params[1];
size_t maddr = cmd_params[2];
alloc_items_add(maddr, nbytes);
#ifdef DEBUG
dbg(" ALLOC cmd for %d bytes --> maddr: %p\n", nbytes, maddr);
#endif
}
void cmdfree()
{
size_t maddr = cmd_params[1];
#ifdef DEBUG
dbg(" FREE cmd for %d at %p\n", maddr, alloc_items_get(maddr));
#endif
alloc_items_remove(maddr);
}
void cmdexecute()
{
void *devdata; 
int exec_result, kernel_id = cmd_params[1];
size_t size = cmd_params[2], devdata_maddr = cmd_params[3];
if (size == 0)
devdata = NULL;
else
{
devdata = alloc_items_add(devdata_maddr, size);
MPI_Recv(devdata, size, MPI_BYTE, 0, MPI_ANY_TAG, communicator,
MPI_STATUS_IGNORE); 
#ifdef DEBUG
dbg(" EXECUTE cmd: alloced devdata (i:%d, u:%p, size:%d)\n", devdata_maddr,
devdata, size);
#endif
}
void *(*kernel_function)(void *);
kernel_function = get_kernel_function_from_id(kernel_id);
#ifdef DEBUG
dbg(" EXECUTE cmd: will execute kernel with id %d at address %p\n",
kernel_id, kernel_function);
#endif
kernel_function((void *)devdata); 
if (devdata)
alloc_items_remove(devdata_maddr);
exec_result = 11; 
MPI_Send(&exec_result, 1, MPI_INT, 0, 0, communicator); 
#ifdef DEBUG
dbg(" EXECUTE cmd done for kernel with id %d\n", kernel_id);
#endif
}
void cmdshutdown()
{
#ifdef DEBUG
dbg("<<< SHUTDOWN cmd, MPI device finalizing and exitting...\n");
#endif
alloc_items_free_all();
free_kerneltable();
MPI_Comm_free(&communicator);
MPI_Finalize();
exit(0);
}
#ifdef DEBUG
#undef DEBUG
#endif
