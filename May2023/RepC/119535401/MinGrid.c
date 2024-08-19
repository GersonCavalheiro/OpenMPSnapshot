#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <omp.h>
#include <sys/param.h>
#define VERSION "0.1"
void
GridMinimum(double *zmg, int *idz,
const double *xb, const unsigned int xbn,
const double *yb, const unsigned int ybn,
const double *x, const double *y, const double *z,
const unsigned int n) {
unsigned int i, k, l, lk, id, lmin;
double xl, yl, zmin;
#pragma omp parallel for private(i,k,l,lk,id,xl,yl,zmin,lmin)
for(i = 0; i < xbn; i++) {
lk = 0;
id = i * ybn;
for(k = 0; k < ybn; k++) {
zmin = 9E9;
lmin = 0;
for(l = lk; l < n; l++) {
yl = y[l];
if(yl >= yb[k+1]) {
lk = l;
break;
}
xl = x[l];
if(xb[i] <= xl && xl < xb[i+1]) {
if(z[l] < zmin) {
zmin = z[l];
lmin = l;
}
}
}
zmg[id+k] = zmin;
idz[lmin] = 1;
}
}
}
static PyObject *
MinGrid_GridMinimum(PyObject *self, PyObject* args) {
PyObject *xbarg, *ybarg, *xarg, *yarg, *zarg;
PyArrayObject *xb, *yb, *x, *y, *z, *zgrid, *idz;
unsigned int n;
npy_intp dim[2];
if(!PyArg_ParseTuple(args, "OOOOO", &xbarg, &ybarg, &xarg, &yarg, &zarg))
return NULL;
xb = (PyArrayObject *) PyArray_ContiguousFromObject(xbarg, PyArray_DOUBLE, 1, 1);
yb = (PyArrayObject *) PyArray_ContiguousFromObject(ybarg, PyArray_DOUBLE, 1, 1);
x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
if(!xb || !yb || !x || !y || !z)
return NULL;
n = x->dimensions[0];
if(n != y->dimensions[0]) {
PyErr_SetString(PyExc_IndexError, "dimension mismatch between x and y coordinates.");
return NULL;
}
if(n != z->dimensions[0]) {
PyErr_SetString(PyExc_IndexError, "dimension mismatch between variable and coordinates.");
return NULL;
}
idz = (PyArrayObject *) PyArray_ZEROS(1, z->dimensions, PyArray_INT, 0);
dim[0] = (xb->dimensions[0] - 1);
dim[1] = (yb->dimensions[0] - 1);
zgrid = (PyArrayObject *) PyArray_ZEROS(2, dim, PyArray_DOUBLE, 0);
if(!zgrid || !idz) {
PyErr_SetString(PyExc_MemoryError, "...");
return NULL;
}
GridMinimum((double *)zgrid->data,
(int *)idz->data,
(double *)xb->data, dim[0],
(double *)yb->data, dim[1],
(double *)x->data,
(double *)y->data,
(double *)z->data, n);
Py_DECREF(xb);
Py_DECREF(yb);
Py_DECREF(x);
Py_DECREF(y);
Py_DECREF(z);
return Py_BuildValue("(OO)", zgrid, idz);
}
static PyMethodDef MinGrid_Methods[] = {
{"GridMinimum", MinGrid_GridMinimum, METH_VARARGS, "please have a look at the code"},
{NULL, NULL, 0, NULL}
};
static struct PyModuleDef ModDef = {
PyModuleDef_HEAD_INIT,
"MinGrid",
NULL,
-1,
MinGrid_Methods
};
PyMODINIT_FUNC
PyInit_MinGrid(void) {
PyObject *mod;
mod = PyModule_Create(&ModDef);
PyModule_AddStringConstant(mod, "__author__", "Aljoscha Rheinwalt <aljoscha.rheinwalt@uni-potsdam.de>");
PyModule_AddStringConstant(mod, "__version__", VERSION);
import_array();
return mod;
}
int
main(int argc, char **argv) {
wchar_t pname[255];
PyImport_AppendInittab("MinGrid", PyInit_MinGrid);
mbstowcs(pname, argv[0], strlen(argv[0])+1);
Py_SetProgramName(pname);
Py_Initialize();
PyImport_ImportModule("MinGrid");
PyMem_RawFree(argv[0]);
return 0;
}
