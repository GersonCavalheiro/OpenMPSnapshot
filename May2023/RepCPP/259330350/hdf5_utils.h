

#pragma once

#include <hdf5.h>



typedef hid_t hdf5_blob;
hdf5_blob hdf5_get_blob(const char *name);
int hdf5_write_blob(hdf5_blob blob, const char *name);
int hdf5_close_blob(hdf5_blob blob);

int hdf5_create(const char *fname);
int hdf5_open(const char *fname);
int hdf5_close();

int hdf5_make_directory(const char *name);
void hdf5_set_directory(const char *path);

int hdf5_write_single_val(const void *val, const char *name, hsize_t hdf5_type);
int hdf5_write_array(const void *data, const char *name, size_t rank,
hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);

int hdf5_exists(const char *name);
int hdf5_read_single_val(void *val, const char *name, hsize_t hdf5_type);
int hdf5_read_array(void *data, const char *name, size_t rank,
hsize_t *fdims, hsize_t *fstart, hsize_t *fcount, hsize_t *mdims, hsize_t *mstart, hsize_t hdf5_type);

hid_t hdf5_make_str_type(size_t len);
int hdf5_write_str_list(const void *data, const char *name, size_t strlen, size_t len);
int hdf5_add_attr(const void *att, const char *att_name, const char *data_name, hsize_t hdf5_type);
int hdf5_add_units(const char *name, const char *unit);
