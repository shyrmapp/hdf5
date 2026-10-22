/* Rebuild: cd testdata/c_interop && h5cc -o /tmp/gen_fill gen_fill.c && /tmp/gen_fill
 * Fixtures for fill values, unallocated storage, slice reads, attribute types
 * and non-default superblock sizes. */
#include "hdf5.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define CHK(x) do { if ((x) < 0) { fprintf(stderr, "failed: %s (line %d)\n", #x, __LINE__); exit(1); } } while (0)

static hid_t chunked_dcpl(int rank, const hsize_t *chunk, hid_t type, const void *fill) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHK(H5Pset_chunk(dcpl, rank, chunk));
    if (fill) CHK(H5Pset_fill_value(dcpl, type, fill));
    return dcpl;
}

static void write_all_int32(hid_t f, const char *name, int rank, const hsize_t *dims, const hsize_t *chunk) {
    hsize_t n = 1;
    for (int i = 0; i < rank; i++) n *= dims[i];
    int32_t *buf = malloc(n * sizeof *buf);
    for (hsize_t i = 0; i < n; i++) buf[i] = (int32_t)i; /* value = linear index */
    hid_t sp = H5Screate_simple(rank, dims, NULL);
    hid_t dcpl = H5P_DEFAULT;
    if (chunk) dcpl = chunked_dcpl(rank, chunk, H5T_NATIVE_INT32, NULL);
    hid_t ds = H5Dcreate2(f, name, H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    CHK(ds);
    CHK(H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
    H5Dclose(ds); H5Sclose(sp); if (chunk) H5Pclose(dcpl);
    free(buf);
}

/* 2-D, chunk 2x2; writes only chunk (0,0) when chunk00 != NULL. */
static void fill_dataset(hid_t f, const char *name, const hsize_t *maxd, hid_t type, const void *fill, const void *chunk00) {
    hsize_t dims[2] = {4, 4}, chunk[2] = {2, 2};
    hid_t sp = H5Screate_simple(2, dims, maxd);
    hid_t dcpl = chunked_dcpl(2, chunk, type, fill);
    hid_t ds = H5Dcreate2(f, name, type, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    CHK(ds);
    if (chunk00) {
        hsize_t start[2] = {0, 0}, count[2] = {2, 2};
        hid_t msp = H5Screate_simple(2, count, NULL);
        CHK(H5Sselect_hyperslab(sp, H5S_SELECT_SET, start, NULL, count, NULL));
        CHK(H5Dwrite(ds, type, msp, sp, H5P_DEFAULT, chunk00));
        H5Sclose(msp);
    }
    H5Dclose(ds); H5Sclose(sp); H5Pclose(dcpl);
}

static void fill_file(const char *path, int latest) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHK(H5Pset_libver_bounds(fapl, latest ? H5F_LIBVER_LATEST : H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST));
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHK(f);

    int16_t m999 = -999, seven = 7, blk[4] = {1, 2, 3, 4};
    fill_dataset(f, "partial", NULL, H5T_NATIVE_INT16, &m999, blk);
    fill_dataset(f, "unwritten_chunked", NULL, H5T_NATIVE_INT16, &seven, NULL);
    hsize_t unl[2] = {H5S_UNLIMITED, 4};
    fill_dataset(f, "unwritten_chunked_unlim", unl, H5T_NATIVE_INT16, &seven, NULL);

    float nc = 9.96921e36f, fblk[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    fill_dataset(f, "nc_fill", NULL, H5T_NATIVE_FLOAT, &nc, fblk);

    /* never-written contiguous with fill 5 (storage allocated late => UNDEF) */
    {
        hsize_t dims[1] = {6};
        int32_t five = 5;
        hid_t sp = H5Screate_simple(1, dims, NULL), dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CHK(H5Pset_fill_value(dcpl, H5T_NATIVE_INT32, &five));
        hid_t ds = H5Dcreate2(f, "unwritten_contig", H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CHK(ds); H5Dclose(ds); H5Sclose(sp); H5Pclose(dcpl);
    }
    /* zero-length unlimited chunked */
    {
        hsize_t dims[1] = {0}, maxd[1] = {H5S_UNLIMITED}, chunk[1] = {4};
        hid_t sp = H5Screate_simple(1, dims, maxd), dcpl = chunked_dcpl(1, chunk, H5T_NATIVE_INT32, NULL);
        hid_t ds = H5Dcreate2(f, "empty_unlim", H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CHK(ds); H5Dclose(ds); H5Sclose(sp); H5Pclose(dcpl);
    }
    if (!latest) {
        /* slice-test datasets, value = linear index */
        hsize_t d1[1] = {30}, c1[1] = {7}, d2[2] = {5, 6}, c2[2] = {2, 4}, d3[3] = {4, 5, 6}, c3[3] = {3, 2, 4};
        write_all_int32(f, "contig1d", 1, d1, NULL);
        write_all_int32(f, "contig2d", 2, d2, NULL);
        write_all_int32(f, "contig3d", 3, d3, NULL);
        write_all_int32(f, "chunk1d", 1, d1, c1);
        write_all_int32(f, "chunk2d", 2, d2, c2);
        write_all_int32(f, "chunk3d", 3, d3, c3);
    }
    H5Fclose(f); H5Pclose(fapl);
}

static void attr_scalar(hid_t f, const char *name, hid_t filetype, hid_t memtype, const void *val) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(f, name, filetype, sp, H5P_DEFAULT, H5P_DEFAULT);
    CHK(a);
    CHK(H5Awrite(a, memtype, val));
    H5Aclose(a); H5Sclose(sp);
}

static void attrs_file(const char *path) {
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHK(f);
    int8_t i8 = -5; int16_t i16 = -300; uint8_t u8 = 250; uint16_t u16 = 60000;
    uint32_t u32 = 4000000000u; uint64_t u64 = UINT64_MAX; double be = 1.5; int32_t bi = -7;
    attr_scalar(f, "i8", H5T_STD_I8LE, H5T_NATIVE_INT8, &i8);
    attr_scalar(f, "i16", H5T_STD_I16LE, H5T_NATIVE_INT16, &i16);
    attr_scalar(f, "u8", H5T_STD_U8LE, H5T_NATIVE_UINT8, &u8);
    attr_scalar(f, "u16", H5T_STD_U16LE, H5T_NATIVE_UINT16, &u16);
    attr_scalar(f, "u32", H5T_STD_U32LE, H5T_NATIVE_UINT32, &u32);
    attr_scalar(f, "u64", H5T_STD_U64LE, H5T_NATIVE_UINT64, &u64);
    attr_scalar(f, "f64be", H5T_IEEE_F64BE, H5T_NATIVE_DOUBLE, &be);
    attr_scalar(f, "i32be", H5T_STD_I32BE, H5T_NATIVE_INT32, &bi);
    H5Fclose(f);
}

static void sb_file(const char *path, int latest, size_t addr, size_t len) {
    hid_t fcpl = H5Pcreate(H5P_FILE_CREATE), fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHK(H5Pset_sizes(fcpl, addr, len));
    CHK(H5Pset_libver_bounds(fapl, latest ? H5F_LIBVER_LATEST : H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST));
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, fapl);
    CHK(f);
    hsize_t dims[1] = {5};
    int32_t v[5] = {0, 1, 2, 3, 4};
    hid_t sp = H5Screate_simple(1, dims, NULL);
    hid_t ds = H5Dcreate2(f, "d", H5T_NATIVE_INT32, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(ds);
    CHK(H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
    H5Dclose(ds); H5Sclose(sp); H5Fclose(f); H5Pclose(fcpl); H5Pclose(fapl);
}

int main(void) {
    fill_file("fill_v3.h5", 0);
    fill_file("fill_latest.h5", 1);
    attrs_file("attr_types.h5");
    sb_file("sb_v0_a4_l8.h5", 0, 4, 8);
    sb_file("sb_v0_a4_l4.h5", 0, 4, 4);
    sb_file("sb_v2_a4_l8.h5", 1, 4, 8);
    sb_file("sb_v2_a4_l4.h5", 1, 4, 4);
    return 0;
}
