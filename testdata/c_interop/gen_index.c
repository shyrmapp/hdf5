/* rebuild: cd testdata/c_interop && h5cc -o /tmp/gen_index gen_index.c && /tmp/gen_index index_cases.h5 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { if ((x) < 0) { fprintf(stderr, "failed: %s\n", #x); exit(1); } } while (0)

int main(int argc, char **argv) {
    const char *name = argc > 1 ? argv[1] : "index_cases.h5";
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CK(H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
    hid_t file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CK(file);

    /* 1. v2 filter pipeline: shuffle + deflate + fletcher32 (value i). */
    {
        hsize_t dims[1] = {1000}, chunk[1] = {100};
        int *buf = malloc(sizeof(int) * 1000);
        for (int i = 0; i < 1000; i++) buf[i] = i;
        hid_t sp = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CK(H5Pset_chunk(dcpl, 1, chunk));
        CK(H5Pset_shuffle(dcpl));
        CK(H5Pset_deflate(dcpl, 6));
        CK(H5Pset_fletcher32(dcpl));
        hid_t ds = H5Dcreate2(file, "pipeline_v2", H5T_NATIVE_INT, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CK(ds);
        CK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        H5Dclose(ds); H5Pclose(dcpl); H5Sclose(sp); free(buf);
    }

    /* 2. Fixed array with 5000 chunks (paged data block): value i, chunk of 2. */
    {
        hsize_t dims[1] = {10000}, chunk[1] = {2};
        int *buf = malloc(sizeof(int) * 10000);
        for (int i = 0; i < 10000; i++) buf[i] = i;
        hid_t sp = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CK(H5Pset_chunk(dcpl, 1, chunk));
        hid_t ds = H5Dcreate2(file, "fixed_5000", H5T_NATIVE_INT, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CK(ds);
        CK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        H5Dclose(ds); H5Pclose(dcpl); H5Sclose(sp); free(buf);
    }

    /* 3. 2D unlimited, 1x1 chunks, 120x120 = 14400 chunks (v2 B-tree): value i*1000+j. */
    {
        hsize_t dims[2] = {120, 120}, maxd[2] = {H5S_UNLIMITED, H5S_UNLIMITED}, chunk[2] = {1, 1};
        int *buf = malloc(sizeof(int) * 14400);
        for (int i = 0; i < 120; i++)
            for (int j = 0; j < 120; j++) buf[i * 120 + j] = i * 1000 + j;
        hid_t sp = H5Screate_simple(2, dims, maxd);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CK(H5Pset_chunk(dcpl, 2, chunk));
        hid_t ds = H5Dcreate2(file, "unlimited_2d", H5T_NATIVE_INT, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CK(ds);
        CK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        H5Dclose(ds); H5Pclose(dcpl); H5Sclose(sp); free(buf);
    }

    H5Fclose(file);
    H5Pclose(fapl);
    return 0;
}
