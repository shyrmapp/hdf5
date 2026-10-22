/* rebuild: cd testdata/c_interop && h5cc -o /tmp/gen_chunks gen_chunks.c && /tmp/gen_chunks chunks_cases.h5 */
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { if ((x) < 0) { fprintf(stderr, "failed: %s\n", #x); exit(1); } } while (0)

/* 1-D int32 dataset of value i, 10 elements, chunk 4 (so chunk 2 is a partial
 * edge chunk), layout v4 with H5Pset_chunk_opts(DONT_FILTER_PARTIAL_CHUNKS). */
static void partial(hid_t file, const char *name, int deflate) {
    hsize_t dims[1] = {10}, chunk[1] = {4};
    int buf[10];
    for (int i = 0; i < 10; i++) buf[i] = i;
    hid_t sp = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CK(H5Pset_chunk(dcpl, 1, chunk));
    CK(H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS));
    CK(H5Pset_shuffle(dcpl));
    if (deflate) CK(H5Pset_deflate(dcpl, 6));
    hid_t ds = H5Dcreate2(file, name, H5T_NATIVE_INT, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    CK(ds);
    CK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
    H5Dclose(ds); H5Pclose(dcpl); H5Sclose(sp);
}

int main(int argc, char **argv) {
    const char *name = argc > 1 ? argv[1] : "chunks_cases.h5";
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CK(H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST));
    hid_t file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CK(file);

    partial(file, "partial_shuffle_deflate", 1);
    partial(file, "partial_shuffle", 0);

    /* Fletcher32 only: float64 1..8, a single chunk. */
    {
        hsize_t dims[1] = {8}, chunk[1] = {8};
        double buf[8];
        for (int i = 0; i < 8; i++) buf[i] = i + 1;
        hid_t sp = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CK(H5Pset_chunk(dcpl, 1, chunk));
        CK(H5Pset_fletcher32(dcpl));
        hid_t ds = H5Dcreate2(file, "fletcher32", H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CK(ds);
        CK(H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        H5Dclose(ds); H5Pclose(dcpl); H5Sclose(sp);
    }

    /* Fletcher32 over an odd number of bytes: uint8 1..7, a single chunk. */
    {
        hsize_t dims[1] = {7}, chunk[1] = {7};
        unsigned char buf[7] = {1, 2, 3, 4, 5, 6, 7};
        hid_t sp = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CK(H5Pset_chunk(dcpl, 1, chunk));
        CK(H5Pset_fletcher32(dcpl));
        hid_t ds = H5Dcreate2(file, "fletcher32_odd", H5T_NATIVE_UCHAR, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CK(ds);
        CK(H5Dwrite(ds, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        H5Dclose(ds); H5Pclose(dcpl); H5Sclose(sp);
    }

    H5Fclose(file);
    return 0;
}
