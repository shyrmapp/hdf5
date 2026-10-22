// Rebuild: cd testdata/c_interop && h5cc -o /tmp/gen_dense gen_dense.c && /tmp/gen_dense .
//
// Fixtures for dense link / dense attribute storage and hard-link aliasing.
// libver latest forces new-style groups; >8 links or attributes go dense.
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>

static hid_t create_ex(const char *dir, const char *name, int latest) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (latest) H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (f < 0) { fprintf(stderr, "create %s failed\n", path); exit(1); }
    return f;
}

static hid_t create(const char *dir, const char *name) { return create_ex(dir, name, 1); }

static hid_t scalar_dset(hid_t loc, const char *name, hid_t dcpl) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t d = H5Dcreate2(loc, name, H5T_NATIVE_INT32, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int v = 7;
    H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Sclose(sp);
    return d;
}

// /g holds n hard links ("l000"...) to one dataset /d.
static void links(const char *dir, int n) {
    char name[64];
    snprintf(name, sizeof name, "links_%d.h5", n);
    hid_t f = create(dir, name);
    hid_t d = scalar_dset(f, "d", H5P_DEFAULT);
    hid_t g = H5Gcreate2(f, "g", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for (int i = 0; i < n; i++) {
        snprintf(name, sizeof name, "l%03d", i);
        H5Lcreate_hard(f, "d", g, name, H5P_DEFAULT, H5P_DEFAULT);
    }
    H5Gclose(g); H5Dclose(d); H5Fclose(f);
}

// n int32 attributes "a000"... (value i*10) on object obj.
static void put_attrs(hid_t obj, int n) {
    hid_t sp = H5Screate(H5S_SCALAR);
    char name[64];
    for (int i = 0; i < n; i++) {
        snprintf(name, sizeof name, "a%03d", i);
        int v = i * 10;
        hid_t a = H5Acreate2(obj, name, H5T_NATIVE_INT32, sp, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a, H5T_NATIVE_INT32, &v);
        H5Aclose(a);
    }
    H5Sclose(sp);
}

// Group /g and dataset /d each carry n attributes.
static void attrs(const char *dir, int n, int corder) {
    char name[64];
    snprintf(name, sizeof name, "attrs_%d_%s.h5", n, corder ? "corder" : "plain");
    hid_t f = create(dir, name);
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (corder) {
        H5Pset_attr_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
        H5Pset_attr_creation_order(dcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    }
    hid_t g = H5Gcreate2(f, "g", H5P_DEFAULT, gcpl, H5P_DEFAULT);
    hid_t d = scalar_dset(f, "d", dcpl);
    put_attrs(g, n);
    put_attrs(d, n);
    H5Pclose(gcpl); H5Pclose(dcpl); H5Gclose(g); H5Dclose(d); H5Fclose(f);
}

// Group /a (with dataset /a/x) is also reachable as /b. latest=0 gives the
// old symbol-table group format.
static void alias(const char *dir, const char *name, int latest) {
    hid_t f = create_ex(dir, name, latest);
    hid_t a = H5Gcreate2(f, "a", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t x = scalar_dset(a, "x", H5P_DEFAULT);
    H5Lcreate_hard(f, "a", f, "b", H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(x); H5Gclose(a); H5Fclose(f);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    int ln[] = {9, 30, 100, 300};
    for (int i = 0; i < 4; i++) links(dir, ln[i]);
    int an[] = {20, 300};
    for (int i = 0; i < 2; i++) { attrs(dir, an[i], 0); attrs(dir, an[i], 1); }
    alias(dir, "alias.h5", 1);
    alias(dir, "alias_old.h5", 0);
    return 0;
}
