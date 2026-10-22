// Rebuild: cd testdata/c_interop && h5cc -o $TMPDIR/gen_diamond gen_diamond.c && $TMPDIR/gen_diamond .
//
// Hard-link diamond: /g0../g20, each /gi holds two hard links (a, b) to
// /g(i+1), so the last group is reachable through 2^20 paths. Eager per-path
// loading is exponential; one load per group is linear.
#include <hdf5.h>
#include <stdio.h>

#define DEPTH 20

int main(int argc, char **argv) {
    char path[512], name[32], next[32];
    snprintf(path, sizeof path, "%s/diamond.h5", argc > 1 ? argv[1] : ".");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    for (int i = 0; i <= DEPTH; i++) {
        snprintf(name, sizeof name, "/g%d", i);
        H5Gclose(H5Gcreate2(f, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    }
    for (int i = 0; i < DEPTH; i++) {
        char a[48], b[48];
        snprintf(next, sizeof next, "/g%d", i + 1);
        snprintf(a, sizeof a, "/g%d/a", i);
        snprintf(b, sizeof b, "/g%d/b", i);
        H5Lcreate_hard(f, next, f, a, H5P_DEFAULT, H5P_DEFAULT);
        H5Lcreate_hard(f, next, f, b, H5P_DEFAULT, H5P_DEFAULT);
    }
    H5Fclose(f);
    H5Pclose(fapl);
    return 0;
}
