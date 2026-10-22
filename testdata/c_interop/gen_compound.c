/* Rebuild: cd testdata/c_interop && h5cc -o /tmp/gen_compound gen_compound.c && /tmp/gen_compound
 * Compound fixtures: compound_v1.h5 (libver earliest) and compound_v3.h5 (latest). */
#include "hdf5.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CHK(x) do { if ((x) < 0) { fprintf(stderr, "failed: %s (line %d)\n", #x, __LINE__); exit(1); } } while (0)

typedef struct { int32_t a; double b; } inner_t;
typedef struct {
    double d; int8_t i8; int16_t i16; int32_t i32; int64_t i64;
    uint8_t u8; uint16_t u16; uint32_t u32; uint64_t u64;
    int32_t be; char fs[5]; char *vs; inner_t in; int32_t arr[3];
} all_t;
typedef struct { double d; int32_t i; uint32_t u; } ddu_t;
typedef struct { double d; char s[300]; int32_t i; } big_t;

static void make(const char *fname, H5F_libver_t lv) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHK(H5Pset_libver_bounds(fapl, lv, H5F_LIBVER_LATEST));
    hid_t f = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHK(f);
    hsize_t n = 2, three = 3;
    hid_t sp = H5Screate_simple(1, &n, NULL);

    /* all member kinds */
    hid_t inner = H5Tcreate(H5T_COMPOUND, sizeof(inner_t));
    H5Tinsert(inner, "a", HOFFSET(inner_t, a), H5T_NATIVE_INT32);
    H5Tinsert(inner, "b", HOFFSET(inner_t, b), H5T_NATIVE_DOUBLE);
    hid_t fs = H5Tcopy(H5T_C_S1); H5Tset_size(fs, 5);
    hid_t vs = H5Tcopy(H5T_C_S1); H5Tset_size(vs, H5T_VARIABLE);
    hid_t arr = H5Tarray_create2(H5T_NATIVE_INT32, 1, &three);
    hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(all_t));
    H5Tinsert(t, "d", HOFFSET(all_t, d), H5T_NATIVE_DOUBLE);
    H5Tinsert(t, "i8", HOFFSET(all_t, i8), H5T_NATIVE_INT8);
    H5Tinsert(t, "i16", HOFFSET(all_t, i16), H5T_NATIVE_INT16);
    H5Tinsert(t, "i32", HOFFSET(all_t, i32), H5T_NATIVE_INT32);
    H5Tinsert(t, "i64", HOFFSET(all_t, i64), H5T_NATIVE_INT64);
    H5Tinsert(t, "u8", HOFFSET(all_t, u8), H5T_NATIVE_UINT8);
    H5Tinsert(t, "u16", HOFFSET(all_t, u16), H5T_NATIVE_UINT16);
    H5Tinsert(t, "u32", HOFFSET(all_t, u32), H5T_NATIVE_UINT32);
    H5Tinsert(t, "u64", HOFFSET(all_t, u64), H5T_NATIVE_UINT64);
    H5Tinsert(t, "be", HOFFSET(all_t, be), H5T_STD_I32BE);
    H5Tinsert(t, "fs", HOFFSET(all_t, fs), fs);
    H5Tinsert(t, "vs", HOFFSET(all_t, vs), vs);
    H5Tinsert(t, "in", HOFFSET(all_t, in), inner);
    H5Tinsert(t, "arr", HOFFSET(all_t, arr), arr);
    all_t a[2] = {
        {1.5, -8, -300, -70000, -5000000000LL, 200, 60000, 3000000000u, 18000000000000000000ULL,
         (int32_t)__builtin_bswap32(-123456), "abcd", "hello", {7, 2.5}, {1, -2, 3}},
        {-2.25, 127, 32767, 2147483647, 9223372036854775807LL, 255, 65535, 4294967295u, 18446744073709551615ULL,
         (int32_t)__builtin_bswap32(654321), "xy", "", {-9, 0.125}, {10, 20, 30}},
    };
    hid_t ds = H5Dcreate2(f, "all", t, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(ds); CHK(H5Dwrite(ds, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, a)); H5Dclose(ds);

    /* same data without the array member: a v1 compound under EARLIEST (arrays force v2) */
    hid_t t1 = H5Tcreate(H5T_COMPOUND, sizeof(all_t));
    H5Tinsert(t1, "d", HOFFSET(all_t, d), H5T_NATIVE_DOUBLE);
    H5Tinsert(t1, "i8", HOFFSET(all_t, i8), H5T_NATIVE_INT8);
    H5Tinsert(t1, "i16", HOFFSET(all_t, i16), H5T_NATIVE_INT16);
    H5Tinsert(t1, "i32", HOFFSET(all_t, i32), H5T_NATIVE_INT32);
    H5Tinsert(t1, "i64", HOFFSET(all_t, i64), H5T_NATIVE_INT64);
    H5Tinsert(t1, "u8", HOFFSET(all_t, u8), H5T_NATIVE_UINT8);
    H5Tinsert(t1, "u16", HOFFSET(all_t, u16), H5T_NATIVE_UINT16);
    H5Tinsert(t1, "u32", HOFFSET(all_t, u32), H5T_NATIVE_UINT32);
    H5Tinsert(t1, "u64", HOFFSET(all_t, u64), H5T_NATIVE_UINT64);
    H5Tinsert(t1, "be", HOFFSET(all_t, be), H5T_STD_I32BE);
    H5Tinsert(t1, "fs", HOFFSET(all_t, fs), fs);
    H5Tinsert(t1, "vs", HOFFSET(all_t, vs), vs);
    H5Tinsert(t1, "in", HOFFSET(all_t, in), inner);
    ds = H5Dcreate2(f, "noarr", t1, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(ds, t1, H5S_ALL, H5S_ALL, H5P_DEFAULT, a)); H5Dclose(ds);

    /* {double,int,uint}: size 24 */
    hid_t t2 = H5Tcreate(H5T_COMPOUND, sizeof(ddu_t));
    H5Tinsert(t2, "d", HOFFSET(ddu_t, d), H5T_NATIVE_DOUBLE);
    H5Tinsert(t2, "i", HOFFSET(ddu_t, i), H5T_NATIVE_INT32);
    H5Tinsert(t2, "u", HOFFSET(ddu_t, u), H5T_NATIVE_UINT32);
    ddu_t b[2] = {{1.5, -7, 3000000000u}, {2.5, 8, 5}};
    ds = H5Dcreate2(f, "ddu", t2, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(ds, t2, H5S_ALL, H5S_ALL, H5P_DEFAULT, b)); H5Dclose(ds);

    /* >256 bytes: 2-byte member offsets */
    hid_t s300 = H5Tcopy(H5T_C_S1); H5Tset_size(s300, 300);
    hid_t t3 = H5Tcreate(H5T_COMPOUND, sizeof(big_t));
    H5Tinsert(t3, "d", HOFFSET(big_t, d), H5T_NATIVE_DOUBLE);
    H5Tinsert(t3, "s", HOFFSET(big_t, s), s300);
    H5Tinsert(t3, "i", HOFFSET(big_t, i), H5T_NATIVE_INT32);
    big_t c[2]; memset(c, 0, sizeof c);
    c[0].d = 3.5; strcpy(c[0].s, "first"); c[0].i = -42;
    c[1].d = 4.5; memset(c[1].s, 'z', 299); c[1].i = 99;
    ds = H5Dcreate2(f, "big", t3, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(ds, t3, H5S_ALL, H5S_ALL, H5P_DEFAULT, c)); H5Dclose(ds);

    H5Sclose(sp); H5Fclose(f); H5Pclose(fapl);
}

int main(void) {
    make("compound_v1.h5", H5F_LIBVER_EARLIEST);
    make("compound_v3.h5", H5F_LIBVER_LATEST);
    return 0;
}
