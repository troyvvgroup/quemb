void AO2MOrestore_nr4to1_gen(double *eri4, double *eri1, int norb1, int norb2)
{
        size_t npair1 = norb1*(norb1+1)/2;
        size_t npair2 = norb2*(norb2+1)/2;
        size_t i, j, ij;
        size_t d2 = norb2 * norb2;
        size_t d3 = norb1 * d2;

        for (ij = 0, i = 0; i < norb1; i++) {
        for (j = 0; j <= i; j++, ij++) {
                NPdunpack_tril(norb2, eri4+ij*npair2, eri1+i*d3+j*d2, HERMITIAN);
                if (i > j) {
                        memcpy(eri1+j*d3+i*d2, eri1+i*d3+j*d2,
                               sizeof(double)*d2);
                }
        } }
}

void AO2MOrestore_nr1to4_gen(double *eri1, double *eri4, int norb1, int norb2)
{
        size_t npair1 = norb1*(norb1+1)/2;
        size_t npair2 = norb2*(norb2+1)/2;
        size_t i, j, k, l, ij, kl;
        size_t d1 = norb2;
        size_t d2 = norb2 * norb2;
        size_t d3 = norb1 * d2;

        for (ij = 0, i = 0; i < norb1; i++) {
        for (j = 0; j <= i; j++, ij++) {
                for (kl = 0, k = 0; k < norb2; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        eri4[ij*npair2+kl] = eri1[i*d3+j*d2+k*d1+l];
                } }
        } }
}
