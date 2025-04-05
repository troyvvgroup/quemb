# Illustrates parallelized BE computation on hexene in large
# basis sets, using IAO localization schemes

from pyscf import cc, gto, scf

from quemb.molbe import BE, fragmentate

# Perform pyscf HF calculation to get mol & mf objects
mol = gto.M(
    atom="""
C       -5.6502267899      0.7485927383     -0.0074809907
C       -4.5584842828      1.7726977952     -0.0418619714
H       -5.5515181382      0.0602177800     -0.8733001951
H       -6.6384111226      1.2516490350     -0.0591711493
H       -5.5928112720      0.1649656434      0.9355613930
C       -3.2701647911      1.4104028701      0.0085107804
C       -2.1789947571      2.4456245961     -0.0265301736
C       -0.7941361337      1.7933691827      0.0427863465
H       -2.3064879754      3.1340763205      0.8376047229
H       -2.2652945230      3.0294980525     -0.9691823088
C        0.3121144607      2.8438276122      0.0072105803
H       -0.6626010212      1.1042272672     -0.8201573062
H       -0.7037327970      1.2086599200      0.9845037688
H        0.2581952957      3.4296615741     -0.9351274363
H        1.3021608456      2.3437050700      0.0587054221
H        0.2170130582      3.5342406229      0.8723087816
H       -4.8264911382      2.8238751191     -0.1088423637
H       -3.0132059318      0.3554469837      0.0754505007
""",
    basis="cc-pVDZ",
    charge=0,
)

# Run reference RHF calculation
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# Perform CCSD calculation to get reference energy for comparison
mc = cc.CCSD(mf, frozen=8)
mc.verbose = 0
ccsd_ecorr = mc.kernel()[0]
print(f"*** CCSD Correlation Energy: {ccsd_ecorr:>14.8f} Ha", flush=True)

# initialize fragments (use frozen core approximation)
# iao_valence_basis is the minimal-like basis upon which you project
# the full, working basis. We recommend minao or sto-3g
# Note: iao_valence_only=True is currently not supported
fobj = fragmentate(
    be_type="be2",
    mol=mol,
    frozen_core=True,
    iao_valence_basis="sto-3g",
    frag_type="autogen",
)

# Initialize BE, specifying other iao parameters:
# The lo_method is used to specify IAO localization
# iao_loc_method is the localization of the generated IAOs and PAOs.
# The default, "SO", calls the get_iao_native routine, while
# "Boys", "PM", and "ER" call get_iao. We recommend using Pipek-Mezey
# if not using "SO"
# We can also specify the init_guess and pop_method here, relevant for
# certain localization schemes.
mybe = BE(
    mf,
    fobj,
    lo_method="IAO",
    iao_loc_method="PM",
)

# Perform BE density matching.
# Uses 2 procs, each fragment calculation assigned OMP_NUM_THREADS to 1
mybe.optimize(solver="CCSD", nproc=1, ompnum=1)

# Compute error
be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (ccsd_ecorr - be_ecorr) * 100.0 / ccsd_ecorr
print(f"*** BE2 Correlation Energy Error (%) : {err_:>8.4f} %")

# expect a be_ecorr of -0.92757108 Ha
