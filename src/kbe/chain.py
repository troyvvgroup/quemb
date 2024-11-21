# Author(s): Oinam Romesh Meitei


# Old fragmentation code - do not use other than debugging!


def findH(mol, nh, tmphlist=[]):
    import math

    coord_ = mol.atom_coords()

    hidx = []
    for idx in range(mol.natm):
        if mol.atom_pure_symbol(idx) == "H":
            d = math.sqrt(
                (nh[0] - coord_[idx][0]) ** 2
                + (nh[1] - coord_[idx][1]) ** 2
                + (nh[2] - coord_[idx][2]) ** 2
            )
            if d * 0.52917721092 < 1.6:
                if idx not in tmphlist:
                    hidx.append(idx)

    return hidx


def polychain(self, mol, frozen_core=False, unitcell=1):
    """
    Hard coded fragmentation for polymer chains. This is not recommended
    for any production level calculations.
    """

    # group H
    hset = [None] * mol.natm
    baslist = mol.aoslice_by_atom()

    mol2 = mol.copy()
    mol2.basis = "sto-3g"
    mol2.build()
    bas2list = mol2.aoslice_by_atom()

    sites__ = []
    sites2 = []
    coreshift = 0
    tmphlist = []
    for adx in range(mol.natm):
        if not mol.atom_pure_symbol(adx) == "H":
            h1 = findH(mol, mol.atom_coord(adx), tmphlist=tmphlist)
            tmphlist.extend(h1)

            bas = baslist[adx]
            bas2 = bas2list[adx]

            start_ = bas[2]
            stop_ = bas[3]

            start2_ = bas2[2]
            stop2_ = bas2[3]

            sites_ = []
            sites2_ = []
            for hidx in h1:
                basH = baslist[hidx]
                basH2 = bas2list[hidx]

                startH_ = basH[2]
                stopH_ = basH[3]

                startH2_ = basH2[2]
                stopH2_ = basH2[3]

                if frozen_core:
                    startH_ -= coreshift
                    stopH_ -= coreshift

                    startH2_ -= coreshift
                    stopH2_ -= coreshift

                b1list = [i for i in range(startH_, stopH_)]
                b2list = [i for i in range(startH2_, stopH2_)]

                sites2_.extend([True if i in b2list else False for i in b1list])
                sites_.extend(b1list)

            if frozen_core:
                start_ -= coreshift
                start2_ -= coreshift
                ncore_ = self.core_list[adx]
                stop_ -= coreshift + ncore_
                stop2_ -= coreshift + ncore_
                coreshift += ncore_
            b1list = [i for i in range(start_, stop_)]
            b2list = [i for i in range(start2_, stop2_)]

            sites_.extend(b1list)
            sites2_.extend([True if i in b2list else False for i in b1list])

            sites__.append(sites_)
            sites2.append(sites2_)

    if unitcell > 1:
        sites = []
        if isinstance(unitcell, int):
            for i in range(unitcell):
                if i:
                    ns_ = nsites[-1][-1]
                    nsites = []
                    for p in sites:
                        nsites.append([q + ns_ + 1 for q in p])
                else:
                    nsites = sites__
                sites = [*sites, *nsites]
        else:
            int_sites = int(unitcell)
            frac_sites = int(len(sites__) * (unitcell - int_sites))
            for i in range(int_sites):
                if i:
                    ns_ = nsites[-1][-1]
                    nsites = []
                    for p in sites:
                        nsites.append([q + ns_ + 1 for q in p])
                else:
                    nsites = sites__
                sites = [*sites, *nsites]
            ns_ = sites[-1][-1]
            nsites = []
            for i in range(frac_sites):
                nsites.append([q + ns_ + 1 for q in sites__[i]])
            sites = [*sites, *nsites]
    elif unitcell < 1:
        frac_sites = int(len(sites__) * (unitcell))
        ns_ = sites__[-1][-1]
        nsites = []
        for i in range(frac_sites):
            nsites.append([q for q in sites__[i]])
        sites = nsites
    else:
        sites = sites__

    if self.be_type == "be2":
        fs = []
        if not self.self_match and not self.allcen:
            for i in range(len(sites) - 2):
                self.fsites.append(sites[i] + sites[i + 1] + sites[i + 2])
                fs.append([sites[i], sites[i + 1], sites[i + 2]])

            self.Nfrag = len(self.fsites)

            self.edge.append([fs[0][2]])
            for i in fs[1:-1]:
                self.edge.append([i[0], i[-1]])
            self.edge.append([fs[-1][0]])

            self.center.append([1])
            for i in range(self.Nfrag - 2):
                self.center.append([i, i + 2])
            self.center.append([self.Nfrag - 2])
            for ix, i in enumerate(self.fsites):
                tmp_ = []
                elist_ = [xx for yy in self.edge[ix] for xx in yy]

                for j in i:
                    if j not in elist_:
                        tmp_.append(i.index(j))

                self.ebe_weight.append([1.0, tmp_])

        elif self.allcen:
            sites_left = [-i - 1 for i in sites[0]]
            ns = len(sites[-1])
            sites_right = [i + ns for i in sites[-1]]

            self.fsites.append(sites_left + sites[0] + sites[1])
            fs.append([sites_left, sites[0], sites[1]])
            for i in range(len(sites) - 2):
                self.fsites.append(sites[i] + sites[i + 1] + sites[i + 2])
                fs.append([sites[i], sites[i + 1], sites[i + 2]])
            self.fsites.append(sites[-2] + sites[-1] + sites_right)
            fs.append([sites[-2], sites[-1], sites_right])

            self.Nfrag = len(self.fsites)
            for i in fs:  # [1:-1]:
                self.edge.append([i[0], i[-1]])

            self.center.append([self.Nfrag - 1, 1])
            for i in range(self.Nfrag - 2):
                self.center.append([i, i + 2])
            self.center.append([self.Nfrag - 2, 0])

            for ix, i in enumerate(self.fsites):
                tmp_ = []

                elist_ = [xx for yy in self.edge[ix] for xx in yy]
                for j in i:
                    if j not in elist_:
                        tmp_.append(i.index(j))
                self.ebe_weight.append([1.0, tmp_])
        else:
            for i in range(len(sites) - 2):
                self.fsites.append(sites[i] + sites[i + 1] + sites[i + 2])
                fs.append([sites[i], sites[i + 1], sites[i + 2]])
            self.Nfrag = len(self.fsites)

            # self.edge.append([fs[0][2]])
            for i in fs:  # [1:-1]:
                self.edge.append([i[0], i[-1]])
            # self.edge.append([fs[-1][0]])

            # Frag (0) outer edge is in Frag (-1)
            self.center.append([self.Nfrag - 1, 1])
            for i in range(self.Nfrag - 2):
                self.center.append([i, i + 2])
            # Frag (-1) outer edge is in Frag (0)
            self.center.append([self.Nfrag - 2, 0])

            for ix, i in enumerate(self.fsites):
                tmp_ = []
                if ix == 0:
                    tmp_.extend([i.index(k) for k in self.edge[ix][0]])
                elist_ = [xx for yy in self.edge[ix] for xx in yy]
                for j in i:
                    if j not in elist_:
                        tmp_.append(i.index(j))
                if ix == self.Nfrag - 1:
                    tmp_.extend([i.index(k) for k in self.edge[ix][1]])
                self.ebe_weight.append([1.0, tmp_])

        # center on each fragments ?? do we add more for PBC
        for i in range(self.Nfrag):
            self.centerf_idx.append([self.fsites[i].index(j) for j in fs[i][1]])
        print(self.fsites)

    elif self.be_type == "be3":
        fs = []
        if not self.self_match and not self.allcen:
            # change back to i,i+1,i+2
            for i in range(len(sites) - 4):
                self.fsites.append(
                    sites[i] + sites[i + 1] + sites[i + 2] + sites[i + 3] + sites[i + 4]
                )
                fs.append(
                    [sites[i], sites[i + 1], sites[i + 2], sites[i + 3], sites[i + 4]]
                )

            self.Nfrag = len(self.fsites)
            self.edge.append([fs[0][3], fs[0][4]])
            # change back 0->1
            for i in fs[1:-1]:
                self.edge.append([i[0], i[1], i[-2], i[-1]])
            self.edge.append([fs[-1][0], fs[-1][1]])

            self.center.append([1, 2])
            self.center.append([0, 0, 2, 3])
            for i in range(self.Nfrag - 4):
                self.center.append([i, i + 1, i + 3, i + 4])
            self.center.append(
                [self.Nfrag - 4, self.Nfrag - 3, self.Nfrag - 1, self.Nfrag - 1]
            )
            self.center.append([self.Nfrag - 3, self.Nfrag - 2])

            for i in range(self.Nfrag):
                self.centerf_idx.append([self.fsites[i].index(j) for j in fs[i][2]])
            for ix, i in enumerate(self.fsites):
                tmp_ = []
                elist_ = [xx for yy in self.edge[ix] for xx in yy]
                for j in i:
                    if j not in elist_:
                        tmp_.append(i.index(j))
                self.ebe_weight.append([1.0, tmp_])

        elif self.allcen:
            ns = len(sites[-1])
            sites_left1 = [-i - 1 for i in sites[0]]
            sites_left0 = [i - ns for i in sites_left1]
            sites_right0 = [i + ns for i in sites[-1]]
            sites_right1 = [i + ns for i in sites_right0]

            self.fsites.append(
                sites_left0 + sites_left1 + sites[0] + sites[1] + sites[2]
            )
            self.fsites.append(sites_left1 + sites[0] + sites[1] + sites[2] + sites[3])
            fs.append([sites_left0, sites_left1, sites[0], sites[1], sites[2]])
            fs.append([sites_left1, sites[0], sites[1], sites[2], sites[3]])
            for i in range(len(sites) - 4):
                self.fsites.append(
                    sites[i] + sites[i + 1] + sites[i + 2] + sites[i + 3] + sites[i + 4]
                )
                fs.append(
                    [sites[i], sites[i + 1], sites[i + 2], sites[i + 3], sites[i + 4]]
                )
            self.fsites.append(
                sites[-4] + sites[-3] + sites[-2] + sites[-1] + sites_right0
            )
            self.fsites.append(
                sites[-3] + sites[-2] + sites[-1] + sites_right0 + sites_right1
            )
            fs.append([sites[-4], sites[-3], sites[-2], sites[-1], sites_right0])
            fs.append([sites[-3], sites[-2], sites[-1], sites_right0, sites_right1])

            self.Nfrag = len(self.fsites)

            # self.edge.append([fs[0][3], fs[0][4]])
            # self.edge.append([fs[1][1], fs[1][3], fs[1][4]])
            for i in fs:  # [2:-2]:
                self.edge.append([i[0], i[1], i[-2], i[-1]])
            # self.edge.append([fs[-2][0],fs[-2][1],fs[-2][3]])
            # self.edge.append([fs[-1][0],fs[-1][1]])

            self.center.append([self.Nfrag - 2, self.Nfrag - 1, 1, 2])
            self.center.append([self.Nfrag - 1, 0, 2, 3])
            for i in range(self.Nfrag - 4):
                self.center.append([i, i + 1, i + 3, i + 4])
            self.center.append([self.Nfrag - 4, self.Nfrag - 3, self.Nfrag - 1, 0])
            self.center.append([self.Nfrag - 3, self.Nfrag - 2, 0, 1])

            for ix, i in enumerate(self.fsites):
                tmp_ = []

                elist_ = [xx for yy in self.edge[ix] for xx in yy]
                for j in i:
                    if j not in elist_:
                        tmp_.append(i.index(j))
                self.ebe_weight.append([1.0, tmp_])

        else:
            for i in range(len(sites) - 4):
                self.fsites.append(
                    sites[i] + sites[i + 1] + sites[i + 2] + sites[i + 3] + sites[i + 4]
                )
                fs.append(
                    [sites[i], sites[i + 1], sites[i + 2], sites[i + 3], sites[i + 4]]
                )

            Nfrag = len(self.fsites)
            self.Nfrag = Nfrag

            for i in fs:
                self.edge.append([i[0], i[1], i[-2], i[-1]])

            self.center.append([Nfrag - 2, Nfrag - 1, 1, Nfrag - 2])
            self.center.append([Nfrag - 1, 0, Nfrag - 2, Nfrag - 1])
            for i in range(self.Nfrag - 4):
                self.center.append([i, i + 1, i + 3, i + 4])

            if Nfrag > 2:
                self.center.append([Nfrag - 4, Nfrag - 3, Nfrag - 1, 0])
                self.center.append([Nfrag - 3, Nfrag - 2, 0, 1])

            for ix, i in enumerate(self.fsites):
                tmp_ = []
                if ix == 0:
                    for edg_ix in range(2):
                        tmp_.extend([i.index(k) for k in self.edge[ix][edg_ix]])
                elist_ = [xx for yy in self.edge[ix] for xx in yy]
                for j in i:
                    if j not in elist_:
                        tmp_.append(i.index(j))
                if ix == self.Nfrag - 1:
                    for edg_ix in range(2):
                        tmp_.extend([i.index(k) for k in self.edge[ix][edg_ix + 2]])
                self.ebe_weight.append([1.0, tmp_])

        # center on each fragments ?? do we add more for PBC
        for i in range(self.Nfrag):
            self.centerf_idx.append([self.fsites[i].index(j) for j in fs[i][2]])

    print(flush=True)
    print("  No. of fragments : ", self.Nfrag, flush=True)
    print(flush=True)

    if not self.be_type == "be1":
        for i in range(self.Nfrag):
            idx = []
            for j in self.edge[i]:
                idx.append([self.fsites[i].index(k) for k in j])
            self.edge_idx.append(idx)
        if not self.self_match and not self.allcen:
            for i in range(self.Nfrag):
                idx = []
                for j in range(len(self.center[i])):
                    idx.append(
                        [
                            self.fsites[self.center[i][j]].index(k)
                            for k in self.edge[i][j]
                        ]
                    )
                self.center_idx.append(idx)
        else:
            for i in range(self.Nfrag):
                idx = []

                for j in self.center[i]:
                    idx__ = []
                    for fidx, fval in enumerate(self.fsites[j]):
                        if not any(fidx in sublist for sublist in self.edge_idx[j]):
                            idx__.append(fidx)
                    idx.append(idx__)
                self.center_idx.append(idx)
