# Author(s): Oinam Romesh Meitei


def chain(self, mol, frozen_core=False, closed=False):
    """
    Hard coded linear chain fragment constructor
    """
    sites = []
    coreshift = 0
    for adx, bas in enumerate(mol.aoslice_by_atom()):
        start_ = bas[2]
        stop_ = bas[3]

        if not frozen_core:
            sites.append([i for i in range(start_, stop_)])
        else:
            start_ -= coreshift
            ncore_ = self.core_list[adx]
            stop_ -= coreshift + ncore_
            coreshift = ncore_ + coreshift
            sites.append([i for i in range(start_, stop_)])
    if closed:
        lnext = [i for i in self.kpt if i > 1]
        if not len(lnext) == 0:
            nk1 = lnext[0]
        else:
            print("Gamma point does not work")
            sys.exit()
        Ns = mol.aoslice_by_atom()[-1][3]

    if self.be_type == "be2":
        fs = []
        if closed:
            # change back to i,i+1,i+2
            sites_left = [-i - 1 for i in sites[0]]
            # sites_left = [i for i in sites[-1]]
            ns = len(sites[-1])
            sites_right = [i + ns for i in sites[-1]]

        if closed:
            self.fsites.append(sites_left + sites[0] + sites[1])
            fs.append([sites_left, sites[0], sites[1]])
        for i in range(mol.natm - 2):
            self.fsites.append(sites[i] + sites[i + 1] + sites[i + 2])
            fs.append([sites[i], sites[i + 1], sites[i + 2]])
        if closed:
            self.fsites.append(sites[-2] + sites[-1] + sites_right)
            fs.append([sites[-2], sites[-1], sites_right])

        self.Nfrag = len(self.fsites)

        if closed:
            for i in fs:  # [1:-1]:
                self.edge.append([i[0], i[-1]])
        else:
            self.edge.append([fs[0][2]])
            # change back 0->1
            for i in fs[1:-1]:
                self.edge.append([i[0], i[-1]])
            self.edge.append([fs[-1][0]])

        if closed:
            self.center.append([self.Nfrag - 1, 1])
            # self.center.append([0,2])
        else:
            self.center.append([1])
        for i in range(self.Nfrag - 2):
            self.center.append([i, i + 2])
        if closed:
            # self.center.append([self.Nfrag-3,self.Nfrag-1])
            self.center.append([self.Nfrag - 2, 0])
        else:
            self.center.append([self.Nfrag - 2])

        if closed:
            for ix, i in enumerate(self.fsites):
                tmp_ = []
                elist_ = [xx for yy in self.edge[ix] for xx in yy]
                for j in i:
                    if not j in elist_:
                        tmp_.append(i.index(j))
                self.ebe_weight.append([1.0, tmp_])

        for i in range(self.Nfrag):
            self.centerf_idx.append([self.fsites[i].index(j) for j in fs[i][1]])

    if self.be_type == "be3" and not closed:
        fs = []

        for i in range(mol.natm - 4):
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

    if self.be_type == "be3" and closed:
        fs = []
        ns = len(sites[-1])
        sites_left1 = [-i - 1 for i in sites[0]]
        sites_left0 = [i - ns for i in sites_left1]
        sites_right0 = [i + ns for i in sites[-1]]
        sites_right1 = [i + ns for i in sites_right0]

        self.fsites.append(sites_left0 + sites_left1 + sites[0] + sites[1] + sites[2])
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
        self.fsites.append(sites[-4] + sites[-3] + sites[-2] + sites[-1] + sites_right0)
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
                if not j in elist_:
                    tmp_.append(i.index(j))
            self.ebe_weight.append([1.0, tmp_])
        for i in range(self.Nfrag):
            self.centerf_idx.append([self.fsites[i].index(j) for j in fs[i][2]])

    if self.be_type == "be4" and not closed:
        fs = []
        for i in range(mol.natm - 6):
            self.fsites.append(
                sites[i]
                + sites[i + 1]
                + sites[i + 2]
                + sites[i + 3]
                + sites[i + 4]
                + sites[i + 5]
                + sites[i + 6]
            )
            fs.append(
                [
                    sites[i],
                    sites[i + 1],
                    sites[i + 2],
                    sites[i + 3],
                    sites[i + 4],
                    sites[i + 5],
                    sites[i + 6],
                ]
            )

        self.Nfrag = len(self.fsites)

        self.edge.append([fs[0][4], fs[0][5], fs[0][6]])
        for i in fs[1:-1]:
            self.edge.append([i[0], i[1], i[2], i[-3], i[-2], i[-1]])
        self.edge.append([fs[-1][0], fs[-1][1], fs[-1][2]])

        self.center.append([1, 2, 3])
        self.center.append([0, 0, 0, 2, 3, 4])
        self.center.append([0, 0, 1, 3, 4, 5])
        for i in range(self.Nfrag - 6):
            self.center.append([i, i + 1, i + 2, i + 4, i + 5, i + 6])

        self.center.append(
            [
                self.Nfrag - 6,
                self.Nfrag - 5,
                self.Nfrag - 4,
                self.Nfrag - 2,
                self.Nfrag - 1,
                self.Nfrag - 1,
            ]
        )
        self.center.append(
            [
                self.Nfrag - 5,
                self.Nfrag - 4,
                self.Nfrag - 3,
                self.Nfrag - 1,
                self.Nfrag - 1,
                self.Nfrag - 1,
            ]
        )
        self.center.append([self.Nfrag - 4, self.Nfrag - 3, self.Nfrag - 2])

        for i in range(self.Nfrag):
            self.centerf_idx.append([self.fsites[i].index(j) for j in fs[i][3]])

    if self.be_type == "be4" and closed:
        print("Will add this soon!")
        sys.exit()
    if not closed:
        for ix, i in enumerate(self.fsites):
            tmp_ = []
            elist_ = [xx for yy in self.edge[ix] for xx in yy]
            for j in i:
                if not j in elist_:
                    tmp_.append(i.index(j))
            self.ebe_weight.append([1.0, tmp_])

    if not self.be_type == "be1":
        for i in range(self.Nfrag):
            idx = []
            for j in self.edge[i]:
                idx.append([self.fsites[i].index(k) for k in j])
            self.edge_idx.append(idx)

        if closed:
            for i in range(self.Nfrag):
                idx = []

                for j in self.center[i]:
                    idx__ = []
                    for fidx, fval in enumerate(self.fsites[j]):
                        if not any(fidx in sublist for sublist in self.edge_idx[j]):
                            idx__.append(fidx)
                    idx.append(idx__)
                self.center_idx.append(idx)
        else:
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
