# Copyright (C) 2017 Janine George

import numpy as np
import matplotlib.pyplot as plt
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN, parse_FORCE_CONSTANTS
from phonopy.units import VaspToCm, VaspToTHz, VaspToEv
from phonopy.phonon.irreps import IrReps


class AtomicContributionsCalculator(object):

    def __init__(
        self,
        poscar_name: str= 'POSCAR',
        forceconstants: bool=False,
        force_filename: str='FORCE_SETS',
        born_filename: str='BORN',
        supercell_matrix=None,
        nac=False,
        symprec=1e-5,
        masses: list=None,
        primitive=None,
        degeneracy_tolerance=1e-4,
        factor=VaspToCm,
        q=None
    ):
        """
        Class that calculates contributions of each atom to the phonon modes at Gamma

        Args:
            poscar_name (str): name of the POSCAR that was used for the phonon calculation
            born_filename (str): name of the file with BORN charges (formatted with outcar-born)
            forceconstants (boolean): If True, ForceConstants are read in. If False, forces are read in.
            force_filename (str): name of the file including force constants or forces
            supercell_matrix (list of lists): reads in supercell
            nac (boolean): If true, NAC is applied. (please be careful if you give a primitive cell. NAC should then be
            calculated for primitive cell)
            symprec (float): contains symprec tag as used in Phonopy
            masses (list): Masses in this list are used instead of the ones prepared in Phonopy. Useful for isotopes.
            primitive (list of lists): contains rotational matrix to arrive at primitive cell
            factor (float): VaspToCm or VaspToTHz or VaspToEv
            q (list of int): q point for the plot. So far only Gamma works

        """

        self._unitcell = read_vasp(poscar_name)
        self._supercell_matrix = supercell_matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        primitive = primitive or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._phonon = Phonopy(self._unitcell, supercell_matrix=self._supercell_matrix, primitive_matrix=primitive,
                               factor=factor, symprec=symprec)
        self.__natoms = self._phonon.get_primitive().get_number_of_atoms()
        self.__symbols = self._phonon.get_primitive().get_chemical_symbols()
        self.__factor = factor
        # If different masses are supplied
        if masses:
            self._phonon.set_masses(masses)
        self._masses = self._phonon.get_primitive().get_masses()

        # Forces or Force Constants
        if not forceconstants:
            self.__set_forcesets(filename=force_filename, phonon=self._phonon)

        if forceconstants:
            self.__set_force_constants(filename=force_filename, phonon=self._phonon)

        # Apply NAC Correction
        if nac:
            born_file = parse_BORN(self._phonon.get_primitive(), filename=born_filename)
            self.__BORN_CHARGES = born_file['born']
            self._phonon.set_nac_params(born_file)

        # frequencies and eigenvectors at Gamma
        if q:
            self.q = list(q)
        else:
            self.q = [0, 0, 0]
        self._frequencies, self._eigvecs = self._phonon.get_frequencies_with_eigenvectors(self.q)

        self.__NumberOfBands = len(self._frequencies)

        # Get Contributions
        self._set_contributions_eigenvector()
        self._set_contributions_eigendisplacements()

        # irrepsobject
        try:
            self.__set_irlabels(phonon=self._phonon,
                                degeneracy_tolerance=degeneracy_tolerance,
                                factor=factor,
                                q=self.q,
                                symprec=symprec)
        except:
            print(
                "Cannot assign IR labels. Play around with symprec, degeneracy_tolerance. The point group could not be implemented.")
            self.__freqlist = {}
            for i in range(0, len(self._frequencies)):
                self.__freqlist[i] = i

    def show_primitivecell(self):
        """
        shows primitive cell used for the plots and evaluations on screen
        """
        print(self._phonon.get_primitive())

    def __set_forcesets(self, filename, phonon):
        """
        sets forces

        """

        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()

    def __set_force_constants(self, filename, phonon):
        """
        sets force constants
        """
        force_constants = parse_FORCE_CONSTANTS(filename=filename)
        phonon.set_force_constants(force_constants)

    def __set_irlabels(self, phonon, degeneracy_tolerance, factor, q, symprec):
        """
        sets list of irreducible labels and list of frequencies without degeneracy
        """
        # phonon.set_dynamical_matrix()
        self.__Irrep = IrReps(dynamical_matrix=phonon._dynamical_matrix,
                              q=q,
                              is_little_cogroup=False,
                              nac_q_direction=None,
                              factor=factor,
                              symprec=symprec,
                              degeneracy_tolerance=degeneracy_tolerance)
        self.__Irrep.run()
        self._IRLabels = self.__Irrep._get_ir_labels()
        self.__ListOfModesWithDegeneracy = self.__Irrep._get_degenerate_sets()
        self.__freqlist = {}
        for band in range(len(self.__ListOfModesWithDegeneracy)):
            self.__freqlist[band] = self.__ListOfModesWithDegeneracy[band][0]

    def _eigenvector(self, atom, band):
        """
        Gives a certain eigenvector corresponding to one specific atom and band

        args:
            atom (int) : number of the atoms (same order as in POSCAR)
            band (int) : number of the frequency (ordered by energy)
        """

        return np.real(self._eigvecs[3*atom:3*atom+3, band])

    def _displacement_vector(self, atom, band):
        """
        Calculate eigendisplacement for specific atom and band

        args:
            atom (int) : number of the atoms (same order as in POSCAR)
            band (int) : number of the frequency (ordered by energy)
        """

        return self._eigenvector(atom, band) / np.sqrt(self._masses[atom])

    def contributions(self):
        """

        contributions : dict

        """

        return self._PercentageAtom

    def _set_contributions_eigenvector(self, squared=True):
        """
        Calculate contribution of each atom to modes
        """
        # TODO: Change this from dict to list/array
        self._PercentageAtom = {}
        modesum = []
        for band in range(len(self._frequencies)):
            modesum_here = 0
            for atom in range(self.__natoms):
                # Use phonon eigenvectors
                eigvec_real = np.linalg.norm(self._eigenvector(atom, band))
                if squared:
                    eigvec_real = np.square(eigvec_real)
                self._PercentageAtom[band, atom] = abs(eigvec_real)
                modesum_here = modesum_here + abs(eigvec_real)
            modesum.append(modesum_here)

        if self.q != [0, 0, 0]:
            import copy
            saver = copy.deepcopy(self._PercentageAtom)
            for band in range(len(self._frequencies)):
                for atom in range(self.__natoms):
                    self._PercentageAtom[band, atom] = saver[band, atom] / modesum[band]

    def _get_contributions_eigenvector(self, band, atom):
        """
        Gives contribution of specific atom to modes with certain frequency
        args:
            band (int): number of the frequency (ordered by energy)
            atom (int):
        """
        return self._PercentageAtom[band, atom]

    def _set_contributions_eigendisplacements(self):
        """
        Calculate contribution of each atom to modes
        Here, eigenvectors divided by sqrt(mass of the atom) are used for the calculation
        """
        # TODO: Change this from dict to list/array
        self.__PercentageAtom_massweight = {}
        atomssum = {}
        saver = {}
        for band in range(len(self._frequencies)):
            atomssum[band] = 0
            for atom in range(self.__natoms):
                eigvec_real = np.linalg.norm(self._displacement_vector(atom, band))
                eigvec_squared = np.square(eigvec_real)
                atomssum[band] = atomssum[band] + abs(eigvec_squared)

                # Hier muss noch was hin, damit rechnung richtig wird
                saver[band, atom] = abs(eigvec_squared)

        for band in range(len(self._frequencies)):
            for atom in range(self.__natoms):
                self.__PercentageAtom_massweight[band, atom] = saver[band, atom] / atomssum[band]

    def __get_contributions_withoutmassweight(self, band, atom):
        """
        Gives contribution of specific atom to modes with certain frequency
        Here, eigenvectors divided by sqrt(mass of the atom) are used for the calculation

        args:
                   band (int): number of the frequency (ordered by energy)
        """
        return self.__PercentageAtom_massweight[band, atom]

    def write_file(self, filename="Contributions.txt"):
        """
        Writes contributions of each atom in file

        args:

            filename (str): filename
        """
        file = open(filename, 'w')
        file.write('Frequency Contributions \n')
        for mode in range(len(self._frequencies)):
            file.write('%s ' % (self._frequencies[mode]))
            for atom in range(self.__natoms):
                file.write('%s ' % (self._get_contributions_eigenvector(mode, atom)))
            file.write('\n ')

        file.close()

    def plot(self,
             atomgroups,
             colorofgroups,
             legendforgroups,
             freqstart: float=None,
             freqend: float=None,
             freqlist: list=None,
             labelsforfreq: list=None,
             irreps_ax=True,
             transmodes=True,
             massincluded=True):
        """
        Plots contributions of atoms/several atoms to modes with certain frequencies (freqlist starts at 1 here)

        args:
            atomgroups (list of list of ints): list that groups atoms, atom numbers start at 1
            colorofgroups (list of str): list that matches a color to each group of atoms
            legendforgroups (list of str): list that gives a legend for each group of atoms
            freqstart (float): min frequency of plot in cm-1
            freqend (float): max frequency of plot in cm-1
            freqlist (list of int): list of frequencies that will be plotted; if no list is given all frequencies in the
            range from freqstart to freqend are plotted, list begins at 1
            labelsforfreq (list of str): list of labels (str) for each frequency
            filename (str): filename for the plot
            transmodes (boolean): if transmode is true than translational modes are shown
            massincluded (boolean): if false, uses eigenvector divided by sqrt(mass of the atom) for the calculation
            instead of the eigenvector
        """

        try:
            if not labelsforfreq:
                labelsforfreq = self._IRLabels
        except:
            print("")

        if not freqlist:
            freqlist = self.__freqlist
        else:
            for freq in range(len(freqlist)):
                freqlist[freq] = freqlist[freq] - 1

        newfreqlist = []
        newlabelsforfreq = []
        for freq in range(len(freqlist)):
            if not transmodes:
                if not freqlist[freq] in [0, 1, 2]:
                    newfreqlist.append(freqlist[freq])
                    try:
                        newlabelsforfreq.append(labelsforfreq[freq])
                    except:
                        newlabelsforfreq.append('')


            else:
                newfreqlist.append(freqlist[freq])
                try:
                    newlabelsforfreq.append(labelsforfreq[freq])
                except:
                    newlabelsforfreq.append('')

        return self._plot(atomgroups=atomgroups, colorofgroups=colorofgroups, legendforgroups=legendforgroups,
                   freqstart=freqstart, freqend=freqend, freqlist=newfreqlist, labelsforfreq=newlabelsforfreq,
                   irreps_ax=irreps_ax, massincluded=massincluded)

    def _plot(self,
              atomgroups,
              colorofgroups,
              legendforgroups,
              freqstart: float=None,
              freqend: float=None,
              freqlist: list=None,
              labelsforfreq: list=None,
              irreps_ax = True,
              massincluded=True):
        """
        Plots contributions of atoms/several atoms to modes with certain frequencies (freqlist starts at 0 here)

        Args:
            atomgroups (list of list of ints): list that groups atoms, atom numbers start at 1
            colorofgroups (list of str): list that matches a color to each group of atoms
            legendforgroups (list of str): list that gives a legend for each group of atoms
            freqstart (float): min frequency of plot in cm-1
            freqend (float): max frequency of plot in cm-1
            freqlist (list of int): list of frequencies that will be plotted; this freqlist starts at 0
            labelsforfreq (list of str): list of labels (str) for each frequency
            massincluded (boolean): if false, uses eigenvector divided by sqrt(mass of the atom) for the calculation
            instead of the eigenvector
        """
        # setting of some parameters in matplotlib: http://matplotlib.org/users/customizing.html
        # import os
        # mpl.rcParams["savefig.directory"] = os.chdir(os.getcwd())
        # mpl.rcParams["savefig.format"] = 'eps'

        fig, ax1 = plt.subplots()
        p = {}
        summe = {}

        for group in range(len(atomgroups)):
            color1 = colorofgroups[group]
            entry = {}
            for freq in range(len(freqlist)):
                entry[freq] = 0
            for number in atomgroups[group]:
                # set the first atom to 0
                atom = int(number) - 1
                for freq in range(len(freqlist)):
                    if massincluded:
                        entry[freq] = entry[freq] + self._get_contributions_eigenvector(freqlist[freq], atom)
                    else:
                        entry[freq] = entry[freq] + self.__get_contributions_withoutmassweight(freqlist[freq], atom)
                    if group == 0:
                        summe[freq] = 0

            # plot bar chart
            p[group] = ax1.barh(np.arange(len(freqlist)),
                                list(entry.values()),
                                left=list(summe.values()),
                                color=color1,
                                edgecolor="black",
                                height=1,
                                label=legendforgroups[group])
            # needed for "left" in the bar chart plot
            for freq in range(len(freqlist)):
                if group == 0:
                    summe[freq] = entry[freq]
                else:
                    summe[freq] = summe[freq] + entry[freq]
        labeling = {}
        for freq in range(len(freqlist)):
            labeling[freq] = round(self._frequencies[freqlist[freq]], 1)
        # details for the plot
        plt.rc("font", size=8)
        ax1.set_yticklabels(list(labeling.values()))
        ax1.set_yticks(np.arange(0.0, len(self._frequencies) + 0.0))
        # start and end of the yrange
        start, end = self.__get_freqbordersforplot(freqstart, freqend, freqlist)
        ax1.set_ylim(start - 0.5, end - 0.5)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel('Contribution of Atoms to Modes')
        if self.__factor == VaspToCm:
            ax1.set_ylabel('Wavenumber (cm$^{-1}$)')
        elif self.__factor == VaspToTHz:
            ax1.set_ylabel('Frequency (THz)')
        elif self.__factor == VaspToEv:
            ax1.set_ylabel('Frequency (eV)')
        else:
            ax1.set_ylabel('Frequency')
        ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                   ncol=len(atomgroups))
        if irreps_ax:
            ax2 = ax1.twinx()
            ax2.set_yticklabels(labelsforfreq)
            ax2.set_yticks(np.arange(0.0, len(self._frequencies) + 0.0))
            ax2.set_ylim(start - 0.5, end - 0.5)

        # plt.savefig(filename, bbox_inches="tight")
        plt.tight_layout()

        return plt

    def __get_freqbordersforplot(self, freqstart, freqend, freqlist):

        if not freqstart:
            start = 0.0
        else:
            for freq in range(len(freqlist)):
                if self._frequencies[freqlist[freq]] > freqstart:
                    start = freq
                    break
                else:
                    start = len(freqlist)
        if not freqend:
            end = len(freqlist)
        else:
            for freq in range(len(freqlist) - 1, 0, -1):
                if self._frequencies[freqlist[freq]] < freqend:
                    end = freq + 1
                    break
                else:
                    end = len(freqlist)

        return start, end

    def plot_irred(self,
                   atomgroups,
                   colorofgroups,
                   legendforgroups,
                   transmodes=False,
                   irreps: list=None,
                   irreps_ax=True,
                   freqstart: float=None,
                   freqend: float=None,
                   massincluded=True):
        """
        Plots contributions of atoms/several atoms to modes with certain irreducible representations (selected by
        Mulliken symbol)
        args:
            atomgroups (list of list of ints): list that groups atoms, atom numbers start at 1
            colorofgroups (list of str): list that matches a color to each group of atoms
            legendforgroups (list of str): list that gives a legend for each group of atoms
            transmodes (boolean): translational modes are included if true
            irreps (list of str): list that includes the irreducible modes that are plotted
            filename (str): filename for the plot
            massincluded (boolean): if false, uses eigenvector divided by sqrt(mass of the atom) for the calculation
            instead of the eigenvector
        """

        freqlist = []
        labelsforfreq = []
        for band in range(len(self.__freqlist)):
            if self._IRLabels[band] in irreps:
                if not transmodes:
                    if not self.__freqlist[band] in [0, 1, 2]:
                        freqlist.append(self.__freqlist[band])
                        labelsforfreq.append(self._IRLabels[band])
                else:
                    freqlist.append(self.__freqlist[band])
                    labelsforfreq.append(self._IRLabels[band])

        if freqlist:
            return self._plot(atomgroups=atomgroups, colorofgroups=colorofgroups, legendforgroups=legendforgroups,
                   freqlist=freqlist, labelsforfreq=labelsforfreq, freqstart=freqstart,
                   freqend=freqend, irreps_ax=irreps_ax, massincluded=massincluded)
        else:
            print("Empty irreducible representations plot.")
            return None
