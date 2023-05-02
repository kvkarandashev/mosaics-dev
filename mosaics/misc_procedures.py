# Several auxiliary functions that appear everywhere.
from .data import NUCLEAR_CHARGE


def canonical_atomtype(atomtype):
    return atomtype[0].upper() + atomtype[1:].lower()


def checked_environ_val(
    environ_name: str, expected_answer=None, default_answer=None, var_class=int
):
    """
    Returns os.environ while checking for exceptions.
    """
    if expected_answer is None:
        try:
            args = (os.environ[environ_name],)
        except LookupError:
            if default_answer is None:
                args = tuple()
            else:
                args = (default_answer,)
        return var_class(*args)
    else:
        return expected_answer


default_parallel_backend = "multiprocessing"

num_procs_name = "MOSAICS_NUM_PROCS"


def default_num_procs(num_procs=None):
    return checked_environ_val(
        num_procs_name, expected_answer=num_procs, default_answer=1
    )


# Dictionnary which is inverse to NUCLEAR_CHARGE in .data
ELEMENTS = None


def str_atom_corr(ncharge):
    """ """
    global ELEMENTS
    if ELEMENTS is None:
        ELEMENTS = {}
        for cur_el, cur_ncharge in NUCLEAR_CHARGE.items():
            ELEMENTS[cur_ncharge] = cur_el
    return ELEMENTS[ncharge]


def int_atom(element):
    """
    Convert string representation of an element to nuclear charge.
    """
    return NUCLEAR_CHARGE[canonical_atomtype(element)]


def int_atom_checked(atom_id):
    """
    Check that input is integer; if string convert to nuclear charge.
    """
    if isinstance(atom_id, str):
        return int_atom(atom_id)
    else:
        return atom_id


# Auxiliary class used for smooth cutoff of positive weights.


class weighted_array(list):
    def normalize_rhos(self, normalization_constant=None):
        if normalization_constant is None:
            normalization_constant = sum(el.rho for el in self)
        for i in range(len(self)):
            self[i].rho /= normalization_constant

    def sort_rhos(self):
        self.sort(key=lambda x: x.rho, reverse=True)

    def normalize_sort_rhos(self):
        self.normalize_rhos()
        self.sort_rhos()

    def cutoff_minor_weights(self, remaining_rho=None):
        if (remaining_rho is not None) and (len(self) > 1):
            ignored_rhos = 0.0
            for remaining_length in range(len(self), 0, -1):
                upper_cutoff = self[remaining_length - 1].rho
                cut_rho = upper_cutoff * remaining_length + ignored_rhos
                if cut_rho > (1.0 - remaining_rho):
                    density_cut = (
                        1.0 - remaining_rho - ignored_rhos
                    ) / remaining_length
                    break
                else:
                    ignored_rhos += upper_cutoff
            del self[remaining_length:]
            for el_id in range(remaining_length):
                self[el_id].rho = max(
                    0.0, self[el_id].rho - density_cut
                )  # max was introduced in case there is some weird numerical noise.
            self.normalize_rhos()
