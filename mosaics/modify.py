from .elementary_mutations import *
from .crossover import *
from .misc_procedures import (
    lookup_or_none,
    random_choice_from_dict,
    random_choice_from_nested_dict,
    str_atom_corr,
)
from .valence_treatment import canonically_permuted_ChemGraph, ChemGraph
from .ext_graph_compound import ExtGraphCompound


global_step_traj_storage_label = "global"
nonglobal_step_traj_storage_label = "local"


def tp_or_chemgraph(tp):
    if isinstance(tp, ChemGraph):
        return tp
    if isinstance(tp, TrajectoryPoint):
        return tp.chemgraph()
    raise Exception()


class TrajectoryPoint:
    def __init__(
        self,
        egc: ExtGraphCompound or None = None,
        cg: ChemGraph or None = None,
        num_visits: int or None = None,
    ):
        """
        This class stores an ExtGraphCompound object along with all the information needed to preserve detailed balance of the random walk.
        egc : ExtGraphCompound object to be stored.
        cg : ChemGraph object used to define egc if the latter is None
        num_visits : initial numbers of visits to the trajectory
        """
        if egc is None:
            if cg is not None:
                egc = ExtGraphCompound(chemgraph=cg)
        self.egc = egc

        if num_visits is not None:
            num_visits = deepcopy(num_visits)
        self.num_visits = num_visits

        self.visit_step_ids = {}
        self.visit_step_num_ids = {}

        self.first_MC_step_encounter = None
        self.first_global_MC_step_encounter = None

        self.first_MC_step_acceptance = None
        self.first_global_MC_step_acceptance = None

        self.first_encounter_replica = None
        self.first_acceptance_replica = None

        # The last time minimized function was looked up for the trajectory point.
        self.last_tot_pot_call_global_MC_step = None

        # Information for keeping detailed balance.
        self.possibility_dict = None

        self.modified_possibility_dict = None

        self.calculated_data = {}

    # TO-DO better way to write this?
    def init_possibility_info(self, **kwargs):
        # self.bond_order_change_possibilities is None - to check whether the init_* procedure has been called before.
        # self.egc.chemgraph.canonical_permutation - to check whether egc.chemgraph.changed() has been called.
        if self.possibility_dict is None:
            self.egc.chemgraph.init_resonance_structures()

            change_prob_dict = lookup_or_none(kwargs, "change_prob_dict")
            if change_prob_dict is None:
                return

            self.possibility_dict = {}
            for change_procedure in change_prob_dict:
                cur_subdict = {}
                pos_label = change_possibility_label[change_procedure]
                cur_pos_generator = possibility_generator_func[change_procedure]
                if pos_label is None:
                    cur_possibilities = cur_pos_generator(self.egc, **kwargs)
                    if len(cur_possibilities) != 0:
                        self.possibility_dict[change_procedure] = cur_possibilities
                else:
                    pos_label_vals = lookup_or_none(kwargs, pos_label)
                    if pos_label_vals is None:
                        raise Exception(
                            "Randomized change parameter "
                            + pos_label
                            + " undefined, leading to problems with "
                            + str(change_procedure)
                            + ". Check code input!"
                        )
                    for pos_label_val in pos_label_vals:
                        cur_possibilities = cur_pos_generator(
                            self.egc, pos_label_val, **kwargs
                        )
                        if len(cur_possibilities) != 0:
                            cur_subdict[pos_label_val] = cur_possibilities
                    if len(cur_subdict) != 0:
                        self.possibility_dict[change_procedure] = cur_subdict

            restricted_tps = lookup_or_none(kwargs, "restricted_tps")

            if restricted_tps is not None:
                self.clear_possibility_info(restricted_tps)

    def clear_possibility_info(self, restricted_tps):
        for poss_func, poss_dict in self.possibility_dict.items():
            poss_key_id = 0
            poss_keys = list(poss_dict.keys())
            while poss_key_id != len(poss_keys):
                poss_key = poss_keys[poss_key_id]
                poss_list = poss_dict[poss_key]

                poss_id = 0
                while poss_id != len(poss_list):
                    result = egc_change_func(
                        self.egc, poss_key, poss_list[poss_id], poss_func
                    )
                    if (result is None) or (
                        TrajectoryPoint(egc=result) in restricted_tps
                    ):
                        poss_id += 1
                    else:
                        del poss_list[poss_id]
                if len(poss_list) == 0:
                    del poss_dict[poss_key]
                    del poss_keys[poss_key_id]
                else:
                    poss_key_id += 1

    def possibilities(self, **kwargs):
        self.init_possibility_info(**kwargs)
        return self.possibility_dict

    def clear_possibility_info(self):
        self.modified_possibility_dict = None
        self.possibility_dict = None

    def calc_or_lookup(self, func_dict, args_dict=None, kwargs_dict=None):
        output = {}
        for quant_name in func_dict.keys():
            if quant_name not in self.calculated_data:
                if args_dict is None:
                    args = ()
                else:
                    args = args_dict[quant_name]
                if kwargs_dict is None:
                    kwargs = {}
                else:
                    kwargs = kwargs_dict[quant_name]
                func = func_dict[quant_name]
                calc_val = func(self, *args, **kwargs)
                # TODO make this hard approach on exceptions optional?
                # try:
                #    calc_val = func(self, *args, **kwargs)
                # except:
                #    print("Exception encountered while evaluating function ", func)
                #    print("Trajectory point:", self)
                #    print("Arguments:", args, kwargs)
                #    print("Previously calculated data:", self.calculated_data)
                #    quit()
                self.calculated_data[quant_name] = calc_val
            output[quant_name] = self.calculated_data[quant_name]
        return output

    def visit_num(self, replica_id):
        if self.num_visits is None:
            return 0
        else:
            return self.num_visits[replica_id]

    def mod_poss_dict_subdict(self, full_modification_path):
        cur_subdict = self.modified_possibility_dict
        for choice in full_modification_path:
            cur_subdict = cur_subdict[choice]
        return cur_subdict

    def delete_mod_poss_dict(self, full_modification_path):
        subdict = self.mod_poss_dict_subdict(full_modification_path[:-1])
        if isinstance(subdict, list):
            subdict.remove(full_modification_path[-1])
        if isinstance(subdict, dict):
            del subdict[full_modification_path[-1]]

    def delete_mod_path(self, full_modification_path):
        fmp_len = len(full_modification_path)
        while len(self.modified_possibility_dict) != 0:
            self.delete_mod_poss_dict(full_modification_path[:fmp_len])
            fmp_len -= 1
            if fmp_len == 0:
                break
            if len(self.mod_poss_dict_subdict(full_modification_path[:fmp_len])) != 0:
                break

    def copy_extra_data_to(self, other_tp, linear_storage=False, omit_data=None):
        """
        Copy all calculated data from self to other_tp.
        """
        for quant_name in self.calculated_data:
            if quant_name not in other_tp.calculated_data:
                if omit_data is not None:
                    if quant_name in omit_data:
                        continue
                other_tp.calculated_data[quant_name] = self.calculated_data[quant_name]
        # Dealing with making sure the order is preserved is too complicated.
        # if self.bond_order_change_possibilities is not None:
        #    if other_tp.bond_order_change_possibilities is None:
        #        other_tp.bond_order_change_possibilities = deepcopy(
        #            self.bond_order_change_possibilities
        #        )
        #        other_tp.chain_addition_possibilities = deepcopy(
        #            self.chain_addition_possibilities
        #        )
        #        other_tp.nuclear_charge_change_possibilities = deepcopy(
        #            self.nuclear_charge_change_possibilities
        #        )
        #        other_tp.atom_removal_possibilities = deepcopy(
        #            self.atom_removal_possibilities
        #        )
        #        other_tp.valence_change_possibilities = deepcopy(
        #            self.valence_change_possibilities
        #        )
        self.egc.chemgraph.copy_extra_data_to(
            other_tp.egc.chemgraph, linear_storage=linear_storage
        )

    def add_visit_step_id(
        self, step_id, beta_id, step_type=global_step_traj_storage_label
    ):
        if step_type not in self.visit_step_ids:
            self.visit_step_ids[step_type] = {}
            self.visit_step_num_ids[step_type] = {}
        if beta_id not in self.visit_step_num_ids[step_type]:
            self.visit_step_num_ids[step_type][beta_id] = 0
            self.visit_step_ids[step_type][beta_id] = np.array([-1])
        if (
            self.visit_step_num_ids[step_type][beta_id]
            == self.visit_step_ids[step_type][beta_id].shape[0]
        ):
            self.visit_step_ids[step_type][beta_id] = np.append(
                self.visit_step_ids[step_type][beta_id],
                np.repeat(-1, self.visit_step_num_ids[step_type][beta_id]),
            )
        self.visit_step_ids[step_type][beta_id][
            self.visit_step_num_ids[step_type][beta_id]
        ] = step_id
        self.visit_step_num_ids[step_type][beta_id] += 1

    def merge_visit_data(self, other_tp):
        """
        Merge visit data with data from TrajectoryPoint in another histogram.
        """
        if other_tp.num_visits is not None:
            if self.num_visits is None:
                self.num_visits = deepcopy(other_tp.num_visits)
            else:
                self.num_visits += other_tp.num_visits

        for step_type, other_visit_step_all_ids in other_tp.visit_step_ids.items():
            for beta_id, other_visit_step_ids in other_visit_step_all_ids.items():
                other_visit_step_num_ids = other_tp.visit_step_num_ids[step_type][
                    beta_id
                ]
                if other_visit_step_num_ids == 0:
                    continue
                if step_type not in self.visit_step_ids:
                    self.visit_step_ids[step_type] = {}
                    self.visit_step_num_ids[step_type] = {}
                if beta_id in self.visit_step_ids[step_type]:
                    new_visit_step_ids = SortedList(
                        self.visit_step_ids[step_type][beta_id][
                            : self.visit_step_num_ids[step_type][beta_id]
                        ]
                    )
                    for visit_step_id in other_visit_step_ids[
                        :other_visit_step_num_ids
                    ]:
                        new_visit_step_ids.add(visit_step_id)
                    self.visit_step_ids[step_type][beta_id] = np.array(
                        new_visit_step_ids
                    )
                    self.visit_step_num_ids[step_type][beta_id] = len(
                        new_visit_step_ids
                    )
                else:
                    self.visit_step_ids[step_type][beta_id] = deepcopy(
                        other_visit_step_ids
                    )
                    self.visit_step_num_ids[step_type][
                        beta_id
                    ] = other_tp.visit_step_num_ids[step_type][beta_id]

    def canonize_chemgraph(self):
        """
        Order heavy atoms inside the ChemGraph object according to canonical ordering. Used to make some tests consistent.
        """
        self.egc = ExtGraphCompound(
            chemgraph=canonically_permuted_ChemGraph(self.chemgraph())
        )
        self.clear_possibility_info()

    def chemgraph(self):
        return self.egc.chemgraph

    def __hash__(self):
        return hash(self.chemgraph())

    # TODO: Is comparison to ChemGraph objects worth the trouble?
    def __lt__(self, tp2):
        return self.chemgraph() < tp_or_chemgraph(tp2)

    def __gt__(self, tp2):
        return self.chemgraph() > tp_or_chemgraph(tp2)

    def __eq__(self, tp2):
        return self.chemgraph() == tp_or_chemgraph(tp2)

    def __str__(self):
        return str(self.egc)

    def __repr__(self):
        return str(self)


# Minimal set of procedures that allow to claim that our MC chains are Markovian.
# replace_heavy_atom is only necessary for this claim to be valid if we are constrained to molecules with only one heavy atom.
minimized_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
]

# Full list of procedures for "simple MC moves" available for simulation.
full_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    change_bond_order_valence,
]

# A list of operations mostly (?) sufficient for exploring chemical space where polyvalent heavy atoms are not protonated.
valence_ha_change_list = [
    add_heavy_atom_chain,
    remove_heavy_atom,
    replace_heavy_atom,
    change_bond_order,
    change_valence_add_atoms,
    change_valence_remove_atoms,
    change_bond_order_valence,
]


# For checking that ExtGraphCompound objects satisfy constraints of the chemical space.
def no_forbidden_bonds(egc: ExtGraphCompound, forbidden_bonds: None or list = None):
    """
    Check that an ExtGraphCompound object has no covalent bonds whose nuclear charge tuple is inside forbidden_bonds.
    egc : checked ExtGraphCompound object
    forbidden_bonds : list of sorted nuclear charge tuples.
    """
    if forbidden_bonds is not None:
        cg = egc.chemgraph
        hatoms = cg.hatoms
        for bond_tuple in cg.bond_orders.keys():
            if connection_forbidden(
                hatoms[bond_tuple[0]].ncharge,
                hatoms[bond_tuple[1]].ncharge,
                forbidden_bonds=forbidden_bonds,
            ):
                return False
    return True


def egc_valid_wrt_change_params(
    egc,
    nhatoms_range=None,
    forbidden_bonds=None,
    possible_elements=None,
    not_protonated=None,
    max_fragment_num=None,
    **other_kwargs,
):
    """
    Check that an ExtGraphCompound object is a member of chemical subspace spanned by change params used throughout chemxpl.modify module.
    egc : ExtGraphCompound object
    nhatoms_range : range of possible numbers of heavy atoms
    forbidden_bonds : ordered tuples of nuclear charges corresponding to elements that are forbidden to have bonds.
    """
    if not no_forbidden_bonds(egc, forbidden_bonds=forbidden_bonds):
        return False
    if not_protonated is not None:
        for ha in egc.chemgraph.hatoms:
            if (ha.ncharge in not_protonated) and (ha.nhydrogens != 0):
                return False
    if nhatoms_range is not None:
        nhas = egc.chemgraph.nhatoms()
        if (nhas < nhatoms_range[0]) or (nhas > nhatoms_range[1]):
            return False
    if possible_elements is not None:
        possible_elements_nc = [int_atom_checked(pe) for pe in possible_elements]
        for ha in egc.chemgraph.hatoms:
            if ha.ncharge not in possible_elements_nc:
                return False
    if max_fragment_num is not None:
        if egc.chemgraph.num_connected() > max_fragment_num:
            return False
    return True


# For randomly applying elementary mutations and maintaining detailed balance.

inverse_procedure = {
    add_heavy_atom_chain: remove_heavy_atom,
    remove_heavy_atom: add_heavy_atom_chain,
    replace_heavy_atom: replace_heavy_atom,
    change_bond_order: change_bond_order,
    change_valence: change_valence,
    change_valence_add_atoms: change_valence_remove_atoms,
    change_valence_remove_atoms: change_valence_add_atoms,
    change_bond_order_valence: change_bond_order_valence,
}

change_possibility_label = {
    add_heavy_atom_chain: "possible_elements",
    remove_heavy_atom: "possible_elements",
    replace_heavy_atom: "possible_elements",
    change_bond_order: "bond_order_changes",
    change_valence_add_atoms: "possible_elements",
    change_valence_remove_atoms: "possible_elements",
    change_valence: None,
    change_bond_order_valence: "bond_order_valence_changes",
}

possibility_generator_func = {
    add_heavy_atom_chain: chain_addition_possibilities,
    remove_heavy_atom: atom_removal_possibilities,
    replace_heavy_atom: atom_replacement_possibilities,
    change_bond_order: bond_change_possibilities,
    change_valence: valence_change_possibilities,
    change_valence_add_atoms: valence_change_add_atoms_possibilities,
    change_valence_remove_atoms: valence_change_remove_atoms_possibilities,
    change_bond_order_valence: valence_bond_change_possibilities,
}


def egc_change_func(
    egc_in: ExtGraphCompound,
    modification_path,
    change_function,
    chain_addition_tuple_possibilities=False,
    **other_kwargs,
) -> ExtGraphCompound:
    """
    Apply a modification defined through modification_path and change_function to ExtGraphCompound instance.
    """
    if (change_function is change_bond_order) or (
        change_function is change_bond_order_valence
    ):
        atom_id_tuple = modification_path[1][:2]
        resonance_structure_id = modification_path[1][-1]
        bo_change = modification_path[0]
        return change_function(
            egc_in,
            *atom_id_tuple,
            bo_change,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is remove_heavy_atom:
        removed_atom_id = modification_path[1][0]
        resonance_structure_id = modification_path[1][1]
        return change_function(
            egc_in,
            removed_atom_id,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is change_valence:
        modified_atom_id = modification_path[0]
        new_valence = modification_path[1][0]
        resonance_structure_id = modification_path[1][1]
        return change_function(
            egc_in,
            modified_atom_id,
            new_valence,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is add_heavy_atom_chain:
        added_element = modification_path[0]
        if chain_addition_tuple_possibilities:
            modified_atom_id = modification_path[1][0]
            added_bond_order = modification_path[1][1]
        else:
            modified_atom_id = modification_path[1]
            added_bond_order = modification_path[2]
        return change_function(
            egc_in,
            modified_atom_id,
            [added_element],
            [added_bond_order],
        )
    if change_function is replace_heavy_atom:
        inserted_atom_type = modification_path[0]
        replaced_atom_id = modification_path[1][0]
        resonance_structure_id = modification_path[1][1]
        return change_function(
            egc_in,
            replaced_atom_id,
            inserted_atom_type,
            resonance_structure_id=resonance_structure_id,
        )
    if change_function is change_valence_add_atoms:
        added_element = modification_path[0]
        modified_atom_id = modification_path[1]
        new_bond_order = modification_path[2]
        return change_function(egc_in, modified_atom_id, added_element, new_bond_order)
    if change_function is change_valence_remove_atoms:
        modified_atom_id = modification_path[1]
        removed_neighbors = modification_path[2][0]
        resonance_structure_id = modification_path[2][1]
        return change_function(
            egc_in,
            modified_atom_id,
            removed_neighbors,
            resonance_structure_id=resonance_structure_id,
        )
    raise Exception()


def inverse_mod_path(
    new_egc,
    old_egc,
    change_procedure,
    forward_path,
    chain_addition_tuple_possibilities=False,
    bond_change_ignore_equivalence=False,
    **other_kwargs,
):
    """
    Find modification path inverse to the forward_path.
    """
    if (change_procedure is change_bond_order) or (
        change_procedure is change_bond_order_valence
    ):
        if bond_change_ignore_equivalence:
            return [-forward_path[0], forward_path[1]]
        else:
            return [-forward_path[0]]
    if change_procedure is remove_heavy_atom:
        removed_atom = forward_path[-1][0]
        removed_elname = str_atom_corr(old_egc.chemgraph.hatoms[removed_atom].ncharge)
        if chain_addition_tuple_possibilities:
            return [removed_elname]
        else:
            neigh = old_egc.chemgraph.neighbors(removed_atom)[0]
            if removed_atom < neigh:
                neigh -= 1
            neigh = new_egc.chemgraph.min_id_equivalent_atom_unchecked(neigh)
            return [removed_elname, neigh]
    if change_procedure is replace_heavy_atom:
        changed_atom = forward_path[-1][0]
        inserted_elname = str_atom_corr(old_egc.chemgraph.hatoms[changed_atom].ncharge)
        return [inserted_elname]
    if change_procedure is add_heavy_atom_chain:
        return [forward_path[0]]
    if change_procedure is change_valence:
        return [new_egc.chemgraph.min_id_equivalent_atom_unchecked(forward_path[0])]
    if change_procedure is change_valence_add_atoms:
        return [
            forward_path[0],
            new_egc.chemgraph.min_id_equivalent_atom_unchecked(forward_path[1]),
            list(range(old_egc.num_heavy_atoms(), new_egc.num_heavy_atoms())),
        ]
    if change_procedure is change_valence_remove_atoms:
        modified_id = forward_path[1]
        new_modified_id = modified_id
        removed_ids = forward_path[2][0]
        for removed_id in removed_ids:
            if removed_id < modified_id:
                new_modified_id -= 1
        bo = old_egc.chemgraph.bond_order(modified_id, removed_ids[0])
        return [
            forward_path[0],
            new_egc.chemgraph.min_id_equivalent_atom_unchecked(new_modified_id),
            bo,
        ]
    raise Exception()


# Special change functions required for changing bond orders while ignoring equivalence.
def get_second_changed_atom_res_struct_list(
    egc: ExtGraphCompound,
    first_changed_atom,
    possible_atom_choices,
    bond_order_change,
    max_fragment_num=None,
    forbidden_bonds=None,
    **other_kwargs,
):
    """
    Which atoms inside an ExtGraphCompound can have their bond with first_changed_atom altered.
    """
    # Note accounting for not_protonated is not needed.
    output = []
    for pos_atom_choice in possible_atom_choices:
        if pos_atom_choice == first_changed_atom:
            continue
        res_structs = bond_order_change_possible_resonance_structures(
            egc,
            first_changed_atom,
            pos_atom_choice,
            bond_order_change,
            max_fragment_num=max_fragment_num,
            forbidden_bonds=forbidden_bonds,
        )
        if res_structs is None:
            continue
        for res_struct in res_structs:
            output.append((pos_atom_choice, res_struct))
    return output


def choose_bond_change_parameters_ignore_equiv(
    egc, possibilities, choices=None, **other_kwargs
):
    # possibilities is structured as dictionnary of "bond order change" : list of potential atoms.
    # First choose the bond order change:
    (
        bond_order_change,
        possible_atom_choices,
        first_atom_log_choice_prob,
    ) = random_choice_from_dict(possibilities, choices=choices)
    first_changed_atom = random.choice(possible_atom_choices)
    possible_second_changed_atom_res_struct_list = (
        get_second_changed_atom_res_struct_list(
            egc,
            first_changed_atom,
            possible_atom_choices,
            bond_order_change,
            **other_kwargs,
        )
    )
    second_atom_res_struct = random.choice(possible_second_changed_atom_res_struct_list)
    log_choice_prob = first_atom_log_choice_prob + np.log(
        float(len(possible_second_changed_atom_res_struct_list))
    )
    mod_path = [bond_order_change, (first_changed_atom, *second_atom_res_struct)]
    return mod_path, log_choice_prob


def inv_prob_bond_change_parameters_ignore_equiv(
    new_egc, inv_poss_dict, inv_mod_path, **other_kwargs
):
    inv_bo_change = inv_mod_path[0]
    first_changed_atom = inv_mod_path[1][0]
    possible_atom_choices, log_choice_prob = random_choice_from_dict(
        inv_poss_dict, get_probability_of=inv_bo_change
    )
    log_choice_prob -= np.log(float(len(possible_atom_choices)))
    second_atom_res_struct_choices = get_second_changed_atom_res_struct_list(
        new_egc,
        first_changed_atom,
        possible_atom_choices,
        inv_bo_change,
        **other_kwargs,
    )
    log_choice_prob -= np.log(float(len(second_atom_res_struct_choices)))
    return log_choice_prob


def randomized_change(
    tp: TrajectoryPoint,
    change_prob_dict=full_change_list,
    visited_tp_list: list or None = None,
    delete_chosen_mod_path: bool = False,
    bond_change_ignore_equivalence: bool = False,
    **other_kwargs,
):
    """
    Randomly modify a TrajectoryPoint object.
    visited_tp_list : list of TrajectoryPoint objects for which data is available.
    bond_change_ignore_equivalence : whether equivalence is accounted for during bond change moves (False is preferable for large systems).
    """
    init_possibilities_kwargs = {
        "change_prob_dict": change_prob_dict,
        "bond_change_ignore_equivalence": bond_change_ignore_equivalence,
        **other_kwargs,
    }
    if delete_chosen_mod_path:
        if tp.modified_possibility_dict is None:
            tp.modified_possibility_dict = deepcopy(
                tp.possibilities(**init_possibilities_kwargs)
            )
        full_possibility_dict = tp.modified_possibility_dict
        if len(full_possibility_dict) == 0:
            return None, None
    else:
        full_possibility_dict = tp.possibilities(**init_possibilities_kwargs)

    cur_change_procedure, possibilities, total_forward_prob = random_choice_from_dict(
        full_possibility_dict, change_prob_dict
    )
    special_bond_change_func = bond_change_ignore_equivalence and (
        cur_change_procedure is change_bond_order
    )

    possibility_dict_label = change_possibility_label[cur_change_procedure]
    possibility_dict = lookup_or_none(other_kwargs, possibility_dict_label)

    old_egc = tp.egc

    if special_bond_change_func:
        modification_path, forward_prob = choose_bond_change_parameters_ignore_equiv(
            old_egc,
            possibilities,
            choices=possibility_dict,
            **init_possibilities_kwargs,
        )
    else:
        modification_path, forward_prob = random_choice_from_nested_dict(
            possibilities, choices=possibility_dict
        )

    if delete_chosen_mod_path and (not special_bond_change_func):
        tp.delete_mod_path([cur_change_procedure] + modification_path)

    total_forward_prob += forward_prob

    new_egc = egc_change_func(
        old_egc, modification_path, cur_change_procedure, **other_kwargs
    )

    if new_egc is None:
        return None, None

    new_tp = TrajectoryPoint(egc=new_egc)
    if visited_tp_list is not None:
        if new_tp in visited_tp_list:
            tp_id = visited_tp_list.index(new_tp)
            visited_tp_list[tp_id].copy_extra_data_to(new_tp)
    new_tp.init_possibility_info(**init_possibilities_kwargs)
    # Calculate the chances of doing the inverse operation
    inv_proc = inverse_procedure[cur_change_procedure]
    inv_pos_label = change_possibility_label[inv_proc]
    inv_poss_dict = lookup_or_none(other_kwargs, inv_pos_label)

    try:
        inv_mod_path = inverse_mod_path(
            new_egc,
            old_egc,
            cur_change_procedure,
            modification_path,
            bond_change_ignore_equivalence=bond_change_ignore_equivalence,
            **other_kwargs,
        )
        inverse_possibilities, total_inverse_prob = random_choice_from_dict(
            new_tp.possibilities(),
            change_prob_dict,
            get_probability_of=inv_proc,
        )
        if special_bond_change_func:
            inverse_prob = inv_prob_bond_change_parameters_ignore_equiv(
                new_egc,
                inverse_possibilities,
                inv_mod_path,
                **init_possibilities_kwargs,
            )
        else:
            inverse_prob = random_choice_from_nested_dict(
                inverse_possibilities, inv_poss_dict, get_probability_of=inv_mod_path
            )
    except KeyError:
        print("NON-INVERTIBLE OPERATION")
        print(old_egc, cur_change_procedure)
        print(new_egc)
        quit()

    total_inverse_prob += inverse_prob

    prob_balance = total_forward_prob - total_inverse_prob

    if special_bond_change_func:
        prob_balance += (
            old_egc.chemgraph.get_log_permutation_factor()
            - new_egc.chemgraph.get_log_permutation_factor()
        )

    return new_tp, total_forward_prob - total_inverse_prob
