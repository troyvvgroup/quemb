# Author(s): Hannah Y. Liu

import numpy as np
import os
import psutil

from pyscf import cc, gto, scf
from quemb.molbe import BE, fragmentate

# ====================================
# Part -1: Independent of this project
# ====================================
def print_memory_usage():
    """Prints the current RAM usage of this script in Megabytes."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEMORY] Current RAM Usage: {mem_mb:.2f} MB")

# ===================================================================
# Part 0: Mock function to mimic Alexa's excite state averaged output
#   In reality, I believe I need to feed in the coordinates of the frag
#   and use Alexa's code to calculate the e.s. energies for each frag.
#   However, now, I'm just going to generate random energies for code testing purposes.
# ===================================================================
def mock_excited_state_averaged(frag, num_es):
    """
    Simulates Alexa's excited state averaged outputs:
    currently generates energies for each fragment and the whole molecule
    returns a single object raw result.
    Eventually, I would need to run multiple of her functions to get all the inputs needed
    """
    num_frag = len(frag)
    ordered_frag = list(frag) # default fragment order her code uses

    # list of n arrays, [np.array([frag1_es1, frag2_es1, ...]), np.array([frag1_es2, frag2_es2, ...]), ...]
    frag_ener_by_state = [] 
    for state_idx in range(1, num_es + 1):
        # random energy for each fragment for this specific excited state
        state_array = np.random.uniform(0.5, 10.0, size=(num_frag,))
        frag_ener_by_state.append(state_array)

    # simulate whole molecule energy as (sum of fragments / 2)
    whole_molecule_ener_by_state = []
    for state_array in frag_ener_by_state:
        simulated_total = np.sum(state_array) / 2.0
        whole_molecule_ener_by_state.append(simulated_total)

    # Pack everything into her raw output format
    mock_output = {
        "ordered_frag": ordered_frag,
        "frag_ener_by_state": frag_ener_by_state,
        "whole_molecule_ener_by_state": whole_molecule_ener_by_state
    }
    print_memory_usage() # TODO: remove later
    return mock_output

# ===========================================================
# Part 1: Cleaning function for standardized input to my code
#   Regardless of Alexa's code's input format 
#   (and to be compatible for future e.s. methods),
#   I will compile all the needed output from the excited state energy code (e.g. Alexa's code)
#   into a standardized format, which should contain all the info I need for computing e.s. grad.
#   Currently, I do not plan to store things I do not need to be efficient,
#   and instead, I plan to add to this cleaning function if there are other things I need later
#   This function will be called everytime the excited state energy code is called to process the data
#   into the desired format, so I need to preventing storing too many things.
#   NOTE: Currently, this is designed to process mock_excited_averaged function
#   TODO: Once Alexa's code is merged into main, 
#   I need to tailor my cleaning function to be compatible with hers
# ===========================================================
def clean_inputs(raw_output):
    """
    Take the output of the excited state energy code 
    (e.g. Mock code now, but eventually the real thing, like Alexa's code),
    and extracts data needed for the excited state gradient part.
    Extra information not needed for es_grad is not stored but can be added to the list later.
    """
    clean_data = {
        "ordered_fragments": raw_output["ordered_frag"],
        "fragment_energies_by_state": raw_output["frag_ener_by_state"],
        "whole_molecule_energies_by_state": raw_output["whole_molecule_ener_by_state"]
    }
    print_memory_usage() # TODO: remove later
    return clean_data

# ====================
# Part 2: NOT DONE YET
# ====================

# =========================
# Final Part: local testing
# =========================
if __name__ == "__main__":
    # NOTE: testing mock function and clean function -- done
    # testing parameters
    my_fragments = ["Frag_A", "Frag_B", "Frag_C"]
    n_states = 2 
    # testing code
    raw_data = mock_excited_state_averaged(my_fragments, n_states)
    standardized_data = clean_inputs(raw_data)
    print("\n--- TEST RUN RESULTS ---")
    print(f"Fragment Order: {standardized_data['ordered_fragments']}")
    print(f"1st excited state Fragment Energies: {standardized_data['fragment_energies_by_state'][0]}")
    print(f"2nd excited state Fragment Energies: {standardized_data['fragment_energies_by_state'][1]}")
    print(f"Whole Molecule Energies by State: {standardized_data['whole_molecule_energies_by_state']}")
    print("------------------------\n")