from pdb2pqr import io, forcefield, hydrogens, debump, pdb
import inspect
from os.path import dirname
from docking.pdb2pqr_example.propka_script import run_propka
from docking.pdb2pqr_example.utility import setup_molecule, is_repairable,assign_titration_state, write_pqr
from docking.pdb2pqr_example.main_parser import build_main_parser
from propka.lib import build_parser as propka_build_parser
from io import StringIO

##? CONSTANTS :
PDB_FILE = "4ey7.pdb"
PQR_OUTPUT = "test.pqr"
PDB2PQ_DATA_DIR = dirname(inspect.getfile(forcefield)) + "/dat/" # FIELDS Default LOCATION, Might change depending on compilation problems
PARSE = PDB2PQ_DATA_DIR + "PARSE.DAT"  #PARSE FIELD
AMBER = PDB2PQ_DATA_DIR + "AMBER.DAT"  #AMBER FIELD
CHARMM = PDB2PQ_DATA_DIR + "CHARMM.DAT"  #CHARMM FIELD
PH = 7


def generate_target_H (PDB_FILE, PQR_OUTPUT, force_field="PARSE", USE_PROPKA=False, PH=7):
    PARSER = build_main_parser()
    PARSER = propka_build_parser(PARSER)

    if isinstance(PDB_FILE, StringIO):      # Try to circumvent ARG Parser to pass a StringIO() ... i hate this
        mockPDB_FILE = "random stuff"
        args = PARSER.parse_args([mockPDB_FILE, PQR_OUTPUT])
        args.input_path = PDB_FILE
    else:
        args = PARSER.parse_args([PDB_FILE, PQR_OUTPUT])
    args.ff = force_field
    args.ph = PH
    
    if (USE_PROPKA):
        args.pka_method = "propka"          # add this to args if using propka as titration method

    if isinstance(PDB_FILE, StringIO):
        ## IF YOUR FILE IS A CIF FILE, THIS WONT WORK AND YOU HAVE TO CALL read_cif from cif module .... i didn't test that
        pdb_file, errors = pdb.read_pdb(args.input_path)  #? USE THIS IF PASSING A STRING IO
        is_cif = False
    else:
        pdb_file, is_cif = io.get_molecule(args.input_path)  #? PDB to open ( can NOT be a StringIO, ONLY TRUE PATHS)

    bml, bmlDef, ligand = setup_molecule(pdb_file, io.get_definitions(), ligand_path= None) #! LIGAND IS NONE FOR TESTING ONLY
    
    ffield = forcefield.Forcefield(force_field, bmlDef, PARSE) # first arg doesnt matter if 3rd arg exist
    #? use elif in case we add other force fields
    if (force_field=="AMBER"):
        ffield = forcefield.Forcefield(force_field, bmlDef, AMBER) # first arg doesnt matter if 3rd arg exist
    elif(force_field=="CHARMM"):
        ffield = forcefield.Forcefield(force_field, bmlDef, CHARMM)
    
    hydrogen_handler = hydrogens.create_handler()
    debumper = debump.Debump(bml)
    hydrogen_routines = hydrogens.HydrogenRoutines(debumper, hydrogen_handler)


    # Update Bonds ? ( no idea what this does i'm just following implenetation )
    bml.update_bonds()
    if(is_repairable(bml, True)):
        bml.repair_heavy()

    # Optimize S-S bridges
    bml.update_ss_bridges()
    try: 
        debumper.debump_biomolecule()
    except:
        print("Cannot debump, skipping")

    # Remove Hydrogens if existing
    bml.remove_hydrogens()

    # Now generate titration states if using PROPKA
    pka_df = None
    if(USE_PROPKA):
        args.pka_method = "propka"
        pka_df, pka_str = run_propka(args=args, biomolecule=bml)  
        bml.apply_pka_values(
            ffield.name,
            PH,
            {f"{row['res_name']} {row['res_num']} {row['chain_id']}": row["pKa"] for row in pka_df if row["group_label"].startswith(row["res_name"])},
        )


    ## Add hydrogens
    bml.add_hydrogens()

    ##Optimize Hydrogens and all molecules, and then cleanup
    hydrogen_routines.set_optimizeable_hydrogens()
    bml.hold_residues(None)
    hydrogen_routines.initialize_full_optimization()
    hydrogen_routines.optimize_hydrogens()
    hydrogen_routines.cleanup()

    #Assign titration states according to force field
    try:
        matched_atoms, missing_atoms = assign_titration_state(bml, ffield)
    except:
        print("Error in titration assignment... aborting")

    ## Regenerate Headers
    reslist, charge = bml.charge
    header = io.print_pqr_header(
        bml.pdblist,
        missing_atoms,
        reslist,
        charge,
        str(ffield),
        None,
        PH,
        None,
        include_old_header= True,
    )

    lines = io.print_biomolecule_atoms(matched_atoms, True)

    results = {
        "lines": lines,
        "header": header,
        "missed_residues": missing_atoms,
        "pka_df": pka_df,
    }

    write_pqr(          # this will write to output file as specified by args, if not needed you can just print(lines)
        args=args,
        pqr_lines=results["lines"],
        header_lines=results["header"],
        missing_lines=results["missed_residues"],
        is_cif=is_cif,
    )
