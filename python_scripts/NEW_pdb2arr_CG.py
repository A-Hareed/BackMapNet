import numpy as np
import sys
import os

# --- Command Line Argument Handling ---
# Ensure the correct number of arguments are provided:
# script_name, cg_pdb_file_path, output_array_path, sequence_name
if len(sys.argv) != 4:
    print("Usage: python your_script_name.py <cg_pdb_file_path> <output_array_path> <sequence_name>")
    sys.exit(1) # Exit the script if command line arguments are incorrect

cg_pdb_file_path = sys.argv[1]
output_array_path = sys.argv[2]
sequence_name = sys.argv[3] # The third argument is used for naming the sequence file
skip_sequence_write = os.environ.get("PDB2ARR_SKIP_SEQUENCE", "0") == "1"

# List to temporarily store coordinates for the current model being processed
current_model_coords = []
# List to store the flattened coordinate arrays for each model found in the PDB file
# The final NumPy array will be created from this list.
full_data = []

# List to store the sequence of residue names.
# We will collect the sequence from the first model encountered that has BB beads.
current_sequence = []
# Variable to track the residue sequence number of the last processed BB bead for sequence extraction.
# Initialized to None, will be updated with the first residue ID found.
processed_res_id = None

# Flag to indicate if the sequence has already been extracted (only extract from the first model).
sequence_extracted = False

# --- Coarse-grained PDB Parsing for BB Beads ---
try:
    # Open the coarse-grained PDB file for reading
    with open(cg_pdb_file_path, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # --- Process ATOM records ---
            # Check if the line starts with "ATOM".
            # Ensure the line is long enough to contain necessary fields for filtering and coordinates.
            # We need up to column 54 (index 53) for coordinates. We also need atom/residue names and res seq.
            if line.startswith('ATOM') and len(line) >= 54:
                try:
                    # --- Extract Bead Information for Filtering and Sequence ---
                    # Extract Atom Name (Bead Type) - columns 13-16 (0-based index 12-15)
                    atom_name = line[12:16].strip()

                    # Extract Residue Name - columns 18-21 (0-based index 17-20)
                    residue_name = line[17:21].strip()

                    # Extract Residue Sequence Number - columns 23-26 (0-based index 22-25)
                    # Convert residue sequence number to an integer
                    res_seq = int(line[22:26].strip())


                    # --- Filtering for BB bead ---
                    # Select only beads with the atom name 'BB'.
                    # We apply this filter *before* extracting coordinates to improve efficiency.
                    if atom_name == 'BB':
                        # --- Coordinate Extraction (ONLY if the 'BB' filter passes) ---
                        try:
                            # Extract x, y, z coordinate strings using fixed-width slicing
                            # x: columns 31-38 (index 30-37)
                            # y: columns 39-46 (index 38-45)
                            # z: columns 47-54 (index 46-53)
                            # Strip whitespace and convert the strings to float numbers
                            x_str = line[30:38].strip()
                            y_str = line[38:46].strip()
                            z_str = line[46:54].strip()

                            x = float(x_str)
                            y = float(y_str)
                            z = float(z_str)

                            # --- Collect Coordinates ---
                            # Extend the list for the current model with the x, y, and z coordinates.
                            # This collects the coordinates for the selected 'BB' beads, flattened within the model.
                            current_model_coords.extend([x, y, z])

                            # --- Sequence Extraction (ONLY for the first model and if a BB bead is found) ---
                            # Collect the sequence of residue names based on the 'BB' beads encountered in the first model.
                            # We check for changes in residue sequence number to identify new residues.
                            # Exclude 'GLY' from the sequence as per your original script's logic.
                            if not sequence_extracted and residue_name != 'CRAZY':
                                # If this is the first residue encountered, or if the residue sequence number has changed
                                if processed_res_id is None or res_seq != processed_res_id:
                                    if res_seq == 1 and len(current_sequence) > 1:
                                        current_sequence.append("|")
                                        current_sequence.append(residue_name)
                                    else:
                                        current_sequence.append(residue_name)
#                                   current_sequence.append(residue_name) # Add the residue name to the sequence list
                                    processed_res_id = res_seq # Update the last processed residue sequence number


                        except ValueError:
                            # Handle errors if coordinate fields are present but cannot be converted to float
                            print(f"Warning: Could not parse coordinates as valid numbers from line: {line.strip()}", file=sys.stderr)
                        except IndexError:
                             # This specific IndexError check is less likely if len(line) >= 54 is done earlier,
                             # but included for safety within the coordinate parsing block.
                             print(f"Warning: Line is too short for coordinates after BB filter: {line.strip()}", file=sys.stderr)

                except ValueError:
                     # Handle errors if residue sequence number cannot be converted to integer
                     print(f"Warning: Could not parse residue sequence number as a valid integer from line: {line.strip()}", file=sys.stderr)
                except IndexError:
                     # Handle cases where line is too short to extract bead info (atom/res name/seq)
                     print(f"Warning: Line is too short to extract bead information (atom/res name/seq): {line.strip()}", file=sys.stderr)


            # --- Handle ENDMDL records ---
            # "ENDMDL" marks the end of a model in a multi-model PDB (trajectory).
            # When encountered, process the coordinates collected for the just-finished model.
            elif line.startswith('ENDMDL'):
                # If we have collected coordinates for the current model (i.e., it wasn't empty)
                if current_model_coords:
                    # Append the list of coordinates for this model to the main data list.
                    # Since we used .extend above, current_model_coords is already a flat list for this model.
                    full_data.append(current_model_coords)

                    # Reset the list to start collecting coordinates for the next model
                    current_model_coords = []

                    # Mark sequence as extracted after the first model's data is added to full_data.
                    # This ensures sequence is only taken from the first complete model found.
                    if not sequence_extracted:
                         sequence_extracted = True
                         # Reset processed_res_id for the next model (sequence won't be collected)
                         processed_res_id = None

                # If ENDMDL is encountered but current_model_coords is empty (e.g., empty model),
                # we just reset the processed_res_id for safety, although sequence extraction is off.
                else:
                     processed_res_id = None


            # This script currently ignores other common PDB record types (e.g., MODEL, TER, HETATM).
            # If you need to handle these, add specific 'elif line.startswith(...):' blocks here.
            # Processing 'MODEL' records could be used to reset state for a new model if needed.

        # --- Handle the last model ---
        # After the loop finishes, if current_model_coords is not empty, it means the last model's
        # data has been collected but ENDMDL was not present at the very end (common in single-model files
        # or if the last ENDMDL is missing). Append these final collected coordinates.
        if current_model_coords:
             full_data.append(current_model_coords)
             # Ensure sequence is marked as extracted if it was the only model processed
             if not sequence_extracted:
                  sequence_extracted = True


# --- Error Handling for File Reading ---
except FileNotFoundError:
    # Handle the case where the input coarse-grained PDB file does not exist
    print(f"Error: Coarse-grained PDB file not found at {cg_pdb_file_path}", file=sys.stderr)
    sys.exit(1) # Exit the script due to a missing input file
except Exception as e:
    # Catch any other unexpected errors during the file reading process
    print(f"An unexpected error occurred while reading the PDB file: {e}", file=sys.stderr)
    sys.exit(1) # Exit the script due to a reading error


if not skip_sequence_write:
    # --- Save Sequence File ---
    # Define the name for the sequence file using the provided sequence_name argument
    sequence_file_name = f'sequence_{sequence_name}_new.txt'

    # Check if the sequence file already exists
    if not os.path.exists(sequence_file_name):
        # Only attempt to save the sequence if it was successfully collected
        if current_sequence:
            try:
                # Join the sequence list into a comma-separated string
                sequence_string = ','.join(current_sequence)
                # Open the file in write mode ('w') and save the sequence string
                with open(sequence_file_name, 'w') as f:
                    f.write(sequence_string)
                print(f"Sequence saved to {sequence_file_name}")
            except Exception as e:
                # Handle any errors during writing the sequence file
                print(f"Error writing sequence file {sequence_file_name}: {e}", file=sys.stderr)
        else:
            # Inform the user if no sequence data was collected
            print(f"No sequence collected (possibly no non-GLY BB beads found in the first model). "
                  f"Sequence file {sequence_file_name} will not be created.", file=sys.stderr)
    else:
        # Inform the user if the sequence file already exists and is being skipped
        print(f"Sequence file {sequence_file_name} already exists. Skipping sequence saving for this run.")


# --- Create NumPy Array from Collected Coordinate Data ---
# Proceed only if data for at least one model was successfully collected
if full_data:
    try:
        # Convert the list of flattened model coordinate lists into a NumPy array.
        # The shape will be (Number of models processed from the PDB, Total coordinates per model).
        # Use float32 for memory efficiency, common for coordinates.
        # np.array will raise a ValueError if the lists within full_data have inconsistent lengths,
        # which would happen if models have different numbers of collected BB beads.
        arr = np.array(full_data, dtype=np.float32)

    except ValueError:
         # Handle the case where models have different numbers of collected BB bead coordinates
         print("Error: Models in the PDB file have inconsistent numbers of collected BB bead coordinates. Cannot create a uniform NumPy array.", file=sys.stderr)
         sys.exit(1) # Exit due to data inconsistency preventing array creation
    except Exception as e:
        # Handle other potential errors during NumPy array creation
        print(f"An error occurred while creating the NumPy array: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Save/Concatenate NumPy Array to File ---
    # This section handles saving the coordinate data array.
    try:
        # Check if the output file specified by output_array_path already exists
        if os.path.exists(output_array_path):
            # If the file exists, load the data currently stored in it
            print(f"File {output_array_path} exists. Loading existing data and concatenating...")

            existing_array = np.load(output_array_path)

            # --- Compatibility Check Before Concatenation ---
            # Ensure that the existing array loaded from the file and the new array (arr)
            # have the same number of columns (the second dimension).
            # This is ABSOLUTELY ESSENTIAL for np.concatenate along axis=0 (stacking rows).
            # All processed PDBs must yield arrays with the same number of coordinates per model.
            if existing_array.shape[1] != arr.shape[1]:
                print(f"Error: Cannot concatenate arrays. Existing array's number of coordinates ({existing_array.shape[1]}) "
                      f"is incompatible with the current array's number of coordinates ({arr.shape[1]}). "
                      f"All processed PDBs must result in models with the same number of BB bead coordinates for concatenation.", file=sys.stderr)
                sys.exit(1) # Exit due to incompatible shapes for concatenation

            # Concatenate the existing array with the new array (from the current PDB) along axis 0.
            # This adds the models processed from the current PDB as new rows at the end of the existing data.
            updated_array = np.concatenate((existing_array, arr), axis=0)

            # Save the updated (concatenated) array back to the same file, overwriting the old file.
            np.save(output_array_path, updated_array)
            print(f"Array concatenated and saved to {output_array_path}. New shape: {updated_array.shape}")

        else:
            # If the output file does not exist, save the array from the current PDB
            # as the initial content of the new file. This creates the file.
            np.save(output_array_path, arr)
            print(f"New array created and saved to {output_array_path} with shape {arr.shape}")

    except Exception as e:
        # Handle any errors that occur during the file loading, concatenation, or saving process
        print(f"An error occurred during the file saving/concatenation process: {e}", file=sys.stderr)
        sys.exit(1) # Exit due to a saving error

else:
    # Handle the case where no bead coordinate data was collected from the current PDB file.
    # This happens if no 'ATOM' lines matched the filtering criteria or if the file was empty.
    print(f"No BB bead coordinates found in {cg_pdb_file_path} after processing. "
          f"No coordinate data will be processed or saved for this PDB run.", file=sys.stderr)
    # The script finishes here gracefully if no relevant data is found.
