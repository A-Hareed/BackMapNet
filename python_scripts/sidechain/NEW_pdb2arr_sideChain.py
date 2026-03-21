import numpy as np
import sys
import os

# --- Command Line Argument Handling ---
# Ensure the correct number of arguments are provided
if len(sys.argv) != 3:
    print("Usage: python NEW_pdb2arr_sideChain.py <pdb_file_path> <output_array_path>")
    sys.exit(1)

pdb_file_path = sys.argv[1]
output_array_path = sys.argv[2]

# List to store all extracted heavy atom coordinates (x, y, z concatenated)
all_heavy_atom_coords = []

# --- PDB Parsing ---
try:
    with open(pdb_file_path, 'r') as f:
        for line in f:
            # Process only standard ATOM records
            # ATOM records start with "ATOM" and have a minimum length to contain coordinate data
            if line.startswith('ATOM') and len(line) >= 54: # Coordinates end at column 54

                # Extract Atom Name using fixed-width slicing (columns 13-16, 0-based index 12-15)
                # Strip whitespace as atom names can be right or left justified
                atom_name = line[12:16].strip()

                # --- Filtering ---
                # Filter out Hydrogen atoms (names starting with 'H'),
                # terminal Oxygen 'OXT', and OT* terminal oxygens (e.g., OT1/OT2)
                # Modify this filtering as needed for your specific requirements
                if not atom_name.startswith('H') and atom_name != 'OXT' and 'OT' not in atom_name:
                    try:
                        # --- Coordinate Extraction ---
                        # Extract x, y, z coordinates using fixed-width slicing
                        # Coordinates are typically in columns:
                        # x: 31-38 (index 30-37)
                        # y: 39-46 (index 38-45)
                        # z: 47-54 (index 46-53)
                        # Strip whitespace and convert to float

                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())

                        # --- Collect Coordinates ---
                        # Extend the list with the x, y, and z coordinates
                        all_heavy_atom_coords.extend([x, y, z])

                    except ValueError:
                        # Handle potential errors if coordinate fields are not valid numbers
                        print(f"Warning: Could not parse coordinates as floats from line: {line.strip()}", file=sys.stderr)
                    except IndexError:
                         # This should ideally not happen if len(line) check is sufficient, but good practice
                         print(f"Warning: Line is too short to contain coordinate fields: {line.strip()}", file=sys.stderr)

            # You can add checks for 'TER', 'MODEL', 'ENDMDL' if they affect which atoms to include,
            # but for a single flat array of *all* heavy atom coords, often they can be ignored
            # unless processing multi-model files and only want the first model.

# --- Error Handling for File Reading ---
except FileNotFoundError:
    # Handle the case where the input PDB file does not exist
    print(f"Error: PDB file not found at {pdb_file_path}", file=sys.stderr)
    sys.exit(1) # Exit the script due to missing input file
except Exception as e:
    # Catch any other unexpected errors during the file reading process
    print(f"An unexpected error occurred while reading the PDB file: {e}", file=sys.stderr)
    sys.exit(1) # Exit the script due to reading error

# --- Create and Reshape NumPy Array ---
# Proceed only if at least one set of heavy atom coordinates was found
if all_heavy_atom_coords:
    # Convert the list of all collected coordinates into a NumPy array
    # Using float32 typically provides sufficient precision and is memory-efficient
    coordinate_array = np.array(all_heavy_atom_coords, dtype=np.float32)

    # Reshape the 1D array into a 2D array with shape (1, N),
    # where N is the total number of coordinates (3 * number of heavy atoms processed)
    coordinate_array = coordinate_array.reshape(1, -1)

    # --- Save/Concatenate NumPy Array to File ---
    try:
        # Check if the output file already exists
        if os.path.exists(output_array_path):
            # If the file exists, load the existing data
            print(f"File {output_array_path} exists. Loading existing data and concatenating...")

            existing_array = np.load(output_array_path)

            # --- Compatibility Check Before Concatenation ---
            # Ensure that the existing array and the new array have the same number of columns
            # This is crucial because concatenation along axis=0 stacks rows, and all rows
            # must have the same length (the total number of coordinates per PDB).
            if existing_array.shape[1] != coordinate_array.shape[1]:
                print(f"Error: Cannot concatenate arrays. Existing array's number of coordinates ({existing_array.shape[1]}) "
                      f"is incompatible with the current array's number of coordinates ({coordinate_array.shape[1]}). "
                      f"All PDBs concatenated must result in the same number of heavy atom coordinates.", file=sys.stderr)
                # Exit the script if concatenation is not possible due to shape mismatch
                sys.exit(1) # Exit due to incompatible data shape

            # Concatenate the existing array with the new array along axis 0 (stacking rows)
            updated_array = np.concatenate((existing_array, coordinate_array), axis=0)

            # Save the updated (concatenated) array back to the same file, overwriting the old one
            np.save(output_array_path, updated_array)
            print(f"Array concatenated and saved to {output_array_path}. New shape: {updated_array.shape}")

        else:
            # If the output file does not exist, save the current array as the initial content
            # This creates the file with the first processed PDB's coordinates as the first row.
            np.save(output_array_path, coordinate_array)
            print(f"New array created and saved to {output_array_path} with shape {coordinate_array.shape}")

    except Exception as e:
        # Handle any errors that occur during the file loading, concatenation, or saving process
        print(f"An error occurred during the file saving/concatenation process: {e}", file=sys.stderr)
        sys.exit(1) # Exit due to saving error

else:
    # Handle the case where no heavy atom coordinates were found in the current PDB file after filtering
    print(f"No heavy atom coordinates (excluding H and OXT) found in {pdb_file_path} after filtering. "
          f"No data will be processed or saved for this PDB run.", file=sys.stderr)
    # The script will finish gracefully without saving any data if no relevant coordinates are found.
    # You could add an exit(1) here if you require every PDB to have data.
