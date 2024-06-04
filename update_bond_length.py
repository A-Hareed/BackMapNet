import numpy as np

unrefined = # the predicted All_atoms coordinates
yhat = # predicted bond lengths. array of shape (frames, all backbone bond lengths)

def set_bond_length(atom1, atom2, new_length):
    pos2 = atom1[:,:3]
    pos1 = atom2[:,:3]
    
    # Calculate the direction vector (d)
    direction = pos2 - pos1
    
    # Calculate the current bond length
    current_length = np.linalg.norm(direction,axis=1).reshape(100,1)
    print(current_length.shape,direction.shape)
    # if current_length == 0:
    #     raise ValueError("Current bond length is zero, cannot update bond length.")
    
    # Normalize the direction vector to get the unit vector (u)
    unit_vector = direction / current_length
    
    # Calculate the new position of atom2
    new_pos2 = pos1 + unit_vector * new_length
    
    # Update the atom2 position
    # atom2['x'], atom2['y'], atom2['z'] = new_pos2
    return new_pos2


final_array = np.zeros((unrefined.shape[0],unrefined.shape[1]))


bond_index = 0
for residue in range(0,19764,12):
    current_array = unrefined[:,residue:residue+12]
    CB = current_array[:,:3]
    updated_N = set_bond_length(CB,current_array[:,3:6],yhat[:,bond_index].reshape(100,1))
    updated_CA = set_bond_length(CB,current_array[:,6:9],yhat[:,(bond_index+1)].reshape(100,1))
    updated_O = set_bond_length(updated_CA,current_array[:,9:],yhat[:,(bond_index+2)].reshape(100,1))
    bond_index+=3
    new_curr = np.concatenate((updated_N,CB,updated_CA,updated_O),axis=1)
    final_array[:,residue:residue+12] = new_curr
    
