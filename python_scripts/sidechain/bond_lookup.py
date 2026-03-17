ATOM_GRAPH = {
    "LYS": {
        0: [("CB", ["CG"]), ("CG", ["CB", "CD"]), ("CD", ["CG"])],
        1: [("CE", ["NZ"]), ("NZ", ["CE"])]
    },
    "ARG": {
        0: [("CB", ["CG"]), ("CG", ["CB", "CD"]), ("CD", ["CG"])],
        1: [("NE", ["CZ"]), ("CZ", ["NE", "NH1", "NH2"]),
            ("NH1", ["CZ"]), ("NH2", ["CZ"])]
    },
    "GLU": {
        0: [("CB", ["CG"]), ("CG", ["CB", "CD"]),
            ("CD", ["CG", "OE1", "OE2"]),
            ("OE1", ["CD"]), ("OE2", ["CD"])]
    },
    "ASP": {
        0: [("CB", ["CG"]), ("CG", ["CB", "OD1", "OD2"]),
            ("OD1", ["CG"]), ("OD2", ["CG"])]
    },
    "ALA": {
        0: [("CB", [])]
    },
    # add others incrementally (safe approach)
    "CYS": { 0: [('CB',['SG']), ('SG',['CB'])]

    },
    "PRO": { 0: [("CB",["CG"]), ("CG",["CB","CD"]),
                 ("CD",["CG"])]

    },
    "VAL": {0: [("CB",["CG1","CG2"]), ("CG1",["CB"]), ("CG2",["CB"]) ]
            
    },
    "GLN": { 0: [("CB",["CG"]), ("CG", ["CB", "CD"]), ("CD", ["CG","OE1","NE2"]), 
                 ("OE1",["CD"]), ("NE2", ["CD"]) ]

    },
    "HIS": { 0: [("CB", ["CG"]), ("CG", ["CB"]) ],
             1: [("ND1", ["CE1"]), ("CE1", ["ND1"]) ],
             2: [("NE2", ["CD2"]), ("CD2", ["NE2"])]

    },
    "TYR": { 0: [("CB", ["CG"]), ("CG",["CB","CD1"]), ("CD1",["CG"])],
             1: [("CE1", ["CZ"]), ("CZ",["CE1","OH"]), ("OH", ["CZ"])],
             2: [("CE2",["CD2"]), ("CD2",["CE2"])]

    },
    "THR": { 0: [("CB",["OG1","CG2"]), ("OG1",["CB"]), ("CG2",["CB"])]

    },
    "ILE": { 0: [("CB",["CG1","CG2"]), ("CG1", ["CB","CD1"]), ("CG2", ["CB"]), ("CD1", ["CG1"]) ]

    },
    "LEU": { 0: [("CB",["CG"]), ("CG", ["CB","CD1","CD2"]), ("CD1",["CG"]), ("CD2", ["CG"])]
        
    },
    "SER": { 0: [("CB", ["OG"]), ("OG",["CB"])]  

    },
    "MET": { 0: [("CB",["CG"]), ("CG",["CB", "SD"]), ("SD", ["CG","CE"]), ("CE", ["SD"]) ]

    },
    "PHE": { 0: [("CB", ["CG"]), ("CG",["CB", "CD1"]), ("CD1", ["CG"]) ],
             1: [("CE1",["CZ"]), ("CZ", ["CE1"])],
             2: [("CD2",["CE2"]), ("CE2",["CD2"])]

    },
    "TRP": { 0: [("CB", ["CG"]), ("CG", ["CB", "CH2"]), ("CH2", ["CG"]) ],
             1: [("CD1",["NE1"]), ("NE1", ["CD1", "CE2"]), ("CE2", ["NE1"])],
             2: [("CZ3", ["CZ2"]), ("CZ2", ["CZ3"])],
             3: [("CD2",["CE3"]), ("CE3", ["CD2"])]

    },
    "ASN": { 0: [("CB", ["CG"]), ("CG", ["CB", "OD1", "ND2"]), ("OD1", ["CG"]), ("ND2", ["CG"]) ]

    }

}

INT_TO_AA = {0: 'ALA',
 1: 'ARG',
 2: 'ASN',
 3: 'ASP',
 4: 'CYS',
 5: 'GLN',
 6: 'GLU',
 7: 'HIS',
 8: 'ILE',
 9: 'LEU',
 10: 'LYS',
 11: 'MET',
 12: 'PHE',
 13: 'PRO',
 14: 'SER',
 15: 'THR',
 16: 'TRP',
 17: 'TYR',
 18: 'VAL'}



ATOM_ORDER = {
    "LYS": {
        0: ["CB","CG", "CD"],
        1: ["CE", "NZ"]
    },
    "ALA": {
        0: ["CB"]
    },
    "CYS": {
        0: ["CB","SG"]
    },
    "GLN": {
        0: ["CB","CG","CD","OE1","NE2"]       
    },
    "VAL": {
        0: ["CB","CG1", "CG2"]
    },
    "ASN": {
        0: ["CB","CG","OD1","ND2"]
    },
    "LEU": {
        0: ["CB","CG","CD1","CD2"]
    },
    "THR": {
        0: ["CB", "OG1","CG2"]
    },
    "PHE": {
        0: ["CB","CG","CD1"],
        1: ["CE1","CZ"],
        2: ["CD2","CE2"]
    },
    "SER": {
        0: ["CB","OG"]
    },
    "PRO": {
        0: ["CB","CG","CD"]
    },
    "TYR": {
        0: ["CB","CG","CD1"],
        1: ["CE1","CZ","OH"],
        2: ["CE2","CD2"]
    },
    "HIS": {
        0: ["CB","CG"],
        1: ["ND1","CE1"],
        2: ["NE2","CD2"]
    },
    "ARG": {
        0: ["CB","CG","CD"],
        1: ["NE","CZ","NH1","NH2"]
    }, 
    "TRP": {
        0: ["CB","CG","CH2"],
        1: ["CD1","NE1","CE2"],
        2: ["CZ3","CZ2"],
        3: ["CD2","CE3"]

    },
    "ILE": {
        0: ["CB","CG1","CG2","CD1"]
    },
    "GLU": {
        0: ["CB","CG","CD","OE1","OE2"]
    },
    "ASP": {
        0: ["CB","CG","OD1","OD2"]
    },
    "MET": { 
        0: ["CB","CG","SD","CE"]
    }
}
