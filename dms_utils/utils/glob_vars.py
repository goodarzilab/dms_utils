
num_to_nt = {
             0 : 'A',
             1 : 'C',
             2 : 'G',
             3 : 'T',
            -1 : '-'
            }

nt_to_num = {
             'A' : 0,
             'C' : 1,
             'G' : 2,
             'T' : 3
            }

bitstring_letters_to_num = {
    'N' : -1,
    '.' : -1,
    '?' : -1,
    '1' : 0, # deletions
    '0' : 0,
    'A' : 1,
    'C' : 1,
    'G' : 1,
    'T' : 1
}