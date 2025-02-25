from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from tdc import Oracle

jnk = Oracle('JNK3')
gsk = Oracle('GSK3B')

def oracle_ss(smiles):
		return [jnk(smiles), gsk(smiles)]
print(oracle_ss('C1=CC2=CN=NC2(Nc2ncccn2)C=C1'))
'''
logp = lambda smiles: MolLogP(Chem.MolFromSmiles(smiles))

full_smiles = "CCCCC"
max_step = 3
print(logp(""))
print("Initial molecule:", full_smiles)
for i in range(max_step):
    s_c = full_smiles + "C"
    s_s = full_smiles + "S"
    full_smiles = s_c if logp(s_c)>=logp(s_s) else s_s
    print(f"step{i}: {s_c}: logp={logp(s_c):.3f}, {s_s}: logp={logp(s_s):.3f}, result: {full_smiles}")
'''

