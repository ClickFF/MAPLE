import os

from malepso.function.engine import engine

if __name__ == '__main__':
    engine = engine()
    
    test_control = 1

    software_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if test_control == 1:
        path = os.path.join(software_dir, 'example', 'opt', 'exp-solv', 'solv2.inp')
        
    # TS: NEB
    if test_control == 2: 
        path = os.path.join(software_dir, 'example', 'ts', 'neb', 'inp2.inp')
    
    # TS: String
    if test_control == 3:
        path = os.path.join(software_dir, 'example', 'ts', 'string', 'inp1.inp')
        
    # TS: Dimer
    if test_control == 4:
        path = os.path.join(software_dir, 'example', 'ts', 'dimer', 'inp1.inp')

    # AIMNet2 LBFGS
    if test_control == 5:
        path = os.path.join(software_dir, 'example', 'opt', 'rfo', 'inp1.inp')
    
    # IRC GS
    if test_control == 6:
        path = os.path.join(software_dir, 'example', 'irc', 'gs', 'inp1.inp')

    # Frequency MW
    if test_control == 7:
        path = os.path.join(software_dir, 'example', 'freq', 'mw', 'inp1.inp')

    engine(path)
