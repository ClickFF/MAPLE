import os

from malepso.function.engine import engine

if __name__ == '__main__':
    engine = engine()
    
    test_control = 3

    software_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if test_control == 1:
        path = os.path.join(software_dir, 'example', 'ts', 'da.inp')
        
    # TS: NEB
    if test_control == 2: 
        path = os.path.join(software_dir, 'example', 'ts', 'neb', 'inp1.inp')
    
    # TS: String
    if test_control == 3:
        path = os.path.join(software_dir, 'example', 'ts', 'string', 'inp1.inp')
        
    # TS: Dimer
    if test_control == 4:
        path = os.path.join(software_dir, 'example', 'ts', 'dimer', 'inp1.inp')
        
    engine(path)