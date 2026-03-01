import sys
import os

def check_module(name):
    print(f"Checking {name}...")
    try:
        mod = __import__(name, fromlist=['*'])
        print(f"  Attributes: {[a for a in dir(mod) if not a.startswith('__')]}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sys.path.insert(0, ".")
    for m in ["src.model.jem.config", "src.model.jem.model", "src.model.jem.sampler", "src.model.jem.loss", "src.model.jem.scaler", "src.model.jem.train"]:
        check_module(m)
