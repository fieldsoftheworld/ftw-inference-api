#!/usr/bin/env python3
"""Quick GPU test script for the FTW inference environment."""

import sys
import os
from pathlib import Path

# Add server directory to Python path (same as run.py)
server_dir = Path(__file__).parent / "server"
sys.path.insert(0, str(server_dir))
os.chdir(server_dir)

def test_system_gpu():
    """Test system-level GPU availability."""
    print("System GPU Check:")

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi available")
            print(f"GPU Info: {result.stdout.split(chr(10))[8].strip()}")  # GPU line
        else:
            print("nvidia-smi failed")
    except Exception as e:
        print(f"nvidia-smi error: {e}")

def test_pytorch_gpu():
    """Test PyTorch GPU availability."""
    print("\nPyTorch GPU Check:")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current GPU: {torch.cuda.current_device()}")
            print(f"   GPU name: {torch.cuda.get_device_name(0)}")
            
            # Test tensor creation on GPU
            x = torch.randn(3, 3).cuda()
            print(f"   GPU tensor test: ({x.device})")
        else:
            print("CUDA not available in PyTorch")
            
    except ImportError:
        print("PyTorch not installed")
    except Exception as e:
        print(f"PyTorch GPU error: {e}")

def test_ftw_tools():
    """Test FTW tools import and GPU detection."""
    print("\nFTW Tools Check:")
    
    try:
        import_attempts = [
            ('ftw_tools', 'ftw_tools'),
            ('ftwtools', 'ftwtools'), 
            ('ftw.tools', 'ftw.tools'),
            ('ftw', 'ftw')
        ]
        
        imported = False
        for import_name, display_name in import_attempts:
            try:
                __import__(import_name)
                print(f"ftw-tools imported as '{import_name}'")
                imported = True
                break
            except ImportError:
                continue
        
        if not imported:
            print("Could not import ftw-tools with any common pattern")
            try:
                import pkgutil
                ftw_modules = [name for _, name, _ in pkgutil.iter_modules() if 'ftw' in name.lower()]
                if ftw_modules:
                    print(f"   Available FTW-related modules: {ftw_modules}")
            except:
                pass
        
    except Exception as e:
        print(f"ftw-tools error: {e}")

def test_config_gpu_setting():
    """Test the application's GPU configuration."""
    print("\n  App GPU Config Check:")
    
    try:
        from app.core.config import get_settings
        settings = get_settings()

        gpu_setting = settings.gpu
        print(f"   Configured GPU: {gpu_setting}")
        
        if gpu_setting is None:
            print("   Mode: CPU")
        elif isinstance(gpu_setting, int):
            print(f"   Mode: GPU {gpu_setting}")
        else:
            print(f"   Mode: {gpu_setting}")
            
    except Exception as e:
        print(f"❌ Config check error: {e}")

if __name__ == "__main__":
    print("FTW Inference API - GPU Environment Test\n")
    
    test_system_gpu()
    test_pytorch_gpu() 
    test_ftw_tools()
    test_config_gpu_setting()
    
    print("\n GPU test complete!")