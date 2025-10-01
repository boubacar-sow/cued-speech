#!/usr/bin/env python3
"""Test MFA detection with pixi environments."""

import sys
import os
import subprocess

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mfa_detection():
    """Test the MFA detection logic."""
    try:
        from cued_speech.generator import CuedSpeechGenerator
        
        # Create a generator instance
        generator = CuedSpeechGenerator()
        
        # Test the MFA detection
        mfa_path = generator._find_mfa_executable()
        
        if mfa_path:
            print(f"✓ MFA found at: {mfa_path}")
            
            # Test that the MFA executable actually works
            try:
                result = subprocess.run([mfa_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
                print(f"✓ MFA version: {result.stdout.strip()}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ MFA executable found but failed to run: {e}")
                return False
        else:
            print("✗ MFA not found in any expected location")
            print("\nExpected locations checked:")
            print("- System PATH")
            print("- Pixi environments (.pixi/envs/*/bin/mfa)")
            print("- Conda environments")
            print("- System conda installations")
            return False
            
    except Exception as e:
        print(f"✗ Error testing MFA detection: {e}")
        return False

def test_pixi_environment():
    """Test if we're in a pixi environment."""
    print("🔍 Checking pixi environment...")
    
    # Check if pixi is available
    try:
        result = subprocess.run(["pixi", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            print(f"✓ Pixi available: {result.stdout.strip()}")
        else:
            print("✗ Pixi not available")
            return False
    except FileNotFoundError:
        print("✗ Pixi not found in PATH")
        return False
    
    # Check for pixi environment
    pixi_env = os.environ.get('PIXI_ENVIRONMENT')
    if pixi_env:
        print(f"✓ Pixi environment: {pixi_env}")
    else:
        print("ℹ️  No PIXI_ENVIRONMENT variable set")
    
    # Check for .pixi directory
    if os.path.exists(".pixi"):
        print("✓ .pixi directory found")
        
        # List available environments
        pixi_envs_dir = ".pixi/envs"
        if os.path.exists(pixi_envs_dir):
            envs = [d for d in os.listdir(pixi_envs_dir) if os.path.isdir(os.path.join(pixi_envs_dir, d))]
            print(f"✓ Available pixi environments: {envs}")
            
            # Check for MFA in each environment
            for env in envs:
                mfa_path = os.path.join(pixi_envs_dir, env, "bin", "mfa")
                if os.path.exists(mfa_path):
                    print(f"✓ MFA found in {env} environment: {mfa_path}")
                else:
                    print(f"ℹ️  MFA not found in {env} environment")
            
            # Also check common pixi paths
            common_pixi_paths = [
                "~/.pixi/envs/default/bin/mfa",
                "~/.pixi/envs/dev/bin/mfa",
                "~/cued_speech/.pixi/envs/default/bin/mfa",
                "~/cued_speech/.pixi/envs/dev/bin/mfa"
            ]
            
            for path in common_pixi_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    print(f"✓ MFA found in common path: {expanded_path}")
                else:
                    print(f"ℹ️  MFA not found in: {expanded_path}")
        else:
            print("ℹ️  No .pixi/envs directory found")
    else:
        print("ℹ️  No .pixi directory found")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing MFA Detection with Pixi Support")
    print("=" * 50)
    
    # Test pixi environment first
    pixi_ok = test_pixi_environment()
    print()
    
    # Test MFA detection
    mfa_ok = test_mfa_detection()
    print()
    
    if mfa_ok:
        print("✅ MFA detection test PASSED")
        print("\n💡 If you're still having issues with cued-speech generate:")
        print("   1. Make sure you're in the pixi shell: pixi shell")
        print("   2. Or run with pixi: pixi run cued-speech generate ...")
        print("   3. Or activate the pixi environment manually")
    else:
        print("❌ MFA detection test FAILED")
        print("\n🔧 To fix this:")
        if pixi_ok:
            print("   1. Make sure MFA is installed in your pixi environment")
            print("   2. Run: pixi add montreal-forced-aligner")
            print("   3. Then try: pixi run cued-speech generate ...")
        else:
            print("   1. Install pixi: curl -fsSL https://pixi.sh/install.sh | bash")
            print("   2. Set up the project: pixi install")
            print("   3. Activate environment: pixi shell")
            print("   4. Run: pixi run cued-speech generate ...") 