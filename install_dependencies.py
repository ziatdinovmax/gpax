import pytoml as toml
import subprocess
import sys

def install_dependencies():
    try:
        # Load the pyproject.toml file
        with open('pyproject.toml', 'r') as f:
            pyproject = toml.load(f)
        print("Successfully loaded pyproject.toml")

        # Extract dependencies
        dependencies = pyproject.get('project', {}).get('dependencies', [])
        optional_dependencies = pyproject.get('project', {}).get('optional-dependencies', {}).get('test', [])

        if not dependencies:
            print("No dependencies found in pyproject.toml")
            return

        # Install each core dependency
        for package_spec in dependencies:
            print(f"Installing {package_spec}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])

        # Optionally install test dependencies
        if optional_dependencies:
            print("\nInstalling test dependencies...")
            for package_spec in optional_dependencies:
                print(f"Installing {package_spec}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])

    except FileNotFoundError:
        print("pyproject.toml file not found.")
    except toml.TomlError as e:
        print(f"Error parsing pyproject.toml: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    install_dependencies()
