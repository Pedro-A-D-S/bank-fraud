import os

def create_folder_structure(project_name):
    """
    Creates the folder structure for the project.

    Parameters
    ----------
    project_name : str
        The name of the project.
    """
    # Create the main project directory
    os.makedirs(project_name)
    
    # Subdirectories
    subdirectories = [
        'data/raw',
        'data/processed',
        'data/images',
        'data/mlflow',
        'notebooks/exploratory',
        'notebooks/preprocessing',
        'notebooks/model_training',
        'src/data_preprocessing',
        'src/model',
        'src/serving',
        'models/random_forest',
        'dvc',
        'pipelines',
        'mlflow/registry',
        'docker',
        'monitoring/prometheus',
        'monitoring/grafana'
    ]
    
    # Create subdirectories
    for subdir in subdirectories:
        os.makedirs(os.path.join(project_name, subdir))
    
    # Create README.md
    readme_content = f"# {project_name}\n\nDescription of your project.\n\n## Setup\n\nInstructions for setting up the project.\n\n## Usage\n\nHow to use the project."
    with open(os.path.join(project_name, 'README.md'), 'w') as readme_file:
        readme_file.write(readme_content)

if __name__ == "__main__":
    project_name = "bank_fraud_detection_project"
    create_folder_structure(project_name)
    print(f"Project folder structure for '{project_name}' has been created.")
