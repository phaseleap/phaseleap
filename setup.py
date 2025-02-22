from setuptools import setup, find_packages


setup(
    name="phaseleap",  # This will be the package name for PyPI
    version="0.1.0",
    author="shant dv",  # Use a nickname or pseudonym if privacy is a concern
    author_email="placeholder@example.com",  # Placeholder for now
    description="PhaseLeap: Adaptive AI Optimization Through Phase Shifts",
    long_description='long_description placeholder',
    long_description_content_type="text/markdown",
    url="https://placeholder.com",  # Placeholder until GitHub repo exists
    project_urls={
        "Bug Tracker": "https://placeholder.com/issues",  # Placeholder
    },
    license="MIT",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy",
        "matplotlib",
        "scipy"
    ],
)
