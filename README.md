# sonar_utils
 Sonar utilities for pre-processing sidescan and optical image data

## Project Setup

### Prerequisities
Make sure you have Python 3.8 or later installed on your system.

### Installation

1. Clone the Repository
```
git clone sonar_utils
cd sonar_utils
```
2. Create a virtual env
```
python -m venv venv
```
3. Activate the virtual env
- on windows
```
venv\Scripts\activate
```

- on Linux or Mac
```
source venv/bin/activate
```

4. Install the required packages
```
pip install -r requirements.txt
```
5. Make sure custom modules are included in project folder
- sonar_utils

## Documentation

This project uses sphinx for documentation. The project uses the sphinx napolean extension for google docstring formatting. 

1. Install sphinx and napolean ext
```
pip install sphinx sphinxcontrib-napoleon sphinx_rtd_theme
```
2. build the documentation
```
make html # in windows ./make.bat html
```

# NOTES - RJS 20250729

- I just got these files from JB. I am reorganizing and refactoring
- docs are also probably unnecessary