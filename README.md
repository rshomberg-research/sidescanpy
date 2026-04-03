# URI research

## TODO

- [ ] work out separating anything proprietary from open source information

- [ ] refactor all code 
- [ ] organize data & outputs
- [ ] rewrite this readme
- [ ] change name 

TODO: evaluate below todo items

- [X] ~~*set up basic file structure*~~ [2025-09-27]
  - [X] ~~*install local python package*~~ [2025-09-27]
- [ ] set up small test data files
- [ ] write basic test Scripts
- [ ] start data pipeline
- [ ] recreate previous work from JB 
  - [ ] import SONAR data
  - [ ] generate waterfall plots
  - [ ] generate depth maps
  - [ ] implement DepthAnything
- [X] ~~*try edgetech software on windows*~~ [2025-09-27]
  - https://www.edgetech.com/resource-center/

NOTE: below here is readme info that I recieved from JB. Probably do not need it
anymore - RJS 20250731

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
- sonar_utilsgit

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
