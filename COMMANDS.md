# DataDecay-X — Command Reference

This file contains the main commands required to set up, run, test, develop, debug, and maintain the DataDecay-X project.

> **Platform:** Windows PowerShell commands are the primary reference. Linux/macOS equivalents are included where useful.

---

## 1. Project Setup

### Create project directory

```powershell
mkdir DataDecay-X
cd DataDecay-X
```

### Open project in VS Code

```powershell
code .
```

### Open project in PyCharm

```powershell
pycharm .
```

If the `pycharm` command is not available, open PyCharm manually and select the `DataDecay-X` folder.

---

# 2. Check Python Installation

```powershell
python --version
```

Alternative:

```powershell
py --version
```

Check Python executable:

```powershell
where.exe python
```

Check pip:

```powershell
python -m pip --version
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

# 3. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show something similar to:

```text
(.venv) PS C:\...\DataDecay-X>
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 4. Virtual Environment Management

Check which Python is being used:

```powershell
python -c "import sys; print(sys.executable)"
```

Check installed packages:

```powershell
python -m pip list
```

Freeze environment:

```powershell
python -m pip freeze > requirements-lock.txt
```

Deactivate:

```powershell
deactivate
```

Delete the virtual environment:

```powershell
Remove-Item -Recurse -Force .venv
```

Then recreate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

# 5. Install Dependencies

Install the project's requirements:

```powershell
python -m pip install -r requirements.txt
```

Upgrade all explicitly listed packages:

```powershell
python -m pip install --upgrade -r requirements.txt
```

Install the core packages individually when debugging installation problems:

```powershell
python -m pip install numpy pandas scipy scikit-learn
```

Dashboard:

```powershell
python -m pip install streamlit plotly
```

Testing:

```powershell
python -m pip install pytest
```

Configuration:

```powershell
python -m pip install pyyaml
```

Optional XGBoost:

```powershell
python -m pip install xgboost
```

Optional LightGBM:

```powershell
python -m pip install lightgbm
```

Optional PyTorch:

```powershell
python -m pip install torch
```

Optional MLflow:

```powershell
python -m pip install mlflow
```

---

# 6. Verify Core Dependencies

Run:

```powershell
python -c "import numpy, pandas, scipy, sklearn; print('Core ML dependencies OK')"
```

Verify Streamlit:

```powershell
python -c "import streamlit; print(streamlit.__version__)"
```

Verify Plotly:

```powershell
python -c "import plotly; print(plotly.__version__)"
```

Verify pytest:

```powershell
python -c "import pytest; print(pytest.__version__)"
```

Verify XGBoost:

```powershell
python -c "import xgboost; print(xgboost.__version__)"
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__)"
```

---

# 7. Project Structure Commands

Create the main directories:

```powershell
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\synthetic
mkdir data\samples
mkdir experiments
mkdir models
mkdir reports
mkdir logs
mkdir tests
mkdir src
mkdir src\datadecay
```

Create test initialization file:

```powershell
New-Item tests\__init__.py -ItemType File
```

Create package initialization file:

```powershell
New-Item src\datadecay\__init__.py -ItemType File
```

---

# 8. Run the Streamlit Application

Primary command:

```powershell
python -m streamlit run app.py
```

Alternative:

```powershell
streamlit run app.py
```

Specify a port:

```powershell
python -m streamlit run app.py --server.port 8501
```

Use another port if 8501 is occupied:

```powershell
python -m streamlit run app.py --server.port 8502
```

Run without automatically opening the browser:

```powershell
python -m streamlit run app.py --server.headless true
```

---

# 9. Stop Streamlit

In the terminal running Streamlit:

```text
CTRL + C
```

---

# 10. Streamlit Debugging

Run with verbose logging:

```powershell
python -m streamlit run app.py --logger.level debug
```

Clear Streamlit cache:

```powershell
streamlit cache clear
```

Run with a specific address:

```powershell
python -m streamlit run app.py --server.address localhost
```

---

# 11. Run the Python Package

If the package provides a CLI:

```powershell
python -m src.datadecay.cli
```

For the package to resolve correctly in a source-layout project, set:

```powershell
$env:PYTHONPATH = "."
```

Then:

```powershell
python -m src.datadecay.cli
```

---

# 12. Temporal Drift Simulator Commands

Example gradual drift experiment:

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type gradual --windows 12 --start-window 4 --duration 4 --magnitude 1.0
```

Example stable scenario:

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type stable --windows 12
```

Example abrupt drift:

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type abrupt --windows 12 --start-window 6 --magnitude 1.0
```

Example recurring drift:

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type recurring --windows 16 --start-window 4 --duration 3 --magnitude 1.0
```

Use a larger drift magnitude:

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type gradual --windows 20 --start-window 5 --duration 8 --magnitude 2.0
```

---

# 13. Run Unit Tests

Run all tests:

```powershell
pytest -q
```

Equivalent:

```powershell
python -m pytest -q
```

Verbose:

```powershell
pytest -v
```

Very verbose:

```powershell
pytest -vv
```

Stop after first failure:

```powershell
pytest -x
```

Show print output:

```powershell
pytest -s
```

Run tests without output capture:

```powershell
python -m pytest -s
```

---

# 14. Run Individual Test Files

Simulator:

```powershell
pytest tests/test_simulator.py -v
```

Drift detection:

```powershell
pytest tests/test_detection.py -v
```

Monitoring:

```powershell
pytest tests/test_monitoring.py -v
```

Forecasting:

```powershell
pytest tests/test_forecasting.py -v
```

Synthetic generation:

```powershell
pytest tests/test_synthetic.py -v
```

Quality gate:

```powershell
pytest tests/test_quality_gate.py -v
```

Recovery:

```powershell
pytest tests/test_recovery.py -v
```

Evaluation:

```powershell
pytest tests/test_evaluation.py -v
```

---

# 15. Run a Specific Test

Search test names:

```powershell
pytest --collect-only -q
```

Run tests matching a keyword:

```powershell
pytest -k simulator -v
```

```powershell
pytest -k forecasting -v
```

```powershell
pytest -k recovery -v
```

Run one exact test:

```powershell
pytest tests/test_simulator.py::test_example -v
```

Replace `test_example` with the actual test function name.

---

# 16. Generate Test Coverage

Install coverage support:

```powershell
python -m pip install pytest-cov
```

Run:

```powershell
pytest --cov=src/datadecay --cov-report=term-missing
```

Generate HTML coverage:

```powershell
pytest --cov=src/datadecay --cov-report=html
```

Open the report:

```powershell
start htmlcov\index.html
```

---

# 17. Python Syntax Checking

Compile the application:

```powershell
python -m py_compile app.py
```

Compile package files:

```powershell
python -m compileall src
```

Compile tests:

```powershell
python -m compileall tests
```

Compile everything:

```powershell
python -m compileall .
```

---

# 18. Import Testing

Test the main package:

```powershell
$env:PYTHONPATH = "."
python -c "import src.datadecay; print('DataDecay-X package import OK')"
```

Test individual modules:

```powershell
$env:PYTHONPATH = "."
python -c "from src.datadecay import simulator; print('Simulator import OK')"
```

```powershell
$env:PYTHONPATH = "."
python -c "from src.datadecay import detection; print('Detection import OK')"
```

```powershell
$env:PYTHONPATH = "."
python -c "from src.datadecay import monitoring; print('Monitoring import OK')"
```

```powershell
$env:PYTHONPATH = "."
python -c "from src.datadecay import forecasting; print('Forecasting import OK')"
```

---

# 19. Git Initialization

Initialize repository:

```powershell
git init
```

Check status:

```powershell
git status
```

Add all files:

```powershell
git add .
```

Create first commit:

```powershell
git commit -m "Initial DataDecay-X implementation"
```

---

# 20. Git Daily Workflow

Check changes:

```powershell
git status
```

View changed files:

```powershell
git diff --stat
```

View detailed changes:

```powershell
git diff
```

Stage everything:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Update DataDecay-X implementation"
```

View commit history:

```powershell
git log --oneline
```

---

# 21. Git Branches

Create feature branch:

```powershell
git checkout -b feature/drift-detection
```

Switch branch:

```powershell
git checkout main
```

List branches:

```powershell
git branch
```

Delete local branch:

```powershell
git branch -d feature/drift-detection
```

---

# 22. GitHub Remote

Add remote:

```powershell
git remote add origin <GITHUB_REPOSITORY_URL>
```

Check remote:

```powershell
git remote -v
```

Rename branch to main:

```powershell
git branch -M main
```

Push:

```powershell
git push -u origin main
```

Later pushes:

```powershell
git push
```

Pull:

```powershell
git pull
```

---

# 23. Git Useful Recovery Commands

Unstage a file:

```powershell
git restore --staged <file>
```

Discard changes to a file:

```powershell
git restore <file>
```

View previous commits:

```powershell
git log --oneline --all
```

Show a commit:

```powershell
git show <commit-hash>
```

---

# 24. Dataset Inspection Commands

List files:

```powershell
Get-ChildItem
```

List CSV files:

```powershell
Get-ChildItem data -Recurse -Filter *.csv
```

List Parquet files:

```powershell
Get-ChildItem data -Recurse -Filter *.parquet
```

List JSON files:

```powershell
Get-ChildItem data -Recurse -Filter *.json
```

Python dataset inspection:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/dataset.csv'); print(df.shape); print(df.columns.tolist()); print(df.head())"
```

Check missing values:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/dataset.csv'); print(df.isna().sum())"
```

Check data types:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/dataset.csv'); print(df.dtypes)"
```

Check duplicate count:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/dataset.csv'); print('Duplicates:', df.duplicated().sum())"
```

---

# 25. Python Interactive Shell

Start Python:

```powershell
python
```

Check Python:

```python
import sys
print(sys.version)
```

Exit:

```python
exit()
```

---

# 26. Run a Python Script

```powershell
python script.py
```

Run a module:

```powershell
python -m module_name
```

Run a package module:

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli
```

---

# 27. Formatting and Code Quality

Optional formatter:

```powershell
python -m pip install black
```

Format the project:

```powershell
black .
```

Check formatting without changing files:

```powershell
black --check .
```

Optional import sorter:

```powershell
python -m pip install isort
```

Run:

```powershell
isort .
```

Check:

```powershell
isort --check-only .
```

Optional linter:

```powershell
python -m pip install ruff
```

Run:

```powershell
ruff check .
```

Auto-fix supported issues:

```powershell
ruff check . --fix
```

---

# 28. Dependency Information

Show installed packages:

```powershell
python -m pip list
```

Show a package:

```powershell
python -m pip show pandas
```

```powershell
python -m pip show scikit-learn
```

Check outdated packages:

```powershell
python -m pip list --outdated
```

Export installed environment:

```powershell
python -m pip freeze > requirements-lock.txt
```

---

# 29. Clean Generated Files

Remove Python cache folders:

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Remove `.pyc` files:

```powershell
Get-ChildItem -Path . -Recurse -Filter *.pyc | Remove-Item -Force
```

Remove pytest cache:

```powershell
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
```

Remove Streamlit cache:

```powershell
Remove-Item -Recurse -Force .streamlit\cache -ErrorAction SilentlyContinue
```

> Do not delete `data/raw`, experiment results, trained models, or reports unless you intentionally want to remove them.

---

# 30. Clean Development Environment

If the environment becomes corrupted:

```powershell
deactivate
```

Delete environment:

```powershell
Remove-Item -Recurse -Force .venv
```

Create again:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```powershell
pytest -q
```

Run application:

```powershell
python -m streamlit run app.py
```

---

# 31. Run the Complete Development Check

Recommended sequence:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python --version
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m compileall src tests
```

```powershell
pytest -q
```

```powershell
python -m streamlit run app.py
```

This is the basic **build → compile → test → run** workflow.

---

# 32. Recommended Full Project Verification

Before a demonstration or submission:

### Step 1 — Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2 — Verify dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3 — Compile

```powershell
python -m compileall src tests
```

### Step 4 — Run unit tests

```powershell
pytest -q
```

### Step 5 — Run simulator

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type gradual --windows 12 --start-window 4 --duration 4 --magnitude 1.0
```

### Step 6 — Launch dashboard

```powershell
python -m streamlit run app.py
```

### Step 7 — Execute demo experiment

Use the Streamlit dashboard's demo/experiment workflow.

### Step 8 — Inspect generated reports

```powershell
Get-ChildItem reports -Recurse
```

### Step 9 — Inspect experiment history

```powershell
Get-ChildItem experiments -Recurse
```

### Step 10 — Check Git status

```powershell
git status
```

---

# 33. Recommended Command Sequence for a Fresh Clone

Use this whenever the project is downloaded/cloned on a new machine:

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd DataDecay-X
```

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m compileall src tests
```

```powershell
pytest -q
```

```powershell
python -m streamlit run app.py
```

---

# 34. Recommended Command Sequence Before Git Push

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m compileall src tests
```

```powershell
pytest -q
```

```powershell
git status
```

```powershell
git add .
```

```powershell
git commit -m "Update DataDecay-X"
```

```powershell
git push
```

---

# 35. Optional Docker Commands

Docker is an optional project component.

Check Docker:

```powershell
docker --version
```

Build image:

```powershell
docker build -t datadecay-x .
```

Run container:

```powershell
docker run -p 8501:8501 datadecay-x
```

Run interactively:

```powershell
docker run -it -p 8501:8501 datadecay-x
```

List images:

```powershell
docker images
```

List running containers:

```powershell
docker ps
```

List all containers:

```powershell
docker ps -a
```

Stop container:

```powershell
docker stop <container-id>
```

Remove container:

```powershell
docker rm <container-id>
```

Remove image:

```powershell
docker rmi datadecay-x
```

---

# 36. Optional MLflow Commands

MLflow is optional.

Install:

```powershell
python -m pip install mlflow
```

Start local MLflow server:

```powershell
mlflow ui
```

Specify port:

```powershell
mlflow ui --port 5000
```

Open:

```text
http://localhost:5000
```

---

# 37. Optional PostgreSQL Commands

PostgreSQL is optional for larger deployments.

Check PostgreSQL client:

```powershell
psql --version
```

Connect:

```powershell
psql -U postgres
```

List databases:

```sql
\l
```

Connect to a database:

```sql
\c datadecay
```

List tables:

```sql
\dt
```

Exit:

```sql
\q
```

The core academic implementation uses SQLite, so PostgreSQL is not required.

---

# 38. SQLite Commands

If SQLite CLI is installed:

```powershell
sqlite3 experiments\datadecay.db
```

List tables:

```sql
.tables
```

Inspect schema:

```sql
.schema
```

Query experiments:

```sql
SELECT * FROM experiments;
```

Exit:

```sql
.quit
```

If the SQLite executable is unavailable, the Python `sqlite3` module can still be used by the application.

---

# 39. Useful Windows File Commands

Current directory:

```powershell
Get-Location
```

List files:

```powershell
Get-ChildItem
```

Detailed list:

```powershell
Get-ChildItem | Format-Table
```

Recursive list:

```powershell
Get-ChildItem -Recurse
```

Create file:

```powershell
New-Item example.txt -ItemType File
```

Create directory:

```powershell
New-Item example -ItemType Directory
```

Copy:

```powershell
Copy-Item source.txt destination.txt
```

Move:

```powershell
Move-Item source.txt destination.txt
```

Delete:

```powershell
Remove-Item example.txt
```

---

# 40. Environment Variables

Set project Python path for the current PowerShell session:

```powershell
$env:PYTHONPATH = "."
```

Check:

```powershell
$env:PYTHONPATH
```

Set a project environment variable:

```powershell
$env:DATADECAY_ENV = "development"
```

Check:

```powershell
$env:DATADECAY_ENV
```

Remove:

```powershell
Remove-Item Env:DATADECAY_ENV
```

---

# 41. Common Errors

## Error: `python is not recognized`

Check:

```powershell
where.exe python
```

Try:

```powershell
py --version
```

If neither works, install Python and enable the PATH option during installation.

---

## Error: PowerShell execution policy

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Error: `No module named ...`

First activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify:

```powershell
python -m pip list
```

---

## Error: `pytest` not recognized

Use:

```powershell
python -m pytest -q
```

instead of:

```powershell
pytest -q
```

---

## Error: `streamlit` not recognized

Use:

```powershell
python -m streamlit run app.py
```

---

## Error: Import from `src.datadecay` fails

Set:

```powershell
$env:PYTHONPATH = "."
```

Then run:

```powershell
python -m src.datadecay.cli
```

---

## Error: Port 8501 already in use

Use:

```powershell
python -m streamlit run app.py --server.port 8502
```

---

## Error: Tests fail after package changes

Run:

```powershell
python -m pip install -r requirements.txt
```

Then:

```powershell
python -m compileall src tests
```

Then:

```powershell
pytest -q
```

---

# 42. One-Command Health Check

After activating the environment:

```powershell
python --version; python -m pip --version; python -c "import numpy,pandas,scipy,sklearn,streamlit,plotly; print('Core imports OK')"; python -m compileall src tests; python -m pytest -q
```

This performs:

```text
Python check
   ↓
pip check
   ↓
Core import check
   ↓
Compilation check
   ↓
Unit tests
```

---

# 43. Quick Start — Copy/Paste Version

For normal development:

```powershell
cd DataDecay-X
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m compileall src tests
python -m pytest -q
python -m streamlit run app.py
```

---

# 44. Quick Start — First Installation

```powershell
cd DataDecay-X
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall src tests
python -m pytest -q
python -m streamlit run app.py
```

---

# 45. Quick Start — Simulator Only

```powershell
cd DataDecay-X
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type gradual --windows 12 --start-window 4 --duration 4 --magnitude 1.0
```

---

# 46. Quick Start — Testing Only

```powershell
cd DataDecay-X
.\.venv\Scripts\Activate.ps1
python -m compileall src tests
python -m pytest -q
```

---

# 47. Quick Start — Dashboard Only

```powershell
cd DataDecay-X
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

---

# 48. Final Submission Verification

Before submitting or demonstrating the project, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python --version
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m compileall src tests
```

```powershell
python -m pytest -q
```

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type gradual --windows 12 --start-window 4 --duration 4 --magnitude 1.0
```

```powershell
python -m streamlit run app.py
```

Then verify manually:

```text
[ ] Application starts
[ ] Dataset page works
[ ] Demo experiment runs
[ ] Drift simulation works
[ ] PSI / KS / Wasserstein are displayed
[ ] Reliability metrics are displayed
[ ] Forecast is generated
[ ] TTD is calculated
[ ] Synthetic data is generated
[ ] Quality gate runs
[ ] Recovery candidates are compared
[ ] Future evaluation is performed
[ ] Experiment is saved
[ ] Report is generated
[ ] No future data is used for training
[ ] Git repository is clean/committed
```

---

# 49. Command Summary

## Most important commands

### Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Test

```powershell
python -m pytest -q
```

### Compile

```powershell
python -m compileall src tests
```

### Simulator

```powershell
$env:PYTHONPATH = "."
python -m src.datadecay.cli --drift-type gradual --windows 12 --start-window 4 --duration 4 --magnitude 1.0
```

### Dashboard

```powershell
python -m streamlit run app.py
```

### Git

```powershell
git status
git add .
git commit -m "Update DataDecay-X"
git push
```

---

# 50. Recommended Development Order

```text
1. Activate .venv
        ↓
2. Install requirements
        ↓
3. Compile project
        ↓
4. Run unit tests
        ↓
5. Run simulator
        ↓
6. Inspect generated data
        ↓
7. Launch Streamlit
        ↓
8. Run demo experiment
        ↓
9. Verify dashboard
        ↓
10. Verify future validation
        ↓
11. Generate reports
        ↓
12. Commit changes
        ↓
13. Push to GitHub
```

This is the standard command workflow for the DataDecay-X academic implementation.
