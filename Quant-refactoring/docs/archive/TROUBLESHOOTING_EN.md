# Troubleshooting Guide

---

## Table of Contents

1. [General Common Errors](#general-common-errors)
2. [Installation & Setup Issues](#installation--setup-issues)
3. [Runtime Issues](#runtime-issues)
4. [Performance Issues](#performance-issues)
5. [Platform-Specific Solutions](#platform-specific-solutions)
6. [API & Configuration Issues](#api--configuration-issues)
7. [Quick Reference](#quick-reference)

---

## General Common Errors

### 1. PyArrow Compatibility Error

**Error Message:**
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```

**Solution:**

Upgrade PyArrow and Datasets packages:

```bash
# Windows (PowerShell / CMD)
pip install --upgrade pyarrow>=14.0.0 datasets>=2.14.0

# Linux / macOS (bash / zsh)
pip install --upgrade pyarrow>=14.0.0 datasets>=2.14.0
```

Or install from requirements file:

```bash
# All platforms
pip install -r requirements_fix.txt
```

**Explanation:**
This error occurs due to version incompatibility between PyArrow and the datasets library. Upgrading to compatible versions resolves this.

---

### 2. Module Import Errors

**Error Messages:**
```
ModuleNotFoundError: No module named 'xxx'
ImportError: cannot import name 'xxx'
```

**Solutions:**

**Step 1: Identify missing module**
```bash
pip list  # Check installed packages
```

**Step 2: Install missing dependency**
```bash
# Windows
pip install [module_name]

# Linux / macOS
pip install [module_name]
```

**Step 3: If still failing, reinstall requirements**
```bash
# Windows (PowerShell)
pip install -r requirements.txt --force-reinstall

# Linux / macOS (bash / zsh)
pip install -r requirements.txt --force-reinstall
```

**Common missing modules:**
- `pyarrow` - Data processing
- `xgboost` - Machine learning
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `requests` - HTTP requests for API calls

---

### 3. Python Version Incompatibility

**Error Message:**
```
SyntaxError: invalid syntax (or similar)
```

**Check Python version:**

```bash
# All platforms
python --version
# or
python3 --version
```

**Requirements:**
- Minimum: Python 3.8
- Recommended: Python 3.10+

**Solution:**

If your Python version is below 3.8:
1. Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Create a new virtual environment with the correct version
3. Reinstall dependencies

---

## Installation & Setup Issues

### 1. Virtual Environment Setup

**Creating a clean environment:**

```bash
# Windows (PowerShell)
python -m venv venv_quant
.\venv_quant\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv_quant
venv_quant\Scripts\activate.bat

# Linux / macOS (bash / zsh)
python3 -m venv venv_quant
source venv_quant/bin/activate
```

**Installing dependencies in virtual environment:**

```bash
# All platforms
pip install --upgrade pip
pip install -r requirements_fix.txt
```

---

### 2. Permission Errors

**Error Message:**
```
PermissionError: [Errno 13] Permission denied
ERROR: Could not install packages due to a EnvironmentError: [Errno 13] Permission denied
```

**Solutions:**

**Option 1: Use --user flag**
```bash
# All platforms
pip install --user [package_name]
```

**Option 2: Use virtual environment (Recommended)**
```bash
# Windows (PowerShell)
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Linux / macOS (bash / zsh)
python3 -m venv myenv
source myenv/bin/activate
```

**Option 3: Linux/macOS - Run with sudo (not recommended)**
```bash
# Linux / macOS (bash / zsh)
# ⚠️ Use only as last resort
sudo pip install [package_name]
```

---

### 3. Directory Structure Issues

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**

Ensure required directories exist:

```bash
# Windows (PowerShell)
mkdir -Force logs, results, config

# Linux / macOS (bash / zsh)
mkdir -p logs results config
```

**Verify directory structure:**

```bash
# Windows (PowerShell)
tree /F  # Shows directory tree

# Linux / macOS (bash / zsh)
tree  # or 'ls -la' for simple listing
```

---

## Runtime Issues

### 1. Data File Not Found

**Error Message:**
```
FileNotFoundError: Price file not found at [path]
```

**Solution:**

**Step 1: Verify data path**

Open `scripts/run_sector_trading.py` and check line ~25:

```python
# Current (possibly incorrect)
DATA_PATH = './VIEW'

# Change to your actual data path
DATA_PATH = '../VIEW'  # If running from scripts directory
# or
DATA_PATH = '/home/user/data/VIEW'  # Absolute path
```

**Step 2: Verify files exist**

```bash
# Windows (PowerShell)
Get-ChildItem DATA_PATH

# Linux / macOS (bash / zsh)
ls -la DATA_PATH
```

**Step 3: Use absolute paths (Recommended)**

```python
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'VIEW')
```

---

### 2. API Key Issues

**Error Message:**
```
401 Unauthorized
Invalid API key
Authentication failed
```

**For FMP (Financial Modeling Prep) API:**

**Step 1: Verify API key**

```bash
# Windows (PowerShell)
$env:FMP_API_KEY

# Linux / macOS (bash / zsh)
echo $FMP_API_KEY
```

**Step 2: Set API key**

**Windows (PowerShell - Temporary):**
```powershell
$env:FMP_API_KEY = "your_actual_api_key_here"
```

**Windows (PowerShell - Permanent):**
```powershell
[System.Environment]::SetEnvironmentVariable("FMP_API_KEY", "your_actual_api_key_here", "User")
# Restart PowerShell or IDE after this
```

**Linux / macOS (bash/zsh - Temporary):**
```bash
export FMP_API_KEY="your_actual_api_key_here"
```

**Linux / macOS (bash/zsh - Permanent):**

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export FMP_API_KEY="your_actual_api_key_here"
```

Then reload:
```bash
source ~/.bashrc
# or
source ~/.zshrc
```

**Step 3: Verify in Python code**

```python
import os
api_key = os.getenv('FMP_API_KEY')
if api_key is None:
    print("Error: FMP_API_KEY not set")
else:
    print(f"API Key loaded: {api_key[:10]}...")
```

---

### 3. Script Execution Issues

**Wrong script location (Common mistake):**

```bash
# ❌ WRONG - Old system
python main.py

# ✅ CORRECT - New system
cd scripts
python run_sector_trading.py
```

**Running scripts correctly:**

```bash
# Windows (PowerShell)
cd C:\Users\YourName\PycharmProjects\Quant\Quant-refactoring\scripts
python run_sector_trading.py

# Windows (CMD)
cd C:\Users\YourName\PycharmProjects\Quant\Quant-refactoring\scripts
python run_sector_trading.py

# Linux / macOS (bash / zsh)
cd /home/user/Quant/Quant-refactoring/scripts
python run_sector_trading.py
```

**Using full pipeline:**

```bash
# All platforms
cd scripts
python run_full_pipeline.py  # Runs everything automatically
```

---

### 4. Memory Issues

**Error Message:**
```
MemoryError
Resource exhausted
Killed (out of memory)
```

**Solutions:**

**Step 1: Check available memory**

```bash
# Windows (PowerShell)
Get-ComputerInfo | Select-Object csPhysicalMemory

# Linux (bash / zsh)
free -h

# macOS (bash / zsh)
vm_stat
```

**Step 2: Reduce data size**

```python
# Modify scripts to use smaller dataset
# Instead of:
data = pd.read_csv('large_file.csv')

# Use:
data = pd.read_csv('large_file.csv', nrows=10000)  # Limit rows
# or
data = pd.read_csv('large_file.csv', usecols=['col1', 'col2'])  # Limit columns
```

**Step 3: Enable memory optimization**

```python
# Convert data types to save memory
data['price'] = data['price'].astype('float32')  # Instead of float64
data['date'] = pd.to_datetime(data['date'])
```

**Step 4: Process in chunks**

```python
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    process_data(chunk)
```

---

### 5. GPU/CUDA Issues

**Error Message:**
```
CUDA out of memory
CUDA runtime error
Could not load dynamic library 'libcuda.so.1'
```

**Check CUDA status:**

```bash
# Windows (PowerShell)
nvidia-smi

# Linux (bash / zsh)
nvidia-smi

# macOS
# (GPU support varies - not available on all models)
system_profiler SPDisplaysDataType
```

**Solution 1: Use CPU instead of GPU**

```python
# In your scripts
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
# or use CPU-specific training parameters
```

**Solution 2: Reduce batch size**

```python
# In your model training
batch_size = 32  # Reduce from 64, 128, etc.
```

**Solution 3: Clear GPU cache**

```python
import torch
torch.cuda.empty_cache()

# or with TensorFlow
import tensorflow as tf
tf.keras.backend.clear_session()
```

---

## Performance Issues

### 1. Slow Execution

**Diagnosis:**

```bash
# Profile your script execution
python -m cProfile -s cumtime scripts/run_sector_trading.py > profile.txt
```

**Check results:**
```bash
# Windows (PowerShell)
Get-Content profile.txt -Head 20

# Linux / macOS (bash / zsh)
head -20 profile.txt
```

**Optimization strategies:**

1. **Reduce API calls:**
   ```python
   # Cache results instead of repeated calls
   import functools

   @functools.lru_cache(maxsize=128)
   def fetch_data(ticker):
       return api.get_data(ticker)
   ```

2. **Use parallel processing:**
   ```python
   from multiprocessing import Pool

   with Pool(4) as p:
       results = p.map(process_ticker, tickers)
   ```

3. **Optimize data operations:**
   ```python
   # Use vectorized operations instead of loops
   # ❌ SLOW
   for i in range(len(data)):
       data['result'][i] = expensive_function(data['value'][i])

   # ✅ FAST
   data['result'] = data['value'].apply(expensive_function)
   ```

---

### 2. High CPU Usage

**Check running processes:**

```bash
# Windows (PowerShell)
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10

# Linux (bash / zsh)
top -o %CPU  # Press 'q' to quit

# macOS (bash / zsh)
top -o %CPU
```

**Kill specific process (if needed):**

```bash
# Windows (PowerShell)
Stop-Process -Name python -Force

# Linux / macOS (bash / zsh)
killall python
# or
kill -9 [PID]  # Get PID from top/ps
```

---

## Platform-Specific Solutions

### Windows

#### PowerShell vs CMD

**Using PowerShell (Recommended):**
```powershell
# Navigate to directory
cd C:\Users\YourName\PycharmProjects\Quant\Quant-refactoring

# Activate virtual environment
.\venv_quant\Scripts\Activate.ps1

# Run script
cd scripts
python run_sector_trading.py
```

**Using CMD:**
```cmd
:: Navigate to directory
cd C:\Users\YourName\PycharmProjects\Quant\Quant-refactoring

:: Activate virtual environment
venv_quant\Scripts\activate.bat

:: Run script
cd scripts
python run_sector_trading.py
```

#### Windows-specific path issues

```python
# Handle backslashes properly
import os
from pathlib import Path

# Option 1: Use pathlib (Recommended)
data_path = Path('..') / 'VIEW' / 'data.csv'

# Option 2: Use forward slashes (works in Python)
data_path = '../VIEW/data.csv'

# Option 3: Use raw string
data_path = r'C:\Users\YourName\data\VIEW'

# Option 4: Use os.path
data_path = os.path.join('..', 'VIEW', 'data.csv')
```

---

### Linux (Ubuntu/Debian)

**Complete setup:**

```bash
# 1. Install Python and pip
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3-pip

# 2. Navigate to project
cd /home/user/Quant/Quant-refactoring

# 3. Create virtual environment
python3.10 -m venv venv_quant

# 4. Activate environment
source venv_quant/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements_fix.txt

# 6. Run script
cd scripts
python run_sector_trading.py
```

**Permission issues (Linux-specific):**

```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run with python explicitly
python scripts/run_sector_trading.py
```

**Cron job setup (for scheduled runs):**

```bash
# Edit crontab
crontab -e

# Add this line to run daily at 2 AM
0 2 * * * cd /home/user/Quant/Quant-refactoring && /home/user/Quant/Quant-refactoring/venv_quant/bin/python scripts/run_sector_trading.py >> logs/cron.log 2>&1
```

---

### macOS

**Complete setup:**

```bash
# 1. Install Python (using Homebrew recommended)
brew install python@3.10

# 2. Navigate to project
cd /Users/YourName/Quant/Quant-refactoring

# 3. Create virtual environment
python3.10 -m venv venv_quant

# 4. Activate environment
source venv_quant/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements_fix.txt

# 6. Run script
cd scripts
python run_sector_trading.py
```

**macOS-specific issues:**

```bash
# If you get "command not found: python3"
# Install from python.org or use:
brew install python@3.10

# If you get certificate verification errors
/Applications/Python\ 3.10/Install\ Certificates.command

# M1/M2 Mac issues (Apple Silicon)
# When installing packages, you might need:
pip install --upgrade pip setuptools wheel
pip install -r requirements_fix.txt
```

**Launchd setup (for scheduled runs on macOS):**

```bash
# Create plist file
nano ~/Library/LaunchAgents/com.quant.trading.plist
```

Add this content:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quant.trading</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YourName/Quant/Quant-refactoring/venv_quant/bin/python</string>
        <string>/Users/YourName/Quant/Quant-refactoring/scripts/run_sector_trading.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/YourName/Quant/Quant-refactoring/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YourName/Quant/Quant-refactoring/logs/launchd_error.log</string>
</dict>
</plist>
```

Then:
```bash
# Load the job
launchctl load ~/Library/LaunchAgents/com.quant.trading.plist

# Check status
launchctl list | grep quant

# Remove if needed
launchctl unload ~/Library/LaunchAgents/com.quant.trading.plist
```

---

## API & Configuration Issues

### 1. FMP API Authentication

**Problem:** API requests returning 401 or similar errors

**Step 1: Get your API key**

Visit https://financialmodelingprep.com/developer/docs and get your free API key.

**Step 2: Store API key securely**

**Windows (Permanent):**
```powershell
# Set as environment variable
[System.Environment]::SetEnvironmentVariable("FMP_API_KEY", "your_key_here", "User")
```

**Linux/macOS (Permanent):**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export FMP_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

**Step 3: Use in your code**

```python
import os
from requests import get

api_key = os.getenv('FMP_API_KEY')
if not api_key:
    raise ValueError("FMP_API_KEY environment variable not set")

# Use in requests
url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
response = get(url)
```

---

### 2. Config File Issues

**Error Message:**
```
ConfigFileNotFoundError
KeyError: 'config_key'
```

**Solution:**

**Step 1: Verify config directory exists**

```bash
# Windows (PowerShell)
Test-Path config

# Linux / macOS (bash / zsh)
test -d config && echo "exists" || echo "not found"
```

**Step 2: Create config directory and sample file**

```bash
# Windows (PowerShell)
mkdir -Force config

# Linux / macOS (bash / zsh)
mkdir -p config
```

**Step 3: Create sample config.yaml**

```bash
# All platforms
cat > config/config.yaml << 'EOF'
api:
  fmp_api_key: ${FMP_API_KEY}
  timeout: 30

data:
  path: ../VIEW
  format: csv

model:
  xgboost:
    max_depth: 6
    n_estimators: 100

logging:
  level: INFO
  file: logs/app.log
EOF
```

---

## Quick Reference

### Most Common Issues & Quick Fixes

| Issue | Command | Platform |
|-------|---------|----------|
| PyArrow error | `pip install --upgrade pyarrow>=14.0.0` | All |
| Missing module | `pip install [module_name]` | All |
| Permission error | Use virtual environment | All |
| Data file not found | Check `DATA_PATH` variable | All |
| API auth failed | Set `FMP_API_KEY` environment variable | All |
| Memory error | Reduce batch size / dataset size | All |
| Slow execution | Enable caching / parallel processing | All |

---

### Complete Fresh Start (All Platforms)

```bash
# Windows (PowerShell)
# ============================================
cd C:\Users\YourName\PycharmProjects\Quant\Quant-refactoring
python -m venv venv_fresh
.\venv_fresh\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_fix.txt
cd scripts
python run_sector_trading.py

# Linux / macOS (bash / zsh)
# ============================================
cd /home/user/Quant/Quant-refactoring
python3 -m venv venv_fresh
source venv_fresh/bin/activate
pip install --upgrade pip
pip install -r requirements_fix.txt
cd scripts
python run_sector_trading.py
```

---

### Testing with Sample Data (All Platforms)

```bash
# Windows (PowerShell)
cd C:\Users\YourName\PycharmProjects\Quant\Quant-refactoring\examples
python comprehensive_example.py

# Linux / macOS (bash / zsh)
cd /home/user/Quant/Quant-refactoring/examples
python comprehensive_example.py
```

---

### Checking Logs

```bash
# Windows (PowerShell)
Get-Content logs\sector_trading.log -Tail 20  # Last 20 lines

# Linux / macOS (bash / zsh)
tail -20 logs/sector_trading.log
```

---

## Getting Help

If you still encounter issues:

1. **Check the logs:**
   ```bash
   # Windows
   Get-Content logs\*.log

   # Linux / macOS
   cat logs/*.log
   ```

2. **Provide information when reporting:**
   - Error message (full traceback)
   - Python version: `python --version`
   - OS: Windows/Linux/macOS
   - Command you ran
   - Relevant log files
   - Virtual environment info: `pip list`

3. **Related documentation:**
   - System overview → [README.md](../README.md)
   - Quick start → [QUICK_START.md](QUICK_START.md)
   - Error fixes → [FIX_ERRORS.md](FIX_ERRORS.md)

---

**Last Updated:** 2025-01-17
**Maintained By:** Quant Development Team
