# Codebase Structure Documentation

## Design Overview

This is an **Options Trading Analysis System** designed to analyze options trading positions, calculate profit/loss (PnL), manage exposure, and screen stocks based on earnings dates. The system is built with a modular Python architecture that separates concerns across multiple specialized libraries.

### Key Design Patterns

1. **Separation of Concerns**: Each module handles a specific aspect (positions, PnL calculation, stock data, etc.)
2. **Data-Driven Analysis**: Heavy use of pandas DataFrames for data manipulation and analysis
3. **Caching Strategy**: Stock data caching to minimize API calls and improve performance
4. **Interactive Notebooks**: Jupyter notebooks provide user-friendly interfaces for analysis
5. **Financial Modeling**: Black-Scholes model implementation for options Greeks calculation

### Architecture Layers

- **Data Layer**: CSV file reading, parsing, and data cleaning ([libs/positionslib.py](libs/positionslib.py), [libs/positions_data.py](libs/positions_data.py))
- **Business Logic Layer**: PnL calculation, exposure analysis, options pairing ([libs/pnllib.py](libs/pnllib.py), [tools/exposure_analysis.py](tools/exposure_analysis.py))
- **Data Access Layer**: Yahoo Finance integration, stock price fetching ([libs/stocks.py](libs/stocks.py), [libs/stock_cache.py](libs/stock_cache.py), [libs/historical_cache.py](libs/historical_cache.py))
- **Utility Layer**: Helper functions, data transformation ([libs/utils.py](libs/utils.py))
- **Tools Layer**: Executable scripts and analysis tools ([tools/screener.py](tools/screener.py), [tools/exposure_analysis.py](tools/exposure_analysis.py), [tools/manage_earnings_database.py](tools/manage_earnings_database.py))
- **Presentation Layer**: Jupyter notebooks for interactive analysis ([notebooks/](notebooks/))

---

## File Descriptions

### Core Library Files (libs/)

#### [libs/pnllib.py](libs/pnllib.py)
**Primary Functionality**: Options trade matching and profit/loss calculation engine

**Role in Project**: The core module for analyzing historical options trades. It handles the complex logic of pairing opening and closing trades, calculating realized PnL, and managing assignments/expirations.

**Key Components**:
- `Trade` class: Main interface for loading and analyzing trades from CSV files
- `parse_option()`: Parses option symbol strings into components (ticker, expiration, strike, type)
- `is_option()`: Validates if a string represents an option symbol
- `pair_option_trades()`: Sophisticated matching algorithm that pairs buy-to-open with sell-to-close trades
- `calculate_option_pnl()`: Computes profit/loss for matched trade pairs
- `get_assigned_prices()`: Retrieves stock prices on assignment dates for accurate PnL calculation
- `filter_by_expiration_month()`: Filters trades by expiration period
- `compute_trade_stats()`: Generates summary statistics (win rate, average win/loss, etc.)
- Logging system with multiple verbosity levels (INFO, WARN, ERROR, SILENT)

**Dependencies**: pandas, numpy, datetime, yfinance, libs.utils

---

#### [libs/positionslib.py](libs/positionslib.py)
**Primary Functionality**: Broker position file parsing and preprocessing

**Role in Project**: Handles reading and cleaning position data exported from brokerage accounts. Transforms raw CSV files into structured DataFrames.

**Key Components**:
- `skip_lines()`: Parses brokerage CSV files with non-standard headers
- `read_positions_file()`: Main function to read and clean position data
- `get_expiration_indexed_positions()`: Creates a dictionary mapping expiration dates to positions
- `get_margin()`: Extracts margin requirement from string format
- Handles numeric conversions, N/A values, and data type cleanup

**Dependencies**: pandas, libs.pnllib, libs.utils

---

#### [libs/positions_data.py](libs/positions_data.py)
**Primary Functionality**: Object-oriented interface for positions data with stock/options separation

**Role in Project**: Provides a cleaner, more maintainable API for working with positions data. Separates stock positions from options positions and integrates with stock cache.

**Key Components**:
- `PositionsReader`: Utility class for reading CSV files with flexible header detection
- `StockData`: Class for managing stock positions with cache integration
- `OptionsData`: Class for managing options positions with symbol parsing
- `_parse_option_symbols()`: Extracts ticker, strike, expiration, and type from option symbols
- Integration with `StockCache` for real-time price and volatility updates

**Dependencies**: pandas, numpy, re, datetime, libs.stock_cache, libs.utils

---

#### [libs/stocks.py](libs/stocks.py)
**Primary Functionality**: Individual stock position tracking and PnL analysis

**Role in Project**: Tracks trading history for specific stocks to calculate realized and unrealized PnL across different time periods. Used for deep-dive analysis of individual stock performance.

**Key Components**:
- `StockTrades` class: Main interface for tracking all trades for a single stock
- `read_data()`: Loads trades from CSV with date filtering, excludes "as of" entries
- `infer_positions()`: Reconstructs position size at each historical point in time
- `infer_pnl()`: Calculates PnL from any historical point to present, accounting for future cash flows
- `find_pnl_from_zero_pos_after_start_date()`: Identifies round-trip trades (open to close to zero position)
- `find_pnl_from_start_date()`: Finds PnL from earliest transaction after specified start date
- `compute_forward_pnl()`: Generates time-series of PnL evolution over time

**Use Case**: Detailed analysis of trading performance for specific stocks, position reconstruction, and entry/exit analysis

**Dependencies**: pandas, libs.historical_cache

---

#### [libs/stock_cache.py](libs/stock_cache.py)
**Primary Functionality**: Persistent caching of stock market data

**Role in Project**: Reduces API calls to Yahoo Finance by maintaining a JSON-based cache of stock prices, volatility, volume, and market cap data.

**Key Components**:
- `StockCache` class: Main cache management interface
- `_calculate_volatility()`: Computes 30-day annualized volatility from historical prices
- `needs_update()`: Checks if cached data has expired (default: 6-hour intervals)
- `update()`: Fetches fresh data from Yahoo Finance with rate limiting
- `lookup()`: Retrieves cached data with automatic refresh if stale
- `display()`: Formats cache data as table, JSON, or dictionary

**Dependencies**: yfinance, pandas, numpy, json, datetime

---

#### [libs/historical_cache.py](libs/historical_cache.py)
**Primary Functionality**: High-performance caching of historical stock price data

**Role in Project**: Provides ~5000x faster access to historical price data compared to direct API calls. Designed for stock screening and backtesting workflows. Each stock's data is stored in a separate JSON file for efficient lazy updates.

**Key Components**:
- `HistoricalCache` class: Main cache management interface with configurable staleness and history
- `update()`: Updates cache for a single symbol from Yahoo Finance
- `update_batch()`: Updates multiple symbols with automatic rate limiting
- `lookup()`: Retrieves historical DataFrame with auto-refresh if stale
- `get_multiple()`: Efficiently retrieves data for multiple symbols
- `needs_update()`: Checks staleness based on days and column changes
- `get_cache_info()`: Returns metadata about cached symbol
- Configurable parameters: staleness_days (default: 1), history_days (default: 252), columns (default: ['Close'])

**Use Cases**:
- Daily stock screening with automatic cache refresh
- Backtesting with frozen historical data (staleness_days=-1)
- Technical analysis with OHLCV data
- Calculating N-day returns efficiently

**Storage**: Per-stock JSON files in `data/historical_cache/{SYMBOL}.json`

**Performance**: <1ms lookup for cached data, ~50KB per stock for 252 days of Close data

**Dependencies**: yfinance, pandas, json, datetime

**Documentation**: [docs/historical_cache.md](historical_cache.md) | [Quick Start](historical_cache_quickstart.md)

---

#### [libs/utils.py](libs/utils.py)
**Primary Functionality**: Statistical analysis utilities and data visualization helpers

**Role in Project**: Provides reusable utilities for stock price analysis, volatility calculation, and conditional expectation analysis.

**Key Components**:
- `option_counts_by_date()`: Counts calls vs puts by expiration date
- `get_stock_closing_prices()`: Fetches historical closing prices from Yahoo Finance
- `clean_numeric()`: Cleans and converts numeric data from CSV files
- `Prices` class: Manages historical price data with advanced analytics
  - `attach_returns()`: Calculates N-day backward or forward returns
  - `add_volatility()`: Computes rolling volatility
  - `add_vix()`: Merges VIX index data
  - `add_high()` / `add_low()`: Tracks N-day highs/lows
  - `add_range_pct()`: Calculates price range as percentage
- `TwoVars` class: Conditional expectation analysis for strategy backtesting
  - Bins independent variable (X) and calculates expected value of dependent variable (Y)
  - Supports percentile calculations for risk analysis

**Dependencies**: pandas, numpy, yfinance, functools

---

### Tools and Analysis Scripts (tools/)

#### [tools/exposure_analysis.py](tools/exposure_analysis.py)
**Primary Functionality**: Options portfolio risk analysis using Greeks

**Role in Project**: Calculates portfolio delta exposure by computing option deltas using the Black-Scholes model. Helps traders understand their directional risk.

**Key Components**:
- `ExposureAnalysis` class: Main analysis engine
- `_get_historical_volatility()`: Calculates annualized volatility with outlier capping
- `_get_option_delta()`: Computes delta using Black-Scholes Greeks (py_vollib library)
- `analyze_positions()`: Processes entire portfolio and adds delta columns
- `get_total_exposure()`: Aggregates exposure by underlying ticker
- Volatility premium adjustment (configurable multiplier)
- Integration with stock cache for efficient data retrieval

**Dependencies**: pandas, numpy, yfinance, datetime, libs.positions_data, py_vollib, libs.stock_cache

**Constants**:
- `VOLATILITY_PREMIUM`: 50% (multiplies historical vol by 1.5)
- `OUTLIER_CAP_MULTIPLIER`: 4x the 90th percentile return

---

#### [tools/screener.py](tools/screener.py)
**Primary Functionality**: Stock screening based on earnings dates and historical returns

**Role in Project**: Identifies stocks suitable for options trading by filtering based on earnings calendar proximity and calculating historical returns using cached data for performance.

**Key Components**:
- `list_stocks()`: Lists all stocks in the earnings database
- `get_stocks_within_days(N)`: Finds stocks with earnings within N days
- `get_stocks_outside_days(N)`: Finds stocks with NO earnings within N days
- `get_stocks_outside_date(end_date_str)`: Finds stocks with NO earnings before specified end date
- `StockData` class: Manages stock price data with optional caching
  - `fetch_price_data()`: Retrieves historical prices via HistoricalCache or yfinance
  - `add_returns(N)`: Calculates N-day returns
  - `attach_earnings_dates()`: Merges earnings dates with stock data
- `main()`: Complete screening workflow
  - Loads symbols from data/stocks_list.csv
  - Optionally updates earnings database
  - Screens for stocks outside earnings window
  - Calculates 5, 22, 66-day returns using HistoricalCache
  - Sorts by 22-day returns
  - Saves to data/screener_output.csv

**Parameters**:
- days_forward (default: 30)
- update_earnings (default: True)
- use_cache (default: True)
- staleness_days (default: 1)

**Output**: data/screener_output.csv with stocks and their historical returns

**Execution**: `python tools/screener.py`

**Dependencies**: pandas, datetime, libs.historical_cache, tools.manage_earnings_database

---

#### [tools/manage_earnings_database.py](tools/manage_earnings_database.py)
**Primary Functionality**: Earnings calendar database management via Finnhub API

**Role in Project**: Manages the earnings date database by fetching data from Finnhub API and storing it in monthly CSV files. Provides command-line interface for database updates.

**Key Components**:
- `get_earnings_dates()`: Fetches earnings dates from Finnhub API for given symbols
- `setup()`: Initializes monthly CSV files (M1.csv - M12.csv) in data/earnings_dates/
- `store()`: Stores single earnings date to appropriate monthly file
- `store_symbols()`: Batch stores earnings data with rate limiting (2s delay)
- `is_valid_symbol()`: Validates symbols (excludes 100+ ETFs, funds, indices like SPY, QQQ, VTI)
- `get_symbols_needing_update()`: Identifies symbols missing data for specified date range
- `update_earnings_database()`: Programmatic API for updating earnings data
- `add_symbol_to_stocks_list()`: Adds symbol to data/stocks_list.csv tracking file

**Command-Line Usage**:
- `python manage_earnings_database.py --update` (update for next 6 months)
- `python manage_earnings_database.py --reset` (wipe and rebuild entire database)
- `python manage_earnings_database.py --add TSLA` (add new symbol to database)

**Storage**: data/earnings_dates/M1.csv through M12.csv (monthly organization)

**API Configuration**: Requires FINNHUB_API_KEY in .env file

**Dependencies**: pandas, finnhub-python, python-dotenv

---

#### [tools/posanalysis.py](tools/posanalysis.py)
**Primary Functionality**: Current positions analysis aggregated by expiration date

**Role in Project**: Analyzes active positions from broker export, grouping by expiration date to show weekly and monthly summaries of premium collected, unrealized PnL, and margin requirements.

**Key Components**:
- `main()`: Main analysis function
  - Reads from data/pos.csv (processed positions file)
  - Groups positions by expiration date
  - Creates weekly aggregated DataFrames showing:
    - Cost Basis (premium collected)
    - Quantity (number of contracts)
    - P&L (unrealized profit/loss)
    - Margin requirements
  - Aggregates monthly totals
  - Displays weekly and monthly summaries

**Output**: Weekly and monthly position summaries printed to console with totals

**Execution**: `python tools/posanalysis.py`

**Dependencies**: pandas, libs.positionslib

---

### Interactive Notebooks (notebooks/)

#### [notebooks/pnl-tool.ipynb](notebooks/pnl-tool.ipynb)
**Primary Functionality**: Monthly PnL analysis and trade statistics

**Role in Project**: Provides an interactive interface for analyzing historical options trading performance by month.

**Key Features**:
- Month-by-month breakdown of trading statistics
- Average win/loss calculation
- Win rate analysis
- Configurable date ranges and expiration filtering
- Symbol-specific filtering

**Usage Pattern**: Users configure `start_month`, `end_month`, `year`, and optional `opt_type` (Puts/Calls) to generate performance reports.

---

#### [notebooks/pos-analysis.ipynb](notebooks/pos-analysis.ipynb)
**Primary Functionality**: Current positions analysis by expiration date

**Role in Project**: Analyzes active positions grouped by expiration date, showing premium collected, unrealized PnL, and margin requirements.

**Key Features**:
- Expiration-date indexed position views
- Premium vs PnL comparison
- Margin requirement tracking
- Gap analysis (difference between premium and PnL)

**Usage Pattern**: Reads latest position file and generates tables showing positions expiring on specific dates.

---

#### [notebooks/screener.ipynb](notebooks/screener.ipynb)
**Primary Functionality**: Stock screening interface based on earnings dates and historical returns

**Role in Project**: Interactive interface for identifying stocks suitable for options trading by filtering based on earnings calendar proximity and analyzing historical performance.

**Key Features**:
- Imports screener module functions
- Interactive workflow for stock screening
- Calls main() function with configurable parameters:
  - days_forward: 30 (screens for stocks with no earnings in next N days)
  - update_earnings: True (refreshes earnings database before screening)
  - use_cache: True (uses HistoricalCache for fast price data retrieval)
  - staleness_days: 1 (refreshes cached data older than 1 day)
- Results automatically saved to data/screener_output.csv
- Shows stocks with no earnings in specified window
- Displays historical returns (5, 22, 66-day) for each stock
- Sorted by 22-day returns for quick identification of trending stocks

---

#### [notebooks/stock_pnl_tool.ipynb](notebooks/stock_pnl_tool.ipynb)
**Primary Functionality**: Individual stock PnL tracking and position evolution analysis

**Role in Project**: Deep-dive analysis of trading performance for specific stocks, tracking position evolution over time and identifying round-trip trades.

**Key Features**:
- Analyzes specific stock symbols over custom time periods
- Uses StockTrades class from libs.stocks to reconstruct historical positions
- Calculates PnL from various start dates (e.g., 01/01/2024, 06/01/2025)
- Filters out "as of" entries from trade data for accurate calculations
- Reads from current positions (data/pos.csv) and historical trades
- Shows PnL evolution from specific start dates to present
- Handles missing data gracefully with error handling
- Tracks ~70 symbols including tech stocks, healthcare, energy sectors

**Use Cases**:
- Historical position reconstruction
- Round-trip trade identification
- Entry/exit analysis for specific stocks
- Performance attribution by stock

---

### Test Programs and Examples (temp/)

#### [temp/example_screener_with_cache.py](temp/example_screener_with_cache.py)
**Primary Functionality**: Example demonstrating HistoricalCache integration with screener

**Role in Project**: Demonstrates the performance benefits of using HistoricalCache (~5000x improvement) and provides example code for integrating the cache into screening workflows.

**Key Features**:
- Shows how to replace direct API calls with HistoricalCache
- Example workflow:
  - Updates cache for test symbols
  - Calculates 5, 22, 66-day returns
  - Merges with earnings dates
  - Displays cache information
  - Performance benchmarking (measures lookup times)
- Uses test symbols: AAPL, GOOGL, MSFT, AMZN, META
- Demonstrates <1ms cached lookup vs ~2s API call performance

**Use Case**: Reference implementation for integrating HistoricalCache into custom tools

---

#### [temp/test_historical_cache.py](temp/test_historical_cache.py)
**Primary Functionality**: Comprehensive test suite for HistoricalCache class

**Role in Project**: Validates all HistoricalCache functionality and provides usage examples for developers.

**Test Coverage**:
1. Cache creation with default parameters
2. Single stock data fetching
3. Fresh cache lookup performance testing
4. Custom staleness thresholds (0 days, -1 for never update)
5. Custom history length (100 days)
6. Multiple columns (Close, Open, Volume, High, Low)
7. Batch updates with rate limiting
8. Mixed fresh/stale detection
9. Multiple symbol retrieval
10. Invalid symbol handling
11. N-day returns calculation
12. Cached symbols listing

**Test Symbols**: AAPL, GOOGL, MSFT, INVALID_XYZ_123

**Execution**: `python temp/test_historical_cache.py`

---

### Configuration and Scripts

#### .env
**Purpose**: Environment variables configuration (excluded from git)

**Contents**:
- FINNHUB_API_KEY: API key for Finnhub earnings data access

**Setup**: Copy from .env.example and add your API key

---

#### .env.example
**Purpose**: Template for environment configuration

**Contents**: Example format for required environment variables

---

#### [run_exposure.sh](run_exposure.sh)
**Purpose**: Bash wrapper script for exposure analysis

**Functionality**:
- Sets correct PYTHONPATH to project root
- Runs exposure analysis: `python tools/exposure_analysis.py`
- Ensures module imports work correctly from any directory

**Execution**: `./run_exposure.sh`

---

## Data Flow Architecture

```
CSV Files (Broker Exports)
        |
        v
libs/positionslib.py / libs/positions_data.py
        |
        v
    DataFrame
        |
        +---> libs/pnllib.py (Trade Matching & PnL Calc)
        |
        +---> tools/exposure_analysis.py (Greeks & Delta Calc)
        |           |
        |           v
        |     libs/stock_cache.py / libs/historical_cache.py
        |           |
        |           v
        |     yfinance API
        |
        v
Jupyter Notebooks (notebooks/) (User Interface)
```

## Key Workflows

### 1. PnL Analysis Workflow
1. Export trades CSV from broker
2. Load trades using `Trade` class in [libs/pnllib.py](libs/pnllib.py)
3. Pair opening and closing trades
4. Calculate realized PnL
5. Generate statistics using `compute_trade_stats()`
6. Visualize in [notebooks/pnl-tool.ipynb](notebooks/pnl-tool.ipynb)

### 2. Exposure Analysis Workflow
1. Export positions CSV from broker
2. Load positions using `PositionsReader` in [libs/positions_data.py](libs/positions_data.py)
3. Split into stocks and options
4. Pre-populate stock cache for all tickers
5. Calculate option deltas using Black-Scholes
6. Aggregate exposure by underlying
7. Display results with [tools/exposure_analysis.py](tools/exposure_analysis.py) main()

### 3. Stock Screening Workflow
1. Populate earnings dates using Finnhub API ([tools/manage_earnings_database.py](tools/manage_earnings_database.py))
2. Filter stocks based on earnings date proximity using [tools/screener.py](tools/screener.py)
3. Use [libs/historical_cache.py](libs/historical_cache.py) for fast historical price data retrieval
4. Calculate historical returns and volatility
5. Identify trading candidates
6. Analyze in [notebooks/screener.ipynb](notebooks/screener.ipynb)

## External Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance API for stock data
- **py_vollib**: Black-Scholes options pricing and Greeks
- **finnhub-python**: Earnings calendar data (referenced but not directly imported)

## Configuration Constants

- **VOLATILITY_PREMIUM** (tools/exposure_analysis.py): 50% (multiplies historical volatility by 1.5)
- **OUTLIER_CAP_MULTIPLIER** (tools/exposure_analysis.py): 4 (caps returns at 4x the 90th percentile)
- **Cache Interval** (libs/stock_cache.py): 6 hours
- **Staleness Days** (libs/historical_cache.py): 1 day (default)
- **History Days** (libs/historical_cache.py): 252 trading days (1 year)
- **Volatility Lookback** (tools/exposure_analysis.py): 252 trading days (1 year)

## Data Storage

- **positions.csv / pos.csv**: Current positions exported from broker
- **all.csv**: Historical trades exported from broker
- **data/earnings_dates/M1.csv - M12.csv**: Monthly earnings calendar data
- **data/stock_cache.json**: Cached stock market data (current prices, volatility)
- **data/historical_cache/{SYMBOL}.json**: Per-stock historical price cache
- **data/pos.csv**: Processed positions data

## Directory Structure

```
optionspnl/
├── libs/                      # Core library modules
│   ├── pnllib.py             # Trade matching and PnL calculation
│   ├── positionslib.py       # Position file parsing
│   ├── positions_data.py     # Object-oriented positions interface
│   ├── stocks.py             # Earnings date management
│   ├── stock_cache.py        # Current stock data cache
│   ├── historical_cache.py   # Historical price data cache
│   └── utils.py              # Statistical utilities
├── tools/                     # Executable scripts
│   ├── screener.py           # Stock screening tool
│   ├── exposure_analysis.py  # Portfolio Greeks analysis
│   └── manage_earnings_database.py  # Earnings database management
├── notebooks/                 # Jupyter notebooks
│   ├── pnl-tool.ipynb        # Monthly PnL analysis
│   ├── pos-analysis.ipynb    # Current positions analysis
│   ├── screener.ipynb        # Stock screening interface
│   └── stock_pnl_tool.ipynb  # Individual stock PnL tracking
├── data/                      # Data storage
│   ├── stock_cache.json      # Stock cache
│   ├── historical_cache/     # Historical price cache
│   └── earnings_dates/       # Earnings calendar data
├── temp/                      # Throwaway test programs
└── docs/                      # Documentation
    └── codebase_structure.md # This file
```

---

This codebase represents a comprehensive options trading analysis platform with strengths in automated trade matching, exposure calculation, and integration with real-time market data. The modular design with clear separation between libraries (`libs/`), tools (`tools/`), and notebooks (`notebooks/`) allows for easy extension and maintenance of individual components.
