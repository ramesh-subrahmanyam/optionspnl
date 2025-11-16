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

- **Data Layer**: CSV file reading, parsing, and data cleaning ([positionslib.py](positionslib.py), [positions_data.py](positions_data.py))
- **Business Logic Layer**: PnL calculation, exposure analysis, options pairing ([pnllib.py](pnllib.py), [exposure_analysis.py](exposure_analysis.py))
- **Data Access Layer**: Yahoo Finance integration, stock price fetching ([stocks.py](stocks.py), [stock_cache.py](stock_cache.py))
- **Utility Layer**: Helper functions, data transformation ([utils.py](utils.py))
- **Presentation Layer**: Jupyter notebooks for interactive analysis

---

## File Descriptions

### Core Library Files

#### [pnllib.py](pnllib.py)
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

**Dependencies**: pandas, numpy, datetime, yfinance, utils.py

---

#### [positionslib.py](positionslib.py)
**Primary Functionality**: Broker position file parsing and preprocessing

**Role in Project**: Handles reading and cleaning position data exported from brokerage accounts. Transforms raw CSV files into structured DataFrames.

**Key Components**:
- `skip_lines()`: Parses brokerage CSV files with non-standard headers
- `read_positions_file()`: Main function to read and clean position data
- `get_expiration_indexed_positions()`: Creates a dictionary mapping expiration dates to positions
- `get_margin()`: Extracts margin requirement from string format
- Handles numeric conversions, N/A values, and data type cleanup

**Dependencies**: pandas, pnllib

---

#### [positions_data.py](positions_data.py)
**Primary Functionality**: Object-oriented interface for positions data with stock/options separation

**Role in Project**: Provides a cleaner, more maintainable API for working with positions data. Separates stock positions from options positions and integrates with stock cache.

**Key Components**:
- `PositionsReader`: Utility class for reading CSV files with flexible header detection
- `StockData`: Class for managing stock positions with cache integration
- `OptionsData`: Class for managing options positions with symbol parsing
- `_parse_option_symbols()`: Extracts ticker, strike, expiration, and type from option symbols
- Integration with `StockCache` for real-time price and volatility updates

**Dependencies**: pandas, numpy, re, datetime, stock_cache

---

#### [stocks.py](stocks.py)
**Primary Functionality**: Earnings date management and stock screening

**Role in Project**: Manages earnings calendar data to identify stocks suitable for options trading based on earnings date proximity.

**Key Components**:
- `get_earnings_dates()`: Fetches earnings dates from Finnhub API
- `store_symbols()`: Persists earnings data to monthly CSV files (M1.csv - M12.csv)
- `get_stocks_outside_date()`: Finds stocks without earnings in a specified date range
- `get_stocks_within_days()`: Finds stocks with earnings within N days
- `StockData` class: Manages stock price data and calculates returns over various periods
- `attach_earnings_dates()`: Merges earnings dates with stock data

**Dependencies**: pandas, datetime, yfinance, logging

---

#### [stock_cache.py](stock_cache.py)
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

#### [historical_cache.py](historical_cache.py)
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

#### [exposure_analysis.py](exposure_analysis.py)
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

**Dependencies**: pandas, numpy, yfinance, datetime, positions_data, py_vollib, stock_cache

**Constants**:
- `VOLATILITY_PREMIUM`: 50% (multiplies historical vol by 1.5)
- `OUTLIER_CAP_MULTIPLIER`: 4x the 90th percentile return

---

#### [utils.py](utils.py)
**Primary Functionality**: Statistical analysis utilities and data visualization helpers

**Role in Project**: Provides reusable utilities for stock price analysis, volatility calculation, and conditional expectation analysis.

**Key Components**:
- `option_counts_by_date()`: Counts calls vs puts by expiration date
- `get_stock_closing_prices()`: Fetches historical closing prices from Yahoo Finance
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

#### [screener.py](screener.py)
**Primary Functionality**: Stock and options position tracking for individual stocks

**Role in Project**: Analyzes trading history for specific stocks to calculate realized and unrealized PnL across different time periods.

**Key Components**:
- `StockTrades` class: Tracks all trades for a single stock
- `read_data()`: Loads trades from CSV with date filtering
- `infer_positions()`: Reconstructs position size at each historical point
- `infer_pnl()`: Calculates PnL from any historical point to present
- `find_pnl_from_zero_pos_after_start_date()`: Identifies round-trip trades (open to close to zero)
- `compute_forward_pnl()`: Generates time-series of PnL evolution

**Dependencies**: pandas

---

### Interactive Notebooks

#### [pnl-tool.ipynb](pnl-tool.ipynb)
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

#### [pos-analysis.ipynb](pos-analysis.ipynb)
**Primary Functionality**: Current positions analysis by expiration date

**Role in Project**: Analyzes active positions grouped by expiration date, showing premium collected, unrealized PnL, and margin requirements.

**Key Features**:
- Expiration-date indexed position views
- Premium vs PnL comparison
- Margin requirement tracking
- Gap analysis (difference between premium and PnL)

**Usage Pattern**: Reads latest position file and generates tables showing positions expiring on specific dates.

---

#### [screener.ipynb](screener.ipynb)
**Primary Functionality**: Stock screening based on earnings dates

**Role in Project**: Helps identify stocks suitable for options trading by filtering based on earnings calendar proximity.

**Expected Features** (based on supporting libraries):
- Filter stocks with earnings outside a date range
- Display next earnings date for each stock
- Return-based screening
- Volatility-based screening

---

#### [stock_pnl_tool.ipynb](stock_pnl_tool.ipynb)
**Primary Functionality**: Individual stock PnL tracking

**Role in Project**: Deep-dive analysis of trading performance for specific stocks, tracking position evolution over time.

**Expected Features** (based on supporting libraries):
- Historical position reconstruction
- PnL evolution charts
- Round-trip trade identification
- Entry/exit analysis

---

## Data Flow Architecture

```
CSV Files (Broker Exports)
        |
        v
positionslib.py / positions_data.py
        |
        v
    DataFrame
        |
        +---> pnllib.py (Trade Matching & PnL Calc)
        |
        +---> exposure_analysis.py (Greeks & Delta Calc)
        |           |
        |           v
        |     stock_cache.py
        |           |
        |           v
        |     yfinance API
        |
        v
Jupyter Notebooks (User Interface)
```

## Key Workflows

### 1. PnL Analysis Workflow
1. Export trades CSV from broker
2. Load trades using `Trade` class in [pnllib.py](pnllib.py)
3. Pair opening and closing trades
4. Calculate realized PnL
5. Generate statistics using `compute_trade_stats()`
6. Visualize in [pnl-tool.ipynb](pnl-tool.ipynb)

### 2. Exposure Analysis Workflow
1. Export positions CSV from broker
2. Load positions using `PositionsReader` in [positions_data.py](positions_data.py)
3. Split into stocks and options
4. Pre-populate stock cache for all tickers
5. Calculate option deltas using Black-Scholes
6. Aggregate exposure by underlying
7. Display results with [exposure_analysis.py](exposure_analysis.py) main()

### 3. Stock Screening Workflow
1. Populate earnings dates using Finnhub API ([stocks.py](stocks.py))
2. Filter stocks based on earnings date proximity
3. Calculate historical returns and volatility
4. Identify trading candidates
5. Analyze in [screener.ipynb](screener.ipynb)

## External Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance API for stock data
- **py_vollib**: Black-Scholes options pricing and Greeks
- **finnhub-python**: Earnings calendar data (referenced but not directly imported)

## Configuration Constants

- **VOLATILITY_PREMIUM** (exposure_analysis.py): 50% (multiplies historical volatility by 1.5)
- **OUTLIER_CAP_MULTIPLIER** (exposure_analysis.py): 4 (caps returns at 4x the 90th percentile)
- **Cache Interval** (stock_cache.py): 6 hours
- **Volatility Lookback** (exposure_analysis.py): 252 trading days (1 year)

## Data Storage

- **positions.csv / pos.csv**: Current positions exported from broker
- **all.csv**: Historical trades exported from broker
- **earnings_dates/M1.csv - M12.csv**: Monthly earnings calendar data
- **data/stock_cache.json**: Cached stock market data
- **data/pos.csv**: Processed positions data

---

This codebase represents a comprehensive options trading analysis platform with strengths in automated trade matching, exposure calculation, and integration with real-time market data. The modular design allows for easy extension and maintenance of individual components.
