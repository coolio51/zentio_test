# 🧬 Real Data Testing for Genetic Algorithm

This directory contains tools for testing the genetic algorithm with actual production data from your live Zentio system.

## 📁 Files

- `export_real_data.py` - Script to export live data to JSON files
- `tests/test_genetic_real.py` - Test that uses real data
- `tests/data/` - Directory containing exported JSON data files

## 🚀 Quick Start

### 1. Export Real Data

First, make sure your Zentio server is running, then export the current data:

```bash
# Export data from live system
make export-data

# Or run directly
uv run python export_real_data.py
```

This will create JSON files in `tests/data/`:

- `real_manufacturing_orders.json` - Manufacturing orders
- `real_resources.json` - Resources and capabilities
- `real_metadata.json` - Export metadata
- `real_combined.json` - All data combined

### 2. Run Real Data Test

```bash
# Run genetic algorithm test with real data
make test-genetic-real

# Or run directly
uv run python -m tests.test_genetic_real
```

## 🔧 Configuration

The real data test uses optimized settings for production-scale data:

- **Population size**: 20 (larger for complex data)
- **Max generations**: 100 (more iterations)
- **Elite size**: 5 (preserve best solutions)
- **Stagnation limit**: 20 generations
- **Operation splits**: Up to 5 splits per operation

## 📊 What You'll See

The test provides detailed output including:

```
🧬 GENETIC ALGORITHM TEST - REAL PRODUCTION DATA

📁 Loading real production data...
📊 Data exported: 2025-08-05T17:30:00
🏢 Organization: xo68UolxkoweysRmTFYmJKN6wCMpau6e
📅 Date range: 60 days ahead

✅ Loaded 50 manufacturing orders
✅ Loaded 17 resources
📋 Extracted 200 operations

🚀 Starting Genetic Optimization with Real Data...

📊 Gen   1: Best=219440.35 Global=219440.35 Avg=237601.86 Ops=145/200 (72.5%) Dropped=55 MOs=50 Splits=12 Machines=145 🔥 Active
📊 Gen  10: Best=198250.20 Global=198250.20 Avg=215430.15 Ops=165/200 (82.5%) Dropped=35 MOs=50 Splits=18 Machines=165 🔥 Active
...

✅ Optimization complete! Ran 45 generations
   Operations scheduled: 180/200 (90.0% success rate)
   ⚠️  Operations dropped: 20 (10.0%)
   📊 Dropped Operations Breakdown:
      • Phase Scheduling Failed: 12 (60.0%)
      • Dependency Dropped: 8 (40.0%)
```

## 🎯 Benefits

### Realistic Testing

- Test with actual production complexity
- Real resource constraints and dependencies
- Actual manufacturing order patterns

### Performance Analysis

- Measure algorithm performance on real data
- Identify bottlenecks and optimization opportunities
- Validate improvements without affecting production

### Reproducible Results

- Consistent test data across runs
- Compare algorithm versions objectively
- Share test scenarios with team

## 💡 Tips

### Data Freshness

- Re-export data regularly to test with current conditions
- Export data represents a 2-month scheduling window
- Data includes actual resource availability and constraints

### Performance Tuning

- Adjust genetic algorithm parameters in `test_genetic_real.py`
- Monitor success rates and dropped operation patterns
- Use results to optimize resource allocation

### Debugging

- Individual operation drop logging has been removed for performance
- Use the summary statistics to identify patterns
- Check dropped operations breakdown for root causes

## 🔄 Workflow

```bash
# 1. Export fresh data from live system
make export-data

# 2. Run test with real data
make test-genetic-real

# 3. Analyze results and tune algorithm

# 4. Repeat as needed
```

This allows you to iterate on the genetic algorithm with confidence that improvements will work in your actual production environment!
