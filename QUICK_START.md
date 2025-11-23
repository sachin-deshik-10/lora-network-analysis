# Quick Start Guide - Enhanced LoRa Network Analysis

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```powershell
pip install pandas numpy streamlit plotly scikit-learn
```

### Step 2: Run the Application
```powershell
streamlit run main.py
```

### Step 3: Configure & Explore
1. Open your browser to `http://localhost:8501`
2. Use the sidebar to configure settings
3. Click the buttons to run analyses

---

## 📊 Feature Overview

### 1. Network Visualization (Auto-displayed)
- **3D Plot**: SNR vs Throughput vs BER
- **SNR vs Distance**: Signal degradation analysis
- **Throughput vs Latency**: Performance trade-offs

### 2. Exploratory Data Analysis (Auto-displayed)
- **Correlation Heatmap**: Feature relationships
- **Distribution Plots**: 6 key metrics
- **Network Health Score**: Real-time health indicator

### 3. Machine Learning Models (Click "Train Models")
- Trains 3 regression models
- Compares R², RMSE, MAE, CV scores
- Displays feature importance
- Saves best model automatically

### 4. Modulation Analysis (Click "Analyze Modulations")
- Compares CSS, FSK, and LoRa
- Recommends optimal scheme
- Shows performance metrics
- Visual comparisons

### 5. Network Efficiency (Auto-displayed)
- Average efficiency (kbps/ms)
- Peak efficiency
- Network utilization %
- Distribution histogram

---

## ⚙️ Configuration Options

### Sidebar Settings

| Setting | Options | Description |
|---------|---------|-------------|
| **Real-Time Simulation** | On/Off | Adds timestamps for time-series |
| **Sample Size** | 100-2000 | Number of data points |
| **Missing Value Strategy** | mean/median | How to fill gaps |
| **Scaling Method** | standard/minmax | Normalization type |
| **Auto-refresh** | On/Off | 30s refresh (real-time only) |

---

## 📈 Understanding the Results

### Model Performance Metrics

- **R² Score**: 0-1 (higher is better)
  - 0.9+ = Excellent
  - 0.7-0.9 = Good
  - <0.7 = Needs improvement

- **RMSE**: Lower is better
  - Measures average prediction error
  
- **MAE**: Lower is better
  - Average absolute error

- **CV Score**: Cross-validation robustness
  - Similar to R² score interpretation

### Network Health Score

- **>80%**: Excellent network performance
- **60-80%**: Good performance
- **40-60%**: Moderate performance
- **<40%**: Network needs attention

### Efficiency Rating

- **>1.0 kbps/ms**: High efficiency
- **0.5-1.0 kbps/ms**: Moderate efficiency
- **<0.5 kbps/ms**: Low efficiency

---

## 🎯 Recommended Workflow

### For Network Optimization
1. Generate data (1000+ samples recommended)
2. Review Network Health Score
3. Check SNR vs Distance plot
4. Click "Analyze Modulations"
5. Implement recommended modulation scheme

### For Predictive Analytics
1. Generate data
2. Click "Train Models"
3. Review R² scores
4. Check feature importance
5. Use saved model for predictions

### For Real-Time Monitoring
1. Enable "Real-Time Simulation" in sidebar
2. Enable "Auto-refresh"
3. Monitor time-series plots
4. Track network health score
5. Review efficiency metrics

---

## 🔧 Troubleshooting

### Issue: Streamlit not found
```powershell
pip install --upgrade streamlit
```

### Issue: Plotly charts not displaying
```powershell
pip install --upgrade plotly
```

### Issue: Model training slow
- Reduce sample size in sidebar
- Use fewer CV folds (requires code edit)

### Issue: Page too wide
- Already configured with `layout="wide"`
- Use browser zoom if needed

---

## 💡 Tips & Best Practices

### Performance Tips
- Use 1000 samples for standard analysis
- Use 2000 samples for detailed insights
- Enable real-time mode only when needed
- Disable auto-refresh when not monitoring

### Analysis Tips
- Check correlation matrix first
- Focus on high-importance features
- Compare all modulation schemes
- Monitor nodes with low health scores

### Visualization Tips
- Hover over plots for detailed info
- Rotate 3D plots by clicking and dragging
- Use animation controls on 3D plot
- Expand/collapse sections as needed

---

## 📱 Browser Compatibility

✅ **Recommended Browsers:**
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

⚠️ **Not Recommended:**
- Internet Explorer (not supported)

---

## 🎓 Understanding Key Concepts

### Signal-to-Noise Ratio (SNR)
- Measures signal quality
- Higher = better
- Typical range: 5-20 dB

### Bit Error Rate (BER)
- Percentage of erroneous bits
- Lower = better
- Typical range: 0.01-0.1

### Throughput
- Data transmission rate
- Higher = better
- Typical range: 50-150 kbps

### Latency
- Delay in data transmission
- Lower = better
- Typical range: 10-100 ms

### Modulation Schemes
- **CSS**: Good for long range, moderate speed
- **FSK**: Simple, reliable, lower complexity
- **LoRa**: Optimized for low power, long range

---

## 📞 Next Steps

1. **Experiment**: Try different configurations
2. **Analyze**: Review all metrics and plots
3. **Optimize**: Implement recommendations
4. **Monitor**: Use real-time mode for tracking
5. **Iterate**: Continuously improve based on insights

---

## 🌟 Pro Tips

### Maximize Insights
- Run multiple analyses with different sample sizes
- Compare results across configurations
- Focus on consistent patterns
- Document optimal configurations

### Export Results
- Take screenshots of key visualizations
- Copy metrics from tables
- Document recommended modulation schemes
- Track health scores over time

### Advanced Usage
- Modify code for custom metrics
- Add your own visualizations
- Integrate with real LoRa devices
- Export trained models for production

---

**Happy Analyzing! 🎉**

For detailed feature documentation, see `ADVANCED_FEATURES.md`
