# 14_project_analysis.py
# 综合项目：城市气温数据分析与可视化
# 涵盖知识：JSON(08) + Requests(09) + Matplotlib(11) + Pandas(12) + SciPy(13)

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set English font (available on all Linux systems)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("  Comprehensive Project: City Weather Data Analysis")
print("  Covers: JSON(08) + Requests(09) + Matplotlib(11) + Pandas(12) + SciPy(13)")
print("=" * 60)

# ============================================================
# Step 1: Simulate loading data from an API (JSON + Requests)
# ============================================================
print("\n[Step 1] Loading raw data...")

# Simulated API response (in real projects, use requests.get() to fetch)
raw_api_data = {
    "source": "Meteorological Data Center API",
    "cities": [
        {"city": "Beijing", "month": "Jan", "temp_high": 2, "temp_low": -8, "rainfall": 3, "humidity": 35},
        {"city": "Beijing", "month": "Feb", "temp_high": 5, "temp_low": -5, "rainfall": 5, "humidity": 38},
        {"city": "Beijing", "month": "Mar", "temp_high": 12, "temp_low": 2, "rainfall": 10, "humidity": 45},
        {"city": "Beijing", "month": "Apr", "temp_high": 21, "temp_low": 9, "rainfall": 25, "humidity": 50},
        {"city": "Beijing", "month": "May", "temp_high": 27, "temp_low": 15, "rainfall": 35, "humidity": 55},
        {"city": "Beijing", "month": "Jun", "temp_high": 31, "temp_low": 20, "rainfall": 70, "humidity": 62},
        {"city": "Beijing", "month": "Jul", "temp_high": 32, "temp_low": 23, "rainfall": 180, "humidity": 72},
        {"city": "Beijing", "month": "Aug", "temp_high": 30, "temp_low": 22, "rainfall": 160, "humidity": 68},
        {"city": "Beijing", "month": "Sep", "temp_high": 26, "temp_low": 16, "rainfall": 50, "humidity": 60},
        {"city": "Beijing", "month": "Oct", "temp_high": 19, "temp_low": 8, "rainfall": 20, "humidity": 52},
        {"city": "Beijing", "month": "Nov", "temp_high": 10, "temp_low": 0, "rainfall": 8, "humidity": 42},
        {"city": "Beijing", "month": "Dec", "temp_high": 3, "temp_low": -6, "rainfall": 3, "humidity": 36},

        {"city": "Shanghai", "month": "Jan", "temp_high": 8, "temp_low": 1, "rainfall": 50, "humidity": 72},
        {"city": "Shanghai", "month": "Feb", "temp_high": 9, "temp_low": 2, "rainfall": 60, "humidity": 73},
        {"city": "Shanghai", "month": "Mar", "temp_high": 13, "temp_low": 5, "rainfall": 80, "humidity": 74},
        {"city": "Shanghai", "month": "Apr", "temp_high": 19, "temp_low": 10, "rainfall": 90, "humidity": 73},
        {"city": "Shanghai", "month": "May", "temp_high": 24, "temp_low": 15, "rainfall": 90, "humidity": 72},
        {"city": "Shanghai", "month": "Jun", "temp_high": 28, "temp_low": 20, "rainfall": 150, "humidity": 78},
        {"city": "Shanghai", "month": "Jul", "temp_high": 33, "temp_low": 25, "rainfall": 140, "humidity": 77},
        {"city": "Shanghai", "month": "Aug", "temp_high": 33, "temp_low": 25, "rainfall": 130, "humidity": 76},
        {"city": "Shanghai", "month": "Sep", "temp_high": 28, "temp_low": 21, "rainfall": 80, "humidity": 74},
        {"city": "Shanghai", "month": "Oct", "temp_high": 23, "temp_low": 15, "rainfall": 50, "humidity": 70},
        {"city": "Shanghai", "month": "Nov", "temp_high": 17, "temp_low": 9, "rainfall": 50, "humidity": 70},
        {"city": "Shanghai", "month": "Dec", "temp_high": 11, "temp_low": 3, "rainfall": 40, "humidity": 70},

        {"city": "Guangzhou", "month": "Jan", "temp_high": 18, "temp_low": 10, "rainfall": 40, "humidity": 70},
        {"city": "Guangzhou", "month": "Feb", "temp_high": 18, "temp_low": 11, "rainfall": 60, "humidity": 74},
        {"city": "Guangzhou", "month": "Mar", "temp_high": 21, "temp_low": 14, "rainfall": 80, "humidity": 78},
        {"city": "Guangzhou", "month": "Apr", "temp_high": 26, "temp_low": 18, "rainfall": 170, "humidity": 80},
        {"city": "Guangzhou", "month": "May", "temp_high": 30, "temp_low": 22, "rainfall": 280, "humidity": 82},
        {"city": "Guangzhou", "month": "Jun", "temp_high": 32, "temp_low": 24, "rainfall": 300, "humidity": 83},
        {"city": "Guangzhou", "month": "Jul", "temp_high": 33, "temp_low": 25, "rainfall": 230, "humidity": 81},
        {"city": "Guangzhou", "month": "Aug", "temp_high": 33, "temp_low": 25, "rainfall": 220, "humidity": 80},
        {"city": "Guangzhou", "month": "Sep", "temp_high": 31, "temp_low": 23, "rainfall": 180, "humidity": 78},
        {"city": "Guangzhou", "month": "Oct", "temp_high": 28, "temp_low": 20, "rainfall": 70, "humidity": 72},
        {"city": "Guangzhou", "month": "Nov", "temp_high": 24, "temp_low": 15, "rainfall": 30, "humidity": 68},
        {"city": "Guangzhou", "month": "Dec", "temp_high": 20, "temp_low": 11, "rainfall": 30, "humidity": 66}
    ]
}

# Write JSON file (simulating API response save)
with open("weather_data.json", "w", encoding="utf-8") as f:
    json.dump(raw_api_data, f, indent=2, ensure_ascii=False)
print("  [OK] Raw data saved to weather_data.json")

# Read back from JSON file
with open("weather_data.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

print(f"  [OK] Source: {loaded_data['source']}")
print(f"  [OK] Records: {len(loaded_data['cities'])}")

# ============================================================
# Step 2: Pandas data cleaning and transformation
# ============================================================
print("\n[Step 2] Data cleaning and transformation...")

df = pd.DataFrame(loaded_data["cities"])

# Add month number for sorting and plotting
month_order = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
df["month_num"] = df["month"].map(month_order)

# Calculate temperature range and average
df["temp_range"] = df["temp_high"] - df["temp_low"]
df["temp_avg"] = (df["temp_high"] + df["temp_low"]) / 2

# Add season column
def get_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["season"] = df["month_num"].apply(get_season)
df = df.sort_values(["city", "month_num"]).reset_index(drop=True)

print(f"  [OK] DataFrame shape: {df.shape}")
print(f"  [OK] Columns: {list(df.columns)}")
print(f"  [OK] Cities: {df['city'].unique().tolist()}")
print(f"  [OK] Preview:")
print(df.head(6).to_string(index=False))

# ============================================================
# Step 3: Descriptive statistics with Pandas
# ============================================================
print("\n[Step 3] Descriptive statistics...")

print("\n  City-level annual statistics:")
city_stats = df.groupby("city").agg(
    Avg_High=("temp_high", "mean"),
    Avg_Low=("temp_low", "mean"),
    Avg_Temp=("temp_avg", "mean"),
    Avg_Range=("temp_range", "mean"),
    Total_Rainfall=("rainfall", "sum"),
    Avg_Humidity=("humidity", "mean")
).round(1)
print(city_stats)

print("\n  Beijing seasonal average temperature:")
beijing_season = df[df["city"] == "Beijing"].groupby("season")["temp_avg"].mean().round(1)
print(beijing_season)

# ============================================================
# Step 4: SciPy statistical analysis
# ============================================================
print("\n[Step 4] SciPy statistical analysis...")

# 1. Two-sample t-test: Beijing vs Shanghai
beijing_temps = df[df["city"] == "Beijing"]["temp_avg"].values
shanghai_temps = df[df["city"] == "Shanghai"]["temp_avg"].values
guangzhou_temps = df[df["city"] == "Guangzhou"]["temp_avg"].values

t_stat, p_value = stats.ttest_ind(beijing_temps, shanghai_temps)
print(f"\n  Beijing vs Shanghai t-test:")
print(f"    t = {t_stat:.4f}, p = {p_value:.4f}")
print(f"    Conclusion: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")

# 2. Correlation: temperature vs humidity
all_temp = df["temp_avg"].values
all_humidity = df["humidity"].values
pearson_r, pearson_p = stats.pearsonr(all_temp, all_humidity)
print(f"\n  Temperature vs Humidity Pearson correlation:")
print(f"    r = {pearson_r:.4f}, p = {pearson_p:.6f}")
print(f"    Conclusion: {'Significantly correlated' if pearson_p < 0.05 else 'Not significantly correlated'}")

# 3. Correlation: temperature vs rainfall
all_rainfall = df["rainfall"].values
pearson_r2, pearson_p2 = stats.pearsonr(all_temp, all_rainfall)
print(f"\n  Temperature vs Rainfall Pearson correlation:")
print(f"    r = {pearson_r2:.4f}, p = {pearson_p2:.6f}")

# 4. Linear regression: month trend for Beijing temperature
x_months = df[df["city"] == "Beijing"]["month_num"].values
y_beijing_temp = df[df["city"] == "Beijing"]["temp_avg"].values
slope, intercept, r_value, p_val, std_err = stats.linregress(x_months[:7], y_beijing_temp[:7])
print(f"\n  Beijing Jan-Jul temperature linear trend:")
print(f"    Equation: temp = {slope:.2f} * month + {intercept:.2f}")
print(f"    R-squared = {r_value**2:.4f}")

# ============================================================
# Step 5: Matplotlib visualization (6-panel dashboard)
# ============================================================
print("\n[Step 5] Generating visualizations...")

fig = plt.figure(figsize=(20, 14))
fig.suptitle("China City Climate Data Analysis Report", fontsize=20, fontweight="bold", y=0.98)

colors_city = {"Beijing": "#e74c3c", "Shanghai": "#3498db", "Guangzhou": "#2ecc71"}

# --- Panel 1: Monthly average temperature line chart ---
ax1 = fig.add_subplot(2, 3, 1)
for city in df["city"].unique():
    city_data = df[df["city"] == city].sort_values("month_num")
    ax1.plot(city_data["month"], city_data["temp_avg"],
             marker="o", linewidth=2, label=city, color=colors_city[city])
ax1.set_title("Monthly Avg Temperature Comparison", fontsize=13, fontweight="bold")
ax1.set_ylabel("Temperature (C)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

# --- Panel 2: Monthly rainfall grouped bar chart ---
ax2 = fig.add_subplot(2, 3, 2)
x_pos = np.arange(12)
width = 0.25
for i, city in enumerate(df["city"].unique()):
    city_data = df[df["city"] == city].sort_values("month_num")
    offset = (i - 1) * width
    ax2.bar(x_pos + offset, city_data["rainfall"], width,
            label=city, color=colors_city[city], alpha=0.85)
ax2.set_title("Monthly Rainfall Comparison", fontsize=13, fontweight="bold")
ax2.set_ylabel("Rainfall (mm)")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df[df["city"] == "Beijing"]["month"].values, rotation=45)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Beijing temperature range area chart ---
ax3 = fig.add_subplot(2, 3, 3)
bj = df[df["city"] == "Beijing"].sort_values("month_num")
ax3.fill_between(bj["month"], bj["temp_low"], bj["temp_high"],
                 alpha=0.4, color="#e74c3c", label="Temp range")
ax3.plot(bj["month"], bj["temp_high"], color="#e74c3c", linewidth=2, marker="s", markersize=4)
ax3.plot(bj["month"], bj["temp_low"], color="#3498db", linewidth=2, marker="^", markersize=4)
ax3.set_title("Beijing Monthly Temp Range", fontsize=13, fontweight="bold")
ax3.set_ylabel("Temperature (C)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis="x", rotation=45)

# --- Panel 4: Annual average temperature bar chart ---
ax4 = fig.add_subplot(2, 3, 4)
city_avg = df.groupby("city")["temp_avg"].mean().sort_values(ascending=False)
bars = ax4.bar(city_avg.index, city_avg.values,
               color=[colors_city[c] for c in city_avg.index], alpha=0.85, edgecolor="black")
for bar, val in zip(bars, city_avg.values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f"{val:.1f}C", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax4.set_title("Annual Avg Temperature by City", fontsize=13, fontweight="bold")
ax4.set_ylabel("Avg Temperature (C)")
ax4.set_ylim(0, max(city_avg.values) * 1.2)
ax4.grid(True, alpha=0.3, axis="y")

# --- Panel 5: Temperature vs humidity scatter plot ---
ax5 = fig.add_subplot(2, 3, 5)
for city in df["city"].unique():
    city_data = df[df["city"] == city]
    ax5.scatter(city_data["temp_avg"], city_data["humidity"],
               c=colors_city[city], s=80, alpha=0.7, label=city, edgecolors="black", linewidth=0.5)
# Trend line
z = np.polyfit(all_temp, all_humidity, 1)
p = np.poly1d(z)
x_line = np.linspace(all_temp.min(), all_temp.max(), 100)
ax5.plot(x_line, p(x_line), "--", color="gray", linewidth=1.5, alpha=0.7,
         label=f"Trend (r={pearson_r:.2f})")
ax5.set_title("Temperature vs Humidity", fontsize=13, fontweight="bold")
ax5.set_xlabel("Avg Temperature (C)")
ax5.set_ylabel("Humidity (%)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# --- Panel 6: Beijing seasonal box plot ---
ax6 = fig.add_subplot(2, 3, 6)
bj_season = df[df["city"] == "Beijing"]
season_order = ["Spring", "Summer", "Autumn", "Winter"]
box_data = [bj_season[bj_season["season"] == s]["temp_avg"].values for s in season_order]
ax6.boxplot(box_data, tick_labels=season_order, patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            flierprops=dict(marker="o", color="red", alpha=0.5))
ax6.set_title("Beijing Seasonal Temp Distribution", fontsize=13, fontweight="bold")
ax6.set_ylabel("Avg Temperature (C)")
ax6.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("weather_analysis_report.png", dpi=150, bbox_inches="tight")
print("  [OK] Report saved to weather_analysis_report.png")
plt.close()

# ============================================================
# Step 6: Generate summary report (JSON + CSV export)
# ============================================================
print("\n[Step 6] Generating summary report...")

# Export cleaned data as CSV
df.to_csv("weather_clean.csv", index=False, encoding="utf-8-sig")
print("  [OK] Cleaned data exported to weather_clean.csv")

# Generate JSON analysis report
report = {
    "title": "China City Climate Data Analysis Report",
    "data_source": raw_api_data["source"],
    "summary": {
        city: {
            "annual_avg_temp": round(df[df["city"] == city]["temp_avg"].mean(), 1),
            "max_monthly_avg": round(df[df["city"] == city]["temp_avg"].max(), 1),
            "min_monthly_avg": round(df[df["city"] == city]["temp_avg"].min(), 1),
            "total_rainfall": int(df[df["city"] == city]["rainfall"].sum()),
            "hottest_month": df[df["city"] == city].loc[df[df["city"] == city]["temp_avg"].idxmax(), "month"],
            "coldest_month": df[df["city"] == city].loc[df[df["city"] == city]["temp_avg"].idxmin(), "month"]
        }
        for city in df["city"].unique()
    },
    "statistics": {
        "beijing_vs_shanghai_ttest": {
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < 0.05)
        },
        "temp_humidity_correlation": {
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(pearson_p), 6)
        }
    }
}

with open("weather_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("  [OK] Analysis report exported to weather_report.json")

# Print report summary
print("\n" + "=" * 60)
print("  Analysis Report Summary")
print("=" * 60)
for city in df["city"].unique():
    ci = report["summary"][city]
    print(f"\n  {city}:")
    print(f"     Annual avg temp: {ci['annual_avg_temp']}C")
    print(f"     Range: {ci['min_monthly_avg']}C ~ {ci['max_monthly_avg']}C")
    print(f"     Total rainfall: {ci['total_rainfall']} mm")
    print(f"     Hottest month: {ci['hottest_month']}")
    print(f"     Coldest month: {ci['coldest_month']}")

sig = report["statistics"]["beijing_vs_shanghai_ttest"]["significant"]
print(f"\n  Beijing vs Shanghai diff: {'Significant' if sig else 'Not significant'}")
print(f"  Temp-Humidity correlation: r = {report['statistics']['temp_humidity_correlation']['pearson_r']}")

# ============================================================
# Cleanup temp files
# ============================================================
for f in ["weather_data.json", "weather_clean.csv", "weather_report.json"]:
    if os.path.exists(f):
        os.remove(f)

print("\n" + "=" * 60)
print("  Project complete!")
print("  Output: weather_analysis_report.png")
print("=" * 60)
