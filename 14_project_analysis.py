import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from playwright.sync_api import sync_playwright

# ============================================================
# Global Configuration & Visualization Styling
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

BASE_URL = "https://cnweather.pages.dev/"
CITIES = ["beijing", "shanghai", "guangzhou", "chengdu"]
START_DATE = "2024-01-01"
END_DATE = "2024-01-31"

# Weather condition translation mapping for English rendering
CONDITION_MAP = {
    "晴": "Sunny",
    "多云": "Cloudy",
    "阴": "Overcast",
    "小雨": "Light Rain",
    "中雨": "Moderate Rain",
    "大雨": "Heavy Rain",
    "阵雨": "Showers",
    "雨夹雪": "Sleet"
}


# ============================================================
# Step 1: Web Fetcher Class (JS Execution via Playwright)
# ============================================================
class WeatherFetcher:
    """
    Fetches weather dynamic static API data using Playwright headless browser
    to execute JavaScript on pure frontend static hosting environment.
    """
    def __init__(self, headless: bool = True):
        self.headless = headless

    def fetch_city_weather(self, city: str, start_date: str, end_date: str) -> dict:
        target_url = f"{BASE_URL}?city={city}&start={start_date}&end={end_date}"
        print(f"  [Fetcher] Requesting URL: {target_url}")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            try:
                page.goto(target_url, wait_until="networkidle", timeout=15000)
                content = page.locator("body").inner_text()
                json_data = json.loads(content)
                browser.close()
                return json_data
            except Exception as e:
                browser.close()
                print(f"  [Fetcher Error] Failed to fetch data for city '{city}': {e}")
                return {"error": True, "message": str(e)}

    def batch_fetch(self, cities: list, start_date: str, end_date: str) -> list:
        all_records = []
        print("\n[Step 1] Starting data fetching task via Playwright JS Executor...")
        
        # Test error response handling with an invalid city name
        invalid_res = self.fetch_city_weather("beijding", start_date, end_date)
        if invalid_res.get("error"):
            print(f"  [Validation] Successfully caught error response: {invalid_res.get('message')}")

        for c in cities:
            res = self.fetch_city_weather(c, start_date, end_date)
            if not res.get("error") and "data" in res:
                all_records.extend(res["data"])
                print(f"  [OK] Retrived {len(res['data'])} records for '{c}'")
            else:
                print(f"  [Warning] Skipping failed fetch for '{c}'")
                
        return all_records


# ============================================================
# Step 2: Data Processing & Analysis Class
# ============================================================
class WeatherAnalyzer:
    """
    Data cleaning, numerical processing, and SciPy statistical modeling.
    """
    def __init__(self, raw_data: list):
        self.raw_data = raw_data
        self.df = pd.DataFrame()

    def clean_and_transform(self) -> pd.DataFrame:
        """Cleans dataset and adds derived features."""
        df = pd.DataFrame(self.raw_data)
        
        # Parse dates
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_month"] = df["date"].dt.day
        df["day_name"] = df["date"].dt.day_name()
        
        # Calculate derived metrics
        df["temp_range"] = df["temp_high"] - df["temp_low"]
        df["temp_avg"] = (df["temp_high"] + df["temp_low"]) / 2.0
        
        # Map condition to English
        df["condition_en"] = df["condition"].map(CONDITION_MAP).fillna(df["condition"])
        
        self.df = df
        print("\n[Step 2] Data cleaning and feature engineering complete.")
        print(f"  DataFrame Shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        print("\nPreview of Cleaned Dataset:")
        print(self.df[["date", "city", "condition_en", "temp_high", "temp_low", "temp_avg", "humidity"]].head().to_string(index=False))
        return self.df

    def compute_numpy_statistics(self) -> dict:
        """Computes mean, median, IQR, std and total stats using NumPy."""
        print("\n[Step 3] Computing NumPy & Pandas Descriptive Statistics...")
        stats_summary = {}
        
        for city in self.df["city"].unique():
            city_df = self.df[self.df["city"] == city]
            temps = city_df["temp_avg"].to_numpy()
            
            stats_summary[city] = {
                "Mean_Temp": float(np.mean(temps)),
                "Median_Temp": float(np.median(temps)),
                "Std_Temp": float(np.std(temps, ddof=1)),
                "Q25_Temp": float(np.percentile(temps, 25)),
                "Q75_Temp": float(np.percentile(temps, 75)),
                "IQR_Temp": float(np.percentile(temps, 75) - np.percentile(temps, 25)),
                "Total_Rainfall": float(np.sum(city_df["rainfall"].to_numpy())),
                "Avg_Humidity": float(np.mean(city_df["humidity"].to_numpy()))
            }
            
        stats_df = pd.DataFrame(stats_summary).T.round(2)
        print("\n--- City Level Summary Statistics ---")
        print(stats_df.to_string())
        return stats_summary

    def run_scipy_hypothesis_tests(self) -> dict:
        """Runs SciPy statistical tests (t-test, correlation, linear regression)."""
        print("\n[Step 4] Running SciPy Statistical Analysis & Hypothesis Testing...")
        
        bj_temps = self.df[self.df["city"] == "beijing"]["temp_avg"].values
        sh_temps = self.df[self.df["city"] == "shanghai"]["temp_avg"].values
        
        # 1. Independent Two-Sample T-Test
        t_stat, p_val_ttest = stats.ttest_ind(bj_temps, sh_temps)
        print(f"\n1. Two-sample T-test (Beijing vs Shanghai Avg Temperature):")
        print(f"   t-statistic: {t_stat:.4f}, p-value: {p_val_ttest:.6f}")
        print(f"   Result: {'Statistically Significant' if p_val_ttest < 0.05 else 'Not Significant'}")
        
        # 2. Pearson Correlation: Temperature vs Humidity
        all_temps = self.df["temp_avg"].values
        all_humidity = self.df["humidity"].values
        r_val, p_val_corr = stats.pearsonr(all_temps, all_humidity)
        print(f"\n2. Pearson Correlation (Temperature vs Humidity):")
        print(f"   r-coefficient: {r_val:.4f}, p-value: {p_val_corr:.6f}")
        
        # 3. Linear Regression: Trend of Beijing daily temp over time
        bj_days = self.df[self.df["city"] == "beijing"]["day_of_month"].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(bj_days, bj_temps)
        print(f"\n3. Linear Regression (Beijing Temp Trend across Month):")
        print(f"   Equation: Temp = {slope:.4f} * Day + {intercept:.4f}")
        print(f"   R-squared: {r_value**2:.4f}, p-value: {p_value:.6f}")
        
        return {
            "ttest_bj_sh": {"t_stat": round(float(t_stat), 4), "p_val": round(float(p_val_ttest), 6)},
            "corr_temp_hum": {"r": round(float(r_val), 4), "p_val": round(float(p_val_corr), 6)},
            "regression_bj": {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4), "r_squared": round(float(r_value**2), 4)}
        }


# ============================================================
# Step 3: Visualization Dashboard Class
# ============================================================
class WeatherVisualizer:
    """
    Renders 6-panel English visualization dashboard using Matplotlib.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.colors = {
            "beijing": "#e74c3c",
            "shanghai": "#3498db",
            "guangzhou": "#2ecc71",
            "chengdu": "#9b59b6"
        }

    def generate_dashboard(self, output_filename: str = "weather_analysis_report.png"):
        print("\n[Step 5] Generating Visualizations Dashboard...")
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("China Dynamic Weather Analytics Dashboard (Jan 2024)", fontsize=18, fontweight="bold", y=0.98)
        
        cities = self.df["city"].unique()

        # Panel 1: Daily Average Temperature Trends
        ax1 = fig.add_subplot(2, 3, 1)
        for city in cities:
            c_df = self.df[self.df["city"] == city].sort_values("date")
            ax1.plot(c_df["day_of_month"], c_df["temp_avg"], marker="o", markersize=3, 
                     linewidth=1.8, label=city.capitalize(), color=self.colors.get(city, "#333333"))
        ax1.set_title("Daily Average Temperature Trend", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Day of Month")
        ax1.set_ylabel("Temperature (°C)")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Median & Average Temperature Comparison by City
        ax2 = fig.add_subplot(2, 3, 2)
        city_stats = self.df.groupby("city")["temp_avg"].agg(["mean", "median"]).reindex(cities)
        x = np.arange(len(cities))
        width = 0.35
        ax2.bar(x - width/2, city_stats["mean"], width, label="Mean", color="#34495e", alpha=0.85)
        ax2.bar(x + width/2, city_stats["median"], width, label="Median", color="#e67e22", alpha=0.85)
        ax2.set_title("Mean vs Median Temperature by City", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.capitalize() for c in cities])
        ax2.set_ylabel("Temperature (°C)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

        # Panel 3: Temperature Range Distribution (Box Plot)
        ax3 = fig.add_subplot(2, 3, 3)
        box_data = [self.df[self.df["city"] == c]["temp_range"].values for c in cities]
        bp = ax3.boxplot(box_data, labels=[c.capitalize() for c in cities], patch_artist=True)
        for patch, city in zip(bp["boxes"], cities):
            patch.set_facecolor(self.colors.get(city, "lightblue"))
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set(color="red", linewidth=2)
        ax3.set_title("Daily Temperature Range (High - Low)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Temperature Difference (°C)")
        ax3.grid(True, alpha=0.3, axis="y")

        # Panel 4: Temperature vs Humidity Scatter & Trendline
        ax4 = fig.add_subplot(2, 3, 4)
        for city in cities:
            c_df = self.df[self.df["city"] == city]
            ax4.scatter(c_df["temp_avg"], c_df["humidity"], label=city.capitalize(), 
                        color=self.colors.get(city, "gray"), s=40, alpha=0.7)
        x_all = self.df["temp_avg"].values
        y_all = self.df["humidity"].values
        z = np.polyfit(x_all, y_all, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax4.plot(x_line, p(x_line), "r--", linewidth=1.5, label="Overall Fit")
        ax4.set_title("Temperature vs Humidity Correlation", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Average Temperature (°C)")
        ax4.set_ylabel("Humidity (%)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Panel 5: Total Rainfall by City
        ax5 = fig.add_subplot(2, 3, 5)
        rain_sum = self.df.groupby("city")["rainfall"].sum().reindex(cities)
        bars = ax5.bar([c.capitalize() for c in cities], rain_sum.values, 
                       color=[self.colors.get(c, "teal") for c in cities], alpha=0.8)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.1f}mm", ha="center", va="bottom", fontsize=8)
        ax5.set_title("Total Cumulative Rainfall", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Rainfall (mm)")
        ax5.grid(True, alpha=0.3, axis="y")

        # Panel 6: Weather Condition Counts (English Mapping)
        ax6 = fig.add_subplot(2, 3, 6)
        cond_counts = self.df.groupby(["city", "condition_en"]).size().unstack(fill_value=0)
        cond_counts.plot(kind="bar", stacked=True, ax=ax6, colormap="tab10", alpha=0.85)
        ax6.set_title("Weather Condition Frequency", fontsize=11, fontweight="bold")
        ax6.set_xlabel("")
        ax6.set_ylabel("Days Count")
        ax6.set_xticklabels([c.capitalize() for c in cities], rotation=0)
        ax6.legend(fontsize=7, title="Condition", loc="upper right")
        ax6.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename, dpi=200)
        plt.close()
        print(f"  [OK] Visualization dashboard exported to '{output_filename}'")


# ============================================================
# Step 4: Data Export & Main Pipeline
# ============================================================
def main():
    print("============================================================")
    print("   China Weather Dynamic Data Analysis & Pipeline Execution")
    print("============================================================")

    # 1. Data Fetching via Playwright
    fetcher = WeatherFetcher(headless=True)
    raw_data = fetcher.batch_fetch(CITIES, START_DATE, END_DATE)

    # Save raw JSON API response
    raw_json_file = "weather_raw_data.json"
    with open(raw_json_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] Saved raw API responses to '{raw_json_file}'")

    # 2. Data Cleaning & Analysis
    analyzer = WeatherAnalyzer(raw_data)
    df_clean = analyzer.clean_and_transform()
    
    # Export Clean CSV
    csv_file = "weather_cleaned_dataset.csv"
    df_clean.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"  [OK] Exported cleaned dataset to '{csv_file}'")

    # Statistical Computation
    stats_summary = analyzer.compute_numpy_statistics()
    scipy_report = analyzer.run_scipy_hypothesis_tests()

    # Save JSON Statistical Report
    summary_json_file = "weather_analysis_report.json"
    report_dict = {
        "metadata": {
            "cities": CITIES,
            "period": f"{START_DATE} to {END_DATE}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "descriptive_stats": stats_summary,
        "hypothesis_tests": scipy_report
    }
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Summary report exported to '{summary_json_file}'")

    # 3. Visualization
    visualizer = WeatherVisualizer(df_clean)
    visualizer.generate_dashboard("weather_analysis_report.png")

    print("\n============================================================")
    print("  Pipeline execution completed successfully!")
    print("  Generated Files:")
    print("   - weather_raw_data.json")
    print("   - weather_cleaned_dataset.csv")
    print("   - weather_analysis_report.json")
    print("   - weather_analysis_report.png")
    print("============================================================")


if __name__ == "__main__":
    main()