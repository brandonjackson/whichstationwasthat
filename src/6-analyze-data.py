"""
Step 6: Data Analysis

Generates visualizations and analysis from the merged data.csv file using
matplotlib and pandas.

Usage:
    python 6-analyze-data.py

Or via Makefile:
    make analyze

Prerequisites:
    - pandas library installed
    - matplotlib library installed
    - cartopy library installed (for heatmap projection; optional but recommended)
    - data.csv file from step 5

Behavior:
    - Loads data.csv from project root
    - Generates several visualizations:
      1. Time series: scatter plot of observations per week with 3-week smoothing
      2. Pie chart of top stations (by station_location_city)
      3. Pie chart of top countries (by station_location_country)
      4. Bar chart showing tracing success rate (success/fail)
      5. Heatmap of station locations (azimuthal equidistant projection centered on London)
      6. Histogram of distances (calculated from reporter location or London if missing)
    - Saves all figures to results/ directory

Output:
    Creates PNG files in results/ directory:
    - observations_per_week.png
    - top_stations_pie.png
    - top_countries_pie.png
    - tracing_success_rate.png
    - station_location_heatmap.png
    - distances_histogram.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

# Try to import cartopy for map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# London coordinates as fallback
LONDON_LAT = 51.5
LONDON_LNG = -0.1

# Map radius for azimuthal equidistant projection (in km, converted to meters for projection)
MAP_RADIUS_KM = 10000
MAP_RADIUS_METERS = MAP_RADIUS_KM * 1000

# Color configuration for map visualization
MAP_COLORS = {
    'land_color': 'lightgray',           # Color for land features
    'land_alpha': 0.2,                   # Transparency of land (0.0 = transparent, 1.0 = opaque)
    'ocean_color': 'lightblue',          # Color for ocean features
    'ocean_alpha': 0.2,                  # Transparency of ocean
    'coastline_color': 'black',          # Color for coastlines
    'coastline_width': 0.5,              # Width of coastline lines
    'borders_color': 'gray',             # Color for country borders
    'borders_width': 0.5,                # Width of border lines
    'borders_alpha': 0.3,                # Transparency of borders
    'heatmap_colormap': 'Reds',          # Colormap for heatmap ('Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'viridis', 'plasma', etc.)
    'heatmap_alpha': 0.7,                # Transparency of heatmap overlay
    'heatmap_levels': 20,                # Number of contour levels in heatmap
    'heatmap_min_threshold': 0.1,        # Minimum value threshold (values below this are transparent/not shown)
    'circle_color': 'gray',              # Color for distance circles
    'circle_alpha': 0.5,                 # Transparency of distance circles
    'circle_linestyle': '--',            # Line style for circles ('-', '--', '-.', ':')
    'circle_linewidth': 0.8,             # Line width for distance circles
    'center_marker_color': 'blue',       # Color for center point marker
    'center_marker_size': 10,            # Size of center point marker
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth (in km).
    Uses the Haversine formula.
    """
    try:
        # Convert to float (handles strings and NaN)
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (ValueError, TypeError):
        return np.nan
    
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    R = 6371  # Earth radius in km
    return R * c


def calculate_distances(df):
    """Calculate distances from reporter location to station location."""
    distances = []
    
    for idx, row in df.iterrows():
        # Use reporter location if available, else London
        try:
            if pd.notna(row['reporter_location_latitude']) and pd.notna(row['reporter_location_longitude']):
                obs_lat = float(row['reporter_location_latitude'])
                obs_lng = float(row['reporter_location_longitude'])
            else:
                obs_lat = LONDON_LAT
                obs_lng = LONDON_LNG
        except (ValueError, TypeError):
            obs_lat = LONDON_LAT
            obs_lng = LONDON_LNG
        
        # Calculate distance if station location is available
        try:
            if pd.notna(row['station_location_latitude']) and pd.notna(row['station_location_longitude']):
                station_lat = float(row['station_location_latitude'])
                station_lng = float(row['station_location_longitude'])
                distance = haversine_distance(obs_lat, obs_lng, station_lat, station_lng)
                distances.append(distance)
            else:
                distances.append(np.nan)
        except (ValueError, TypeError):
            distances.append(np.nan)
    
    return distances


def smooth_data(values, window=3):
    """Apply moving average smoothing with window size."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        # Use a centered window when possible, otherwise use available data
        half_window = window // 2
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        smoothed.append(np.mean(values[start:end]))
    return np.array(smoothed)


def plot_time_series(df, output_dir):
    """Plot time series of observations per week with smoothing."""
    # Count observations per week
    weekly_counts = df.groupby('issue_date').size().reset_index(name='count')
    weekly_counts['issue_date'] = pd.to_datetime(weekly_counts['issue_date'])
    weekly_counts = weekly_counts.sort_values('issue_date')
    
    # Apply 3-week smoothing
    smoothed_counts = smooth_data(weekly_counts['count'].values, window=3)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(weekly_counts['issue_date'], weekly_counts['count'], 
                alpha=0.6, s=30, label='Weekly observations')
    plt.plot(weekly_counts['issue_date'], smoothed_counts, 
             color='red', linewidth=2, label='3-week moving average')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Observations', fontsize=12)
    plt.title('Observations Per Week Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'observations_per_week.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: observations_per_week.png")


def plot_top_stations_pie(df, output_dir, top_n=10):
    """Plot pie chart of top stations."""
    # Count stations by city
    station_counts = df['station_location_city'].value_counts().head(top_n)
    
    # Combine others
    others_count = df['station_location_city'].value_counts().iloc[top_n:].sum()
    
    if others_count > 0:
        station_counts['Others'] = others_count
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(station_counts)))
    plt.pie(station_counts.values, labels=station_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors)
    plt.title(f'Top {top_n} Stations by Observation Count', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_stations_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: top_stations_pie.png")


def plot_top_countries_pie(df, output_dir, top_n=10):
    """Plot pie chart of top countries."""
    # Count countries
    country_counts = df['station_location_country'].value_counts().head(top_n)
    
    # Combine others
    others_count = df['station_location_country'].value_counts().iloc[top_n:].sum()
    
    if others_count > 0:
        country_counts['Others'] = others_count
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Pastel1(range(len(country_counts)))
    plt.pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors)
    plt.title(f'Top {top_n} Countries by Observation Count', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_countries_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: top_countries_pie.png")


def categorize_observation(row):
    """
    Categorize an observation for tracing success rate:
    - 'success': Successfully traced/identified (has station location or meaningful observation text)
    - 'fail': Untraceable (explicitly cannot trace, or no information)
    """
    # Check station location fields
    station_city = row.get('station_location_city')
    station_country = row.get('station_location_country')
    has_station_location = (pd.notna(station_city) and str(station_city).strip() != '' and str(station_city).strip().lower() != 'nan') or \
                          (pd.notna(station_country) and str(station_country).strip() != '' and str(station_country).strip().lower() != 'nan')
    
    # Check observation text and full text for keywords
    obs_text = str(row.get('observation_text', '')).lower()
    full_text = str(row.get('full_text', '')).lower()
    combined_text = obs_text + ' ' + full_text
    
    # Explicit untraceable keywords = fail
    if ('cannot trace' in combined_text or
        'not traceable' in combined_text or
        'untraceable' in combined_text or
        'not sufficient' in combined_text or
        'too vague' in combined_text):
        return 'fail'
    
    # If we have station location, it's a success
    if has_station_location:
        return 'success'
    
    # If no station location but observation text suggests a station was identified
    # (has meaningful content), consider it a success
    obs_text_orig = str(row.get('observation_text', ''))
    if obs_text_orig and obs_text_orig.strip() and obs_text_orig.strip() not in ['', '.']:
        # Has observation text suggesting a station was identified, even if not geocoded
        return 'success'
    
    # No station location and no meaningful observation text = fail
    return 'fail'


def plot_tracing_success_rate(df, output_dir):
    """Plot tracing success rate (success/fail)."""
    # Categorize each observation
    categories = df.apply(categorize_observation, axis=1)
    category_counts = categories.value_counts()
    
    # Ensure consistent ordering: success first, then fail
    ordered_counts = pd.Series({
        'success': category_counts.get('success', 0),
        'fail': category_counts.get('fail', 0)
    })
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#e74c3c']  # green for success, red for fail
    bars = plt.bar(ordered_counts.index, ordered_counts.values, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11)
    
    # Calculate and display success rate
    success_rate = (ordered_counts.get('success', 0) / len(df)) * 100
    
    plt.xlabel('Tracing Result', fontsize=12)
    plt.ylabel('Number of Observations', fontsize=12)
    plt.title(f'Tracing Success Rate (Success: {success_rate:.1f}%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'tracing_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: tracing_success_rate.png")
    print(f"   Success: {ordered_counts.get('success', 0)} ({success_rate:.1f}%)")
    print(f"   Fail: {ordered_counts.get('fail', 0)} ({100-success_rate:.1f}%)")


def plot_station_heatmap(df, output_dir):
    """Plot heatmap of station locations using azimuthal equidistant projection centered on London."""
    if not CARTOPY_AVAILABLE:
        print("⚠️  Cartopy not available. Install with: pip install cartopy")
        print("   Skipping heatmap generation.")
        return
    
    # Filter to rows with valid station coordinates
    valid_stations = df[
        (df['station_location_latitude'].notna()) & 
        (df['station_location_longitude'].notna())
    ].copy()
    
    # Convert to float, handling any string values
    try:
        valid_stations['lat'] = pd.to_numeric(valid_stations['station_location_latitude'], errors='coerce')
        valid_stations['lon'] = pd.to_numeric(valid_stations['station_location_longitude'], errors='coerce')
        valid_stations = valid_stations[
            valid_stations['lat'].notna() & valid_stations['lon'].notna()
        ]
    except Exception as e:
        print(f"⚠️  Error processing coordinates: {e}")
        return
    
    if len(valid_stations) == 0:
        print("⚠️  No valid station coordinates found for heatmap")
        return
    
    # Create figure with azimuthal equidistant projection centered on London
    fig = plt.figure(figsize=(14, 14))
    proj = ccrs.AzimuthalEquidistant(
        central_latitude=LONDON_LAT,
        central_longitude=LONDON_LNG
    )
    ax = plt.axes(projection=proj)
    
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, 
                   linewidth=MAP_COLORS['coastline_width'],
                   edgecolor=MAP_COLORS['coastline_color'])
    ax.add_feature(cfeature.BORDERS, 
                   linewidth=MAP_COLORS['borders_width'], 
                   alpha=MAP_COLORS['borders_alpha'],
                   edgecolor=MAP_COLORS['borders_color'])
    ax.add_feature(cfeature.LAND, 
                   alpha=MAP_COLORS['land_alpha'], 
                   color=MAP_COLORS['land_color'])
    ax.add_feature(cfeature.OCEAN, 
                   alpha=MAP_COLORS['ocean_alpha'], 
                   color=MAP_COLORS['ocean_color'])
    
    # Prepare data for heatmap
    lons = valid_stations['lon'].values
    lats = valid_stations['lat'].values
    
    # Create 2D histogram for heatmap
    # Use a grid that covers a wide area (20,000 km radius covers most of the globe)
    # Since the projection is centered on London, use global bounds for binning
    lon_bins = np.linspace(-180, 180, 120)
    lat_bins = np.linspace(-90, 90, 90)
    
    # Count observations in each bin
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    
    # Create meshgrid for plotting
    lon_centers = (xedges[:-1] + xedges[1:]) / 2
    lat_centers = (yedges[:-1] + yedges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Transpose heatmap to match meshgrid orientation
    heatmap = heatmap.T
    
    # Plot heatmap
    # Use log scale for better visualization of distribution
    heatmap_log = np.log1p(heatmap)  # log(1+x) to handle zeros
    
    im = ax.contourf(
        lon_mesh, lat_mesh, heatmap_log,
        levels=MAP_COLORS['heatmap_levels'],
        transform=ccrs.PlateCarree(),
        cmap=MAP_COLORS['heatmap_colormap'],
        alpha=MAP_COLORS['heatmap_alpha']
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Log of Observation Count', fontsize=10)
    
    # Mark London as the center
    ax.plot(LONDON_LNG, LONDON_LAT, 'o', 
            color=MAP_COLORS['center_marker_color'], 
            markersize=MAP_COLORS['center_marker_size'],
            transform=ccrs.PlateCarree(), label='London (center)')
    
    # Add distance circles at 2k km increments
    # In azimuthal equidistant projection, circles are drawn in projection coordinates (meters)
    circle_increment_km = 2000
    for distance_km in range(circle_increment_km, MAP_RADIUS_KM + 1, circle_increment_km):
        circle_radius_m = distance_km * 1000
        circle = mpatches.Circle((0, 0), circle_radius_m, 
                                 transform=proj, 
                                 fill=False, 
                                 edgecolor=MAP_COLORS['circle_color'],
                                 linestyle=MAP_COLORS['circle_linestyle'],
                                 alpha=MAP_COLORS['circle_alpha'],
                                 linewidth=MAP_COLORS['circle_linewidth'])
        ax.add_patch(circle)
        
        # Add distance label on the right side of the circle
        # Place label at 90 degrees (top) from center in projection coordinates
        label_x = 0
        label_y = circle_radius_m
        ax.text(label_x, label_y, f'{distance_km:,} km',
                transform=proj,
                ha='center', va='bottom',
                fontsize=8, color='gray', alpha=0.7)
    
    # Set the map extent in projection coordinates (meters from center)
    # Do this last to ensure circular aspect ratio is maintained
    ax.set_xlim(-MAP_RADIUS_METERS, MAP_RADIUS_METERS)
    ax.set_ylim(-MAP_RADIUS_METERS, MAP_RADIUS_METERS)
    
    plt.title(f'Station Location Heatmap\n(Azimuthal Equidistant Projection, Centered on London, {MAP_RADIUS_KM:,} km radius)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'station_location_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: station_location_heatmap.png")
    print(f"   Plotted {len(valid_stations)} station observations")


def plot_distances_histogram(df, output_dir):
    """Plot histogram of distances."""
    # Calculate distances
    distances = calculate_distances(df)
    df_distances = pd.Series(distances)
    df_distances = df_distances.dropna()
    
    if len(df_distances) == 0:
        print("⚠️ No valid distances to plot (missing coordinates)")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(df_distances, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Distances from Observer to Station', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'distances_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: distances_histogram.png")
    print(f"   Distance statistics: mean={df_distances.mean():.1f} km, "
          f"median={df_distances.median():.1f} km, "
          f"max={df_distances.max():.1f} km")


def analyze_data(data_csv: str = "data.csv", results_dir: str = "results"):
    """Main analysis function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load data
    data_path = project_root / data_csv
    if not data_path.exists():
        print(f"❌ Error: {data_csv} not found at {data_path}")
        return
    
    print(f"📊 Loading data from {data_path}...")
    df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
    print(f"✅ Loaded {len(df)} rows")
    
    # Create results directory
    results_path = project_root / results_dir
    results_path.mkdir(exist_ok=True)
    # Note: plt.savefig() overwrites existing files by default
    print(f"📁 Results will be saved to: {results_path} (overwrites existing files)")
    print()
    
    # Generate visualizations
    plot_time_series(df, results_path)
    plot_top_stations_pie(df, results_path)
    plot_top_countries_pie(df, results_path)
    plot_tracing_success_rate(df, results_path)
    plot_station_heatmap(df, results_path)
    plot_distances_histogram(df, results_path)
    
    print()
    print("✅ Analysis complete! All figures saved to results/ directory")


if __name__ == "__main__":
    analyze_data()

