import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import argparse
from tqdm import tqdm
import logging
from tags_config import process_instruments, equipment_tags, calculated_tags

# Setup logging
logging.basicConfig(filename='pi_data_generator.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_comprehensive_pi_data(start_date=None, end_date=None, frequency='1min', max_tags=None, key_tags_only=False):
    """
    Generate COMPREHENSIVE and REALISTIC PI Data Export for Crude Oil Separation Train
    
    Parameters:
    - start_date: Start date (default: 2025-07-01)
    - end_date: End date (default: 2025-07-03)
    - frequency: Data frequency - '1min', '5min', '1h', '1D' (default: '1min')
    - max_tags: Maximum number of tags to include (default: all)
    - key_tags_only: If True, generate only 8 key tags from mock data (default: False)
    """
    
    logging.info("Starting PI data generation")
    
    # Set default date range
    if start_date is None:
        start_date = datetime(2025, 7, 1, 0, 0, 0)
    if end_date is None:
        end_date = datetime(2025, 7, 3, 0, 0, 0)
    
    # Validation
    if end_date <= start_date:
        logging.error("End date must be after start date")
        raise ValueError("End date must be after start date")
    valid_freq = ['1min', '5min', '1h', '1D']
    if frequency not in valid_freq:
        logging.error(f"Invalid frequency: {frequency}. Must be one of {valid_freq}")
        raise ValueError(f"Invalid frequency. Use one of {valid_freq}")
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    print("⏰ Configuration:")
    print(f"   • Start Date: {start_date}")
    print(f"   • End Date: {end_date}")
    print(f"   • Frequency: {frequency}")
    print(f"   • Total Timestamps: {len(timestamps):,}")
    
    # Import tags from external config
    all_tags = {**process_instruments, **equipment_tags, **calculated_tags}
    
    # Limit to key tags if specified
    key_tags = ['TT-201', 'PT-201', 'TT-202', 'TT-203', 'FT-201', 'AT-201', 'P-101A_VIB', 'P-101B_VIB']
    if key_tags_only:
        all_tags = {k: v for k, v in all_tags.items() if k in key_tags}
    
    if max_tags and max_tags < len(all_tags):
        critical_tags = {k: v for k, v in all_tags.items() if v.get('critical', False)}
        other_tags = {k: v for k, v in all_tags.items() if not v.get('critical', False)}
        remaining_slots = max_tags - len(critical_tags)
        if remaining_slots > 0:
            other_tags_limited = dict(list(other_tags.items())[:remaining_slots])
            all_tags = {**critical_tags, **other_tags_limited}
        else:
            all_tags = dict(list(critical_tags.items())[:max_tags])
    
    print("\n🏷️  TAG CONFIGURATION:")
    print(f"   • Process Instruments: {len(process_instruments)} tags")
    print(f"   • Equipment Status: {len(equipment_tags)} tags") 
    print(f"   • Calculated/Derived: {len(calculated_tags)} tags")
    print(f"   • TOTAL PI TAGS: {len(all_tags)} tags")
    
    estimated_records = len(all_tags) * len(timestamps)
    estimated_size_mb = estimated_records * 100 / 1024 / 1024
    
    print("\n📊 DATASET SIZE ESTIMATE:")
    print(f"   • Total Records: {estimated_records:,}")
    print(f"   • Estimated Size: ~{estimated_size_mb:.1f} MB")
    
    if estimated_records > 1000000:
        print("⚠️  Large dataset detected!")
        print("   Consider using smaller time range or higher frequency interval")
        print("   Or use max_tags parameter to limit tag count")
    
    # Create process events relative to the actual date range
    total_duration = end_date - start_date
    
    process_events = [
        {'start': start_date + timedelta(hours=14), 'end': start_date + timedelta(hours=18), 'type': 'upset', 'desc': 'Feed quality deviation - High sulfur crude', 'impact': 'feed_quality'},
        {'start': start_date + timedelta(days=1, hours=8), 'end': start_date + timedelta(days=1, hours=12), 'type': 'maintenance', 'desc': 'Pump P-101A maintenance', 'impact': 'pump_a_down'},
        {'start': start_date + timedelta(days=2, hours=9), 'end': start_date + timedelta(days=2, hours=15), 'type': 'upset', 'desc': 'Cooling water temperature spike', 'impact': 'cooling_issue'},
        {'start': start_date + timedelta(days=3, hours=10), 'end': start_date + timedelta(days=3, hours=14), 'type': 'upset', 'desc': 'Power fluctuation', 'impact': 'control_issues'},
    ]
    
    # Filter events to only include those within the date range
    process_events = [event for event in process_events if event['start'] <= end_date]
    
    # Initialize state variables
    all_data = []
    flow_base = {}
    prev_flow_base = {}
    controller_values = {}
    upset_state = {tag: {'active': False, 'duration': 0, 'multiplier': 1.0} for tag in all_tags}
    range_drift = {tag: 1.0 for tag in all_tags}
    volatility_counter = 0
    volatility_active = False
    
    for timestamp in tqdm(timestamps, desc="Generating PI data"):
        hour = timestamp.hour
        total_days = (end_date - start_date).days + 1
        days_since_start = (timestamp - start_date).days + (hour / 24.0)
        
        daily_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (hour - 6) / 24)
        if total_days > 7:
            seasonal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * days_since_start / total_days)
        else:
            seasonal_factor = 1.0 + 0.02 * days_since_start
    
        event_factors = {'base': 1.0, 'noise_mult': 1.0}
        for event in process_events:
            if event['start'] <= timestamp <= event['end']:
                if event['impact'] == 'feed_quality':
                    event_factors['analyzer'] = 0.85
                    event_factors['temperature'] = 1.05
                elif event['impact'] == 'pump_a_down':
                    event_factors['pump_a'] = 0.0
                elif event['impact'] == 'cooling_issue':
                    event_factors['cooling'] = 1.2
                elif event['impact'] == 'control_issues':
                    event_factors['control'] = 1.1
                    event_factors['noise_mult'] = 2.0
                event_factors['bad_prob_mult'] = 4.0
        
        bad_prob = 0.02 * event_factors.get('bad_prob_mult', 1.0)
        pump_a_down = 'pump_a' in event_factors and event_factors['pump_a'] == 0.0
        if pump_a_down:
            event_factors['pump_b_load'] = 1.2
        
        # Volatility phase
        volatility_counter += 1
        if volatility_counter > random.randint(50, 100):
            volatility_active = True
            volatility_duration = random.randint(5, 15)
            volatility_counter = 0
        if volatility_active:
            global_noise_boost = random.uniform(1.5, 3.0)
            volatility_duration -= 1
            if volatility_duration <= 0:
                volatility_active = False
        else:
            global_noise_boost = 1.0
        
        # Tag loop
        for tag, config in all_tags.items():
            if config.get('type') == 'digital':
                value = 1
                if 'P-101A' in tag and pump_a_down:
                    value = 0
                elif 'P-101B' in tag and pump_a_down:
                    value = 1
                elif 'control_issues' in event_factors and random.random() < 0.1:
                    value = random.randint(0, 1)
                status = 'Good' if random.random() > bad_prob else 'Bad'
                units = 'status'
            else:
                min_val, max_val = config['range']
                mean_val = (min_val + max_val) / 2
                range_drift[tag] += np.random.normal(0, 0.01)
                range_drift[tag] = max(0.7, min(1.3, range_drift[tag]))
                base_deviation = (max_val - min_val) / 2 * range_drift[tag]
                base_value = mean_val + np.random.uniform(-base_deviation, base_deviation) * daily_factor * seasonal_factor
                
                if 'trend' in config:
                    base_value *= config['trend'] ** days_since_start
                
                # Apply event factors
                if 'analyzer' in event_factors and 'AT-' in tag:
                    base_value *= event_factors['analyzer']
                if 'temperature' in event_factors and 'TT-' in tag:
                    base_value *= event_factors['temperature']
                if 'cooling' in event_factors and ('condenser' in config['desc'].lower() or 'cool' in config['desc'].lower()):
                    base_value *= event_factors['cooling']
                if 'control' in event_factors and ('C-' in tag):
                    base_value *= event_factors['control']
                if 'pump_b_load' in event_factors and 'P-101B' in tag:
                    base_value *= event_factors['pump_b_load']
                
                if 'FT-201' in prev_flow_base and 'TT-' in tag:
                    prev_flow = prev_flow_base['FT-201']
                    base_value *= (1 + 0.05 * (prev_flow - (180 + 250) / 2) / ((180 + 250) / 2))
                
                # Sudden drops/rises
                if upset_state[tag]['active']:
                    base_value *= upset_state[tag]['multiplier']
                    upset_state[tag]['duration'] -= 1
                    if upset_state[tag]['duration'] <= 0:
                        upset_state[tag]['active'] = False
                elif random.random() < 0.02:
                    upset_state[tag]['active'] = True
                    upset_state[tag]['duration'] = random.randint(1, 3)
                    upset_state[tag]['multiplier'] = random.uniform(0.7, 0.9) if random.random() < 0.5 else random.uniform(1.1, 1.3)
                
                # Noise with heavy tails
                use_heavy_tails = True
                effective_noise_std = config['noise'] * event_factors['noise_mult'] * global_noise_boost
                noise = np.random.standard_t(df=3) * effective_noise_std if use_heavy_tails else np.random.normal(0, effective_noise_std)
                value = base_value + noise
                value = max(0, min(max_val * 1.4, max(min_val * 0.6, value)))
                value = round(value, 2)
                
                status = 'Good' if random.random() > bad_prob else 'Bad'
                units = config.get('units', 'N/A')
            
            if '_POS' in tag and 'related_controller' in config:
                controller_tag = config['related_controller']
                if controller_tag in controller_values:
                    value = controller_values[controller_tag] + np.random.normal(0, config['noise'])
                    value = max(0, min(100, value))
            
            all_data.append({
                'Timestamp': timestamp,
                'Tag': tag,
                'Description': config['desc'],
                'Value': value,
                'Units': units,
                'Status': status
            })
        
        if 'FC-' in tag or 'LC-' in tag or 'PC-' in tag or 'TC-' in tag:
            controller_values[tag] = value
        if 'FT-' in tag:
            flow_base[tag] = value
    
        prev_flow_base = flow_base.copy()
    
    df_full = pd.DataFrame(all_data)
    print("\n✅ DATASET GENERATION COMPLETE:")
    print(f"   • Total Records: {len(df_full):,}")
    print(f"   • Unique Tags: {df_full['Tag'].nunique()}")
    
    # Comparisons
    comparisons = []
    lab_timestamps = pd.date_range(start=start_date, end=end_date, freq='12h')
    for ts in lab_timestamps:
        analyzer_val = df_full[(df_full['Timestamp'] == ts) & (df_full['Tag'] == 'AT-201')]['Value'].values
        if len(analyzer_val) > 0:
            analyzer_val = analyzer_val[0]
            lab_val = analyzer_val * random.uniform(0.95, 1.05)
            alignment = round(100 * (1 - abs(analyzer_val - lab_val) / lab_val), 1)
            status = 'On-Spec' if 85 <= analyzer_val <= 95 else 'Off-Spec'
            comparisons.append({
                'Timestamp': ts,
                'Parameter': 'Light Ends (mol%)',
                'Analyzer_Value': round(analyzer_val, 2),
                'Lab_Value': round(lab_val, 2),
                'Alignment_%': alignment,
                'Spec_Range': '85-95',
                'Status': status
            })
            rvp_val = random.uniform(8, 12)
            rvp_analyzer = rvp_val * random.uniform(0.95, 1.05)
            alignment = round(100 * (1 - abs(rvp_analyzer - rvp_val) / rvp_val), 1)
            status = 'On-Spec' if 8 <= rvp_analyzer <= 12 else 'Off-Spec'
            comparisons.append({
                'Timestamp': ts,
                'Parameter': 'Naphtha RVP (psi)',
                'Analyzer_Value': round(rvp_analyzer, 2),
                'Lab_Value': round(rvp_val, 2),
                'Alignment_%': alignment,
                'Spec_Range': '8-12',
                'Status': status
            })
    df_comparisons = pd.DataFrame(comparisons)
    
    # Trends
    df_full['Date'] = df_full['Timestamp'].dt.date
    df_trends = df_full.groupby(['Date', 'Tag'])['Value'].mean().reset_index()
    df_trends = df_trends.pivot(index='Date', columns='Tag', values='Value')
    df_trends_pct = df_trends.pct_change() * 100
    df_trends_pct = df_trends_pct.round(2)
    df_trends_pct.columns = [col + '_pct_change' for col in df_trends_pct.columns]
    df_trends = pd.concat([df_trends, df_trends_pct], axis=1).reset_index()
    
    # Tag List
    tag_list = pd.DataFrame([{'Tag': k, **v} for k, v in all_tags.items()])
    
    # Events
    df_events = pd.DataFrame([{
        'Start_Date': event['start'].strftime('%Y-%m-%d %H:%M'),
        'End_Date': event['end'].strftime('%Y-%m-%d %H:%M'),
        'Event_Type': event['type'],
        'Description': event['desc'],
        'Impact': event['impact'],
        'Duration_Hours': int((event['end'] - event['start']).total_seconds() / 3600)
    } for event in process_events])
    
    # Statistics
    df_stats = df_full.groupby(['Tag', 'Description', 'Units'])['Value'].agg(['count', 'mean', 'std', 'min', 'max']).round(2).reset_index()
    df_stats['Good_%'] = df_full.groupby('Tag')['Status'].apply(lambda x: (x == 'Good').sum() / len(x) * 100).round(1).values
    
    # Data Status
    df_status = df_full.pivot_table(index='Timestamp', columns='Tag', values='Status', aggfunc='first').reset_index()
    
    # Documentation
    doc_content = f"""# COMPREHENSIVE PI DATA EXPORT DOCUMENTATION
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Plant: Crude Oil Separation Train
Time Range: {start_date} to {end_date}
Frequency: {frequency}

## DATASET OVERVIEW
- Total Records: {len(df_full):,}
- Total PI Tags: {len(all_tags)}
- Process Instruments: {len(process_instruments)}
- Equipment Status: {len(equipment_tags)}
- Calculated Tags: {len(calculated_tags)}
- Process Events: {len(process_events)}
- Data Quality: {(df_full['Status'] == 'Good').sum() / len(df_full) * 100:.1f}% Good

## OUTPUT STRUCTURE
- **Excel File**: PI_Export_Crude_Separation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{frequency}_{len(all_tags)}tags.xlsx
- **Tabs**:
  - **Raw Data**: Time-series values for each tag (pivot format, Timestamp vs. Tag).
  - **Trends**: Daily averages and % changes for each tag, useful for tracking changes.
  - **Comparisons**: Analyzer vs. lab samples (every 12 hours) for Light Ends (mol%) and Naphtha RVP (psi), with alignment % and On/Off-Spec status.
  - **Tag List**: Full list of tags with descriptions, units, ranges, and criticality.
  - **Process Events**: Events affecting data (e.g., upsets, maintenance) with start/end times and impacts.
  - **Statistics Summary**: Per-tag stats (count, mean, std, min, max, good data %) for data quality analysis.
  - **Data Status**: Pivot of status (Good/Bad) per tag and timestamp.

## FREQUENCY EXPLANATION
{frequency} = {'1 minute intervals' if frequency == '1min' else 
             '5 minute intervals' if frequency == '5min' else
             'Hourly intervals' if frequency == '1h' else
             'Daily intervals' if frequency == '1D' else 'Unknown frequency'}

## TAG CATEGORIES

### PROCESS INSTRUMENTS ({len(process_instruments)} tags)
"""
    for tag, config in process_instruments.items():
        range_str = f"{config.get('range', ('N/A', 'N/A'))[0]}-{config.get('range', ('N/A', 'N/A'))[1]} {config.get('units', '')}"
        doc_content += f"{tag}: {config['desc']} ({range_str})\n"
    
    doc_content += f"\n### EQUIPMENT STATUS ({len(equipment_tags)} tags)\n"
    for tag, config in equipment_tags.items():
        if config.get('type') == 'digital':
            doc_content += f"{tag}: {config['desc']} (Digital)\n"
        else:
            range_str = f"{config.get('range', ('N/A', 'N/A'))[0]}-{config.get('range', ('N/A', 'N/A'))[1]} {config.get('units', '')}"
            doc_content += f"{tag}: {config['desc']} ({range_str})\n"
    
    doc_content += f"\n### CALCULATED TAGS ({len(calculated_tags)} tags)\n"
    for tag, config in calculated_tags.items():
        range_str = f"{config.get('range', ('N/A', 'N/A'))[0]}-{config.get('range', ('N/A', 'N/A'))[1]} {config.get('units', '')}"
        doc_content += f"{tag}: {config['desc']} ({range_str})\n"
    
    doc_content += f"\n## PROCESS EVENTS ({len(process_events)} events)\n"
    for event in process_events:
        duration = int((event['end'] - event['start']).total_seconds() / 3600)
        doc_content += f"{event['start'].strftime('%Y-%m-%d %H:%M')} to {event['end'].strftime('%Y-%m-%d %H:%M')} ({duration}h): {event['desc']} ({event['type']}, Impact: {event['impact']})\n"
    
    doc_content += """
## USAGE NOTES
- **What's changed since yesterday?**: Use Trends tab for daily % changes (e.g., TT-201 temp increase ~2%/day).
- **How did things run over the weekend?**: Raw Data covers Jul 31-Aug 3; Trends summarizes daily averages.
- **How are things running today?**: Filter Raw Data for Aug 3 or check Trends for latest day.
- **Are we on specification?**: Comparisons tab shows On/Off-Spec for Light Ends (AT-201) and Naphtha RVP.
- **Have there been changes in vibration readings?**: Check P-101A_VIB, P-101B_VIB in Raw Data/Trends (e.g., ~2.2 to ~3.8 mm/s).
- **Analyzer vs. lab alignment?**: Comparisons tab shows alignment % (~99%) for AT-201 and RVP.
- **Data Quality**: Statistics Summary provides Good % per tag; Data Status shows Good/Bad over time.
"""
    
    # Define file_name and date_str before saving
    date_str = start_date.strftime('%Y%m%d') + '_' + end_date.strftime('%Y%m%d')
    file_name = f"PI_Export_Crude_Separation_{date_str}_{frequency}_{len(all_tags)}tags.xlsx"
    
    # Save Excel
    try:
        import openpyxl
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            df_full.pivot_table(index='Timestamp', columns='Tag', values='Value', aggfunc='first').reset_index().to_excel(writer, sheet_name='Raw Data', index=False)
            df_trends.to_excel(writer, sheet_name='Trends', index=False)
            df_comparisons.to_excel(writer, sheet_name='Comparisons', index=False)
            tag_list.to_excel(writer, sheet_name='Tag List', index=False)
            df_events.to_excel(writer, sheet_name='Process Events', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics Summary', index=False)
            df_status.to_excel(writer, sheet_name='Data Status', index=False)
        logging.info(f"Saved Excel: {file_name}")
        print(f"✅ Saved Excel: {file_name}")
    except (ImportError, Exception) as e:
        print(f"⚠️ Excel save failed ({e}), saving CSV...")
        logging.error(f"Excel save failed: {e}")
        df_full.to_csv(f"PI_Export_Raw_{date_str}.csv", index=False)
        df_comparisons.to_csv(f"PI_Comparisons_{date_str}.csv", index=False)
        df_trends.to_csv(f"PI_Trends_{date_str}.csv", index=False)
        tag_list.to_csv(f"PI_Tag_List_{date_str}.csv", index=False)
        df_events.to_csv(f"PI_Events_{date_str}.csv", index=False)
        df_stats.to_csv(f"PI_Stats_{date_str}.csv", index=False)
        df_status.to_csv(f"PI_Status_{date_str}.csv", index=False)
        print(f"✅ Saved CSVs: PI_Export_Raw_{date_str}.csv, etc.")
        logging.info(f"Saved CSVs for all tabs")
    
    # Save Documentation
    doc_file = f"PI_Export_Documentation_{date_str}_{frequency}.txt"
    with open(doc_file, 'w') as f:
        f.write(doc_content)
    
    return df_full, df_trends, df_comparisons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PI data for crude separation')
    parser.add_argument('--start', default=None, help='Start date (YYYY-MM-DD), default: 2025-07-01')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD), default: 2025-07-03')
    parser.add_argument('--freq', default='1min', help='Data frequency (1min, 5min, 1h, 1D), default: 1min')
    parser.add_argument('--tags', type=int, default=None)
    parser.add_argument('--key-tags-only', action='store_true', help='Generate only 8 key tags')
    args = parser.parse_args()
    
    # Handle date parsing with proper defaults
    start_date = None
    end_date = None
    
    if args.start:
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
        except ValueError as e:
            print(f"❌ Invalid start date format: {e}")
            logging.error(f"Invalid start date format: {e}")
            exit(1)
    
    if args.end:
        try:
            end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(hour=23, minute=59)
        except ValueError as e:
            print(f"❌ Invalid end date format: {e}")
            logging.error(f"Invalid end date format: {e}")
            exit(1)
    
    # Explicitly define frequency with default
    frequency = args.freq if args.freq else '1min'
    
    np.random.seed(42)
    random.seed(42)
    
    try:
        df_full, df_trends, df_comparisons = generate_comprehensive_pi_data(
            start_date, end_date, frequency, args.tags, args.key_tags_only
        )
        logging.info("PI dataset generated successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Error in generation: {e}")
        print("Required packages: pandas, numpy, tqdm. Optional: openpyxl")
