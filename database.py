import sqlite3
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'srivilliputhur.db')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        full_name TEXT NOT NULL
    )
    ''')

    # 2. Wards table (33 wards)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS wards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ward_no INTEGER NOT NULL,
        ward_name TEXT NOT NULL,
        population_2011 INTEGER,
        lgd_code INTEGER,
        data_source_type TEXT NOT NULL DEFAULT 'VERIFIED_PUBLIC'
    )
    ''')

    # 3. Municipality stats
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS municipality_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter TEXT NOT NULL,
        value TEXT NOT NULL,
        year TEXT,
        source TEXT NOT NULL,
        data_source_type TEXT NOT NULL DEFAULT 'HISTORICAL_PUBLIC'
    )
    ''')

    # 4. Verified Drainage records
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS drains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        drain_id TEXT UNIQUE NOT NULL,
        ward_no INTEGER,
        street_name TEXT NOT NULL,
        drain_length_m REAL,
        drain_width_m REAL,
        drain_depth_m REAL,
        drain_type TEXT,
        culvert_count TEXT,
        source TEXT,
        source_url TEXT,
        data_source_type TEXT NOT NULL DEFAULT 'VERIFIED_PUBLIC'
    )
    ''')

    # 5. Synthetic ML Dataset (50 rows)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        drain_id TEXT UNIQUE NOT NULL,
        ward_no INTEGER,
        street_name TEXT,
        latitude REAL,
        longitude REAL,
        rainfall_mm_7day REAL,
        rainfall_mm_30day REAL,
        plastic_accumulation_level TEXT,
        plastic_accumulation_score INTEGER,
        drain_width_m REAL,
        drain_depth_m REAL,
        drain_type TEXT,
        days_since_cleaning INTEGER,
        previous_overflow_count_1yr INTEGER,
        water_level_cm REAL,
        blockage_percent REAL,
        overflow_risk_label TEXT,
        data_source_type TEXT NOT NULL DEFAULT 'SYNTHETIC',
        notes TEXT
    )
    ''')

    # 6. Historical Incidents (12 events)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id TEXT UNIQUE NOT NULL,
        date TEXT NOT NULL,
        location_street_area TEXT NOT NULL,
        ward_no TEXT,
        incident_type TEXT NOT NULL,
        rainfall_mm TEXT,
        reported_cause TEXT,
        impact TEXT,
        source TEXT,
        source_url TEXT,
        data_source_type TEXT NOT NULL DEFAULT 'HISTORICAL_PUBLIC'
    )
    ''')

    # 7. Rainfall Portals
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rainfall_portals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portal_name TEXT NOT NULL,
        data_type TEXT NOT NULL,
        coverage TEXT,
        format TEXT,
        source_url TEXT,
        data_source_type TEXT NOT NULL DEFAULT 'VERIFIED_PUBLIC'
    )
    ''')

    # 8. Plastic Info
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plastic_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT NOT NULL,
        ward_no TEXT,
        waste_plastic_issue TEXT NOT NULL,
        evidence_description TEXT,
        date_year TEXT,
        source TEXT,
        data_source_type TEXT NOT NULL
    )
    ''')

    # 9. Field Surveys (25 planned points)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS field_surveys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        survey_point_id TEXT UNIQUE NOT NULL,
        drain_id TEXT,
        survey_date TEXT,
        survey_time_ist TEXT,
        surveyor_id TEXT,
        ward_no INTEGER,
        street_or_location TEXT NOT NULL,
        landmark TEXT,
        latitude_wgs84 REAL,
        longitude_wgs84 REAL,
        gps_accuracy_m REAL,
        gps_method TEXT DEFAULT 'Phone GPS',
        drain_present TEXT,
        drain_type TEXT,
        drain_cover_status TEXT,
        drain_shape TEXT,
        drain_length_observed_m REAL,
        drain_width_m REAL,
        drain_depth_m REAL,
        water_depth_cm REAL,
        flow_observed TEXT,
        plastic_level TEXT,
        plastic_description TEXT,
        other_debris_level TEXT,
        sediment_depth_cm REAL,
        blockage_percent REAL,
        blockage_method TEXT,
        overflow_observed TEXT,
        overflow_evidence TEXT,
        overflow_severity TEXT,
        last_cleaning_date TEXT,
        cleaning_info_source TEXT,
        days_since_cleaning INTEGER,
        weather_at_visit TEXT,
        rainfall_recent_notes TEXT,
        notes TEXT,
        record_status TEXT NOT NULL DEFAULT 'PLANNED',
        data_quality_check TEXT
    )
    ''')

    # 10. Citizen Reports
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS citizen_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id TEXT UNIQUE NOT NULL,
        name TEXT,
        phone TEXT,
        email TEXT,
        ward_no INTEGER,
        street_location TEXT NOT NULL,
        gps_lat REAL,
        gps_lng REAL,
        issue_type TEXT NOT NULL,
        description TEXT NOT NULL,
        photo_filename TEXT,
        status TEXT NOT NULL DEFAULT 'Submitted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 11. Maintenance Records
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS maintenance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id TEXT UNIQUE NOT NULL,
        drain_id TEXT,
        ward_no INTEGER,
        street TEXT NOT NULL,
        issue_description TEXT NOT NULL,
        cleaning_type TEXT,
        cleaning_date TEXT,
        waste_removed_kg REAL,
        assigned_team TEXT,
        status TEXT NOT NULL DEFAULT 'Pending',
        completion_date TEXT,
        photo_evidence TEXT,
        remarks TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Seed Default Users
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES ('admin', 'pbkdf2:sha256:admin123', 'admin', 'Municipal Engineering Admin')")
        cursor.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES ('inspector', 'pbkdf2:sha256:staff123', 'staff', 'Drainage Field Inspector')")
        cursor.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES ('citizen', 'pbkdf2:sha256:citizen123', 'citizen', 'Srivilliputhur Resident')")

    conn.commit()
    conn.close()
    print("Database schema initialized.")

def seed_data():
    conn = get_db()
    cursor = conn.cursor()

    layer1_path = os.path.join(RAW_DIR, 'Srivilliputhur_Complete_Dataset_Layer1_Layer2.xlsx')
    survey_path = os.path.join(RAW_DIR, 'Srivilliputhur_Drainage_Field_Survey_Template (1).xlsx')
    synthetic_path = os.path.join(RAW_DIR, 'Srivilliputhur_Synthetic_ML_Dataset.xlsx')

    # 1. Seed Wards (1D_Ward_Data)
    cursor.execute("DELETE FROM wards")
    df_wards = pd.read_excel(layer1_path, sheet_name='1D_Ward_Data')
    for _, r in df_wards.iterrows():
        cursor.execute('''
        INSERT INTO wards (ward_no, ward_name, population_2011, lgd_code, data_source_type)
        VALUES (?, ?, ?, ?, ?)
        ''', (int(r['Ward_No']), str(r['Ward_Name']), int(r['Population_2011']), int(r['LGD_Code']), str(r['Data_Source_Type'])))

    # 2. Seed Municipality Stats (1C_Municipality_Stats)
    cursor.execute("DELETE FROM municipality_stats")
    df_stats = pd.read_excel(layer1_path, sheet_name='1C_Municipality_Stats')
    for _, r in df_stats.iterrows():
        cursor.execute('''
        INSERT INTO municipality_stats (parameter, value, year, source, data_source_type)
        VALUES (?, ?, ?, ?, ?)
        ''', (str(r['Parameter']), str(r['Value']), str(r['Year']) if pd.notna(r['Year']) else None, str(r['Source']), str(r['Data_Source_Type'])))

    # 3. Seed Verified Drainage (1B_Verified_Drainage)
    cursor.execute("DELETE FROM drains")
    df_drains = pd.read_excel(layer1_path, sheet_name='1B_Verified_Drainage')
    for _, r in df_drains.iterrows():
        cursor.execute('''
        INSERT INTO drains (drain_id, ward_no, street_name, drain_length_m, drain_width_m, drain_depth_m, drain_type, culvert_count, source, source_url, data_source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(r['Drain_ID']), int(r['Ward_No']) if pd.notna(r['Ward_No']) else None, str(r['Street_Name']),
              float(r['Drain_Length_m']) if pd.notna(r['Drain_Length_m']) else None,
              float(r['Drain_Width_m']) if pd.notna(r['Drain_Width_m']) else None,
              float(r['Drain_Depth_m']) if pd.notna(r['Drain_Depth_m']) else None,
              str(r['Drain_Type']) if pd.notna(r['Drain_Type']) else None,
              str(r['Culvert_Count']) if pd.notna(r['Culvert_Count']) else None,
              str(r['Source']) if pd.notna(r['Source']) else None,
              str(r['Source_URL']) if pd.notna(r['Source_URL']) else None,
              str(r['Data_Source_Type'])))

    # 4. Seed Synthetic ML Dataset
    cursor.execute("DELETE FROM ml_dataset")
    df_syn = pd.read_excel(synthetic_path, sheet_name='Synthetic_Drains')
    for _, r in df_syn.iterrows():
        cursor.execute('''
        INSERT INTO ml_dataset (drain_id, ward_no, street_name, latitude, longitude,
                                rainfall_mm_7day, rainfall_mm_30day, plastic_accumulation_level,
                                plastic_accumulation_score, drain_width_m, drain_depth_m,
                                drain_type, days_since_cleaning, previous_overflow_count_1yr,
                                water_level_cm, blockage_percent, overflow_risk_label,
                                data_source_type, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(r['Drain_ID']), int(r['Ward_No']), str(r['Street_Name']), float(r['Latitude']), float(r['Longitude']),
              float(r['Rainfall_mm_7day']), float(r['Rainfall_mm_30day']), str(r['Plastic_Accumulation_Level']),
              int(r['Plastic_Accumulation_Score']), float(r['Drain_Width_m']), float(r['Drain_Depth_m']),
              str(r['Drain_Type']), int(r['Days_Since_Cleaning']), int(r['Previous_Overflow_Count_1yr']),
              float(r['Water_Level_cm']), float(r['Blockage_Percent']), str(r['Overflow_Risk_Label']),
              str(r['Data_Source_Type']), str(r['Notes'])))

    # 5. Seed Historical Incidents (1A_Historical_Incidents)
    cursor.execute("DELETE FROM incidents")
    df_inc = pd.read_excel(layer1_path, sheet_name='1A_Historical_Incidents')
    for _, r in df_inc.iterrows():
        cursor.execute('''
        INSERT INTO incidents (incident_id, date, location_street_area, ward_no, incident_type,
                               rainfall_mm, reported_cause, impact, source, source_url, data_source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(r['Incident_ID']), str(r['Date']), str(r['Location_Street_Area']), str(r['Ward_No']),
              str(r['Incident_Type']), str(r['Rainfall_mm']), str(r['Reported_Cause']), str(r['Impact']),
              str(r['Source']), str(r['Source_URL']), str(r['Data_Source_Type'])))

    # 6. Seed Rainfall Portals (1E_Rainfall_Portals)
    cursor.execute("DELETE FROM rainfall_portals")
    df_rain = pd.read_excel(layer1_path, sheet_name='1E_Rainfall_Portals')
    for _, r in df_rain.iterrows():
        cursor.execute('''
        INSERT INTO rainfall_portals (portal_name, data_type, coverage, format, source_url, data_source_type)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (str(r['Portal_Name']), str(r['Data_Type']), str(r['Coverage']), str(r['Format']),
              str(r['Source_URL']), str(r['Data_Source_Type'])))

    # 7. Seed Plastic Info (1F_Plastic_Info)
    cursor.execute("DELETE FROM plastic_info")
    df_plas = pd.read_excel(layer1_path, sheet_name='1F_Plastic_Info')
    for _, r in df_plas.iterrows():
        cursor.execute('''
        INSERT INTO plastic_info (location, ward_no, waste_plastic_issue, evidence_description, date_year, source, data_source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(r['Location']), str(r['Ward_No']), str(r['Waste_Plastic_Issue']), str(r['Evidence_Description']),
              str(r['Date_Year']), str(r['Source']), str(r['Data_Source_Type'])))

    # 8. Seed Field Surveys (25 planned survey points)
    cursor.execute("DELETE FROM field_surveys")
    df_survey = pd.read_excel(survey_path, sheet_name='Drain_Survey')
    for _, r in df_survey.iterrows():
        point_id = str(r['Survey_Point_ID'])
        status = str(r['Record_Status']) if pd.notna(r['Record_Status']) else 'PLANNED'
        street = str(r['Street_or_Location']) if pd.notna(r['Street_or_Location']) else 'Planned Survey Location'
        cursor.execute('''
        INSERT INTO field_surveys (survey_point_id, drain_id, ward_no, street_or_location, landmark,
                                   latitude_wgs84, longitude_wgs84, gps_accuracy_m, gps_method,
                                   drain_type, plastic_level, blockage_percent, record_status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (point_id,
              str(r['Drain_ID']) if pd.notna(r['Drain_ID']) else None,
              int(r['Ward_No']) if pd.notna(r['Ward_No']) else None,
              street,
              str(r['Landmark']) if pd.notna(r['Landmark']) else None,
              float(r['Latitude_WGS84']) if pd.notna(r['Latitude_WGS84']) else None,
              float(r['Longitude_WGS84']) if pd.notna(r['Longitude_WGS84']) else None,
              float(r['GPS_Accuracy_m']) if pd.notna(r['GPS_Accuracy_m']) else None,
              str(r['GPS_Method']) if pd.notna(r['GPS_Method']) else 'Phone GPS',
              str(r['Drain_Type']) if pd.notna(r['Drain_Type']) else None,
              str(r['Plastic_Level']) if pd.notna(r['Plastic_Level']) else None,
              float(r['Blockage_Percent']) if pd.notna(r['Blockage_Percent']) else None,
              status,
              str(r['Notes']) if pd.notna(r['Notes']) else 'Planned field observation point awaiting physical survey visit.'))

    # Seed Sample Initial Citizen Reports & Maintenance (demonstrative workflow)
    cursor.execute("DELETE FROM citizen_reports")
    cursor.execute('''
    INSERT INTO citizen_reports (complaint_id, name, phone, email, ward_no, street_location, gps_lat, gps_lng, issue_type, description, status)
    VALUES ('SVP-REP-2026-0001', 'K. Murugan', '9876543210', 'kmurugan@example.com', 8, 'Temple Street, near car stand', 9.5165, 77.6241, 'Drain Overflow', 'Plastic cups and bags blocking open channel; greywater spilling onto pedestrian walkway.', 'In Progress'),
           ('SVP-REP-2026-0002', 'S. Lakshmi', '9443123456', 'slakshmi@example.com', 19, 'Netaji Road, near bazaar', 9.5080, 77.6320, 'Plastic Waste', 'Accumulation of single-use carry bags in roadside drain culvert.', 'Assigned'),
           ('SVP-REP-2026-0003', 'M. Ramanathan', '9842109876', 'ramanathan@example.com', 32, 'Bharathi Nagar 3rd Street', 9.5050, 77.6350, 'Blocked Drain', 'Silt and leaf litter restricting inlet culvert flow.', 'Submitted')
    ''')

    cursor.execute("DELETE FROM maintenance_records")
    cursor.execute('''
    INSERT INTO maintenance_records (ticket_id, drain_id, ward_no, street, issue_description, cleaning_type, cleaning_date, waste_removed_kg, assigned_team, status, remarks)
    VALUES ('SVP-MAINT-101', 'SVP-SYN-D001', 8, 'Temple Street', 'Severe plastic and silt accumulation', 'Manual De-silting', '2026-09-10', 45.0, 'Sanitation Team B', 'In Progress', 'Scheduled for secondary jetting'),
           ('SVP-MAINT-102', 'SVP-W32-D001', 32, 'Bharathi Nagar 3rd Street Road', 'Culvert inlet inspection and clearance', 'Culvert Jetting', '2026-08-28', 22.5, 'Sanitation Team A', 'Completed', 'Flow restored through chainage 73m culvert')
    ''')

    conn.commit()
    conn.close()
    print("Database seeding completed successfully.")

if __name__ == '__main__':
    init_db()
    seed_data()
