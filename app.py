import os
import json
import sqlite3
import random
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'srivilliputhur.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure database exists on fresh deployment
if not os.path.exists(DB_PATH):
    try:
        import database
        database.init_db()
        database.seed_data()
    except Exception as e:
        print(f"Database auto-init notice: {e}")

app = Flask(__name__)
app.secret_key = 'srivilliputhur_academic_prototype_secret_key_2026'
CORS(app)

# Load ML Models and Metrics
def get_ml_assets():
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None, None, None, None, None
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    try:
        rf_causal = joblib.load(os.path.join(MODELS_DIR, 'rf_causal.joblib'))
        lr_causal = joblib.load(os.path.join(MODELS_DIR, 'logistic_causal.joblib'))
        rf_leakage = joblib.load(os.path.join(MODELS_DIR, 'rf_leakage.joblib'))
        lr_leakage = joblib.load(os.path.join(MODELS_DIR, 'logistic_leakage.joblib'))
    except Exception as e:
        print(f"Error loading models: {e}")
        rf_causal, lr_causal, rf_leakage, lr_leakage = None, None, None, None
    return metrics, rf_causal, lr_causal, rf_leakage, lr_leakage

METRICS, RF_CAUSAL, LR_CAUSAL, RF_LEAKAGE, LR_LEAKAGE = get_ml_assets()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Helper context for templates
@app.context_processor
def inject_global_context():
    return {
        'project_title': 'AI-Based Plastic Waste & Drainage Overflow Risk Prediction System',
        'municipality_name': 'Srivilliputhur Municipality',
        'state_name': 'Tamil Nadu',
        'prototype_disclaimer': 'Academic Demonstration Prototype — Built using public municipal records and a 50-row synthetic ML dataset. Not an official municipal deployment.',
        'current_year': '2026'
    }

# =========================================================================
# Web Page Routes
# =========================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM wards")
    ward_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM incidents")
    incident_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM drains")
    verified_drain_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM field_surveys")
    planned_surveys = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ml_dataset")
    synthetic_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ml_dataset WHERE overflow_risk_label = 'MEDIUM'")
    medium_risk_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ml_dataset WHERE overflow_risk_label = 'LOW'")
    low_risk_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM citizen_reports")
    citizen_reports_count = c.fetchone()[0]
    c.execute("SELECT * FROM municipality_stats")
    stats = [dict(row) for row in c.fetchall()]
    conn.close()

    return render_template('dashboard.html',
                           ward_count=ward_count,
                           incident_count=incident_count,
                           verified_drain_count=verified_drain_count,
                           planned_surveys=planned_surveys,
                           synthetic_count=synthetic_count,
                           medium_risk_count=medium_risk_count,
                           low_risk_count=low_risk_count,
                           citizen_reports_count=citizen_reports_count,
                           stats=stats)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html', metrics=METRICS)

@app.route('/map')
def risk_map():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM ml_dataset")
    synthetic_points = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM drains")
    verified_drains = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM incidents")
    incidents = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM field_surveys")
    surveys = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('map.html',
                           synthetic_points=synthetic_points,
                           verified_drains=verified_drains,
                           incidents=incidents,
                           surveys=surveys)

@app.route('/drains')
def drains():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM drains")
    verified_drains = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM ml_dataset")
    synthetic_drains = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('drains.html',
                           verified_drains=verified_drains,
                           synthetic_drains=synthetic_drains)

@app.route('/rainfall')
def rainfall():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM rainfall_portals")
    portals = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM incidents WHERE rainfall_mm != 'NOT_AVAILABLE' OR reported_cause LIKE '%rain%'")
    rain_incidents = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('rainfall.html', portals=portals, rain_incidents=rain_incidents)

@app.route('/incidents')
def incidents():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM incidents ORDER BY date DESC")
    incident_list = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('incidents.html', incidents=incident_list)

@app.route('/plastic-waste')
def plastic_waste():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM plastic_info")
    plastic_records = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('plastic_waste.html', plastic_records=plastic_records)

@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        conn = get_db()
        c = conn.cursor()
        complaint_id = f"SVP-REP-2026-{random.randint(1000, 9999)}"
        name = request.form.get('name', 'Anonymous Citizen')
        phone = request.form.get('phone', 'Not Provided')
        email = request.form.get('email', 'Not Provided')
        ward_no = request.form.get('ward_no')
        street_location = request.form.get('street_location')
        issue_type = request.form.get('issue_type')
        description = request.form.get('description')
        gps_lat = request.form.get('gps_lat') or None
        gps_lng = request.form.get('gps_lng') or None

        c.execute('''
        INSERT INTO citizen_reports (complaint_id, name, phone, email, ward_no, street_location, gps_lat, gps_lng, issue_type, description, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Submitted')
        ''', (complaint_id, name, phone, email, int(ward_no) if ward_no else None, street_location,
              float(gps_lat) if gps_lat else None, float(gps_lng) if gps_lng else None, issue_type, description))
        conn.commit()
        conn.close()
        flash(f"Complaint registered successfully! Your tracking ID is {complaint_id}", 'success')
        return redirect(url_for('report', tracked_id=complaint_id))

    tracked_id = request.args.get('tracked_id')
    tracked_report = None
    if tracked_id:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM citizen_reports WHERE complaint_id = ?", (tracked_id,))
        row = c.fetchone()
        if row:
            tracked_report = dict(row)
        conn.close()

    return render_template('report.html', tracked_report=tracked_report)

@app.route('/maintenance')
def maintenance():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM maintenance_records ORDER BY updated_at DESC")
    records = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('maintenance.html', records=records)

@app.route('/field-survey')
def field_survey():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM field_surveys")
    surveys = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('field_survey.html', surveys=surveys)

@app.route('/analytics')
def analytics():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT overflow_risk_label, COUNT(*) as cnt FROM ml_dataset GROUP BY overflow_risk_label")
    risk_dist = {row['overflow_risk_label']: row['cnt'] for row in c.fetchall()}
    c.execute("SELECT drain_type, COUNT(*) as cnt FROM ml_dataset GROUP BY drain_type")
    type_dist = {row['drain_type']: row['cnt'] for row in c.fetchall()}
    c.execute("SELECT plastic_accumulation_level, COUNT(*) as cnt FROM ml_dataset GROUP BY plastic_accumulation_level")
    plastic_dist = {row['plastic_accumulation_level']: row['cnt'] for row in c.fetchall()}
    conn.close()
    return render_template('analytics.html',
                           metrics=METRICS,
                           risk_dist=risk_dist,
                           type_dist=type_dist,
                           plastic_dist=plastic_dist)

@app.route('/sources')
def sources():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM rainfall_portals")
    portals = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('sources.html', portals=portals)

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/limitations')
def limitations():
    return render_template('limitations.html')

@app.route('/admin')
def admin():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM citizen_reports ORDER BY created_at DESC")
    reports = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM maintenance_records ORDER BY updated_at DESC")
    maint = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM field_surveys")
    surveys = [dict(row) for row in c.fetchall()]
    conn.close()
    return render_template('admin.html', reports=reports, maintenance=maint, surveys=surveys)

# =========================================================================
# JSON REST API Endpoints
# =========================================================================

@app.route('/api/stats')
def api_stats():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM wards")
    ward_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM incidents")
    incident_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM drains")
    verified_drain_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM field_surveys")
    planned_surveys = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ml_dataset")
    synthetic_count = c.fetchone()[0]
    conn.close()
    return jsonify({
        'total_wards': ward_count,
        'historical_incidents': incident_count,
        'verified_drainage_records': verified_drain_count,
        'planned_field_survey_points': planned_surveys,
        'synthetic_ml_records': synthetic_count,
        'status': 'success'
    })

@app.route('/api/drains')
def api_drains():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM drains")
    verified = [dict(row) for row in c.fetchall()]
    c.execute("SELECT * FROM ml_dataset")
    synthetic = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify({
        'verified_drains': verified,
        'synthetic_drains': synthetic,
        'disclaimer': 'Verified records derived from official municipal tenders. Synthetic records labeled as SYNTHETIC prototype data.'
    })

@app.route('/api/incidents')
def api_incidents():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM incidents")
    incidents = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(incidents)

@app.route('/api/field-surveys', methods=['GET', 'POST'])
def api_field_surveys():
    conn = get_db()
    c = conn.cursor()
    if request.method == 'POST':
        data = request.get_json() or {}
        point_id = data.get('survey_point_id')
        status = data.get('record_status', 'COMPLETED')
        plastic_level = data.get('plastic_level', 'LOW')
        blockage_pct = data.get('blockage_percent', 0.0)
        c.execute('''
        UPDATE field_surveys
        SET record_status = ?, plastic_level = ?, blockage_percent = ?, notes = ?
        WHERE survey_point_id = ?
        ''', (status, plastic_level, float(blockage_pct), data.get('notes', ''), point_id))
        conn.commit()
        conn.close()
        return jsonify({'message': f'Survey point {point_id} updated to {status}', 'status': 'success'})

    c.execute("SELECT * FROM field_surveys")
    surveys = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(surveys)

@app.route('/api/citizen-reports', methods=['GET', 'POST'])
def api_citizen_reports():
    conn = get_db()
    c = conn.cursor()
    if request.method == 'POST':
        data = request.get_json() or {}
        complaint_id = f"SVP-REP-2026-{random.randint(1000, 9999)}"
        c.execute('''
        INSERT INTO citizen_reports (complaint_id, name, phone, email, ward_no, street_location, gps_lat, gps_lng, issue_type, description, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Submitted')
        ''', (complaint_id, data.get('name', 'Anonymous Citizen'), data.get('phone', 'Not Provided'),
              data.get('email', 'Not Provided'), data.get('ward_no'), data.get('street_location'),
              data.get('gps_lat'), data.get('gps_lng'), data.get('issue_type'), data.get('description')))
        conn.commit()
        conn.close()
        return jsonify({'complaint_id': complaint_id, 'status': 'Submitted', 'message': 'Report received'})

    c.execute("SELECT * FROM citizen_reports ORDER BY created_at DESC")
    reports = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(reports)

@app.route('/api/maintenance', methods=['GET', 'POST', 'PATCH'])
def api_maintenance():
    conn = get_db()
    c = conn.cursor()
    if request.method in ['POST', 'PATCH']:
        data = request.get_json() or {}
        ticket_id = data.get('ticket_id')
        new_status = data.get('status')
        team = data.get('assigned_team')
        remarks = data.get('remarks')
        if ticket_id:
            c.execute('''
            UPDATE maintenance_records
            SET status = COALESCE(?, status),
                assigned_team = COALESCE(?, assigned_team),
                remarks = COALESCE(?, remarks),
                updated_at = CURRENT_TIMESTAMP
            WHERE ticket_id = ?
            ''', (new_status, team, remarks, ticket_id))
            conn.commit()
            conn.close()
            return jsonify({'message': f'Ticket {ticket_id} updated', 'status': 'success'})
        else:
            new_ticket = f"SVP-MAINT-{random.randint(200, 999)}"
            c.execute('''
            INSERT INTO maintenance_records (ticket_id, drain_id, ward_no, street, issue_description, cleaning_type, assigned_team, status, remarks)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Assigned', ?)
            ''', (new_ticket, data.get('drain_id'), data.get('ward_no'), data.get('street'),
                  data.get('issue_description'), data.get('cleaning_type'), data.get('assigned_team'), data.get('remarks')))
            conn.commit()
            conn.close()
            return jsonify({'ticket_id': new_ticket, 'status': 'Assigned', 'message': 'Maintenance ticket created'})

    c.execute("SELECT * FROM maintenance_records ORDER BY updated_at DESC")
    records = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(records)

@app.route('/api/ml-metrics')
def api_ml_metrics():
    return jsonify(METRICS)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    model_type = data.get('model_type', 'rf') # 'rf' or 'lr'
    pipeline_type = data.get('pipeline_type', 'causal') # 'causal' or 'leakage'

    try:
        rainfall_7 = float(data.get('rainfall_mm_7day', 45.0))
        rainfall_30 = float(data.get('rainfall_mm_30day', 180.0))
        plastic_score = int(data.get('plastic_accumulation_score', 2))
        drain_width = float(data.get('drain_width_m', 0.8))
        drain_depth = float(data.get('drain_depth_m', 0.9))
        drain_type = str(data.get('drain_type', 'OPEN')).upper()
        days_cleaning = int(data.get('days_since_cleaning', 15))
        prev_overflow = int(data.get('previous_overflow_count_1yr', 1))
        water_level = float(data.get('water_level_cm', 12.0))
        blockage_pct = float(data.get('blockage_percent', 35.0))

        if pipeline_type == 'leakage':
            row = pd.DataFrame([{
                'rainfall_mm_7day': rainfall_7,
                'rainfall_mm_30day': rainfall_30,
                'plastic_accumulation_score': plastic_score,
                'drain_width_m': drain_width,
                'drain_depth_m': drain_depth,
                'days_since_cleaning': days_cleaning,
                'previous_overflow_count_1yr': prev_overflow,
                'water_level_cm': water_level,
                'blockage_percent': blockage_pct,
                'drain_type': drain_type
            }])
            model = RF_LEAKAGE if model_type == 'rf' else LR_LEAKAGE
        else:
            row = pd.DataFrame([{
                'rainfall_mm_7day': rainfall_7,
                'rainfall_mm_30day': rainfall_30,
                'plastic_accumulation_score': plastic_score,
                'drain_width_m': drain_width,
                'drain_depth_m': drain_depth,
                'days_since_cleaning': days_cleaning,
                'previous_overflow_count_1yr': prev_overflow,
                'water_level_cm': water_level,
                'drain_type': drain_type
            }])
            model = RF_CAUSAL if model_type == 'rf' else LR_CAUSAL

        pred_code = model.predict(row)[0]
        probs = model.predict_proba(row)[0]

        risk_label = 'MEDIUM' if pred_code == 1 else 'LOW'
        confidence = round(float(probs[pred_code]) * 100, 1)

        disclaimer = (
            "This prediction is generated from a synthetic prototype dataset created for academic demonstration. "
            "It is not a real-time municipal flood prediction and has not been validated against complete real-world drainage observations."
        )

        high_risk_notice = (
            "Model Scope Notice: The synthetic training dataset contains 32 LOW and 18 MEDIUM instances with ZERO HIGH records. "
            "Consequently, the model cannot legitimately predict HIGH risk because no HIGH examples exist in the ground truth. "
            "To prevent false confidence, HIGH risk must be identified via physical municipal field survey."
        )

        return jsonify({
            'status': 'success',
            'predicted_risk_label': risk_label,
            'confidence_percent': confidence,
            'probabilities': {
                'LOW': round(float(probs[0]) * 100, 1),
                'MEDIUM': round(float(probs[1]) * 100, 1),
                'HIGH': 0.0
            },
            'model_used': f"{'Random Forest' if model_type == 'rf' else 'Logistic Regression'} ({'Causal Pipeline' if pipeline_type == 'causal' else 'Diagnostic Pipeline with Blockage %'})",
            'disclaimer': disclaimer,
            'high_risk_notice': high_risk_notice,
            'inputs_evaluated': {
                'rainfall_mm_7day': rainfall_7,
                'rainfall_mm_30day': rainfall_30,
                'plastic_accumulation_score': plastic_score,
                'drain_width_m': drain_width,
                'drain_depth_m': drain_depth,
                'drain_type': drain_type,
                'days_since_cleaning': days_cleaning,
                'previous_overflow_count_1yr': prev_overflow,
                'water_level_cm': water_level,
                'blockage_percent': blockage_pct if pipeline_type == 'leakage' else 'Excluded from Causal Pipeline (Target Leakage Mitigation)'
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
