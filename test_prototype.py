import unittest
import json
import sqlite3
import pandas as pd
from app import app, get_db

class TestSrivilliputhurPrototype(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    # 1. Database Integrity Tests
    def test_database_record_counts(self):
        conn = get_db()
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM wards")
        self.assertEqual(c.fetchone()[0], 33, "Wards count must be exactly 33")

        c.execute("SELECT COUNT(*) FROM incidents")
        self.assertEqual(c.fetchone()[0], 12, "Historical incidents count must be exactly 12")

        c.execute("SELECT COUNT(*) FROM drains")
        self.assertEqual(c.fetchone()[0], 1, "Verified drains count must be exactly 1 (Tender W32)")

        c.execute("SELECT COUNT(*) FROM ml_dataset")
        self.assertEqual(c.fetchone()[0], 50, "Synthetic ML dataset rows must be exactly 50")

        c.execute("SELECT COUNT(*) FROM field_surveys")
        self.assertEqual(c.fetchone()[0], 25, "Field survey points must be exactly 25")

        c.execute("SELECT COUNT(*) FROM rainfall_portals")
        self.assertEqual(c.fetchone()[0], 4, "Verified rainfall portals must be exactly 4")

        c.execute("SELECT COUNT(*) FROM plastic_info")
        self.assertEqual(c.fetchone()[0], 3, "Plastic info entries must be exactly 3")

        c.execute("SELECT COUNT(*) FROM municipality_stats")
        self.assertEqual(c.fetchone()[0], 7, "Municipality stats must be exactly 7")
        conn.close()

    # 2. Page HTTP Status Tests (All 16 routes)
    def test_all_web_routes(self):
        routes = [
            '/',
            '/dashboard',
            '/prediction',
            '/map',
            '/drains',
            '/rainfall',
            '/incidents',
            '/plastic-waste',
            '/report',
            '/maintenance',
            '/field-survey',
            '/analytics',
            '/sources',
            '/methodology',
            '/limitations',
            '/admin'
        ]
        for r in routes:
            res = self.client.get(r)
            self.assertEqual(res.status_code, 200, f"Route {r} should return status 200 OK")
            # Verify academic prototype banner exists on page
            self.assertIn(b"ACADEMIC PROTOTYPE NOTICE", res.data, f"Route {r} missing academic disclaimer")

    # 3. REST API & ML Inference Tests
    def test_api_stats(self):
        res = self.client.get('/api/stats')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['total_wards'], 33)
        self.assertEqual(data['historical_incidents'], 12)
        self.assertEqual(data['verified_drainage_records'], 1)
        self.assertEqual(data['planned_field_survey_points'], 25)
        self.assertEqual(data['synthetic_ml_records'], 50)

    def test_api_predict_causal_pipeline(self):
        payload = {
            'model_type': 'rf',
            'pipeline_type': 'causal',
            'rainfall_mm_7day': 60.0,
            'rainfall_mm_30day': 220.0,
            'plastic_accumulation_score': 3,
            'drain_width_m': 0.6,
            'drain_depth_m': 0.8,
            'drain_type': 'OPEN',
            'days_since_cleaning': 35,
            'previous_overflow_count_1yr': 2,
            'water_level_cm': 18.0
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data['predicted_risk_label'], ['LOW', 'MEDIUM'])
        self.assertIn('synthetic prototype dataset created for academic demonstration', data['disclaimer'])
        self.assertIn('Model Scope Notice', data['high_risk_notice'])

    def test_api_predict_logistic_regression(self):
        payload = {
            'model_type': 'lr',
            'pipeline_type': 'causal',
            'rainfall_mm_7day': 20.0,
            'rainfall_mm_30day': 80.0,
            'plastic_accumulation_score': 0,
            'drain_width_m': 1.2,
            'drain_depth_m': 1.2,
            'drain_type': 'COVERED',
            'days_since_cleaning': 5,
            'previous_overflow_count_1yr': 0,
            'water_level_cm': 4.0
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['predicted_risk_label'], 'LOW')

    def test_citizen_report_submission(self):
        payload = {
            'name': 'Test Citizen',
            'phone': '9988776655',
            'ward_no': 5,
            'street_location': 'Test Street Road',
            'issue_type': 'Blocked Drain',
            'description': 'Plastic bottle choking the mouth of the roadside culvert.'
        }
        res = self.client.post('/api/citizen-reports', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['complaint_id'].startswith('SVP-REP-2026-'))
        self.assertEqual(data['status'], 'Submitted')

    def test_maintenance_ticket_creation(self):
        payload = {
            'drain_id': 'SVP-SYN-D005',
            'ward_no': 12,
            'street': 'Bazaar Lane',
            'issue_description': 'Silt accumulation',
            'cleaning_type': 'Manual De-silting',
            'assigned_team': 'Sanitation Squad 3',
            'remarks': 'Clear before monsoon spell'
        }
        res = self.client.post('/api/maintenance', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['ticket_id'].startswith('SVP-MAINT-'))

if __name__ == '__main__':
    unittest.main()
