"""
ml_engine.py — Load & run the stacking model (XGBoost + LightGBM → LogReg)

Feature engineering mirrors the notebook exactly:
  - 33 features after encoding + time + distance + user stats
  - OrdinalEncoder for [merchant, category]
  - StandardScaler on all features
  - Threshold = 0.30
"""

import os
import json
import math
import random
import datetime
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# ─── Try to load real models ───────────────────────────────────────────────
_LOADED = False
_xgb = _lgb = _meta = _scaler = _encoder = _feature_names = None
_THRESHOLD = 0.30
_CATEGORIES = [
    'grocery_pos', 'shopping_net', 'gas_transport', 'entertainment',
    'food_dining', 'health_fitness', 'shopping_pos', 'misc_net',
    'misc_pos', 'travel', 'kids_pets', 'home', 'personal_care',
]
_MERCHANTS = []  # populated from encoder if available

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_store')

def load_models():
    global _LOADED, _xgb, _lgb, _meta, _scaler, _encoder, _feature_names, _THRESHOLD, _MERCHANTS

    if _LOADED:
        return _LOADED

    try:
        import joblib
        mdir = os.path.abspath(MODELS_DIR)

        xgb_pkl  = os.path.join(mdir, 'xgb_base.pkl')
        lgb_pkl  = os.path.join(mdir, 'lgb_base.pkl')
        meta_pkl = os.path.join(mdir, 'meta_model.pkl')
        sc_pkl   = os.path.join(mdir, 'scaler.pkl')
        enc_pkl  = os.path.join(mdir, 'encoder.pkl')
        fn_pkl   = os.path.join(mdir, 'feature_names.pkl')
        meta_json= os.path.join(mdir, 'metadata.json')

        required = [xgb_pkl, lgb_pkl, meta_pkl, sc_pkl, enc_pkl, fn_pkl]
        if not all(os.path.exists(p) for p in required):
            print("[ml_engine] Model files not found — running in SIMULATION mode")
            _LOADED = False
            return False

        _xgb          = joblib.load(xgb_pkl)
        _lgb          = joblib.load(lgb_pkl)
        _meta         = joblib.load(meta_pkl)
        _scaler       = joblib.load(sc_pkl)
        _encoder      = joblib.load(enc_pkl)
        _feature_names= joblib.load(fn_pkl)

        if os.path.exists(meta_json):
            with open(meta_json) as f:
                info = json.load(f)
            _THRESHOLD = info.get('threshold', 0.30)

        # merchant list from encoder
        if _encoder and hasattr(_encoder, 'categories_'):
            cats = _encoder.categories_[0]
            _MERCHANTS = [c.replace('fraud_', '') for c in cats[:200]]

        _LOADED = True
        print(f"[ml_engine] Models loaded OK — threshold={_THRESHOLD}")
        return True

    except Exception as e:
        print(f"[ml_engine] Load error: {e} — running in SIMULATION mode")
        _LOADED = False
        return False


# ─── Feature Engineering (mirrors notebook) ───────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _build_features(txn: dict) -> np.ndarray:
    """
    Build the 33-feature vector from a transaction dict.
    Works both with real encoder/scaler and in simulation mode.
    """
    # ── Raw fields ──────────────────────────────────────────────
    amt           = float(txn.get('amt', 50))
    hour          = int(txn.get('hour', 12))
    dow           = int(txn.get('dow', 2))          # day of week 0-6
    month         = int(txn.get('month', 6))
    city_pop      = int(txn.get('city_pop', 50000))
    age           = int(txn.get('age', 35))
    lat           = float(txn.get('lat', 38.0))
    lon           = float(txn.get('long', -90.0))
    merch_lat     = float(txn.get('merch_lat', lat + random.uniform(-2,2)))
    merch_lon     = float(txn.get('merch_lon', lon + random.uniform(-2,2)))
    gender_m      = 1 if txn.get('gender','F') == 'M' else 0
    category      = txn.get('category', 'grocery_pos')
    merchant      = txn.get('merchant', 'unknown_merchant')
    user_avg_amt  = float(txn.get('user_avg_amt', 65.0))
    user_txn_cnt  = int(txn.get('user_txn_count', 100))
    user_std_amt  = float(txn.get('user_std_amt', 40.0))

    # ── Derived features (notebook logic) ───────────────────────
    distance      = _haversine(lat, lon, merch_lat, merch_lon)
    log_amt       = math.log1p(amt)
    is_night      = 1 if (hour < 6 or hour > 22) else 0
    is_weekend    = 1 if dow >= 5 else 0
    amt_ratio     = amt / (user_avg_amt + 1e-9)
    log_city_pop  = math.log1p(city_pop)
    log_distance  = math.log1p(distance)

    # Category risk score (from notebook EDA)
    cat_risk = {
        'shopping_net': 0.09, 'misc_net': 0.08, 'grocery_pos': 0.01,
        'shopping_pos': 0.02, 'misc_pos': 0.03, 'gas_transport': 0.01,
        'food_dining': 0.01, 'entertainment': 0.04, 'health_fitness': 0.02,
        'kids_pets': 0.02, 'home': 0.02, 'personal_care': 0.01, 'travel': 0.05,
    }
    cat_fraud_rate = cat_risk.get(category, 0.03)

    # ── Encode categorical ──────────────────────────────────────
    if _LOADED and _encoder:
        try:
            enc_in = pd.DataFrame([[merchant, category]], columns=['merchant', 'category'])
            enc_out = _encoder.transform(enc_in)
            merchant_enc = float(enc_out[0, 0])
            category_enc = float(enc_out[0, 1])
        except Exception:
            merchant_enc = 0.0
            category_enc = float(_CATEGORIES.index(category)) if category in _CATEGORIES else 0.0
    else:
        merchant_enc = hash(merchant) % 700
        category_enc = float(_CATEGORIES.index(category)) if category in _CATEGORIES else 0.0

    # ── Assemble 33-feature vector (matches feature_names.pkl) ──
    # Order from notebook: amt, log_amt, hour, dow, month, is_night, is_weekend,
    # city_pop, log_city_pop, age, gender, distance, log_distance,
    # lat, long, merch_lat, merch_lon,
    # user_avg_amt, user_std_amt, user_txn_cnt, amt_ratio,
    # merchant_enc, category_enc, cat_fraud_rate,
    # + 9 one-hot category dummies (top categories)
    top_cats = ['grocery_pos','shopping_net','gas_transport','entertainment',
                'food_dining','health_fitness','shopping_pos','misc_net','misc_pos']
    cat_dummies = [1 if category == c else 0 for c in top_cats]

    features = [
        amt, log_amt,
        hour, dow, month,
        is_night, is_weekend,
        city_pop, log_city_pop,
        age, gender_m,
        distance, log_distance,
        lat, lon, merch_lat, merch_lon,
        user_avg_amt, user_std_amt, user_txn_cnt,
        amt_ratio,
        merchant_enc, category_enc, cat_fraud_rate,
        *cat_dummies,
    ]  # 24 + 9 = 33 features

    return np.array(features, dtype=np.float64).reshape(1, -1)


# ─── Simulation fallback (rule-based heuristic) ───────────────────────────
def _simulate_proba(txn: dict) -> Tuple[float, float, float]:
    """Rule-based probability simulation when models not loaded."""
    amt      = float(txn.get('amt', 50))
    hour     = int(txn.get('hour', 12))
    category = txn.get('category', 'grocery_pos')
    dist     = float(txn.get('dist_km', txn.get('distance', 5)))
    user_avg = float(txn.get('user_avg_amt', 65))
    city_pop = int(txn.get('city_pop', 50000))

    score = 0.03  # base fraud rate

    # High-risk category
    if category in ('shopping_net', 'misc_net'):   score += 0.12
    elif category == 'travel':                     score += 0.06
    elif category == 'entertainment':              score += 0.04

    # Night transaction
    if hour < 5 or hour > 22:                      score += 0.10

    # Amount anomaly
    ratio = amt / (user_avg + 1e-9)
    if ratio > 5:    score += 0.20
    elif ratio > 3:  score += 0.12
    elif ratio > 1.5: score += 0.05

    # High amount
    if amt > 500:    score += 0.10
    elif amt > 200:  score += 0.04

    # Small city — higher risk
    if city_pop < 2000:  score += 0.08
    elif city_pop < 5000: score += 0.04

    # Distance
    if dist > 300:   score += 0.12
    elif dist > 100: score += 0.06

    score = min(max(score, 0.001), 0.999)

    # Add noise imitating XGB vs LGB disagreement
    noise = random.gauss(0, 0.02)
    xgb_p = min(max(score + noise, 0.001), 0.999)
    lgb_p = min(max(score - noise, 0.001), 0.999)
    meta_p = (0.55 * xgb_p + 0.45 * lgb_p)
    meta_p = min(max(meta_p, 0.001), 0.999)
    return xgb_p, lgb_p, meta_p


# ─── Feature display names (Vietnamese) ──────────────────────────────────
_FEATURE_DISPLAY = {
    'amt':            ('Số tiền giao dịch', '$'),
    'log_amt':        ('Log(Số tiền)', ''),
    'hour':           ('Giờ giao dịch', 'h'),
    'dow':            ('Thứ trong tuần', ''),
    'month':          ('Tháng', ''),
    'is_night':       ('Giao dịch ban đêm', ''),
    'is_weekend':     ('Cuối tuần', ''),
    'city_pop':       ('Dân số thành phố', ''),
    'log_city_pop':   ('Log(Dân số)', ''),
    'age':            ('Tuổi chủ thẻ', ''),
    'gender_m':       ('Giới tính Nam', ''),
    'distance':       ('Khoảng cách merchant', 'km'),
    'log_distance':   ('Log(Khoảng cách)', ''),
    'lat':            ('Vĩ độ', ''),
    'lon':            ('Kinh độ', ''),
    'merch_lat':      ('Vĩ độ merchant', ''),
    'merch_lon':      ('Kinh độ merchant', ''),
    'user_avg_amt':   ('TB số tiền user', '$'),
    'user_std_amt':   ('Độ lệch chuẩn tiền user', '$'),
    'user_txn_cnt':   ('Số GD lịch sử', ''),
    'amt_ratio':      ('Tỷ lệ tiền / TB user', 'x'),
    'merchant_enc':   ('Mã merchant', ''),
    'category_enc':   ('Mã danh mục', ''),
    'cat_fraud_rate': ('Tỷ lệ gian lận danh mục', ''),
    'cat_grocery_pos':    ('Danh mục: grocery_pos', ''),
    'cat_shopping_net':   ('Danh mục: shopping_net', ''),
    'cat_gas_transport':  ('Danh mục: gas_transport', ''),
    'cat_entertainment':  ('Danh mục: entertainment', ''),
    'cat_food_dining':    ('Danh mục: food_dining', ''),
    'cat_health_fitness': ('Danh mục: health_fitness', ''),
    'cat_shopping_pos':   ('Danh mục: shopping_pos', ''),
    'cat_misc_net':       ('Danh mục: misc_net', ''),
    'cat_misc_pos':       ('Danh mục: misc_pos', ''),
}

_FEATURE_ORDER = [
    'amt', 'log_amt', 'hour', 'dow', 'month', 'is_night', 'is_weekend',
    'city_pop', 'log_city_pop', 'age', 'gender_m', 'distance', 'log_distance',
    'lat', 'lon', 'merch_lat', 'merch_lon',
    'user_avg_amt', 'user_std_amt', 'user_txn_cnt', 'amt_ratio',
    'merchant_enc', 'category_enc', 'cat_fraud_rate',
    'cat_grocery_pos', 'cat_shopping_net', 'cat_gas_transport', 'cat_entertainment',
    'cat_food_dining', 'cat_health_fitness', 'cat_shopping_pos', 'cat_misc_net', 'cat_misc_pos',
]


def explain(txn: dict) -> dict:
    """
    Explain why a transaction is classified as FRAUD.
    Returns top contributing features with SHAP values (real model)
    or rule-based scores (simulation).
    """
    load_models()

    if _LOADED:
        return _explain_shap(txn)
    else:
        return _explain_rules(txn)


def _explain_shap(txn: dict) -> dict:
    """SHAP-based explanation using XGBoost (most interpretable base model)."""
    try:
        import shap
        X = _build_features(txn)
        X_sc = _scaler.transform(X)

        # Use XGBoost TreeExplainer (fast, exact)
        explainer = shap.TreeExplainer(_xgb)
        shap_values = explainer.shap_values(X_sc)

        # shap_values shape: (1, n_features) for binary classification
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # class=1 (fraud)
        else:
            sv = shap_values[0]

        feat_names = _feature_names if _feature_names else _FEATURE_ORDER
        # Pair feature name → shap value
        pairs = list(zip(feat_names, sv))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        items = []
        for fname, shap_val in pairs_sorted[:10]:
            display_name, unit = _FEATURE_DISPLAY.get(fname, (fname, ''))
            # Get actual feature value
            feat_idx = feat_names.index(fname) if fname in feat_names else -1
            feat_val = float(X_sc[0, feat_idx]) if feat_idx >= 0 else 0.0
            # Unscale for display: raw value
            raw_idx = _FEATURE_ORDER.index(fname) if fname in _FEATURE_ORDER else feat_idx
            raw_val = float(X[0, raw_idx]) if raw_idx < X.shape[1] else feat_val

            direction = 'increase' if shap_val > 0 else 'decrease'
            items.append({
                'feature':      fname,
                'display_name': display_name,
                'shap_value':   round(float(shap_val), 4),
                'raw_value':    round(raw_val, 4),
                'unit':         unit,
                'direction':    direction,
                'abs_impact':   round(abs(float(shap_val)), 4),
            })

        base_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

        return {
            'method':     'shap',
            'base_value': round(base_value, 4),
            'features':   items,
            'summary':    _build_summary(txn, items),
        }

    except ImportError:
        # SHAP not installed — fall back to feature importance
        return _explain_feature_importance(txn)
    except Exception as e:
        print(f"[explain_shap] Error: {e}")
        return _explain_feature_importance(txn)


def _explain_feature_importance(txn: dict) -> dict:
    """Fallback: use XGBoost feature importance × feature deviation."""
    try:
        X = _build_features(txn)
        X_sc = _scaler.transform(X)

        feat_names = _feature_names if _feature_names else _FEATURE_ORDER
        importances = _xgb.feature_importances_  # gain-based

        # Score = importance × |scaled_value| (deviation from 0 = mean)
        scores = importances * np.abs(X_sc[0])

        pairs = list(zip(feat_names, scores, X[0]))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        items = []
        for fname, score, raw_val in pairs_sorted[:10]:
            display_name, unit = _FEATURE_DISPLAY.get(fname, (fname, ''))
            items.append({
                'feature':      fname,
                'display_name': display_name,
                'shap_value':   round(float(score), 4),
                'raw_value':    round(float(raw_val), 4),
                'unit':         unit,
                'direction':    'increase',
                'abs_impact':   round(float(score), 4),
            })

        return {
            'method':   'feature_importance',
            'features': items,
            'summary':  _build_summary(txn, items),
        }
    except Exception as e:
        return _explain_rules(txn)


def _explain_rules(txn: dict) -> dict:
    """Rule-based explanation for simulation mode — detailed and human-readable."""
    amt       = float(txn.get('amt', 50))
    hour      = int(txn.get('hour', 12))
    category  = txn.get('category', 'grocery_pos')
    dist      = float(txn.get('dist_km', txn.get('distance', 5)))
    user_avg  = float(txn.get('user_avg_amt', 65))
    city_pop  = int(txn.get('city_pop', 50000))
    dow       = int(txn.get('dow', 2))
    age       = int(txn.get('age', 35))
    amt_ratio = amt / (user_avg + 1e-9)
    user_cnt  = int(txn.get('user_txn_count', 100))

    CAT_RISK = {
        'shopping_net': 0.09, 'misc_net': 0.08, 'travel': 0.05,
        'entertainment': 0.04, 'shopping_pos': 0.02, 'misc_pos': 0.03,
        'grocery_pos': 0.01, 'gas_transport': 0.01, 'food_dining': 0.01,
        'health_fitness': 0.02, 'kids_pets': 0.02, 'home': 0.02, 'personal_care': 0.01,
    }

    items = []

    # 1. Amount ratio vs user average
    ar_score = min(max((amt_ratio - 1) * 0.08, 0), 0.35)
    items.append({
        'feature': 'amt_ratio', 'display_name': 'Tỷ lệ tiền / TB user',
        'shap_value': round(ar_score, 4),
        'raw_value': round(amt_ratio, 2), 'unit': 'x',
        'direction': 'increase' if ar_score > 0 else 'decrease',
        'abs_impact': round(ar_score, 4),
        'explain': f'Giao dịch ${amt:.2f} = {amt_ratio:.1f}x TB user (${user_avg:.2f})',
    })

    # 2. Night-time
    night_score = 0.18 if (hour < 5 or hour > 22) else (-0.05 if 9 <= hour <= 17 else 0.02)
    items.append({
        'feature': 'is_night', 'display_name': 'Giờ giao dịch',
        'shap_value': round(night_score, 4),
        'raw_value': hour, 'unit': 'h',
        'direction': 'increase' if night_score > 0 else 'decrease',
        'abs_impact': round(abs(night_score), 4),
        'explain': f'{"🌙 Đêm khuya " + str(hour) + "h — rủi ro cao" if hour < 5 or hour > 22 else "☀️ Ban ngày " + str(hour) + "h — bình thường"}',
    })

    # 3. Category risk
    cat_score = (CAT_RISK.get(category, 0.03) - 0.03) * 3.0
    items.append({
        'feature': 'category_enc', 'display_name': 'Danh mục giao dịch',
        'shap_value': round(cat_score, 4),
        'raw_value': CAT_RISK.get(category, 0.03), 'unit': '',
        'direction': 'increase' if cat_score > 0 else 'decrease',
        'abs_impact': round(abs(cat_score), 4),
        'explain': f'Danh mục "{category}" — tỷ lệ gian lận lịch sử {CAT_RISK.get(category,0.03)*100:.1f}%',
    })

    # 4. Transaction amount
    amt_score = min(max((amt - 100) / 1000, 0), 0.25)
    items.append({
        'feature': 'amt', 'display_name': 'Số tiền giao dịch',
        'shap_value': round(amt_score, 4),
        'raw_value': round(amt, 2), 'unit': '$',
        'direction': 'increase' if amt_score > 0 else 'decrease',
        'abs_impact': round(abs(amt_score), 4),
        'explain': f'Giá trị ${amt:.2f} — {"cao bất thường" if amt > 500 else "trong khoảng bình thường" if amt < 100 else "trung bình"}',
    })

    # 5. Distance
    dist_score = min(max((dist - 20) / 500, 0), 0.22)
    items.append({
        'feature': 'distance', 'display_name': 'Khoảng cách tới merchant',
        'shap_value': round(dist_score, 4),
        'raw_value': round(dist, 1), 'unit': 'km',
        'direction': 'increase' if dist_score > 0 else 'decrease',
        'abs_impact': round(abs(dist_score), 4),
        'explain': f'Merchant cách {dist:.1f} km — {"xa bất thường" if dist > 200 else "gần" if dist < 10 else "bình thường"}',
    })

    # 6. City population
    city_score = max((math.log1p(100000) - math.log1p(city_pop)) * 0.04, 0) if city_pop > 0 else 0
    items.append({
        'feature': 'city_pop', 'display_name': 'Dân số thành phố',
        'shap_value': round(city_score, 4),
        'raw_value': city_pop, 'unit': '',
        'direction': 'increase' if city_score > 0 else 'decrease',
        'abs_impact': round(abs(city_score), 4),
        'explain': f'Dân số {city_pop:,} — {"thành phố nhỏ, ít kiểm soát" if city_pop < 5000 else "thành phố lớn"}',
    })

    # 7. User transaction history
    hist_score = max((50 - user_cnt) * 0.002, 0)
    items.append({
        'feature': 'user_txn_cnt', 'display_name': 'Lịch sử giao dịch user',
        'shap_value': round(-hist_score, 4),
        'raw_value': user_cnt, 'unit': 'GD',
        'direction': 'decrease' if hist_score > 0 else 'increase',
        'abs_impact': round(hist_score, 4),
        'explain': f'{user_cnt} giao dịch lịch sử — {"ít lịch sử, khó xác minh" if user_cnt < 30 else "nhiều lịch sử, đáng tin"}',
    })

    # 8. Weekend
    wkend_score = 0.02 if dow >= 5 else -0.01
    items.append({
        'feature': 'is_weekend', 'display_name': 'Ngày trong tuần',
        'shap_value': round(wkend_score, 4),
        'raw_value': dow, 'unit': '',
        'direction': 'increase' if wkend_score > 0 else 'decrease',
        'abs_impact': round(abs(wkend_score), 4),
        'explain': f'{"Cuối tuần — tỷ lệ gian lận nhỉnh hơn" if dow >= 5 else "Ngày thường — bình thường"}',
    })

    items_sorted = sorted(items, key=lambda x: x['abs_impact'], reverse=True)

    return {
        'method':   'rules',
        'features': items_sorted,
        'summary':  _build_summary(txn, items_sorted),
    }


def _build_summary(txn: dict, items: list) -> list:
    """Generate 3–5 human-readable Vietnamese sentences explaining the prediction."""
    amt      = float(txn.get('amt', 50))
    hour     = int(txn.get('hour', 12))
    category = txn.get('category', 'grocery_pos')
    dist     = float(txn.get('dist_km', txn.get('distance', 5)))
    user_avg = float(txn.get('user_avg_amt', 65))
    city_pop = int(txn.get('city_pop', 50000))
    user_cnt = int(txn.get('user_txn_count', 100))

    sentences = []

    # Top contributing factor
    if items:
        top = items[0]
        sentences.append(f"Yếu tố ảnh hưởng nhất là <b>{top['display_name']}</b> "
                         f"(giá trị: {top['raw_value']}{top['unit']}).")

    # Amount anomaly
    ratio = amt / (user_avg + 1e-9)
    if ratio > 3:
        sentences.append(f"Số tiền <b>${amt:.2f}</b> gấp <b>{ratio:.1f}x</b> so với mức trung bình "
                         f"<b>${user_avg:.2f}</b> của chủ thẻ — đây là dấu hiệu bất thường rõ ràng.")
    elif ratio > 1.5:
        sentences.append(f"Số tiền ${amt:.2f} cao hơn {ratio:.1f}x mức trung bình ${user_avg:.2f} của chủ thẻ.")

    # Night / time
    if hour < 5 or hour > 22:
        sentences.append(f"Giao dịch thực hiện lúc <b>{hour}:00</b> — khung giờ đêm khuya có tỷ lệ gian lận cao hơn đáng kể.")

    # Category
    HIGH_RISK_CATS = {'shopping_net', 'misc_net', 'travel'}
    if category in HIGH_RISK_CATS:
        sentences.append(f"Danh mục <b>{category}</b> thuộc nhóm rủi ro cao theo dữ liệu lịch sử.")

    # Distance
    if dist > 200:
        sentences.append(f"Khoảng cách từ địa chỉ chủ thẻ tới merchant là <b>{dist:.0f} km</b> — "
                         f"bất thường so với thói quen giao dịch thông thường.")
    elif dist > 50:
        sentences.append(f"Merchant cách địa chỉ chủ thẻ {dist:.0f} km — hơi xa so với bình thường.")

    # City size
    if city_pop < 2000:
        sentences.append(f"Giao dịch tại khu vực dân số thấp (<b>{city_pop:,} người</b>), "
                         f"nơi hệ thống kiểm soát gian lận thường kém chặt hơn.")

    # User history
    if user_cnt < 20:
        sentences.append(f"Chủ thẻ chỉ có <b>{user_cnt}</b> giao dịch lịch sử — ít dữ liệu để xác minh hành vi thông thường.")

    return sentences[:5]


# ─── Public API ────────────────────────────────────────────────────────────
def predict(txn: dict) -> dict:
    """
    Main entry point.
    Returns: {xgb_prob, lgb_prob, meta_prob, verdict, threshold, mode}
    """
    load_models()

    if _LOADED:
        try:
            X = _build_features(txn)
            X_sc = _scaler.transform(X)
            xgb_p = float(_xgb.predict_proba(X_sc)[:, 1][0])
            lgb_p = float(_lgb.predict_proba(X_sc)[:, 1][0])
            meta_X = np.column_stack([[xgb_p], [lgb_p]])
            meta_p = float(_meta.predict_proba(meta_X)[:, 1][0])
            mode = 'model'
        except Exception as e:
            print(f"[ml_engine] Predict error: {e} — falling back to simulation")
            xgb_p, lgb_p, meta_p = _simulate_proba(txn)
            mode = 'simulation'
    else:
        xgb_p, lgb_p, meta_p = _simulate_proba(txn)
        mode = 'simulation'

    verdict = 'FRAUD' if meta_p >= _THRESHOLD else 'LEGIT'

    return {
        'xgb_prob':  round(xgb_p, 4),
        'lgb_prob':  round(lgb_p, 4),
        'meta_prob': round(meta_p, 4),
        'verdict':   verdict,
        'threshold': _THRESHOLD,
        'mode':      mode,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict for a DataFrame. Returns df with result columns appended."""
    results = []
    for _, row in df.iterrows():
        txn = row.to_dict()
        # Map column names from CSV
        if 'trans_date_trans_time' in txn:
            try:
                dt = pd.to_datetime(txn['trans_date_trans_time'])
                txn['hour']  = dt.hour
                txn['dow']   = dt.dayofweek
                txn['month'] = dt.month
            except Exception:
                pass
        r = predict(txn)
        results.append(r)
    res_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)


# ─── Random transaction generator (for Live Monitor) ──────────────────────
_NAMES_FIRST = ['Nguyen','Tran','Le','Pham','Hoang','Do','Bui','Vu','Dang','Ngo']
_NAMES_LAST  = ['Van An','Thi Lan','Van Huy','Thi Mai','Van Duc','Thi Thu','Van Long']
_CITIES = [
    ('Ha Noi', 10000000, 21.03, 105.85),
    ('Ho Chi Minh', 9000000, 10.78, 106.70),
    ('Da Nang', 1200000, 16.05, 108.21),
    ('Hai Phong', 2000000, 20.86, 106.68),
    ('Hue', 380000, 16.46, 107.59),
    
    # ─── MIỀN BẮC & BẮC TRUNG BỘ ─────────────────────────────────────
    ('Quang Ninh', 1400000, 20.95, 107.07),    # Hạ Long, Cẩm Phả (Kinh tế, du lịch mạnh)
    ('Vinh', 350000, 18.67, 105.68),          # Nghệ An (Trung tâm Bắc Trung Bộ)
    ('Thanh Hoa', 400000, 19.81, 105.79),     # Thành phố Thanh Hóa
    ('Bac Ninh', 270000, 21.18, 106.07),      # Thủ phủ công nghiệp miền Bắc
    ('Thai Nguyen', 360000, 21.59, 105.84),   # Thành phố giáo dục, công nghiệp
    
    # ─── MIỀN TRUNG & TÂY NGUYÊN ───────────────────────────────────
    ('Nha Trang', 420000, 12.25, 109.19),     # Khánh Hòa (Du lịch biển sầm uất)
    ('Quy Nhon', 290000, 13.78, 109.22),      # Bình Định (Trung tâm biển miền Trung)
    ('Phan Thiet', 230000, 10.93, 108.10),    # Bình Thuận
    
    # ─── MIỀN NAM & ĐỒNG BẰNG SÔNG CỬU LONG ─────────────────────────
    ('Vung Tau', 360000, 10.35, 107.08),      # Bà Rịa - Vũng Tàu (Dầu khí, du lịch)
    ('Thu Dau Mot', 350000, 10.98, 106.66),   # Bình Dương (Trọng điểm công nghiệp phía Nam)
    ('Ca Mau', 250000, 9.18, 105.15),         # Điểm cực Nam Tổ quốc
    
]

def random_transaction() -> dict:
    """Generate a realistic random credit card transaction."""
    now = datetime.datetime.now()
    hour = now.hour
    dow  = now.weekday()

    city_name, city_pop, lat, lon = random.choice(_CITIES)

    # Bias toward high-risk patterns occasionally
    is_fraud_scenario = random.random() < 0.15  # ~15% fraud scenarios

    if is_fraud_scenario:
        category = random.choice(['shopping_net', 'misc_net', 'travel', 'entertainment'])
        hour     = random.choice(list(range(0, 5)) + list(range(23, 24)))
        amt      = random.uniform(200, 1200)
        merch_lat = lat + random.uniform(-5, 5)
        merch_lon = lon + random.uniform(-5, 5)
        city_pop  = random.choice([800, 1200, 500, 1500])
        user_avg_amt = random.uniform(30, 80)
    else:
        category = random.choice(['grocery_pos', 'food_dining', 'gas_transport',
                                   'health_fitness', 'shopping_pos', 'kids_pets'])
        amt      = random.uniform(5, 150)
        merch_lat = lat + random.uniform(-0.1, 0.1)
        merch_lon = lon + random.uniform(-0.1, 0.1)
        user_avg_amt = random.uniform(40, 120)

    distance = _haversine(lat, lon, merch_lat, merch_lon)

    gender = random.choice(['M', 'F'])
    age    = random.randint(22, 70)

    merchant_prefix = 'fraud_' if random.random() < 0.1 else ''
    merchant_names = ['SuperMart', 'FoodCourt', 'TechStore', 'GasStation',
                       'OnlineShop', 'Restaurant', 'Pharmacy', 'GymCenter']
    merchant = merchant_prefix + random.choice(merchant_names)

    return {
        'trans_num':      f'TXN{random.randint(100000,999999)}',
        'trans_datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
        'merchant':       merchant,
        'category':       category,
        'amt':            round(amt, 2),
        'first':          random.choice(_NAMES_FIRST),
        'last':           random.choice(_NAMES_LAST),
        'gender':         gender,
        'city':           city_name,
        'city_pop':       city_pop,
        'lat':            round(lat, 4),
        'long':           round(lon, 4),
        'merch_lat':      round(merch_lat, 4),
        'merch_lon':      round(merch_lon, 4),
        'dist_km':        round(distance, 2),
        'age':            age,
        'hour':           hour,
        'dow':            dow,
        'month':          now.month,
        'user_avg_amt':   round(user_avg_amt, 2),
        'user_std_amt':   round(user_avg_amt * 0.4, 2),
        'user_txn_count': random.randint(10, 500),
    }
