import json
import time
import random
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .ml_engine import predict, predict_batch, random_transaction, load_models, explain, _THRESHOLD


# ─── Load models on startup ────────────────────────────────────────────────
load_models()


# ─── Page views ───────────────────────────────────────────────────────────
def index(request):
    return render(request, 'detector/index.html')

def single_predict(request):
    return render(request, 'detector/single.html')

def upload_csv(request):
    return render(request, 'detector/upload.html')

def dashboard(request):
    return render(request, 'detector/dashboard.html')

def live_monitor(request):
    return render(request, 'detector/live.html')


# ─── API: Single prediction ────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(['POST'])
def api_predict(request):
    try:
        data = json.loads(request.body)
        result = predict(data)

        # Enrich with risk level
        p = result['meta_prob']
        if p >= 0.70:   risk = 'CRITICAL'
        elif p >= 0.50: risk = 'HIGH'
        elif p >= 0.30: risk = 'MEDIUM'
        else:           risk = 'LOW'
        result['risk_level'] = risk

        # Top risk factors
        factors = []
        amt       = float(data.get('amt', 0))
        hour      = int(data.get('hour', 12))
        category  = data.get('category', '')
        dist      = float(data.get('dist_km', data.get('distance', 0)))
        user_avg  = float(data.get('user_avg_amt', 65))
        city_pop  = int(data.get('city_pop', 50000))

        if hour < 5 or hour > 22:
            factors.append(f'Giao dịch lúc {hour}h (đêm khuya)')
        if category in ('shopping_net', 'misc_net'):
            factors.append(f'Danh mục rủi ro cao ({category})')
        if amt / (user_avg + 1e-9) > 3:
            factors.append(f'Số tiền bất thường (x{amt/user_avg:.1f} so với TB)')
        if dist > 200:
            factors.append(f'Khoảng cách xa ({dist:.0f} km)')
        if city_pop < 3000:
            factors.append('Thành phố dân số thấp')
        if amt > 500:
            factors.append(f'Giao dịch lớn (${amt:.2f})')
        if not factors:
            factors.append('Không phát hiện dấu hiệu bất thường')

        result['risk_factors'] = factors[:4]
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ─── API: Upload CSV & batch analyze ──────────────────────────────────────
@csrf_exempt
@require_http_methods(['POST'])
def api_upload_analyze(request):
    try:
        csv_file = request.FILES.get('file')
        if not csv_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        df = pd.read_csv(csv_file, nrows=500)  # limit for demo

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Parse datetime if available
        for col in ('trans_date_trans_time', 'trans_datetime', 'datetime', 'date'):
            if col in df.columns:
                try:
                    dt = pd.to_datetime(df[col])
                    df['hour']  = dt.dt.hour
                    df['dow']   = dt.dt.dayofweek
                    df['month'] = dt.dt.month
                except Exception:
                    pass
                break

        # Required columns fallback
        defaults = {
            'amt': 50, 'hour': 12, 'dow': 2, 'month': 6,
            'city_pop': 50000, 'age': 35, 'lat': 38.0, 'long': -90.0,
            'merch_lat': 38.0, 'merch_lon': -90.0,
            'gender': 'F', 'category': 'grocery_pos',
            'merchant': 'unknown', 'user_avg_amt': 65,
            'user_std_amt': 30, 'user_txn_count': 100,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        result_df = predict_batch(df)

        # ── Summary stats ────────────────────────────────────────
        total     = len(result_df)
        n_fraud   = int((result_df['verdict'] == 'FRAUD').sum())
        n_legit   = total - n_fraud
        fraud_rate= round(n_fraud / total * 100, 2)
        avg_prob  = round(float(result_df['meta_prob'].mean()), 4)
        total_amt = round(float(result_df['amt'].sum()), 2)
        fraud_amt = round(float(result_df.loc[result_df['verdict']=='FRAUD', 'amt'].sum()), 2)

        # ── Charts data ──────────────────────────────────────────
        # 1. Fraud by category
        cat_col = 'category' if 'category' in result_df.columns else None
        cat_data = {}
        if cat_col:
            grp = result_df.groupby(cat_col)['verdict'].apply(
                lambda x: (x == 'FRAUD').sum()
            ).sort_values(ascending=False).head(10)
            cat_data = grp.to_dict()

        # 2. Fraud by hour
        hour_col = 'hour' if 'hour' in result_df.columns else None
        hour_data = {}
        if hour_col:
            grp = result_df.groupby('hour')['verdict'].apply(
                lambda x: (x == 'FRAUD').sum()
            )
            hour_data = {str(k): int(v) for k, v in grp.to_dict().items()}

        # 3. Amount distribution buckets
        bins   = [0, 20, 50, 100, 200, 500, 1000, 99999]
        labels = ['<$20','$20-50','$50-100','$100-200','$200-500','$500-1k','>$1k']
        result_df['amt_bucket'] = pd.cut(result_df['amt'], bins=bins, labels=labels)
        amt_dist_all   = result_df['amt_bucket'].value_counts().reindex(labels, fill_value=0).to_dict()
        amt_dist_fraud = result_df[result_df['verdict']=='FRAUD']['amt_bucket'].value_counts().reindex(labels, fill_value=0).to_dict()

        # 4. Probability histogram
        prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        prob_labels = [f'{int(b*100)}-{int(prob_bins[i+1]*100)}%' for i, b in enumerate(prob_bins[:-1])]
        result_df['prob_bucket'] = pd.cut(result_df['meta_prob'], bins=prob_bins, labels=prob_labels, include_lowest=True)
        prob_hist = result_df['prob_bucket'].value_counts().reindex(prob_labels, fill_value=0).to_dict()

        # 5. Top fraud transactions
        top_fraud = (
            result_df[result_df['verdict'] == 'FRAUD']
            .nlargest(10, 'meta_prob')
            [['amt', 'category', 'hour', 'meta_prob', 'xgb_prob', 'lgb_prob']]
            .round(4)
            .to_dict(orient='records')
        )

        # 6. DOW distribution
        dow_names = ['T2','T3','T4','T5','T6','T7','CN']
        dow_data = {}
        if 'dow' in result_df.columns:
            grp = result_df.groupby('dow')['verdict'].apply(lambda x: (x=='FRAUD').sum())
            dow_data = {dow_names[k]: int(v) for k, v in grp.to_dict().items() if k < 7}

        return JsonResponse({
            'summary': {
                'total': total, 'n_fraud': n_fraud, 'n_legit': n_legit,
                'fraud_rate': fraud_rate, 'avg_prob': avg_prob,
                'total_amt': total_amt, 'fraud_amt': fraud_amt,
            },
            'charts': {
                'by_category': cat_data,
                'by_hour':     hour_data,
                'amt_dist':    {'all': amt_dist_all, 'fraud': amt_dist_fraud},
                'prob_hist':   prob_hist,
                'by_dow':      dow_data,
            },
            'top_fraud': top_fraud,
        })

    except Exception as e:
        import traceback
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)


# ─── API: Live monitor SSE stream ─────────────────────────────────────────
def api_live_stream(request):
    """Server-Sent Events endpoint — pushes one transaction per second."""

    def event_stream():
        count = 0
        while True:
            txn = random_transaction()
            result = predict(txn)

            p = result['meta_prob']
            if p >= 0.70:   risk = 'CRITICAL'
            elif p >= 0.50: risk = 'HIGH'
            elif p >= 0.30: risk = 'MEDIUM'
            else:           risk = 'LOW'

            payload = {**txn, **result, 'risk_level': risk, 'seq': count}
            data = json.dumps(payload, ensure_ascii=False)
            yield f"data: {data}\n\n"
            count += 1
            time.sleep(1.2)

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


# ─── API: Explain prediction ───────────────────────────────────────────────
@csrf_exempt
@require_http_methods(['POST'])
def api_explain(request):
    try:
        data = json.loads(request.body)
        result = explain(data)
        return JsonResponse(result)
    except Exception as e:
        import traceback
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)
