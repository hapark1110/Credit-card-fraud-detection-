# FraudGuard — Django Fraud Detection Demo

Ứng dụng web phát hiện gian lận thẻ tín dụng sử dụng **Stacking Model**:
- Base: XGBoost + LightGBM
- Meta-learner: Logistic Regression
- Threshold: 0.30 | ROC-AUC ~0.98

---

## 🚀 Cài đặt & Chạy

### 1. Cài thư viện
```bash
pip install -r requirements.txt
```

### 2. Đặt model vào thư mục `models_store/`

Copy các file sau từ notebook Kaggle vào `fraud_django/models_store/`:

```
models_store/
  xgb_base.pkl       ← XGBoost model
  lgb_base.pkl       ← LightGBM model
  meta_model.pkl     ← Logistic Regression meta
  scaler.pkl         ← StandardScaler
  encoder.pkl        ← OrdinalEncoder
  feature_names.pkl  ← list 33 features
  metadata.json      ← threshold, metrics
  user_stats.csv     ← (optional) user statistics
```

> **Nếu không có file model:** Hệ thống sẽ tự động chạy ở chế độ **Simulation** (rule-based heuristic) và gắn nhãn `⚙ Simulation` trên giao diện.

### 3. Khởi động server
```bash
cd fraud_django
python manage.py runserver
```

Mở trình duyệt: **http://127.0.0.1:8000**

---

## 📋 Tính năng

### 🔍 Giao dịch đơn lẻ (`/single/`)
- Nhập 10 thông số giao dịch
- Hiển thị xác suất từng model: XGBoost, LightGBM, Stack
- Phân tích yếu tố rủi ro
- 4 mẫu thử nhanh (2 gian lận, 2 bình thường)

### 📊 Upload CSV & Dashboard (`/upload/` → `/dashboard/`)
- Upload CSV tối đa 500 dòng
- Vẽ 5 biểu đồ: theo danh mục, theo giờ, phân phối tiền, histogram xác suất, theo thứ
- Bảng Top 10 giao dịch gian lận nghi ngờ nhất

### 📡 Live Monitor (`/live/`)
- Server-Sent Events: tạo giao dịch ngẫu nhiên mỗi ~1.2 giây
- Feed real-time với màu sắc phân biệt gian lận / bình thường
- 4 biểu đồ cập nhật liên tục: xác suất, tỷ lệ gian lận, danh mục, cảnh báo

---

## 🗂 Cấu trúc project

```
fraud_django/
├── manage.py
├── requirements.txt
├── models_store/          ← đặt file .pkl vào đây
├── fraud_project/
│   ├── settings.py
│   └── urls.py
└── detector/
    ├── ml_engine.py       ← feature engineering + inference
    ├── views.py           ← Django views + API endpoints
    ├── urls.py
    ├── templates/detector/
    │   ├── base.html
    │   ├── index.html
    │   ├── single.html
    │   ├── upload.html
    │   ├── dashboard.html
    │   └── live.html
    └── static/detector/
        ├── css/main.css
        └── js/main.js
```

---

## 🔧 API Endpoints

| Method | URL | Mô tả |
|--------|-----|--------|
| POST | `/api/predict/` | Dự đoán 1 giao dịch (JSON) |
| POST | `/api/upload-analyze/` | Upload CSV, trả về dữ liệu dashboard |
| GET  | `/api/live-stream/` | SSE stream giao dịch ngẫu nhiên |
