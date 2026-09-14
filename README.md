# Stock Analytics Platform 📈

Hệ thống tự động thu thập dữ liệu giá giao dịch cổ phiếu từ CafeF trên cả 3 sàn (HOSE, HNX, UPCOM), tính toán các tín hiệu kỹ thuật theo chiến lược RSI và trực quan hóa qua Dashboard tương tác để hỗ trợ ra quyết định đầu tư.

---

## 📌 Tính năng chính

- **Thu thập dữ liệu tự động:** Cào dữ liệu lịch sử giá và khối lượng giao dịch từ nguồn CafeF cho các mã cổ phiếu trên 3 sàn HOSE, HNX và UPCOM.
- **Phân tích kỹ thuật & Tín hiệu giao dịch:** Tính toán chỉ số RSI (Relative Strength Index) và tự động nhận diện các vùng quá mua/quá bán (Overbought/Oversold) để phát tín hiệu mua/bán.
- **Trực quan hóa (Dashboard):** Giao diện tương tác trực quan giúp theo dõi biểu đồ nến, biến động RSI và danh sách tín hiệu lọc theo thời gian thực.
- **Pipeline tự động hóa:** Chạy toàn bộ quy trình từ tải dữ liệu, tính toán tín hiệu đến khởi chạy dashboard chỉ qua 1 dòng lệnh hoặc script batch.

---

## 📁 Cấu trúc thư mục

stock-analytics-platform/
├── outputs/             # Chứa dữ liệu thô và kết quả phân tích (.csv)
├── crawl_data.py        # Script cào và trích xuất dữ liệu giá từ CafeF
├── signal_rsi.py        # Logic tính toán chỉ số RSI và sinh tín hiệu giao dịch
├── dashboard.py         # Ứng dụng hiển thị biểu đồ & bảng điều khiển
├── main.py              # File điều phối luồng thực thi (pipeline orchestrator)
├── run_pipeline.bat     # Script chạy tự động trên môi trường Windows
├── .gitattributes       # Cấu hình Git LFS quản lý các file dữ liệu lớn
└── README.md            # Tài liệu hướng dẫn dự án

---

Chiến lược RSI:
- Vùng quá bán --> RSI < 30, cân nhắc mua vào
- Vùng quá mua --> RSI > 70, cân nhắc chốt lời
- Phân kỳ RSI: Cảnh báo đảo chiều xu hướng dựa trên sự lệch pha giữa đường giá và đường chỉ 
