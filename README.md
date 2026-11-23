<div align="center">

# EnPreNiNet: Night Vision Multi-Object Detection

[![Status](https://img.shields.io/badge/Status-Initial_Setup_(v0.1.0)-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-Discover_AI_2026-green.svg)](https://doi.org/10.1007/s44163-026-02028-y)
[![Dataset](https://img.shields.io/badge/Dataset-BDD100K_(Night)-orange.svg)](https://bdd-data.berkeley.edu/)
[![Platform](https://img.shields.io/badge/Platform-Google_Colab-F9AB00.svg?logo=googlecolab&logoColor=white)]()
[![Classes](https://img.shields.io/badge/Classes-Car_%7C_Person_%7C_TrafficLight-purple.svg)]()

**Tái hiện và mô phỏng kiến trúc học sâu tổ hợp EnPreNiNet kết hợp bộ tiền xử lý tăng cường ảnh GHE Ensemble cho bài toán phát hiện đa đối tượng ban đêm trên xe tự hành.**

[Tổng quan](#overview) • [Kiến trúc Mô hình](#architecture) • [Dữ liệu & Phạm vi](#dataset--scope) • [Thông số Tham chiếu](#benchmark-reference) • [Trích dẫn](#citation)

</div>

---

## Overview

Trong điều kiện ánh sáng yếu hoặc ban đêm, các mô hình thị giác máy tính thường suy giảm độ chính xác nghiêm trọng do nhiễu cao, phân bố cường độ điểm ảnh không đồng đều và mất chi tiết giữa tiền cảnh với hậu cảnh. 

Repository này được xây dựng nhằm mô phỏng lại phương pháp đề xuất trong bài báo *"Improved multi object detection in night vision environment for autonomous vehicle using EnPreNiNet model"* (Ranjitha & Atham, 2026). Hệ thống kết hợp kỹ thuật cân bằng tương phản cục bộ (**GHE Ensemble**) cùng cơ chế hợp nhất quyết định từ hai kiến trúc phát hiện vật thể (**Enhanced Faster R-CNN** và **Enhanced YOLOv5**).

> **Note:** Dự án hiện đang ở giai đoạn khởi tạo (`v0.1.0 - Initial Setup`), mới thiết lập tài liệu đặc tả kỹ thuật và chưa bao gồm mã nguồn huấn luyện/suy luận.

---

## Architecture

Quy trình xử lý của **EnPreNiNet** bao gồm 3 giai đoạn nối tiếp nhau theo thiết kế của bài báo gốc:

```text
[Raw Night Frame] 
       │
       ▼  (Resize 640x640 + Grayscale)
┌──────────────────────────────────────────────────────────┐
│  Stage 1: GHE Ensemble Pre-processing (15-Step Pipeline) │
│  • Gaussian Filtering (Kernel 3x3, σ = 2)                │
│  • Sliding Window (W×W) Local Histogram Equalization     │
└─────────────────────────────┬────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Stage 2A: Faster R-CNN    │   │ Stage 2B: YOLOv5 (Big)    │
│ • Backbone: ResNet-150    │   │ • Backbone: CSPDarknet    │
│ • Neck/Head: RPN + RoI    │   │ • Neck/Head: PANet + YOLO │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │ (NMS + Thresholding)          │ (NMS + Thresholding)
              └───────────────┬───────────────┘
                              ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: EnPreNiNet Fusion Module                       │
│  • Combine Detections & Global NMS (Highest Confidence)  │
└─────────────────────────────┬────────────────────────────┘
                              ▼
             [Final Multi-Object Predictions]
```

### Chi tiết Kỹ thuật từng Module

| Giai đoạn | Thành phần | Thông số & Cơ chế hoạt động (Theo bài báo) |
| :--- | :--- | :--- |
| **1. Tiền xử lý** | **GHE Ensemble** | • Chuẩn hóa kích thước ảnh về **$640 \times 640$** và chuyển sang **Grayscale**.<br>• Lọc nhiễu bằng nhân Gaussian $3 \times 3$ ($\sigma = 2$) đã chuẩn hóa.<br>• Cân bằng lược đồ xám trên cửa sổ trượt $W \times W$: $PDF(x) = \frac{f(x)}{H}$, $C(x) = \sum_{j=0}^{x} PDF(j)$, $s'(x) = \text{round}(H \times C(x))$. |
| **2. Nhận diện** | **Dual Detectors** | • **Enhanced Faster R-CNN:** Mô hình hai giai đoạn (two-stage) dùng backbone **ResNet-150**, chuyên biệt cho các vật thể kích thước nhỏ.<br>• **Enhanced YOLOv5:** Mô hình một giai đoạn (one-stage) biến thể lớn (**Big / `yolov5l`**) dùng backbone **CSPDarknet** và **PANet** neck.<br>• Cả hai nhánh được huấn luyện **độc lập** trên cùng tập ảnh đã qua GHE Ensemble. |
| **3. Hợp nhất** | **Decision Fusion** | • Chạy **Non-Maximum Suppression (NMS)** riêng lẻ trên đầu ra của từng nhánh.<br>• Áp dụng **Thresholding** để lọc bỏ các khung bao có độ tin cậy thấp.<br>• Gộp danh sách dự đoán và giữ lại khung bao có điểm tin cậy cao nhất cho từng đối tượng thuộc mỗi lớp. |

---

## Dataset & Scope

* **Bộ dữ liệu gốc:** Trích xuất các khung hình điều kiện ban đêm (`timeofday == "night"`) từ tập **BDD100K** (tương tự cách bài báo xây dựng tập *BDD-Darko*).
* **Tỷ lệ phân chia dữ liệu:** **70%** Training / **20%** Validation / **10%** Testing.
* **Môi trường thực nghiệm dự kiến:** Huấn luyện và đánh giá trên **Google Colab**, sử dụng các công cụ được đề cập trong bài báo (`OpenCV-python`, `scikit-image`, `YOLOv5`, `Faster R-CNN`).
* **Phạm vi lớp đối tượng (3 Classes):** Thay vì 6 lớp mở rộng như bài báo gốc, bản mô phỏng này tập trung vào **3 lớp giao thông tiêu chuẩn** có sẵn nhãn bounding box trong BDD100K:
  * `0` — **Car**
  * `1` — **Person**
  * `2` — **TrafficLight**

---

## Benchmark Reference

Các chỉ số đánh giá được sử dụng trong nghiên cứu bao gồm **Precision**, **Recall**, **F1-Score**, và **mean Average Precision (mAP)** tại các mức ngưỡng **25%**, **50%**, và **75%**. Dưới đây là kết quả đối chứng được công bố trong bài báo gốc (trên tập BDD-Darko 6 lớp):

| Phương pháp / Cấu hình | Số lượng ảnh | Tổng số đối tượng | mAP |
| :--- | :---: | :---: | :---: |
| Baseline YOLOv5 (Ảnh gốc chưa tiền xử lý) | 20,000 | 79,413 | 33.4% |
| GHE-YOLOv5 (YOLOv5 + GHE Ensemble) | 25,000 | 90,126 | 65.3% |
| **EnPreNiNet (Proposed Ensemble Model)** | **27,000** | **105,030** | **76.3%** |

---

## Citation

Nếu bạn sử dụng ý tưởng kiến trúc từ nghiên cứu gốc, vui lòng trích dẫn bài báo của các tác giả:

```bibtex
@article{ranjitha2026enpreninet,
  title={Improved multi object detection in night vision environment for autonomous vehicle using EnPreNiNet model},
  author={Ranjitha, P. and Atham, Saira Banu},
  journal={Discover Artificial Intelligence},
  volume={6},
  pages={971},
  year={2026},
  publisher={Springer},
  doi={10.1007/s44163-026-02028-y}
}
```