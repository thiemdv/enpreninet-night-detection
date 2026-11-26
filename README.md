<div align="center">

# EnPreNiNet: Night Vision Multi-Object Detection

[![Status](https://img.shields.io/badge/Status-Pipeline_Ready_(v1.0.0)-brightgreen.svg)]()
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

> **Note:** Toàn bộ pipeline `F01–F08` (`src/01_prepare_bdd.py` đến `src/06_evaluate.py`, bộ kiểm thử bất biến `tests/test_invariants.py`, và `notebooks/Colab_Runner.ipynb`) đã hoàn thiện và được kiểm chứng offline trên CPU; sẵn sàng chạy huấn luyện/đánh giá đầy đủ E0–E5 trên Google Colab GPU.

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

## Implementation Roadmap (F01–F08) & Experiment Matrix (E0–E5)

| Feature | Module | Mô tả |
| :--- | :--- | :--- |
| **F01** | `configs/config.yaml`, `requirements.txt`, `tests/test_invariants.py` | Khởi tạo cấu hình trung tâm và bộ kiểm thử bất biến |
| **F02** | `src/01_prepare_bdd.py` | Lọc ảnh ban đêm BDD100K (3 lớp), chia 70/20/10, xuất YOLO `.txt` & COCO `.json` |
| **F03** | `src/02_ghe_ensemble.py` | Tiền xử lý GHE Ensemble 15 bước (3 chế độ: `fig4_literal`, `fig4_normalized`, `standard_local_he`) |
| **F04** | `src/03_train_infer_yolo.py` | Huấn luyện & suy luận YOLOv5l theo schema dự đoán thống nhất (`xyxy`, nhãn `0..2`) |
| **F05** | `src/04_train_infer_frcnn.py` | Huấn luyện & suy luận Faster R-CNN (`resnet_fpn_backbone("resnet152")`), map nhãn `1..3 -> 0..2` |
| **F06** | `src/05_fusion.py` | Hợp nhất mức điểm số EnPreNiNet (intra-NMS, ngưỡng tin cậy, bật/tắt global NMS) |
| **F07** | `src/06_evaluate.py` | Đánh giá Precision, Recall, F1, AP@50/mAP@50, AP@75/mAP@75 và báo cáo ablation |
| **F08** | `notebooks/Colab_Runner.ipynb` | Notebook điều phối mỏng chạy ma trận thực nghiệm **E0–E5** trên Google Colab & lưu Google Drive |

**Ma trận thực nghiệm (E0–E5):**
- **GPU Pilot (Khuyến nghị trước full training):** 200 ảnh ban đêm thật, 2 epochs để kiểm chứng bộ nhớ GPU T4 (`batch_size=2` cho Faster R-CNN ResNet-152), loss hữu hạn, `dataset_fingerprint` và đường ống E0–E5 đầu cuối.
- **E0:** Baseline YOLOv5l (`v7.0`, COCO pretrained) trên ảnh ban đêm gốc (chưa GHE).
- **E1:** GHE-YOLOv5l trên ảnh `standard_local_he` (hỗ trợ đối chứng mở rộng với `fig4_literal` & `fig4_normalized`).
- **E2:** Baseline Faster R-CNN ResNet-152 (ImageNet-pretrained backbone + detection head khởi tạo mới) trên ảnh ban đêm gốc.
- **E3:** GHE-Faster R-CNN ResNet-152 trên ảnh `standard_local_he`.
- **E4:** EnPreNiNet Fusion (không dùng Global NMS — ablation `global_nms_enabled: false`).
- **E5:** **EnPreNiNet Full** (GHE Ensemble + YOLOv5l + Faster R-CNN ResNet-152 + Global NMS tại `tau in {0.25, 0.50, 0.75}`).

> **Ghi chú học thuật & Giới hạn triển khai:**
> - Bộ tiền xử lý GHE tái hiện thuật toán 15 bước tại **Hình 4** của bài báo (`standard_local_he` làm cấu hình chính, `fig4_literal`/`fig4_normalized` làm đối chứng), không bao gồm các nhánh mở rộng của Hình 6.
> - Bộ đánh giá `src/06_evaluate.py` sử dụng quy ước khớp 1-1 kiểu **Pascal VOC** (greedy best-IoU) kết hợp nội suy PR toàn điểm (**all-point AP@50/AP@75**, loại lớp có `0` GT khỏi trung bình `mAP`), khác với COCO 101-point mAP@[.50:.95].
> - Khi huấn luyện thật, `YOLOv5l` khởi tạo từ trọng số detector pretrained COCO trong khi `Faster R-CNN` khởi tạo từ backbone `ResNet-152` pretrained ImageNet; sự khác biệt khởi tạo này cần được ghi nhận khi phân tích kết quả thực nghiệm.

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