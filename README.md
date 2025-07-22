# 🌎 Global Satellite Precipitation Estimation – PUnet

Access to high-quality, high-resolution, near-real-time precipitation data is essential for hydrologic research, weather forecasting, and disaster mitigation. While traditional tools like rain gauges and radar networks are effective, they face challenges such as sparse coverage in remote regions and high operational costs. In contrast, satellite-based approaches offer global coverage with higher temporal and spatial resolutions.

---

## 🛰️ Background

Satellite precipitation products such as:

- **Hydro Estimator (HE)**
- **IMERG (Integrated Multi-satellitE Retrievals for GPM)**
- **PERSIANN**

utilize both geostationary infrared (IR) and passive microwave (PMW) observations.  
- **PMW sensors** provide detailed atmospheric profiles but with higher latency.  
- **IR sensors** offer low latency, capturing cloud-top information, making them ideal for real-time estimation.

---

## 🔍 Introducing PERSIANN-UNet (PUnet)

Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs), have enhanced precipitation estimation.  
**PERSIANN-UNet (PUnet)** is a global, deep learning-based satellite precipitation estimation framework that leverages:

- IR satellite data
- Monthly climatological precipitation inputs

### 📌 Key Characteristics:
- **Spatial Resolution:** 0.04° × 0.04°  
- **Temporal Resolution:** 30 minutes  
- **Training:** Global, using full-disk images (no tiling needed)  
- **Preprocessing:** Image resizing, guided filtering, quantile mapping

PUnet is evaluated against **HE**, **IMERG**, and **PDIR-Now** over the 2022–2023 test period and shows notable improvements over traditional IR-based methods.

<div align="center">
  <img width="100%" alt="PUnet Workflow" src="https://github.com/user-attachments/assets/6da23218-57a4-498a-95db-399f271c5a60" />
  <em>Flowchart of the proposed PUnet workflow, showing main processing stages.</em>
</div>

---

## 👥 Development and Contributors

Developed at the **Center for Hydrometeorology and Remote Sensing (CHRS)** at **UC Irvine** by:

- Phu Nguyen
- Vu Dao
- Tu Ung
- Claudia Jimenez Arellano
- Kuolin Hsu
- Soroosh Sorooshian

With collaborators:

- **George J. Huffman** (NASA GSFC)  
- **F. Martin Ralph** (Scripps Institution of Oceanography, UC San Diego)

---

## ⚙️ Key Features

- **UNet Architecture:** Fully convolutional, tailored for spatial data
- **Full Global Processing:** Ingests entire image, minimizes edge artifacts
- **Monthly Training:** One model per calendar month for improved seasonal accuracy

### 🛠️ Post-Processing Steps

1. Temporarily downscale resolution to **0.234°** for model efficiency  
2. Apply **bias correction** using IMERG Final  
3. Restore high resolution (0.04°) using **guided filtering** with IR inputs

---

## 📊 Model Evaluation

PUnet outperforms IR-based products such as **PDIR-Now** and **HE** in:

- **Correlation**
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**

It aligns closely with **IMERG Final**, particularly in tropical and midlatitude regions.  
Although IMERG performs best over **CONUS**, PUnet remains competitive and reliable.

PUnet also effectively captures **extreme events**, maintaining low bias and RMSE.  
It aligns well with indices such as **R10mm**, **CDD**, and **CWD**, although it may slightly overestimate during very wet days.

<div align="center">
  <img width="100%" alt="Rainfall Event Louisiana" src="https://github.com/user-attachments/assets/0f69dd84-818b-4d8f-a54d-3586ce3e55dd" />
  <em>PUnet accurately captured heavy rainfall over Louisiana on May 7, 2025.</em>
</div>

---

## 🔮 Outlook and Applications

The PUnet framework is expected to become the next operational product in the **PERSIANN** family.  
Its exclusive use of globally available IR data positions it for reconstructing high-resolution **climate data records (CDRs)** dating back to **1983**.

PUnet outputs are visualized through the [**CHRS iRain system**](http://irain.eng.uci.edu/), making it useful for:

- Water resource management  
- Risk and disaster response  
- Climate monitoring

---

## 📌 Citation

If you use PUnet in your research, please cite:

> Dao, V., Nguyen, P., Arellano, C. J., Ung, T., Hsu, K., Sorooshian, S.,
