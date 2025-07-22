🌍 Global Precipitation Estimation and Climate Data Reconstruction Using Deep Learning

Access to high-quality, high-resolution, near-real-time precipitation data is essential for hydrologic research, weather forecasting, and disaster mitigation. While traditional tools like rain gauges and radar networks are effective, they face challenges such as sparse coverage in remote regions and high operational costs. In contrast, satellite-based approaches offer global coverage with higher temporal and spatial resolutions.

Satellite precipitation products such as the Hydro Estimator (HE), IMERG (Integrated Multi-satellitE Retrievals for GPM), and PERSIANN utilize both geostationary infrared (IR) and passive microwave (PMW) observations. While PMW sensors provide detailed atmospheric profiles, they come with greater latency. IR sensors, though limited to cloud-top information, offer rapid refresh rates, making them attractive for real-time precipitation estimation.

🧩 Introducing PERSIANN-UNet (PUnet)
Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs), have significantly enhanced precipitation estimation capabilities. This project introduces PERSIANN-Unet (PUnet)—a global, deep learning-based satellite precipitation estimation framework that leverages IR data and monthly climatological precipitation inputs.

PUnet produces near-real-time estimates at 0.04° × 0.04° spatial and 30-minute temporal resolution. Unlike many models trained over small regions, PUnet is trained globally using full-disk images, eliminating the need for tiling and avoiding edge artifacts. Key preprocessing steps include image resizing, guided filtering, and quantile mapping, enabling effective global-scale training.

The model is evaluated against HE, IMERG, and PDIR-Now over an independent test period (2022–2023), and shows substantial improvements over IR-based baselines in global and regional performance.

<div align="center"> <img width="1800" height="1046" alt="PUnet Workflow" src="https://github.com/user-attachments/assets/6da23218-57a4-498a-95db-399f271c5a60" /> <br/> <em>Flowchart of the proposed PUnet workflow, showing main processing stages.</em> </div>
Development and Contributors
PUnet was developed by researchers at the Center for Hydrometeorology and Remote Sensing (CHRS) at the University of California, Irvine, including Phu Nguyen, Vu Dao, Tu Ung, Claudia Jimenez Arellano, Kuolin Hsu, and Soroosh Sorooshian. The project also involved collaboration with George J. Huffman (NASA GSFC) and F. Martin Ralph (Scripps Institution of Oceanography, UC San Diego).

The primary objective is to provide low-latency, globally available precipitation data with low computational overhead—crucial for applications in forecasting, climate monitoring, and hazard response.

🔑 Key Features of PUnet
UNet Architecture: Fully convolutional network tailored for spatial data.

Full Global Processing: Avoids tiling by ingesting the entire image, minimizing edge effects.

Monthly Model Training: Separate training for each calendar month enhances seasonal accuracy.

🧠 Post-Processing Pipeline:

Resolution is temporarily downscaled from 0.04° to 0.234° for training.

Bias correction using IMERG Final.

High-resolution restoration using guided filtering on original IR input.

🧪 Model Evaluation
Evaluation shows that PUnet outperforms IR-based products such as PDIR-Now and HE, particularly in metrics like correlation, RMSE, and MAE. It aligns well with IMERG Final, especially in tropical and midlatitude regions. While IMERG remains strongest over CONUS, PUnet provides competitive performance and robust detection capabilities.

PUnet also shows promise in capturing extreme precipitation events. For example, it performed well during a Western U.S. event, avoiding overestimation seen in other products and maintaining low bias and RMSE. For indices such as R10mm, CDD, and CWD, PUnet showed good agreement with ground truth, though it may slightly overestimate total rainfall from very wet days.

<div align="center"> <img width="2195" height="1261" alt="Rainfall Event Louisiana" src="https://github.com/user-attachments/assets/0f69dd84-818b-4d8f-a54d-3586ce3e55dd" /> <br/> <em>PUnet accurately captured heavy rainfall over Louisiana on May 7, 2025.</em> </div>
Outlook and Applications
The research team envisions PUnet becoming the next operational product in the PERSIANN family. Its exclusive use of globally available IR data also makes it a strong candidate for reconstructing high-resolution climate data records (CDRs) dating back to 1983.

PUnet outputs are accessible for visualization via the CHRS iRain system, promoting broad use in water resource management, risk assessment, and climate science.
